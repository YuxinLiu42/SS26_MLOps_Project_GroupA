"""Inference optimization benchmarks for the fine-tuned model.

Two Typer commands, both run on a CUDA GPU (4-bit needs bitsandbytes/CUDA), on
Vertex via ``cloud/run_optimize.sh``. Results saved as JSON.

``benchmark`` loads the adapter in a few configurations and measures load time,
peak GPU memory, and per-generate latency on a fixed set of test samples:

- ``bf16``            : the serving default.
- ``int4``           : 4-bit weights via bitsandbytes (QLoRA-style) — CUDA only.
- ``bf16+compile``   : ``torch.compile`` of the bf16 model.

``prune-sweep`` merges the LoRA adapter into the base, then global
magnitude-prunes the Linear weights to several sparsity levels and measures test
accuracy at each. Latency is reported too, but it stays flat by design:
unstructured pruning only zeros weights, so the dense kernels still do the full
matmul — there is no speedup without sparse kernels. The deliverable is the
accuracy-vs-sparsity curve, not a latency win.
"""

import gc
import json
import logging
import resource
import time
from pathlib import Path

import torch
import typer
from peft import PeftModel
from rich.logging import RichHandler
from torch.nn.utils import prune
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from scipali.data.data import DATASET_SUBSET, PROCESSED_DATA_DIR, DataModule
from scipali.models.model import MODEL_NAME, extract_answer_letter

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
log = logging.getLogger(__name__)
app = typer.Typer(help="Benchmark quantization/compilation of the fine-tuned model.")


def _sample_batch(adapter_dir: Path, n: int):
    """Build one eval batch of ``n`` samples plus the matching processor."""
    processor = AutoProcessor.from_pretrained(str(adapter_dir))
    data = DataModule(
        processed_dir=PROCESSED_DATA_DIR,
        subset=DATASET_SUBSET,
        processor=processor,
        batch_size=n,
        num_workers=0,
    )
    data.setup()
    batch = next(iter(data.test_dataloader()))
    return batch


def _load(adapter_dir: Path, mode: str):
    """Load base + adapter in the requested mode; return (model, load_seconds)."""
    kwargs: dict = {"torch_dtype": torch.bfloat16}
    if mode == "int4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )
        kwargs["device_map"] = "cuda"
    t = time.time()
    base = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_NAME, **kwargs)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    if mode != "int4":
        model = model.to("cuda")
    if mode == "bf16+compile":
        model = torch.compile(model)  # type: ignore[assignment]
    model.eval()
    return model, time.time() - t


def prune_linear_layers(model: torch.nn.Module, amount: float) -> float:
    """Global L1-unstructured prune of every ``nn.Linear`` weight to ``amount``.

    Uses a single global magnitude threshold across all Linear weights (so the
    sparsity budget is allocated adaptively, like ``prune.global_unstructured``).
    The threshold is found from a fine HISTOGRAM of ``|w|`` rather than by pooling
    every weight: pooling all ~3B weights into a float32 buffer (plus the kthvalue
    selection workspace) peaked ~49GB of host RAM and forced a 64GB machine. The
    histogram is O(bins) memory and only ever holds one layer's ``|w|`` at a time,
    so the sweep fits a 32GB host. Each layer is masked with ``|w| > threshold``
    and the mask is baked into a dense zeroed weight (``prune.remove``).
    (``prune-finetune`` keeps these zeros frozen during training by masking the
    gradient -- see ``_mask_pruned_grads`` -- not a live ``prune`` reparametrization,
    which would ~triple weight memory and OOM the L4.)

    Returns the ACHIEVED sparsity (fraction with ``|w| <= threshold``), computed
    exactly from the zeroed weights -- the histogram's bin width only perturbs the
    threshold by ``max|w| / bins``, so achieved tracks ``amount`` to well within
    1%. The sweep records this achieved value, so the curve is plotted against
    real sparsity. ``amount == 0`` is a no-op.
    """
    linears = [(m, "weight") for m in model.modules() if isinstance(m, torch.nn.Linear)]
    total = sum(m.weight.numel() for m, _ in linears)
    # hi == 0 only for an all-zero model (nothing meaningful to prune; the achieved
    # calc below already reports it as fully sparse).
    hi = max((m.weight.detach().abs().max().item() for m, _ in linears), default=0.0)
    if amount > 0 and hi > 0:
        # Global magnitude cutoff via a histogram of |w| (O(bins) memory): sum
        # per-layer histograms, freeing each layer's |w| copy before the next --
        # no multi-GB pool, no kthvalue workspace, so peak host RAM stays at
        # ~one layer's worth.
        #
        # Counts are EXACT int64: torch.histc returns float32, which rounds bin
        # counts above 2**24 (~16.7M) -- a bf16 magnitude spike can exceed that and
        # skew the threshold. Bucketize + bincount (int64) avoids that entirely.
        bins = 1_000_000
        hist = torch.zeros(bins, dtype=torch.int64)
        for module, _ in linears:
            w = module.weight.detach().abs().float().flatten()
            idx = torch.clamp((w / hi * bins).long(), max=bins - 1)
            hist += torch.bincount(idx, minlength=bins).cpu()
            del w, idx
        # Smallest bin whose cumulative count reaches k -> prune |w| <= its edge.
        k = int(amount * total)
        cumulative = torch.cumsum(hist, dim=0)  # int64 -> exact, no overflow at 3B
        cutoff_bin = int(torch.searchsorted(cumulative, torch.tensor(k)).item())
        threshold = (cutoff_bin + 1) / bins * hi  # right edge of the cutoff bin
        for module, name in linears:
            mask = (module.weight.detach().abs() > threshold).to(module.weight.dtype)
            prune.custom_from_mask(module, name, mask)  # 1 = keep, 0 = prune
            prune.remove(module, name)  # bake the zeros into the dense weight
    zeros = sum(int((m.weight == 0).sum()) for m, _ in linears)
    return zeros / total if total else 0.0


def _log_rss(tag: str) -> None:
    """Log peak process RSS so host-RAM OOM points are visible in the job logs.

    ``ru_maxrss`` is in KB on Linux (the training image), so /1e6 gives GB.
    """
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    log.info("[mem] %s: peak RSS = %.2f GB", tag, kb / 1e6)


def _mask_pruned_grads(model: torch.nn.Module) -> None:
    """Zero the gradient at pruned (zero) Linear weights so the model stays sparse.

    This is the memory-lean alternative to a live ``prune`` reparametrization,
    which keeps weight_orig + weight_mask + the computed weight (~3x weight
    memory) and OOMs the 24GB L4. Here the weights are dense (already zeroed) and
    only their *gradients* are masked, via a backward hook.
    Frozen params get no gradient, so only trainable Linears need it.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and module.weight.requires_grad:
            module.weight.register_hook(lambda grad, w=module.weight: grad * (w != 0))


def _load_merged(adapter_dir: Path):
    """Load base+adapter in bf16, merge LoRA into the base, return a CUDA model."""
    base = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    return model.to("cuda").eval()


def _score_accuracy(model, processor, loader, n_batches: int) -> tuple[int, int]:
    """Accuracy over up to ``n_batches`` (0 = all); returns (correct, total).

    Mirrors evaluate.py: generate, decode the continuation, and compare the
    extracted choice letter against the dataset's ``answer_texts``.
    """
    correct = total = 0
    for batch_idx, batch in enumerate(loader):
        if n_batches and batch_idx >= n_batches:
            break
        input_ids = batch["input_ids"].to("cuda")
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=batch["attention_mask"].to("cuda"),
            pixel_values=batch["pixel_values"].to("cuda", torch.bfloat16),
            max_new_tokens=10,
            do_sample=False,
        )
        preds = processor.batch_decode(
            generated_ids[:, input_ids.shape[1] :], skip_special_tokens=True
        )
        for pred, target in zip(preds, batch["answer_texts"]):
            correct += int(extract_answer_letter(pred) == extract_answer_letter(target))
            total += 1
    return correct, total


@app.command()
def benchmark(
    adapter_dir: Path = typer.Argument(..., help="LoRA adapter directory."),
    n_samples: int = typer.Option(8, help="Samples per generate call."),
    iters: int = typer.Option(5, help="Timed generate iterations (after warmup)."),
    output_path: Path = typer.Option(Path("optimize_results.json")),
) -> None:
    """Benchmark bf16 vs int4 vs bf16+compile and save a results table."""
    if not torch.cuda.is_available():
        typer.echo("CUDA required (4-bit + meaningful latency need a GPU).", err=True)
        raise typer.Exit(code=1)

    batch = _sample_batch(adapter_dir, n_samples)
    gen_kwargs = dict(
        input_ids=batch["input_ids"].to("cuda"),
        attention_mask=batch["attention_mask"].to("cuda"),
        pixel_values=batch["pixel_values"].to("cuda", torch.bfloat16),
        max_new_tokens=10,
        do_sample=False,
    )

    results = []
    for mode in ("bf16", "int4", "bf16+compile"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model, load_s = _load(adapter_dir, mode)
        with torch.inference_mode():
            model.generate(**gen_kwargs)  # warmup (also triggers compile)
            t = time.time()
            for _ in range(iters):
                model.generate(**gen_kwargs)
            latency = (time.time() - t) / iters
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        row = {
            "mode": mode,
            "load_s": round(load_s, 1),
            "latency_s_per_batch": round(latency, 3),
            "latency_s_per_sample": round(latency / n_samples, 3),
            "peak_gpu_gb": round(peak_gb, 2),
        }
        results.append(row)
        log.info("%s", row)
        del model

    output_path.write_text(
        json.dumps({"n_samples": n_samples, "results": results}, indent=2)
    )
    log.info("Saved benchmark to %s", output_path)


@app.command()
def prune_sweep(
    adapter_dir: Path = typer.Argument(..., help="LoRA adapter directory."),
    sparsities: str = typer.Option(
        "0.0,0.3,0.5,0.7", help="Comma-separated target sparsity levels."
    ),
    n_batches: int = typer.Option(
        0, help="Test batches scored per level (0 = whole test split)."
    ),
    batch_size: int = typer.Option(8, help="Eval/latency batch size."),
    iters: int = typer.Option(5, help="Timed generate iterations for latency."),
    output_path: Path = typer.Option(Path("prune_results.json")),
) -> None:
    """Prune the merged model to each sparsity and measure accuracy + latency.

    Accuracy is the headline (pruning degrades the adapter, which was trained on
    the un-pruned base); latency is reported only to confirm it does not drop.
    """
    if not torch.cuda.is_available():
        typer.echo("CUDA required (bf16 generate + meaningful latency).", err=True)
        raise typer.Exit(code=1)

    levels = [float(s) for s in sparsities.split(",") if s.strip()]
    processor = AutoProcessor.from_pretrained(str(adapter_dir))
    data = DataModule(
        processed_dir=PROCESSED_DATA_DIR,
        subset=DATASET_SUBSET,
        processor=processor,
        batch_size=batch_size,
        num_workers=0,  # avoid forking DataLoader workers once the model is on
        # CUDA -- forking a CUDA-initialised process can balloon host RAM.
    )
    data.setup()
    _log_rss("after dataset setup")

    results = []
    for amount in levels:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # Reload + re-merge each level: prune.remove is destructive, so every
        # sparsity must start from the clean baseline weights.
        model = _load_merged(adapter_dir)
        _log_rss(f"sparsity={amount} after load_merged")
        achieved = prune_linear_layers(model, amount)
        _log_rss(f"sparsity={amount} after prune ({achieved:.3f})")

        with torch.inference_mode():
            correct, total = _score_accuracy(
                model, processor, data.test_dataloader(), n_batches
            )
            _log_rss(f"sparsity={amount} after score")
            batch = next(iter(data.test_dataloader()))
            gen_kwargs = dict(
                input_ids=batch["input_ids"].to("cuda"),
                attention_mask=batch["attention_mask"].to("cuda"),
                pixel_values=batch["pixel_values"].to("cuda", torch.bfloat16),
                max_new_tokens=10,
                do_sample=False,
            )
            model.generate(**gen_kwargs)  # warmup
            t = time.time()
            for _ in range(iters):
                model.generate(**gen_kwargs)
            latency = (time.time() - t) / iters

        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        row = {
            "sparsity_requested": amount,
            "sparsity_achieved": round(achieved, 4),
            "accuracy": round(correct / total, 4) if total else 0.0,
            "correct": correct,
            "total": total,
            "latency_s_per_batch": round(latency, 3),
            "peak_gpu_gb": round(peak_gb, 2),
        }
        results.append(row)
        log.info("%s", row)
        del model
        gc.collect()  # free the CPU copy before the next level reloads the base

    output_path.write_text(
        json.dumps({"batch_size": batch_size, "results": results}, indent=2)
    )
    log.info("Saved pruning sweep to %s", output_path)


@app.command()
def prune_finetune(
    adapter_dir: Path = typer.Argument(..., help="LoRA adapter directory."),
    sparsity: float = typer.Option(0.5, help="Target sparsity to prune to."),
    steps: int = typer.Option(300, help="Fine-tuning optimizer steps."),
    batch_size: int = typer.Option(
        1, help="Train/eval batch size (raise if VRAM allows)."
    ),
    lr: float = typer.Option(1e-5, help="AdamW learning rate."),
    eval_batches: int = typer.Option(
        0, help="Test batches scored for accuracy (0 = all)."
    ),
    output_path: Path = typer.Option(Path("prune_finetune_results.json")),
) -> None:
    """Prune to ``sparsity``, then masked-fine-tune to recover accuracy.

    One-shot pruning degrades accuracy sharply; this re-trains the *surviving*
    weights to recover it. The weights are pruned (dense, zeroed) and kept sparse
    by masking the *gradient* at zeroed positions (``_mask_pruned_grads``), so they
    never re-grow. The vision tower is frozen and only the language model is
    fine-tuned (8-bit AdamW + gradient checkpointing) to fit the 24GB L4. Reports
    test accuracy before (one-shot) and after fine-tuning -- the recovery is the
    headline.
    """
    if not torch.cuda.is_available():
        typer.echo("CUDA required (fine-tuning + generation).", err=True)
        raise typer.Exit(code=1)
    import bitsandbytes as bnb  # 8-bit AdamW; installed at runtime (run_optimize.sh)

    processor = AutoProcessor.from_pretrained(str(adapter_dir))
    data = DataModule(
        processed_dir=PROCESSED_DATA_DIR,
        subset=DATASET_SUBSET,
        processor=processor,
        batch_size=batch_size,
        num_workers=0,
    )
    data.setup()

    # Prune into DENSE zeroed weights. A live prune reparametrization (keeping
    # weight_orig + weight_mask + the computed weight per layer) is ~3x the weight
    # memory and OOM'd the 24GB L4, so we instead bake the zeros and mask the
    # GRADIENTS below, so pruned weights never re-grow with no extra weight copies.
    model = _load_merged(adapter_dir)
    achieved = prune_linear_layers(model, sparsity)
    _log_rss(f"after prune (sparsity {achieved:.3f})")

    # One-shot accuracy (pre-fine-tune) -- the baseline this run tries to recover.
    model.eval()
    with torch.inference_mode():
        c0, t0 = _score_accuracy(model, processor, data.test_dataloader(), eval_batches)
    oneshot = c0 / t0 if t0 else 0.0
    log.info(
        "one-shot accuracy @ sparsity %.3f: %.4f (%d/%d)", achieved, oneshot, c0, t0
    )

    # The merged model came from a PeftModel, so ALL its params are frozen
    # (requires_grad=False -- PEFT trains only the LoRA adapter). Unfreeze
    # everything, then re-freeze the vision tower -> fine-tune the language model
    # only (less VRAM, and that's where the task capacity is). Gradient
    # checkpointing trades compute for memory so the 3B model + optimizer fit the L4.
    for param in model.parameters():
        param.requires_grad = True
    for param in model.model.vision_tower.parameters():
        param.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    log.info("trainable params: %d (%.2fB)", n_trainable, n_trainable / 1e9)
    if not trainable:
        raise RuntimeError(
            "no trainable parameters after freezing -- check freeze logic"
        )
    # Keep the model sparse during training: zero the gradient at pruned weights.
    _mask_pruned_grads(model)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.train()
    optimizer = bnb.optim.AdamW8bit(trainable, lr=lr)

    keys = ("input_ids", "attention_mask", "pixel_values", "token_type_ids", "labels")
    step = 0
    log.info("fine-tuning: %d steps, batch_size=%d, lr=%.1e", steps, batch_size, lr)
    while step < steps:
        for batch in data.train_dataloader():
            if step >= steps:
                break
            inputs = {}
            for key in keys:
                value = batch.get(key)
                if torch.is_tensor(value):
                    value = value.to("cuda")
                    inputs[key] = (
                        value.to(torch.bfloat16) if key == "pixel_values" else value
                    )
            optimizer.zero_grad(set_to_none=True)
            loss = model(**inputs).loss
            loss.backward()
            optimizer.step()
            step += 1
            if step == 1 or step % 20 == 0:
                gpu_gb = torch.cuda.max_memory_allocated() / 1e9
                log.info(
                    "step %d/%d  loss=%.4f  peak_gpu=%.1fGB",
                    step,
                    steps,
                    loss.item(),
                    gpu_gb,
                )
                _log_rss(f"step {step}")

    # Post-fine-tune accuracy (mask still live -> the model is still sparse).
    model.eval()
    with torch.inference_mode():
        c1, t1 = _score_accuracy(model, processor, data.test_dataloader(), eval_batches)
    finetuned = c1 / t1 if t1 else 0.0
    log.info(
        "fine-tuned accuracy @ sparsity %.3f: %.4f (%d/%d)", achieved, finetuned, c1, t1
    )

    result = {
        "sparsity_requested": sparsity,
        "sparsity_achieved": round(achieved, 4),
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "accuracy_oneshot": round(oneshot, 4),
        "accuracy_finetuned": round(finetuned, 4),
        "recovery_points": round((finetuned - oneshot) * 100, 2),
        "total": t1,
    }
    output_path.write_text(json.dumps(result, indent=2))
    log.info("Saved prune-finetune result to %s: %s", output_path, result)


if __name__ == "__main__":
    app()

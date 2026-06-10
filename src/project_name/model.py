"""Defines the model architecture for the project."""

import logging
from pathlib import Path

import lightning as L
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch
from rich.logging import RichHandler
from peft import get_peft_model, LoraConfig, PeftModel

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)
log = logging.getLogger(__name__)

MODEL_NAME = "google/paligemma2-3b-pt-224"


def build_prompt(
    question: str,
    choices: list[str],
    hint: str = "",
    lecture: str = "",
) -> str:
    """Build the PaliGemma2 VQA prompt for a sample.

    Uses the standard PaliGemma2 task prefix 'answer en' for English VQA.
    Choices are formatted as '(A) choice1 (B) choice2 ...'
    Optional hint and lecture are appended when provided.

    Args:
        question: The question text.
        choices: List of answer choice strings.
        hint: Optional hint text appended after the question.
        lecture: Optional background knowledge appended after the hint.

    Returns:
        Formatted prompt string ready for the processor.
    """
    choices_str = " ".join(
        f"({chr(65 + i)}) {choice}" for i, choice in enumerate(choices)
    )
    parts = [f"answer en {question}"]
    if hint:
        parts.append(f"Hint: {hint}")
    if lecture:
        parts.append(f"Lecture: {lecture}")
    parts.append(f"Choices: {choices_str}")
    return " ".join(parts)


class PaliGemmaModule(L.LightningModule):
    """PaliGemma2 fine-tuning module for ScienceQA.

    Wraps PaliGemmaForConditionalGeneration in a LightningModule,
    handling training, validation, and test steps, as well as
    optimizer and scheduler configuration.

    The vision encoder is frozen by default since ScienceQA images
    do not require visual feature adaptation — only the language model
    needs to learn the answer format.

    Args:
        model_name: HuggingFace model identifier.
        learning_rate: AdamW learning rate.
        torch_dtype: Weight dtype. bfloat16 recommended for modern GPUs.
        freeze_vision_encoder: Whether to freeze the SigLIP vision encoder.
        freeze_language_model: Whether to freeze the Gemma language model.
        gradient_checkpointing: Whether to enable gradient checkpointing
                                for memory efficiency.
        use_lora: Whether to apply LoRA fine-tuning to the language model.
        lora_r: LoRA rank (number of trainable parameters = 2 * r * hidden_size).
        lora_alpha: LoRA scaling factor.
        lora_dropout: LoRA dropout rate.
        lora_target_modules: Which language-model projection layers to wrap with
            LoRA. Default (q,k,v,o) covers full attention; pass a subset like
            (q_proj, v_proj) for fewer trainable params.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        learning_rate: float = 2e-5,
        torch_dtype: torch.dtype = torch.bfloat16,
        freeze_vision_encoder: bool = True,
        freeze_language_model: bool = False,
        gradient_checkpointing: bool = False,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    ) -> None:
        """Initialize the PaliGemmaModule with model and processor loading."""
        super().__init__()
        self.save_hyperparameters(ignore=["torch_dtype"])
        self.torch_dtype = torch_dtype

        log.info("Loading processor from %s ...", model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

        log.info("Loading model %s ...", model_name)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )

        if freeze_vision_encoder:
            log.info("Freezing vision encoder parameters.")
            for param in self.model.model.vision_tower.parameters():  # type: ignore[union-attr]
                param.requires_grad = False

        if freeze_language_model:
            log.info("Freezing language model parameters.")
            for param in self.model.model.language_model.parameters():  # type: ignore[union-attr]
                param.requires_grad = False

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.model.enable_input_require_grads()  # type: ignore[union-attr]

        if use_lora:
            if not lora_target_modules:
                raise ValueError(
                    "lora_target_modules must contain "
                    "at least one module name when use_lora=True."
                )
            # Match the chosen projection layers of the language model only
            # (vision tower stays frozen).
            target_re = rf".*language_model.*\.({'|'.join(lora_target_modules)})$"
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_re,
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)  # type: ignore[assignment]

            # log case and adapter shape of every module LoRA actually wrapped
            for name, module in self.model.named_modules():
                if not hasattr(module, "lora_A"):
                    continue
                lora_a = module.lora_A  # type: ignore[attr-defined]
                if len(lora_a) == 0:
                    continue
                adapter = next(iter(lora_a))
                base = module.base_layer  # type: ignore[attr-defined]
                log.info(
                    "LoRA: %s | base %d→%d | A %s | B %s",
                    name,
                    base.in_features,
                    base.out_features,
                    tuple(lora_a[adapter].weight.shape),
                    tuple(module.lora_B[adapter].weight.shape),  # type: ignore[attr-defined]
                )

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        log.info(
            "Trainable parameters: %.1fM / %.1fM total.",
            trainable / 1e6,
            total / 1e6,
        )

    def forward(self, **inputs):
        """Forward pass through the model."""
        return self.model(**inputs)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Compute cross-entropy loss on a training batch.

        Args:
            batch: Dict of tensors from DataModule._collate, including 'labels'.
            batch_idx: Index of the current batch (unused).

        Returns:
            Scalar loss tensor.
        """
        # Keep only tensor inputs; DataModule._collate attaches metadata lists
        # ("subjects", "answer_texts") for analysis/scoring that are NOT model
        # inputs and must not flow into the model's **kwargs.
        model_inputs = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        outputs = self.model(**model_inputs)
        loss = outputs.loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Compute validation loss on a validation batch.

        Args:
            batch: Dict of tensors from DataModule._collate, including 'labels'.
            batch_idx: Index of the current batch (unused).
        """
        # Keep only tensor inputs; drop metadata lists (see training_step).
        model_inputs = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        outputs = self.model(**model_inputs)
        loss = outputs.loss
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        """Generate predictions and compute exact-match accuracy on a test batch.

        Decodes generated tokens and compares against the ground-truth
        answer_text taken directly from the dataset (carried through _collate),
        not by decoding the -100-masked labels.

        Args:
            batch: Dict from DataModule._collate, including 'answer_texts'.
            batch_idx: Index of the current batch (unused).
        """
        input_ids = batch["input_ids"]
        pixel_values = batch.get("pixel_values")
        attention_mask = batch.get("attention_mask")

        generated_ids = self.model.generate(  # type: ignore[misc]
            input_ids=input_ids,
            attention_mask=attention_mask,  # avoid attending to padding
            pixel_values=pixel_values,
            max_new_tokens=10,
            do_sample=False,
        )

        # Decode only newly generated tokens, skipping the prompt
        input_len = input_ids.shape[1]
        preds = self.processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
        )

        # Ground truth = raw answer_text from the dataset (e.g. "A"), the
        # authoritative label — never a tokenize→mask→decode round-trip.
        targets = batch["answer_texts"]

        correct = sum(
            p.strip().upper() == t.strip().upper() for p, t in zip(preds, targets)
        )
        acc = correct / len(preds)
        self.log("test/accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """Configure AdamW optimizer with cosine annealing scheduler.

        Returns:
            Dict with optimizer and step-level lr_scheduler.
        """
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.hparams.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def save_adapter(self, save_directory: str | Path) -> None:
        """Save ONLY the trained LoRA adapter (+ processor), not the 3B base.

        The full Lightning .ckpt is ~6 GB because it serializes the entire base
        model; the LoRA adapter is a few MB. For the model registry and serving
        we only need the adapter — the base is re-downloaded from the Hub at
        load time. The processor is saved alongside so serving can reload both
        from one directory.

        Args:
            save_directory: Target directory for adapter_config.json,
                adapter_model.safetensors, and the processor files.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        # self.model is the PEFT-wrapped model; save_pretrained writes adapter only.
        self.model.save_pretrained(str(save_directory))
        self.processor.save_pretrained(str(save_directory))
        log.info("Saved LoRA adapter + processor to %s", save_directory)

    @classmethod
    def load_adapter(
        cls,
        adapter_dir: str | Path,
        model_name: str = MODEL_NAME,
        device: torch.device | None = None,
    ) -> "PaliGemmaModule":
        """Load a base model from the Hub and attach a trained LoRA adapter.

        The inverse of save_adapter: builds the base PaliGemma (no fresh LoRA),
        then loads the trained adapter weights via PEFT. Returns an eval-mode
        module whose .model / .processor work exactly like a fine-tuned module,
        so predict_single / evaluate need no special-casing.

        Args:
            adapter_dir: Directory produced by save_adapter.
            model_name: Base model identifier (must match training).
            device: Target device; auto-selected (CUDA > MPS > CPU) if None.

        Returns:
            PaliGemmaModule in eval mode with the trained adapter attached.
        """
        adapter_dir = Path(adapter_dir)
        if device is None:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        # use_lora=False → plain base model; we attach the *trained* adapter next.
        module = cls(model_name=model_name, use_lora=False)
        # PeftModel wraps the base; same intentional reassignment as __init__'s
        # get_peft_model (self.model is typed as PaliGemmaForConditionalGeneration).
        module.model = PeftModel.from_pretrained(  # type: ignore[assignment]
            module.model, str(adapter_dir)
        )
        # Prefer the processor saved next to the adapter; fall back to the base.
        if (adapter_dir / "preprocessor_config.json").exists():
            module.processor = AutoProcessor.from_pretrained(str(adapter_dir))
        module.eval()
        module.to(device)
        log.info("Loaded base %s + adapter %s on %s", model_name, adapter_dir, device)
        return module

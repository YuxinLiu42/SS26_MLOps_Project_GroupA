"""Evaluation for PaliGemma2 fine-tuned on ScienceQA."""

import json
import logging
from collections import defaultdict
from pathlib import Path
import typer
from rich.logging import RichHandler
from rich.table import Table
from rich import print as rprint

from project_name.data import DataModule, PROCESSED_DATA_DIR, DATASET_SUBSET
from project_name.model import PaliGemmaModule

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)
log = logging.getLogger(__name__)
app = typer.Typer(
    help="Evaluate a fine-tuned PaliGemma2 checkpoint " "on ScienceQA test set."
)


@app.command()
def evaluate(
    ckpt_path: Path = typer.Argument(
        ..., help="Path to the model checkpoint (.ckpt file)."
    ),
    processed_dir: Path = typer.Option(
        PROCESSED_DATA_DIR, help="Directory containing the processed dataset."
    ),
    subset: str = typer.Option(DATASET_SUBSET, help="Dataset subset name."),
    batch_size: int = typer.Option(8, help="Batch size for evaluation."),
    num_workers: int = typer.Option(2, help="Number of DataLoader worker processes."),
    output_path: Path = typer.Option(
        Path("eval_results.json"), help="Path to save evaluation results as JSON."
    ),
    by_subject: bool = typer.Option(
        False, help="Whether to report accuracy broken down by subject."
    ),
) -> None:
    """Evaluate a fine-tuned PaliGemma2 checkpoint on the ScienceQA test set.

    Loads the model from a checkpoint, runs inference on the full test split,
    and reports exact-match accuracy. Results are printed to the terminal
    and saved as a JSON file.

    Args:
        ckpt_path: Path to the model checkpoint (.ckpt file).
        processed_dir: Directory containing the processed dataset.
        subset: Dataset subset name.
        batch_size: Batch size for evaluation.
        num_workers: Number of DataLoader worker processes.
        output_path: Path to save evaluation results as JSON.
        by_subject: Whether to report accuracy broken down by subject.
    """
    log.info("Loading model from checkpoint: %s", ckpt_path)
    model = PaliGemmaModule.load_from_checkpoint(ckpt_path)
    model.eval()

    log.info(
        "Initializing data module from %s ...",
        processed_dir / subset,
    )
    data = DataModule(
        processed_dir=processed_dir,
        subset=subset,
        processor=model.processor,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    data.setup()
    test_loader = data.test_dataloader()
    log.info("Test set size: %d batches", len(test_loader))

    total_correct = 0
    total_samples = 0
    subject_stats = dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )

    log.info("Running inference on test set ...")
    for batch in test_loader:
        input_ids = batch["input_ids"].to(model.device)
        pixel_values = batch.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(model.device)
        labels = batch["labels"].to(model.device)
        subjects = batch.get("subject", [])

        generated_ids = model.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=10,
            do_sample=False,
        )

        input_len = input_ids.shape[1]
        preds = model.processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
        )

        label_ids = labels.clone()
        label_ids[label_ids == -100] = model.processor.tokenizer.pad_token_id
        targets = model.processor.batch_decode(label_ids, skip_special_tokens=True)

        for i, (pred, target) in enumerate(zip(preds, targets)):
            is_correct = pred.strip() == target.strip()
            total_correct += is_correct
            total_samples += 1
            if by_subject and subjects:
                subj = subjects[i]
                subject_stats[subj]["correct"] += int(is_correct)
                subject_stats[subj]["total"] += 1

    overall_acc = total_correct / total_samples if total_samples > 0 else 0.0
    log.info(
        "Evaluation complete: %d / %d correct | Accuracy: %.4f%%",
        total_correct,
        total_samples,
        overall_acc,
    )

    if by_subject and subject_stats:
        table = Table(title="Accuracy by Subject")
        table.add_column("Subject", style="cyan")
        table.add_column("Correct", justify="right")
        table.add_column("Total", justify="right")
        table.add_column("Accuracy", justify="right", style="green")

        for subj, stats in sorted(subject_stats.items()):
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            table.add_row(
                subj,
                str(stats["correct"]),
                str(stats["total"]),
                f"{acc:.4f}",
            )

        rprint(table)

    results = {
        "checkpoint": str(ckpt_path),
        "total_correct": total_correct,
        "total_samples": total_samples,
        "accuracy": overall_acc,
    }
    if by_subject and subject_stats:
        results["by_subject"] = {
            subj: {
                "correct": stats["correct"],
                "total": stats["total"],
                "accuracy": stats["correct"] / stats["total"]
                if stats["total"] > 0
                else 0.0,
            }
            for subj, stats in subject_stats.items()
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to: %s", output_path)


if __name__ == "__main__":
    app()

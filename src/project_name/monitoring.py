"""Data-drift monitoring for ScienceQA inputs (M27).

Builds an Evidently data-drift report comparing a reference split (train) with a
current split (test, or live-collected inputs). We don't have raw tabular
features, so we derive lightweight ones from each sample — question length,
number of choices, hint/lecture presence, image dimensions, subject — which is
enough to catch distribution shift in the inputs the model sees.

Run (single-command Typer app, so no subcommand name):
    uv run --group serving python -m project_name.monitoring
"""

import logging
from pathlib import Path

import pandas as pd
import typer
from datasets import load_from_disk
from rich.logging import RichHandler

from project_name.data import DATASET_SUBSET, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
log = logging.getLogger(__name__)

app = typer.Typer(help="Data-drift monitoring for ScienceQA inputs.")

RESULTS_DIR = Path("reports/figures")


def _features(split) -> pd.DataFrame:
    """Derive a tabular feature frame from a ScienceQA split for drift checks.

    Args:
        split: A HuggingFace Dataset split with image/question/choices/... cols.

    Returns:
        A DataFrame with one row per sample and numeric/categorical features.
    """
    rows = []
    for s in split:
        img = s.get("image")
        rows.append(
            {
                "question_char_len": len(s["question"]),
                "question_word_len": len(s["question"].split()),
                "num_choices": len(s["choices"]),
                "hint_present": int(bool(s.get("hint"))),
                "lecture_present": int(bool(s.get("lecture"))),
                "image_width": img.width if img is not None else 0,
                "image_height": img.height if img is not None else 0,
                "subject": s.get("subject", "unknown"),
            }
        )
    return pd.DataFrame(rows)


@app.command()
def drift(
    processed_dir: Path = typer.Option(PROCESSED_DATA_DIR),
    reference: str = typer.Option("train", help="Reference split."),
    current: str = typer.Option("test", help="Current split to compare."),
    output_dir: Path = typer.Option(RESULTS_DIR, "--output-dir", "-o"),
) -> None:
    """Generate an Evidently data-drift report (reference vs current split)."""
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    dataset = load_from_disk(processed_dir / DATASET_SUBSET)
    ref_df = _features(dataset[reference])
    cur_df = _features(dataset[current])
    log.info(
        "Reference (%s): %d rows | Current (%s): %d rows",
        reference,
        len(ref_df),
        current,
        len(cur_df),
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "drift_report.html"
    report.save_html(str(html_path))

    result = report.as_dict()["metrics"][0]["result"]
    log.info(
        "Dataset drift detected: %s | drifted columns: %d/%d",
        result.get("dataset_drift"),
        result.get("number_of_drifted_columns"),
        result.get("number_of_columns"),
    )
    log.info("Saved drift report to %s", html_path)


if __name__ == "__main__":
    app()

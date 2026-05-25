"""Module for downloading, preprocessing, and loading the dataset."""

import logging
from pathlib import Path
import typer
from torch.utils.data import DataLoader
from datasets import DatasetDict, load_dataset, load_from_disk
from rich.logging import RichHandler
import shutil
import lightning as L
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)
log = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

DATASET_NAME = "lmms-lab/ScienceQA"
DATASET_SUBSET = "ScienceQA-IMG"

app = typer.Typer(help="Data pipeline: download and preprocess ScienceQA dataset.")


@app.command()
def download(
    raw_dir: Path = typer.Option(
        RAW_DATA_DIR, help="Directory to save the raw dataset."
    ),
) -> None:
    """Download the ScienceQA dataset from HuggingFace Hub and save it to disk.

    Only the image-based subset of the dataset is downloaded
    since the model is designed to handle multimodal data.
    Raw data is saved under raw_dir/ScienceQA-IMG/ without any modification.

    Args:
        raw_dir: Directory to save the raw dataset.
    """
    save_path = raw_dir / DATASET_SUBSET

    if save_path.exists():
        log.info("Raw dataset already exists at %s, skipping download.", save_path)
        return

    save_path.mkdir(parents=True, exist_ok=True)

    log.info(
        "Downloading '%s' subset '%s' from HuggingFace Hub...",
        DATASET_NAME,
        DATASET_SUBSET,
    )
    dataset = load_dataset(DATASET_NAME, DATASET_SUBSET)
    dataset.save_to_disk(save_path)

    log.info("Raw dataset saved to %s", save_path)
    log.info(
        "Available splits: %s | validation=%d | test=%d",
        list(dataset.keys()),
        len(dataset["validation"]),
        len(dataset["test"]),
    )


# All columns retained by default for training and evaluation
COLUMNS_TO_KEEP = [
    "image",  # Visual input
    "question",  # Question text
    "choices",  # Choice list, used to construct prompt
    "hint",  # Question hint, present at inference time
    "lecture",  # Background knowledge, present at inference time
    "answer",  # Original index, used during evaluation
    "answer_text",  # Model prediction target
    "subject",  # Analyze performance by subject
    "topic",  # Analyze performance by topic
]

# Columns that are always removed regardless of user input
# solution contains ground-truth reasoning steps and must never be seen during training
COLUMNS_ALWAYS_DROP = ["solution", "task", "grade", "category", "skill"]

# Columns that can be optionally dropped via --drop-cols
# Required columns (image, question, choices, answer, answer_text) cannot be dropped
COLUMNS_OPTIONAL = ["hint", "lecture", "subject", "topic"]


@app.command()
def preprocess(
    raw_dir: Path = typer.Option(
        RAW_DATA_DIR, help="Directory containing the raw dataset."
    ),
    processed_dir: Path = typer.Option(
        PROCESSED_DATA_DIR, help="Directory to save the processed dataset."
    ),
    subject: str = typer.Option(
        "",
        help="Subject to filter the dataset by (e.g., 'physics', 'chemistry'). "
        "Empty keeps all.",
    ),
    val_ratio: float = typer.Option(
        0.8, help="Ratio of validation samples to keep (between 0 and 1)."
    ),
    drop_cols: str = typer.Option(
        "",
        help=(
            "Comma-separated list of optional columns to drop. "
            f"Allowed values: {', '.join(COLUMNS_OPTIONAL)}. "
            "Required columns (image, question, choices, answer, answer_text) "
            "cannot be dropped. "
            f"Always-dropped columns regardless of input: "
            f"{', '.join(COLUMNS_ALWAYS_DROP)}."
        ),
    ),
    overwrite: bool = typer.Option(
        False, help="Overwrite existing processed dataset if it already exists."
    ),
) -> None:
    """Preprocess the raw ScienceQA-IMG dataset and save the processed version to disk.

    Processing steps:
      1. Add answer_text field derived from choices and answer index.
      2. Filter rows by subject if specified.
      3. Drop columns not in the whitelist, plus any extra columns from --drop-cols.
      4. Split the original validation set into train/val using val_ratio,
         since the pt model requires a train split for fine-tuning.

    Args:
        raw_dir: Directory containing the raw dataset.
        processed_dir: Directory to save the processed dataset.
        subject: Subject to filter the dataset by (e.g., 'physics', 'chemistry').
                 Empty keeps all.
        val_ratio: Ratio of validation samples to keep (between 0 and 1).
        drop_cols: Comma-separated optional columns to drop.
                   Allowed: hint, lecture, subject, topic.
        overwrite: Overwrite existing processed dataset if it already exists.
    """
    raw_path = raw_dir / DATASET_SUBSET
    processed_path = processed_dir / DATASET_SUBSET

    if not raw_path.exists():
        log.error("Raw data not found at %s. Run the download command first.", raw_path)
        raise typer.Exit(code=1)

    if processed_path.exists():
        if not overwrite:
            log.info(
                "Processed dataset already exists at %s, skipping.", processed_path
            )
            return
        log.info("Overwrite enabled, removing existing dataset at %s.", processed_path)
        shutil.rmtree(processed_path)

    # Parse and validate optional extra columns to drop
    extra_drop: list[str] = [c.strip() for c in drop_cols.split(",") if c.strip()]
    invalid = [c for c in extra_drop if c not in COLUMNS_OPTIONAL]
    if invalid:
        log.error(
            "Cannot drop unknown or required columns: %s. "
            "Only optional columns can be dropped: %s",
            invalid,
            COLUMNS_OPTIONAL,
        )
        raise typer.Exit(code=1)

    effective_keep = [c for c in COLUMNS_TO_KEEP if c not in extra_drop]
    log.info("Columns kept: %s", effective_keep)
    if extra_drop:
        log.info("Extra columns dropped by user: %s", extra_drop)
    log.info("Columns always dropped: %s", COLUMNS_ALWAYS_DROP)

    log.info("Loading raw dataset from %s ...", raw_path)
    dataset = load_from_disk(raw_path)
    log.info(
        "Raw sizes: validation=%d | test=%d",
        len(dataset["validation"]),
        len(dataset["test"]),
    )

    # Step 1: Add answer_text field derived from choices and answer index.
    def _add_answer_text(sample: dict) -> dict:
        """Replace integer answer index with the corresponding answer string."""
        sample["answer_text"] = sample["choices"][sample["answer"]]
        return sample

    dataset = dataset.map(_add_answer_text)
    log.info("Add 'answer_text' field to the dataset.")

    # Step 2: Filter rows by subject if specified.
    if subject:
        before = {split: len(dataset[split]) for split in dataset}
        dataset = dataset.filter(lambda x: x["subject"] == subject)
        for split in dataset:
            log.info(
                "Subject filter '%s': %s %d -> %d",
                subject,
                split,
                before[split],
                len(dataset[split]),
            )

    # Step 3: Drop unused columns to reduce storage size.
    for split in dataset:
        columns_to_drop = [
            c for c in dataset[split].column_names if c not in effective_keep
        ]
        if columns_to_drop:
            dataset[split] = dataset[split].remove_columns(columns_to_drop)
    log.info("Remaining columns: %s", dataset["validation"].column_names)

    # Step 4: Split the original validation set into train/val using val_ratio,
    # since the pt model requires a train split for fine-tuning.
    split_result = dataset["validation"].train_test_split(
        test_size=1 - val_ratio, seed=42
    )

    final_dataset = DatasetDict(
        {
            "train": split_result["train"],
            "validation": split_result["test"],
            "test": dataset["test"],
        }
    )

    final_dataset.save_to_disk(processed_path)
    log.info("Processed dataset saved to %s", processed_path)
    log.info(
        "Final sizes: train=%d | validation=%d | test=%d",
        len(final_dataset["train"]),
        len(final_dataset["validation"]),
        len(final_dataset["test"]),
    )


@app.command()
def subset_data(
    processed_dir: Path = typer.Option(
        PROCESSED_DATA_DIR, help="Directory containing the processed dataset."
    ),
    n_samples: int = typer.Option(
        200, help="Number of samples to select per split " "for fast debugging."
    ),
) -> None:
    """Select a small subset of the processed dataset for fast debugging.

    Does NOT overwrite the full processed dataset.
    The debug subset is saved separately under processed_dir/ScienceQA-IMG_debug/.

    Args:
        processed_dir: Directory containing the processed dataset.
        n_samples: Number of samples to select per split for fast debugging.
    """
    processed_path = processed_dir / DATASET_SUBSET
    debug_path = processed_dir / f"{DATASET_SUBSET}_debug"

    if not processed_path.exists():
        log.error(
            "Processed data not found at %s. Run the preprocess command first.",
            processed_path,
        )
        raise typer.Exit(code=1)

    if debug_path.exists():
        log.info("Debug subset already exists at %s, skipping.", debug_path)
        return

    log.info("Loading processed dataset from %s ...", processed_path)
    dataset = load_from_disk(processed_path)

    debug_dataset = DatasetDict(
        {
            split: dataset[split].select(range(min(n_samples, len(dataset[split]))))
            for split in dataset
        }
    )

    debug_dataset.save_to_disk(debug_path)
    log.info("Debug subset saved to %s", debug_path)
    log.info(
        "Debug sizes: train=%d | validation=%d | test=%d",
        len(debug_dataset["train"]),
        len(debug_dataset["validation"]),
        len(debug_dataset["test"]),
    )


class DataModule(L.LightningDataModule):
    """LightningDataModule for the dataset.

    Expects data already downloaded and preprocessed by the CLI commands.
    Handles dataset loading, prompt construction, image/text processing,
    and label masking for seq2seq-style fine-tuning.

    Args:
        processed_dir: Root directory of processed data.
        subset: Dataset subset name.
        processor: AutoProcessor instance from the model module.
        batch_size: Dataloader batch size.
        num_workers: Number of dataloader worker processes.
        max_length: Max token length for text prompt inputs.
        max_label_length: Max token length for answer label inputs.
    """

    def __init__(
        self,
        processed_dir: Path = PROCESSED_DATA_DIR,
        subset: str = DATASET_SUBSET,
        processor=None,
        batch_size: int = 4,
        num_workers: int = 2,
        max_length: int = 256,
        max_label_length: int = 32,
    ) -> None:
        """Initialize the DataModule with dataset paths and processing parameters."""
        super().__init__()
        self.data_path = processed_dir / subset
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.max_label_length = max_label_length
        self.dataset: DatasetDict | None = None

    def setup(self, stage: str | None = None) -> None:
        """Load the processed dataset from disk.

        Called by the Trainer before any dataloader is created.
        Loads all splits (train/validation/test) into memory.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict').
                   Not used here since all splits are loaded at once.
        """
        log.info("Loading processed dataset from %s ...", self.data_path)
        self.dataset = load_from_disk(self.data_path)
        log.info(
            "Dataset loaded: train=%d | validation=%d | test=%d",
            len(self.dataset["train"]),
            len(self.dataset["validation"]),
            len(self.dataset["test"]),
        )

    def _collate(self, samples: list[dict]) -> dict:
        """Collate a batch of samples into model-ready tensors.

        For each sample, constructs the text prompt and retrieves the image.
        Missing images are replaced with a blank RGB placeholder.
        Label tokens corresponding to padding are masked with -100 so they
        are ignored by the loss function.

        Args:
            samples: List of dataset samples, each containing keys:
                     'question', 'choices', 'answer_text', and 'image'.

        Returns:
            A dict of tensors ready for the model, including 'labels'.
        """
        # Avoid circular import by importing processor here
        from project_name.model import build_prompt

        prompts, answer_texts, images = [], [], []
        subjects = []
        for s in samples:
            prompts.append(build_prompt(s["question"], s["choices"]))
            answer_texts.append(s["answer_text"])
            images.append(
                s["image"] if s["image"] is not None else Image.new("RGB", (224, 224))
            )
            subjects.append(s.get("subject", "unknown"))

        prompts_with_image = [f"<image> {p}" for p in prompts]

        inputs = self.processor(
            text=prompts_with_image,
            images=images,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_length,
        )

        labels = inputs["input_ids"].clone()

        prompt_encodings = self.processor.tokenizer(
            prompts_with_image,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
        )
        for i, prompt_ids in enumerate(prompt_encodings["input_ids"]):
            prompt_len = len(prompt_ids)
            labels[i, :prompt_len] = -100  # mask prompt

        labels[labels == self.processor.tokenizer.pad_token_id] = -100  # mask padding

        inputs["labels"] = labels
        inputs["subjects"] = subjects  # for analysis, not used by model

        return dict(inputs)

    def train_dataloader(self) -> DataLoader:
        """Return the DataLoader for the training split.

        Returns:
            DataLoader with shuffling enabled for training.
        """
        assert self.dataset is not None, "Call setup() before accessing dataloaders."
        return DataLoader(
            self.dataset["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the DataLoader for the validation split.

        Returns:
            DataLoader without shuffling for deterministic evaluation.
        """
        assert self.dataset is not None, "Call setup() before accessing dataloaders."
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the DataLoader for the test split.

        Returns:
            DataLoader without shuffling for deterministic evaluation.
        """
        assert self.dataset is not None, "Call setup() before accessing dataloaders."
        return DataLoader(
            self.dataset["test"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate,
        )


if __name__ == "__main__":
    app()

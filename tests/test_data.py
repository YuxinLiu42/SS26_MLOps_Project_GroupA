"""Tests for the data pipeline: CLI commands and DataModule."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from datasets import Dataset, DatasetDict
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler
from typer.testing import CliRunner

from project_name.data import (
    COLUMNS_ALWAYS_DROP,
    DATASET_SUBSET,
    DataModule,
    app,
)

runner = CliRunner()


def _make_sample(
    answer: int = 0,
    subject: str = "physics",
    image: Image.Image | None = None,
) -> dict:
    """Return a minimal ScienceQA sample dict.

    Args:
        answer: Index into the choices list that is the correct answer.
        subject: Subject label for the sample.
        image: PIL image to attach; defaults to a small blank RGB image.

    Returns:
        A dict with all required fields populated.
    """
    return {
        "image": image or Image.new("RGB", (32, 32)),
        "question": "What is the color of the sky?",
        "choices": ["blue", "red", "green"],
        "hint": "Look up.",
        "lecture": "The sky appears blue due to Rayleigh scattering.",
        "answer": answer,
        "subject": subject,
        "topic": "optics",
        "solution": "Step 1: ...",
        "task": "classification",
        "grade": 5,
        "category": "science",
        "skill": "observation",
    }


def _make_dataset(n_validation: int = 10, n_test: int = 4) -> DatasetDict:
    """Return a minimal DatasetDict with validation and test splits.

    Mirrors the raw dataset structure before preprocessing.

    Args:
        n_validation: Number of samples in the validation split.
        n_test: Number of samples in the test split.

    Returns:
        A DatasetDict with 'validation' and 'test' splits.
    """
    return DatasetDict(
        {
            "validation": Dataset.from_list(
                [_make_sample() for _ in range(n_validation)]
            ),
            "test": Dataset.from_list([_make_sample() for _ in range(n_test)]),
        }
    )


def _make_processed_dataset(
    n_train: int = 8,
    n_validation: int = 2,
    n_test: int = 4,
) -> DatasetDict:
    """Return a minimal processed DatasetDict with all three splits.

    Mirrors the dataset structure after preprocessing: train split present,
    answer_text added, always-drop columns removed.

    Args:
        n_train: Number of samples in the training split.
        n_validation: Number of samples in the validation split.
        n_test: Number of samples in the test split.

    Returns:
        A DatasetDict with 'train', 'validation', and 'test' splits.
    """

    def _processed_sample() -> dict:
        """Return a single preprocessed sample with answer_text and no dropped cols."""
        s = _make_sample()
        s["answer_text"] = s["choices"][s["answer"]]
        for col in COLUMNS_ALWAYS_DROP:
            s.pop(col, None)
        return s

    return DatasetDict(
        {
            "train": Dataset.from_list([_processed_sample() for _ in range(n_train)]),
            "validation": Dataset.from_list(
                [_processed_sample() for _ in range(n_validation)]
            ),
            "test": Dataset.from_list([_processed_sample() for _ in range(n_test)]),
        }
    )


# Tests for the download CLI command.
class TestDownload:
    """Tests for the download CLI command."""

    def test_skips_if_already_exists(self, tmp_path: Path) -> None:
        """Download is skipped when the target directory already exists."""
        save_path = tmp_path / DATASET_SUBSET
        save_path.mkdir(parents=True)

        result = runner.invoke(app, ["download", "--raw-dir", str(tmp_path)])

        assert result.exit_code == 0
        assert save_path.exists()

    @patch("project_name.data.load_dataset")
    def test_downloads_and_saves(self, mock_load: MagicMock, tmp_path: Path) -> None:
        """Dataset is downloaded from HuggingFace and saved to disk."""
        mock_dataset = _make_dataset()
        mock_load.return_value = mock_dataset

        with patch.object(mock_dataset, "save_to_disk"):
            result = runner.invoke(app, ["download", "--raw-dir", str(tmp_path)])

        assert result.exit_code == 0
        mock_load.assert_called_once()


# Tests for the preprocess CLI command.
class TestPreprocess:
    """Tests for the preprocess CLI command."""

    def test_exits_if_raw_data_missing(self, tmp_path: Path) -> None:
        """Preprocessing exits with code 1 when raw data directory is absent."""
        result = runner.invoke(
            app,
            [
                "preprocess",
                "--raw-dir",
                str(tmp_path),
                "--processed-dir",
                str(tmp_path / "processed"),
            ],
        )

        assert result.exit_code == 1

    def test_skips_if_processed_exists(self, tmp_path: Path) -> None:
        """Preprocessing is skipped when the processed directory already exists."""
        (tmp_path / "raw" / DATASET_SUBSET).mkdir(parents=True)
        (tmp_path / "processed" / DATASET_SUBSET).mkdir(parents=True)

        result = runner.invoke(
            app,
            [
                "preprocess",
                "--raw-dir",
                str(tmp_path / "raw"),
                "--processed-dir",
                str(tmp_path / "processed"),
            ],
        )

        assert result.exit_code == 0
        assert (tmp_path / "processed" / DATASET_SUBSET).exists()

    @patch("project_name.data.load_from_disk")
    def test_answer_text_added(self, mock_load: MagicMock, tmp_path: Path) -> None:
        """answer_text is correctly derived from choices and answer index."""
        (tmp_path / "raw" / DATASET_SUBSET).mkdir(parents=True)
        mock_load.return_value = _make_dataset()

        saved: list[DatasetDict] = []

        def _capture(ds: DatasetDict, path: Path) -> None:
            saved.append(ds)

        with patch.object(DatasetDict, "save_to_disk", _capture):
            result = runner.invoke(
                app,
                [
                    "preprocess",
                    "--raw-dir",
                    str(tmp_path / "raw"),
                    "--processed-dir",
                    str(tmp_path / "processed"),
                ],
            )

        assert result.exit_code == 0
        assert "answer_text" in saved[0]["train"].column_names

    @patch("project_name.data.load_from_disk")
    def test_always_drop_columns_removed(
        self, mock_load: MagicMock, tmp_path: Path
    ) -> None:
        """Columns in COLUMNS_ALWAYS_DROP are never present in the output."""
        (tmp_path / "raw" / DATASET_SUBSET).mkdir(parents=True)
        mock_load.return_value = _make_dataset()

        saved: list[DatasetDict] = []

        def _capture(ds: DatasetDict, path: Path) -> None:
            saved.append(ds)

        with patch.object(DatasetDict, "save_to_disk", _capture):
            runner.invoke(
                app,
                [
                    "preprocess",
                    "--raw-dir",
                    str(tmp_path / "raw"),
                    "--processed-dir",
                    str(tmp_path / "processed"),
                ],
            )

        for col in COLUMNS_ALWAYS_DROP:
            assert col not in saved[0]["validation"].column_names

    @patch("project_name.data.load_from_disk")
    def test_subject_filter_removes_other_subjects(
        self, mock_load: MagicMock, tmp_path: Path
    ) -> None:
        """Only samples matching the subject filter are retained."""
        (tmp_path / "raw" / DATASET_SUBSET).mkdir(parents=True)
        samples = [_make_sample(subject="physics")] * 6 + [
            _make_sample(subject="chemistry")
        ] * 4
        mock_load.return_value = DatasetDict(
            {
                "validation": Dataset.from_list(samples),
                "test": Dataset.from_list([_make_sample(subject="physics")] * 2),
            }
        )

        saved: list[DatasetDict] = []

        def _capture(ds: DatasetDict, path: Path) -> None:
            saved.append(ds)

        with patch.object(DatasetDict, "save_to_disk", _capture):
            result = runner.invoke(
                app,
                [
                    "preprocess",
                    "--raw-dir",
                    str(tmp_path / "raw"),
                    "--processed-dir",
                    str(tmp_path / "processed"),
                    "--subject",
                    "physics",
                ],
            )

        assert result.exit_code == 0
        subjects = {s["subject"] for s in saved[0]["validation"]}
        assert subjects == {"physics"}

    @patch("project_name.data.load_from_disk")
    def test_rejects_invalid_drop_cols(
        self, mock_load: MagicMock, tmp_path: Path
    ) -> None:
        """Exit code 1 is returned when a required column is passed to --drop-cols."""
        (tmp_path / "raw" / DATASET_SUBSET).mkdir(parents=True)
        mock_load.return_value = _make_dataset()

        result = runner.invoke(
            app,
            [
                "preprocess",
                "--raw-dir",
                str(tmp_path / "raw"),
                "--processed-dir",
                str(tmp_path / "processed"),
                "--drop-cols",
                "answer",
            ],
        )

        assert result.exit_code == 1

    @patch("project_name.data.load_from_disk")
    def test_overwrite_removes_existing(
        self, mock_load: MagicMock, tmp_path: Path
    ) -> None:
        """Existing processed directory is removed when --overwrite is set."""
        (tmp_path / "raw" / DATASET_SUBSET).mkdir(parents=True)
        existing = tmp_path / "processed" / DATASET_SUBSET
        existing.mkdir(parents=True)
        sentinel = existing / "old_file.txt"
        sentinel.write_text("old")
        mock_load.return_value = _make_dataset()

        with patch("project_name.data.DatasetDict.save_to_disk"):
            result = runner.invoke(
                app,
                [
                    "preprocess",
                    "--raw-dir",
                    str(tmp_path / "raw"),
                    "--processed-dir",
                    str(tmp_path / "processed"),
                    "--overwrite",
                ],
            )

        assert result.exit_code == 0
        assert not sentinel.exists()


# Tests for the subset_data CLI command.
class TestSubsetData:
    """Tests for the subset_data CLI command."""

    def test_exits_if_processed_missing(self, tmp_path: Path) -> None:
        """subset_data exits with code 1 when processed data directory is absent."""
        result = runner.invoke(app, ["subset-data", "--processed-dir", str(tmp_path)])

        assert result.exit_code == 1

    def test_skips_if_debug_exists(self, tmp_path: Path) -> None:
        """subset_data is skipped when the debug directory already exists."""
        (tmp_path / DATASET_SUBSET).mkdir(parents=True)
        debug_path = tmp_path / f"{DATASET_SUBSET}_debug"
        debug_path.mkdir(parents=True)

        result = runner.invoke(app, ["subset-data", "--processed-dir", str(tmp_path)])

        assert result.exit_code == 0
        assert debug_path.exists()

    @patch("project_name.data.load_from_disk")
    def test_subset_size_capped_at_n_samples(
        self, mock_load: MagicMock, tmp_path: Path
    ) -> None:
        """Each split in the debug subset contains at most n_samples rows."""
        (tmp_path / DATASET_SUBSET).mkdir(parents=True)
        mock_load.return_value = _make_processed_dataset(
            n_train=50, n_validation=20, n_test=20
        )

        saved: list[DatasetDict] = []

        def _capture(ds: DatasetDict, path: Path) -> None:
            saved.append(ds)

        with patch.object(DatasetDict, "save_to_disk", _capture):
            result = runner.invoke(
                app,
                [
                    "subset-data",
                    "--processed-dir",
                    str(tmp_path),
                    "--n-samples",
                    "10",
                ],
            )

        assert result.exit_code == 0
        assert len(saved[0]["train"]) == 10

    @patch("project_name.data.load_from_disk")
    def test_subset_does_not_exceed_split_size(
        self, mock_load: MagicMock, tmp_path: Path
    ) -> None:
        """n_samples larger than the split size returns the full split without error."""
        (tmp_path / DATASET_SUBSET).mkdir(parents=True)
        mock_load.return_value = _make_processed_dataset(
            n_train=3, n_validation=2, n_test=2
        )

        with patch("project_name.data.DatasetDict.save_to_disk"):
            result = runner.invoke(
                app,
                [
                    "subset-data",
                    "--processed-dir",
                    str(tmp_path),
                    "--n-samples",
                    "200",
                ],
            )

        assert result.exit_code == 0


# Tests for DataModule setup, collation, and dataloaders.
class TestDataModule:
    """Tests for DataModule setup, collation, and dataloaders."""

    @pytest.fixture()
    def processed_dataset(self) -> DatasetDict:
        """Return a small processed DatasetDict for DataModule tests."""
        return _make_processed_dataset(n_train=8, n_validation=2, n_test=4)

    @pytest.fixture()
    def mock_processor(self) -> MagicMock:
        """Return a mock AutoProcessor that returns plausible tensor dicts.

        The processor and its tokenizer are configured to return MagicMock
        objects so that _collate can run without real model weights.
        """
        processor = MagicMock()
        tokenizer = MagicMock()

        # processor(...) returns a dict-like object with tensor values
        processor.return_value = {"input_ids": torch.zeros(2, 10, dtype=torch.long)}

        # tokenizer(...) returns label encodings with pad_token_id = 0
        tokenizer.return_value = {"input_ids": torch.zeros(2, 5, dtype=torch.long)}
        tokenizer.pad_token_id = 0

        processor.tokenizer = tokenizer
        return processor

    @pytest.fixture()
    def data_module(
        self,
        tmp_path: Path,
        processed_dataset: DatasetDict,
        mock_processor: MagicMock,
    ) -> DataModule:
        """Return a fully set-up DataModule backed by a temp directory.

        Args:
            tmp_path: Pytest-provided temporary directory.
            processed_dataset: Small in-memory DatasetDict.
            mock_processor: Mock AutoProcessor instance.

        Returns:
            A DataModule with setup() already called.
        """
        with patch("project_name.data.load_from_disk", return_value=processed_dataset):
            dm = DataModule(
                processed_dir=tmp_path,
                processor=mock_processor,
                batch_size=2,
                num_workers=0,
            )
            dm.setup()
        return dm

    def test_setup_loads_all_splits(self, data_module: DataModule) -> None:
        """setup() populates self.dataset with all three splits."""
        assert data_module.dataset is not None
        assert set(data_module.dataset.keys()) == {"train", "validation", "test"}

    def test_setup_correct_split_sizes(self, data_module: DataModule) -> None:
        """setup() loads the expected number of samples per split."""
        assert data_module.dataset is not None
        assert len(data_module.dataset["train"]) == 8
        assert len(data_module.dataset["validation"]) == 2
        assert len(data_module.dataset["test"]) == 4

    def test_collate_returns_labels(self, data_module: DataModule) -> None:
        """_collate() output dict contains a 'labels' key."""
        assert data_module.dataset is not None
        samples = [data_module.dataset["train"][i] for i in range(2)]

        with patch("project_name.model.build_prompt", return_value="Q: ..."):
            batch = data_module._collate(samples)

        assert "labels" in batch

    def test_collate_masks_padding(self, data_module: DataModule) -> None:
        """Padding token positions in labels are replaced with -100."""
        assert data_module.dataset is not None
        samples = [data_module.dataset["train"][i] for i in range(2)]

        with patch("project_name.model.build_prompt", return_value="Q: ..."):
            batch = data_module._collate(samples)

        assert (batch["labels"] == -100).any()

    def test_collate_replaces_missing_image(self, data_module: DataModule) -> None:
        """_collate() replaces None images with a blank RGB placeholder."""
        assert data_module.dataset is not None
        samples = [data_module.dataset["train"][i] for i in range(2)]
        samples[0] = dict(samples[0], image=None)

        received_images: list = []

        def _capture(text, images, **kwargs) -> dict:
            received_images.extend(images)
            return {"input_ids": torch.zeros(len(text), 10, dtype=torch.long)}

        data_module.processor.side_effect = _capture

        with patch("project_name.model.build_prompt", return_value="Q: ..."):
            data_module._collate(samples)

        assert isinstance(received_images[0], Image.Image)
        assert received_images[0].size == (224, 224)

    def test_train_dataloader_shuffles(self, data_module: DataModule) -> None:
        """train_dataloader is configured with shuffle=True."""
        dl = data_module.train_dataloader()

        assert dl.batch_size == data_module.batch_size
        assert isinstance(dl.sampler, RandomSampler)

    def test_val_dataloader_no_shuffle(self, data_module: DataModule) -> None:
        """val_dataloader is configured with shuffle=False."""
        dl = data_module.val_dataloader()

        assert isinstance(dl.sampler, SequentialSampler)

    def test_test_dataloader_no_shuffle(self, data_module: DataModule) -> None:
        """test_dataloader is configured with shuffle=False."""
        dl = data_module.test_dataloader()

        assert isinstance(dl.sampler, SequentialSampler)

    def test_data_path_uses_subset(self, tmp_path: Path) -> None:
        """data_path is constructed as processed_dir / subset without extra nesting."""
        dm = DataModule(processed_dir=tmp_path, subset=DATASET_SUBSET)

        assert dm.data_path == tmp_path / DATASET_SUBSET

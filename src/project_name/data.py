"""Module for training the model."""

from pathlib import Path

import typer
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:
        """Initialize the dataset with the path to the raw data."""
        self.data_path = data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return 0

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        return None

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""


def preprocess(data_path: Path, output_folder: Path) -> None:
    """Preprocess the raw data and save it to the output folder."""
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)

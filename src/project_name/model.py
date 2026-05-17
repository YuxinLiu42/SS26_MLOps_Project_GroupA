"""Defines the model architecture for the project."""

import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)


class Model(nn.Module):
    """Simple fully-connected classifier for MNIST."""

    def __init__(self, input_size: int = 784, num_classes: int = 10) -> None:
        """Initialize the model."""
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input shape: (batch, 1, 28, 28)."""
        x = x.view(x.size(0), -1)  # flatten
        return self.network(x)

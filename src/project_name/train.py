"""Module for training the model."""

import hydra
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from project_name.model import Model
import logging

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="train", version_base=None)
def train(cfg: DictConfig) -> None:
    """Train the model on MNIST and save checkpoint."""
    transform = transforms.ToTensor()
    dataset = datasets.MNIST("data/raw", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg.epochs):
        total_loss = 0.0
        for imgs, labels in dataloader:
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{cfg.epochs} — loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "models/model.pt")
    logger.info("Model saved to models/model.pt")


if __name__ == "__main__":
    train()

"""Module for training the model."""

import logging

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from typing import cast
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from project_name.model import Model

logger = logging.getLogger(__name__)

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "loss", "goal": "minimize"},
    "parameters": {
        "lr": {"min": 1e-4, "max": 1e-2},
        "batch_size": {"values": [32, 64, 128]},
        "epochs": {"values": [5, 10]},
    },
}


def sweep_train() -> None:
    """Training function used by W&B sweep agent."""
    wandb.init()
    cfg = wandb.config

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
        wandb.log({"loss": avg_loss, "epoch": epoch + 1})

    wandb.finish()


@hydra.main(config_path="../../configs", config_name="train", version_base=None)
def train(cfg: DictConfig) -> None:
    """Train the model on MNIST and save checkpoint."""
    transform = transforms.ToTensor()
    dataset = datasets.MNIST("data/raw", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    wandb.init(
        project="project_name",
        config=cast(
            dict, OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        ),
    )

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
        wandb.log({"loss": avg_loss, "epoch": epoch + 1})

    torch.save(model.state_dict(), "models/model.pt")
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file("models/model.pt")
    wandb.log_artifact(artifact)
    logger.info("Model saved to models/model.pt")

    wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    if "--sweep" in sys.argv:
        sys.argv.remove("--sweep")
        sweep_id = wandb.sweep(SWEEP_CONFIG, project="project_name")
        wandb.agent(sweep_id, function=sweep_train, count=10)
    else:
        train()

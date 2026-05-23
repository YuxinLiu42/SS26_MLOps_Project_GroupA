"""Module for training the model."""

import lightning as L
import logging

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from typing import cast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from project_name.model import PaliGemmaModule

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
    """Train the model for a W&B sweep."""
    wandb.init()
    cfg = wandb.config

    transform = transforms.ToTensor()
    dataset = datasets.MNIST("data/raw", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = PaliGemmaModule(learning_rate=cfg.lr)
    trainer = L.Trainer(max_epochs=cfg.epochs, enable_checkpointing=False)
    trainer.fit(model, dataloader)
    wandb.finish()


@hydra.main(config_path="../../configs", config_name="train", version_base=None)
def train(cfg: DictConfig) -> None:
    """Train the model and log results to W&B."""
    transform = transforms.ToTensor()
    dataset = datasets.MNIST("data/raw", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = PaliGemmaModule(learning_rate=cfg.lr)

    wandb.init(
        project="project_name",
        config=cast(
            dict, OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        ),
    )

    trainer = L.Trainer(max_epochs=cfg.epochs)
    trainer.fit(model, dataloader)

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

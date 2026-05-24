"""Training entry point for fine-tuning PaliGemma2 on ScienceQA."""

import logging
from pathlib import Path

import hydra
import lightning as L
import wandb
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler

from project_name.data import DataModule
from project_name.model import PaliGemmaModule

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)
log = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("checkpoints")

# Number of test samples logged as a W&B prediction artifact.
N_PREDICTION_SAMPLES = 32


class PredictionLogger(Callback):
    """Lightning callback that logs a sample of test predictions to W&B.

    After the test epoch ends, collects up to n_samples rows of
    (image, question, prediction, ground_truth, correct) and uploads them
    as a wandb.Table artifact named 'test-predictions'.

    Only runs when a WandbLogger is active; silently skips otherwise.

    Args:
        n_samples: Maximum number of rows to include in the artifact table.
    """

    def __init__(self, n_samples: int = N_PREDICTION_SAMPLES) -> None:
        """Initialize PredictionLogger with the desired sample count."""
        super().__init__()
        self.n_samples = n_samples
        self._rows: list[list] = []

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Collect predictions from each test batch until n_samples is reached.

        Decodes model predictions and ground-truth labels from the batch,
        then appends one row per sample to the internal buffer.
        Images are converted to wandb.Image for rich display in the W&B UI.

        Args:
            trainer: The active Lightning Trainer instance.
            pl_module: The PaliGemmaModule being evaluated.
            outputs: Return value of test_step (unused).
            batch: Dict of tensors from DataModule._collate.
            batch_idx: Index of the current test batch.
            dataloader_idx: Index of the dataloader (unused).
        """
        assert isinstance(pl_module, PaliGemmaModule)
        if len(self._rows) >= self.n_samples:
            return

        input_ids = batch["input_ids"]
        pixel_values = batch.get("pixel_values")
        labels = batch["labels"]

        generated_ids = pl_module.model.generate(  # type: ignore[misc]
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=10,
            do_sample=False,
        )

        input_len = input_ids.shape[1]
        preds = pl_module.processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
        )

        labels_ids = labels.clone()
        labels_ids[labels_ids == -100] = pl_module.processor.tokenizer.pad_token_id
        targets = pl_module.processor.batch_decode(labels_ids, skip_special_tokens=True)

        # pixel_values shape: (B, C, H, W)
        for i, (pred, target) in enumerate(zip(preds, targets)):
            if len(self._rows) >= self.n_samples:
                break
            img = (
                wandb.Image(pixel_values[i].float().cpu())
                if pixel_values is not None
                else None
            )
            self._rows.append(
                [img, pred.strip(), target.strip(), pred.strip() == target.strip()]
            )

    def on_test_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Upload the collected prediction rows as a W&B Table artifact.

        Creates a wandb.Artifact of type 'predictions' containing a Table
        with columns: image, prediction, ground_truth, correct.
        Logs the artifact to the active W&B run, then clears the buffer.

        Args:
            trainer: The active Lightning Trainer instance.
            pl_module: The PaliGemmaModule being evaluated (unused).
        """
        if not isinstance(trainer.logger, WandbLogger) or not self._rows:
            return

        table = wandb.Table(columns=["image", "prediction", "ground_truth", "correct"])

        for row in self._rows:
            table.add_data(*row)

        artifact = wandb.Artifact("test-predictions", type="predictions")
        artifact.add(table, "predictions")
        trainer.logger.experiment.log_artifact(artifact)

        log.info("W&B prediction artifact logged | row=%d", len(self._rows))
        self._rows.clear()


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def train(cfg: DictConfig) -> float:
    """Fine-tune PaliGemma2 on the preprocessed ScienceQA-IMG dataset.

    Instantiates PaliGemmaModule and DataModule, wires them together via
    a shared processor, then runs Trainer.fit followed by Trainer.test
    using the best checkpoint.

    The vision encoder is frozen by default since ScienceQA images do not
    require visual feature adaptation. Training uses AdamW with cosine
    annealing, gradient clipping, and early stopping on val/loss.

    Hydra manages config composition and override from the command line.
    W&B logging and hyperparameter sweeps are enabled via cfg.trainer.wandb.

    Args:
        cfg: Hydra DictConfig composed from configs/train.yaml.

    Returns:
        Final val/loss value, used by W&B sweep agent as the optimisation target.
    """
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    L.seed_everything(cfg.seed, workers=True)

    log.info("Initializing model from %s ...", cfg.model.model_name)
    model = PaliGemmaModule(
        model_name=cfg.model.model_name,
        learning_rate=cfg.model.learning_rate,
        freeze_vision_encoder=cfg.model.freeze_vision_encoder,
        freeze_language_model=cfg.model.freeze_language_model,
        gradient_checkpointing=cfg.model.gradient_checkpointing,
        use_lora=cfg.model.use_lora,
    )

    log.info(
        "Initializing data module from %s ...",
        Path(cfg.data.processed_dir) / cfg.data.subset,
    )
    data = DataModule(
        processed_dir=Path(cfg.data.processed_dir),
        subset=cfg.data.subset,
        processor=model.processor,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        max_length=cfg.data.max_length,
        max_label_length=cfg.data.max_label_length,
    )

    logger = None
    if cfg.trainer.wandb.project:
        logger = WandbLogger(
            project=cfg.trainer.wandb.project,
            name=cfg.trainer.wandb.run_name or None,
            tags=list(cfg.trainer.wandb.tags) if cfg.trainer.wandb.tags else None,
            log_model=cfg.trainer.wandb.log_model,
        )
        params = OmegaConf.to_container(cfg, resolve=True)
        logger.log_hyperparams(params)  # type: ignore[arg-type]
        log.info(
            "W&B logging enabled | project=%s, run=%s",
            cfg.trainer.wandb.project,
            cfg.trainer.wandb.run_name,
        )
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.trainer.ckpt_dir,
            filename="paligemma2-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            verbose=True,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=cfg.trainer.patience,
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        PredictionLogger(n_samples=cfg.trainer.wandb.n_prediction_samples),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,  # type: ignore[arg-type]
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        val_check_interval=cfg.trainer.val_check_interval,
        profiler=cfg.trainer.profiler or None,
        callbacks=callbacks,
        logger=logger if logger else True,
        log_every_n_steps=10,
        deterministic=False,
        fast_dev_run=cfg.trainer.fast_dev_run,
    )

    log.info(
        "Starting training | max_epochs=%d | batch_size=%d "
        "| accumulate=%d | effective_batch=%d",
        cfg.trainer.max_epochs,
        cfg.data.batch_size,
        cfg.trainer.accumulate_grad_batches,
        cfg.data.batch_size * cfg.trainer.accumulate_grad_batches,
    )
    trainer.fit(model=model, datamodule=data)

    log.info("Training complete. Running test set evaluation with best checkpoint ...")
    ckpt_path = None if cfg.trainer.fast_dev_run else "best"
    trainer.test(model=model, datamodule=data, ckpt_path=ckpt_path)
    best_val_loss = trainer.checkpoint_callback.best_model_score  # type: ignore[union-attr]
    log.info(
        "Best checkpoint saved at %s | val/loss=%.4f",
        trainer.checkpoint_callback.best_model_path,  # type: ignore[union-attr]
        best_val_loss,
    )

    # Return val/loss
    if cfg.trainer.wandb.enabled:
        wandb.finish()

    return float(best_val_loss)


if __name__ == "__main__":
    train()

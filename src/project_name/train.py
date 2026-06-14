"""Training entry point for fine-tuning PaliGemma2 on ScienceQA."""

import math
import os
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
from urllib.parse import urlparse

from project_name.data import DataModule
from project_name.model import PaliGemmaModule

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)
log = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("checkpoints")
_CONFIGS_DIR = str(Path(__file__).parent.parent.parent / "configs")
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
        if not isinstance(trainer.logger, WandbLogger):
            return
        assert isinstance(pl_module, PaliGemmaModule)
        if len(self._rows) >= self.n_samples:
            return

        input_ids = batch["input_ids"]
        pixel_values = batch.get("pixel_values")
        attention_mask = batch.get("attention_mask")

        generated_ids = pl_module.model.generate(  # type: ignore[misc]
            input_ids=input_ids,
            attention_mask=attention_mask,  # avoid attending to padding
            pixel_values=pixel_values,
            max_new_tokens=10,
            do_sample=False,
        )

        input_len = input_ids.shape[1]
        preds = pl_module.processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
        )

        # Ground truth = answer letter derived from the dataset's `answer`
        # index (carried through _collate), not a decode of the masked labels.
        targets = batch["answer_texts"]

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
                [
                    img,
                    pred.strip().upper(),
                    target.strip().upper(),
                    pred.strip().upper() == target.strip().upper(),
                ]
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


def upload_to_gcs(local_path: Path, gcs_dir: str) -> str:
    """Upload a local file to a gs:// directory.

    Args:
        local_path: Path to the local file to upload.
        gcs_dir: Destination directory as a gs://bucket/prefix URI.

    Returns:
        The full gs:// URI of the uploaded object.
    """
    from google.cloud import storage  # type: ignore[attr-defined]

    parsed = urlparse(gcs_dir)
    blob_name = f"{parsed.path.lstrip('/')}/{local_path.name}".lstrip("/")
    client = storage.Client()
    blob = client.bucket(parsed.netloc).blob(blob_name)
    blob.upload_from_filename(str(local_path))
    return f"gs://{parsed.netloc}/{blob_name}"


def upload_dir_to_gcs(local_dir: Path, gcs_dir: str) -> str:
    """Upload every file under local_dir to gs://.../<local_dir.name>/.

    Used to publish the small LoRA adapter directory (adapter weights + config +
    processor) to the model registry / serving location.

    Args:
        local_dir: Directory whose files are uploaded recursively.
        gcs_dir: Destination directory as a gs://bucket/prefix URI.

    Returns:
        The full gs:// URI of the uploaded directory.
    """
    from google.cloud import storage  # type: ignore[attr-defined]

    parsed = urlparse(gcs_dir)
    prefix = f"{parsed.path.lstrip('/')}/{local_dir.name}".lstrip("/")
    client = storage.Client()
    bucket = client.bucket(parsed.netloc)
    for f in sorted(local_dir.rglob("*")):
        if f.is_file():
            blob = bucket.blob(f"{prefix}/{f.relative_to(local_dir).as_posix()}")
            blob.upload_from_filename(str(f))
    return f"gs://{parsed.netloc}/{prefix}"


def log_adapter_artifact(
    logger: WandbLogger,
    adapter_dir: Path,
    run_tag: str,
    cfg: DictConfig,
    trainer: L.Trainer,
    best_val_acc: float,
) -> None:
    """Version the run's LoRA adapter as a W&B artifact (model registry).

    Every run logs a new version of the same artifact lineage
    ('scienceqa-paligemma2-lora'), tagged with a per-run alias. Promotion is a
    separate step: it adds the `production` alias to the chosen version and
    copies the adapter to the stable GCS serving path — serving reads GCS,
    never W&B (no W&B key in the API container).

    Args:
        logger: Active WandbLogger whose run logs the artifact.
        adapter_dir: Directory written by save_adapter.
        run_tag: W&B run name used to tag this version.
        cfg: Resolved Hydra config (hyperparams recorded in metadata).
        trainer: Trainer after fit/test; provides test/accuracy if available.
        best_val_acc: Best val/accuracy from the checkpoint callback.
    """
    test_acc = trainer.callback_metrics.get("test/accuracy")
    artifact = wandb.Artifact(
        "scienceqa-paligemma2-lora",
        type="model",
        metadata={
            "run_name": run_tag,
            "val_accuracy": float(best_val_acc),
            "test_accuracy": float(test_acc) if test_acc is not None else None,
            "learning_rate": float(cfg.model.learning_rate),
            "effective_batch": int(
                cfg.data.batch_size * cfg.trainer.accumulate_grad_batches
            ),
        },
    )
    artifact.add_dir(str(adapter_dir))
    logger.experiment.log_artifact(artifact, aliases=[f"run-{run_tag}"])
    log.info("Logged W&B adapter artifact version (alias run-%s)", run_tag)


def resolve_learning_rate(cfg: DictConfig) -> float:
    """Resolve the learning rate, applying the sqrt batch-size rule if needed.

    An explicit model.learning_rate always wins (scaling bypassed). Otherwise
    the lr is derived from model.base_learning_rate — defined at
    model.reference_effective_batch — via lr = base * sqrt(eff_batch / ref).
    This keeps a swept base_learning_rate comparable across effective batch
    sizes, so the sweep's LR and accumulation dimensions stay decoupled.

    Args:
        cfg: Composed Hydra config with model/data/trainer groups.

    Returns:
        The learning rate to train with.

    Raises:
        ValueError: If neither model.learning_rate nor model.base_learning_rate
            is set.
    """
    if cfg.model.learning_rate is not None:
        log.info(
            "Explicit model.learning_rate=%.3e set; sqrt scaling bypassed.",
            cfg.model.learning_rate,
        )
        return float(cfg.model.learning_rate)
    if cfg.model.base_learning_rate is None:
        raise ValueError(
            "Set model.learning_rate or model.base_learning_rate in the config."
        )
    eff_batch = cfg.data.batch_size * cfg.trainer.accumulate_grad_batches
    lr = cfg.model.base_learning_rate * math.sqrt(
        eff_batch / cfg.model.reference_effective_batch
    )
    log.info(
        "Derived lr %.3e from base_lr %.3e (sqrt scaling, eff_batch %d, ref %d)",
        lr,
        cfg.model.base_learning_rate,
        eff_batch,
        cfg.model.reference_effective_batch,
    )
    return float(lr)


@hydra.main(version_base="1.3", config_path=_CONFIGS_DIR, config_name="train")
def train(cfg: DictConfig) -> float:
    """Fine-tune PaliGemma2 on the preprocessed ScienceQA-IMG dataset.

    Instantiates PaliGemmaModule and DataModule, wires them together via
    a shared processor, then runs Trainer.fit followed by Trainer.test
    using the best checkpoint.

    The vision encoder is frozen by default since ScienceQA images do not
    require visual feature adaptation. Training uses AdamW with cosine
    annealing, gradient clipping, and early stopping on val/accuracy.

    Hydra manages config composition and override from the command line.
    W&B logging and hyperparameter sweeps are enabled via cfg.trainer.wandb.

    Args:
        cfg: Hydra DictConfig composed from configs/train.yaml.

    Returns:
        Best val/accuracy, the metric the W&B sweep optimises (the agent reads
        it from the logged run history).
    """
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    L.seed_everything(cfg.seed, workers=True)

    # Resolve before model init so the derived value reaches both the module
    # hparams and the W&B-logged config.
    cfg.model.learning_rate = resolve_learning_rate(cfg)

    log.info("Initializing model from %s ...", cfg.model.model_name)
    model = PaliGemmaModule(
        model_name=cfg.model.model_name,
        learning_rate=cfg.model.learning_rate,
        freeze_vision_encoder=cfg.model.freeze_vision_encoder,
        freeze_language_model=cfg.model.freeze_language_model,
        gradient_checkpointing=cfg.model.gradient_checkpointing,
        use_lora=cfg.model.use_lora,
        lora_r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        lora_target_modules=tuple(cfg.model.lora_target_modules),
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
    if cfg.trainer.wandb.enabled:
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
            # auto_insert_metric_name=False so the "val/accuracy" metric's slash
            # is NOT inserted literally (which made Lightning create a 'val/'
            # subdir and mangled the uploaded path to '.../model//...ckpt').
            filename="paligemma2-ep{epoch:02d}-va{val/accuracy:.4f}",
            auto_insert_metric_name=False,
            # Select by ACCURACY, not val/loss: sweep #1 showed they disagree
            # (the best-val/loss trial scored 3 pts below baseline on test).
            monitor="val/accuracy",
            mode="max",
            save_top_k=1,  # full Lightning ckpt is ~6GB; keep just the best
            save_last=False,
            verbose=True,
        ),
        EarlyStopping(
            monitor="val/accuracy",
            patience=cfg.trainer.patience,
            mode="max",
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
    if cfg.trainer.fast_dev_run:
        log.info("fast_dev_run complete — skipping checkpoint summary.")
        return 0.0

    best_val_acc = trainer.checkpoint_callback.best_model_score  # type: ignore[union-attr]
    best_ckpt = trainer.checkpoint_callback.best_model_path  # type: ignore[union-attr]
    log.info(
        "Best checkpoint saved at %s | val/accuracy=%.4f",
        best_ckpt,
        best_val_acc,
    )

    # Export ONLY the LoRA adapter (a few MB) for the registry / serving — NOT
    # the ~6GB full Lightning ckpt. trainer.test(ckpt_path="best") above loaded
    # the best weights into `model`, so it now holds them.
    # Tag the dir with the W&B run so baseline + each sweep trial upload to a
    # distinct GCS path instead of overwriting one another.
    run_tag = "run"
    if logger is not None:
        try:
            run_tag = logger.experiment.name or logger.experiment.id or "run"
        except Exception:  # pragma: no cover - wandb offline / no run
            run_tag = cfg.trainer.wandb.run_name or "run"
    adapter_dir = Path(cfg.trainer.ckpt_dir) / f"adapter-{run_tag}"
    # The adapter/ckpt is saved locally and (for LoRA) already logged to W&B
    # above, so a transient GCS error on the staging upload must NOT fail the
    # run — that would mark an otherwise-successful sweep trial 'failed' and
    # lose only the recoverable GCS copy. Warn and continue.
    if cfg.model.use_lora:
        model.save_adapter(adapter_dir)
        if logger is not None:
            log_adapter_artifact(
                logger, adapter_dir, run_tag, cfg, trainer, best_val_acc
            )
        model_dir = os.environ.get("AIP_MODEL_DIR")
        if model_dir:
            try:
                uri = upload_dir_to_gcs(adapter_dir, model_dir)
                log.info("Uploaded LoRA adapter to %s", uri)
            except Exception:
                log.exception(
                    "GCS upload of the adapter to %s failed — continuing "
                    "(adapter is in W&B; re-upload separately if needed)",
                    model_dir,
                )
    else:
        # No LoRA → fall back to uploading the full checkpoint file.
        model_dir = os.environ.get("AIP_MODEL_DIR")
        if model_dir and best_ckpt:
            try:
                uri = upload_to_gcs(Path(best_ckpt), model_dir)
                log.info("Uploaded full checkpoint to %s", uri)
            except Exception:
                log.exception(
                    "GCS upload of the checkpoint to %s failed — continuing",
                    model_dir,
                )

    if cfg.trainer.wandb.enabled:
        wandb.finish()

    return float(best_val_acc)


if __name__ == "__main__":
    train()

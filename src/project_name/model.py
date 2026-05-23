"""Defines the model architecture for the project."""

import logging
import lightning as L
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch
from rich.logging import RichHandler
from typing import cast

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)
log = logging.getLogger(__name__)

MODEL_NAME = "google/paligemma2-3b-pt-224"


def build_prompt(question: str, choices: list[str]) -> str:
    """Build the PaliGemma2 VQA prompt for a sample.

    Uses the standard PaliGemma2 task prefix 'answer en' for English VQA.
    Choices are formatted as '(A) choice1 (B) choice2 ...'

    Args:
        question: The question text.
        choices: List of answer choice strings.

    Returns:
        Formatted prompt string ready for the processor.
    """
    choices_str = " ".join(
        f"({chr(65+i)}) {choice}" for i, choice in enumerate(choices)
    )
    prompt = f"answer en {question} Choices: {choices_str}"
    return prompt


class PaliGemmaModule(L.LightningModule):
    """PaliGemma2 fine-tuning module for ScienceQA.

    Wraps PaliGemmaForConditionalGeneration in a LightningModule,
    handling training, validation, and test steps, as well as
    optimizer and scheduler configuration.

    The vision encoder is frozen by default since ScienceQA images
    do not require visual feature adaptation — only the language model
    needs to learn the answer format.

    Args:
        model_name: HuggingFace model identifier.
        learning_rate: AdamW learning rate.
        torch_dtype: Weight dtype. bfloat16 recommended for modern GPUs.
        freeze_vision_encoder: Whether to freeze the SigLIP vision encoder.
        freeze_language_model: Whether to freeze the Gemma language model.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        learning_rate: float = 2e-5,
        torch_dtype: torch.dtype = torch.bfloat16,
        freeze_vision_encoder: bool = True,
        freeze_language_model: bool = False,
    ) -> None:
        """Initialize the PaliGemmaModule with model and processor loading."""
        super().__init__()
        self.save_hyperparameters(ignore=["torch_dtype"])
        self.torch_dtype = torch_dtype

        log.info("Loading processor from %s ...", model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

        log.info("Loading model %s ...", model_name)
        self.model = cast(
            PaliGemmaForConditionalGeneration,
            PaliGemmaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
            ),
        )

        if freeze_vision_encoder:
            log.info("Freezing vision encoder parameters.")
            for param in self.model.vision_tower.parameters():
                param.requires_grad = False

        if freeze_language_model:
            log.info("Freezing language model parameters.")
            for param in self.model.language_model.parameters():
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        log.info(
            "Trainable parameters: %.1fM / %.1fM total.",
            trainable / 1e6,
            total / 1e6,
        )

    def forward(self, **inputs):
        """Forward pass through the model."""
        return self.model(**inputs)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Compute cross-entropy loss on a training batch.

        Args:
            batch: Dict of tensors from DataModule._collate, including 'labels'.
            batch_idx: Index of the current batch (unused).

        Returns:
            Scalar loss tensor.
        """
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Compute validation loss on a validation batch.

        Args:
            batch: Dict of tensors from DataModule._collate, including 'labels'.
            batch_idx: Index of the current batch (unused).
        """
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        """Generate predictions and compute exact-match accuracy on a test batch.

        Decodes generated tokens and compares against ground-truth answer_text.
        Padding tokens (-100) in labels are replaced with pad_token_id before decoding.

        Args:
            batch: Dict of tensors from DataModule._collate, including 'labels'.
            batch_idx: Index of the current batch (unused).
        """
        input_ids = batch["input_ids"]
        pixel_values = batch.get("pixel_values")
        labels = batch["labels"]

        generated_ids = self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=10,
            do_sample=False,
        )

        # Decode only newly generated tokens, skipping the prompt
        input_len = input_ids.shape[1]
        preds = self.processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
        )

        # Replace -100 padding mask with pad_token_id before decoding labels
        label_ids = labels.clone()
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        targets = self.processor.batch_decode(label_ids, skip_special_tokens=True)

        correct = sum(p.strip() == t.strip() for p, t in zip(preds, targets))
        acc = correct / len(preds)
        self.log("test/accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """Configure AdamW optimizer with cosine annealing scheduler.

        Returns:
            Dict with optimizer and step-level lr_scheduler.
        """
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.hparams.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

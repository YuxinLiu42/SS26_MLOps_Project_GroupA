"""Defines the model architecture for the project."""

import logging
import lightning as L
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch
from rich.logging import RichHandler
from peft import get_peft_model, LoraConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)
log = logging.getLogger(__name__)

MODEL_NAME = "google/paligemma2-3b-pt-224"


def build_prompt(
    question: str,
    choices: list[str],
    hint: str = "",
    lecture: str = "",
) -> str:
    """Build the PaliGemma2 VQA prompt for a sample.

    Uses the standard PaliGemma2 task prefix 'answer en' for English VQA.
    Choices are formatted as '(A) choice1 (B) choice2 ...'
    Optional hint and lecture are appended when provided.

    Args:
        question: The question text.
        choices: List of answer choice strings.
        hint: Optional hint text appended after the question.
        lecture: Optional background knowledge appended after the hint.

    Returns:
        Formatted prompt string ready for the processor.
    """
    choices_str = " ".join(
        f"({chr(65 + i)}) {choice}" for i, choice in enumerate(choices)
    )
    parts = [f"answer en {question}"]
    if hint:
        parts.append(f"Hint: {hint}")
    if lecture:
        parts.append(f"Lecture: {lecture}")
    parts.append(f"Choices: {choices_str}")
    return " ".join(parts)


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
        gradient_checkpointing: Whether to enable gradient checkpointing
                                for memory efficiency.
        use_lora: Whether to apply LoRA fine-tuning to the language model.
        lora_r: LoRA rank (number of trainable parameters = 2 * r * hidden_size).
        lora_alpha: LoRA scaling factor.
        lora_dropout: LoRA dropout rate.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        learning_rate: float = 2e-5,
        torch_dtype: torch.dtype = torch.bfloat16,
        freeze_vision_encoder: bool = True,
        freeze_language_model: bool = False,
        gradient_checkpointing: bool = False,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ) -> None:
        """Initialize the PaliGemmaModule with model and processor loading."""
        super().__init__()
        self.save_hyperparameters(ignore=["torch_dtype"])
        self.torch_dtype = torch_dtype

        log.info("Loading processor from %s ...", model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

        log.info("Loading model %s ...", model_name)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )

        if freeze_vision_encoder:
            log.info("Freezing vision encoder parameters.")
            for param in self.model.model.vision_tower.parameters():  # type: ignore[union-attr]
                param.requires_grad = False

        if freeze_language_model:
            log.info("Freezing language model parameters.")
            for param in self.model.model.language_model.parameters():  # type: ignore[union-attr]
                param.requires_grad = False

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.model.enable_input_require_grads()  # type: ignore[union-attr]

        if use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                # only match the q_proj and v_proj of the language model
                target_modules=r".*language_model.*\.(q_proj|v_proj)$",
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)  # type: ignore[assignment]

            # log case and adapter shape of every module LoRA actually wrapped
            for name, module in self.model.named_modules():
                if not hasattr(module, "lora_A"):
                    continue
                lora_a = module.lora_A  # type: ignore[attr-defined]
                if len(lora_a) == 0:
                    continue
                adapter = next(iter(lora_a))
                base = module.base_layer  # type: ignore[attr-defined]
                log.info(
                    "LoRA: %s | base %d→%d | A %s | B %s",
                    name,
                    base.in_features,
                    base.out_features,
                    tuple(lora_a[adapter].weight.shape),
                    tuple(module.lora_B[adapter].weight.shape),  # type: ignore[attr-defined]
                )

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        log.info(
            "Trai nable parameters: %.1fM / %.1fM total.",
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

        generated_ids = self.model.generate(  # type: ignore[misc]
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

        correct = sum(
            p.strip().upper() == t.strip().upper() for p, t in zip(preds, targets)
        )
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

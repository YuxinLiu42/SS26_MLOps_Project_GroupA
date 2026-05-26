"""Running single-sample predictions with a fine-tuned PaliGemma2 model."""

import logging
from pathlib import Path

import torch
import typer
from PIL import Image
from rich.logging import RichHandler

from project_name.model import PaliGemmaModule, build_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)
log = logging.getLogger(__name__)

app = typer.Typer(
    help="Run single-sample predictions with a fine-tuned PaliGemma2 checkpoint."
)


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device | None = None,
) -> PaliGemmaModule:
    """Load a trained PaliGemmaModule from a Lightning checkpoint.

    If no device is specified, the best available device is selected
    automatically: CUDA > MPS > CPU.

    Args:
        checkpoint_path: Path to the .ckpt file produced by train.py.
        device: Target device. Auto-detected if None.

    Returns:
        PaliGemmaModule in eval mode, moved to the target device.
    """
    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    log.info("Loading checkpoint from %s on %s ...", checkpoint_path, device)
    module = PaliGemmaModule.load_from_checkpoint(  # type: ignore[operator]
        checkpoint_path,
        map_location=device,
    )
    module.eval()
    log.info("Checkpoint loaded successfully.")
    return module


def predict_single(
    module: PaliGemmaModule,
    image: Image.Image | None,
    question: str,
    choices: list[str],
    max_new_tokens: int = 10,
    **prompt_kwargs: str,
) -> str:
    """Run a single prediction with the PaliGemmaModule.

    Constructs the prompt via build_prompt, runs a greedy generate pass,
    and decodes only the newly generated tokens (input prompt is stripped).
    Optional fields such as hint and lecture are forwarded via prompt_kwargs
    only if they are present in the processed dataset.

    Args:
        module: Loaded PaliGemmaModule in eval mode.
        image: PIL Image, or None for text-only samples.
        question: The question string.
        choices: List of answer choice strings.
        max_new_tokens: Maximum number of tokens to generate in the answer.
        **prompt_kwargs: Optional prompt fields forwarded to build_prompt,
                         e.g. hint="...", lecture="...".

    Returns:
        Tuple of (answer_text, answer_letter) where answer_text is the raw
        decoded string and answer_letter is the corresponding "A"/"B"/"C"/"D",
        or "?" if the prediction does not match any choice.
    """
    built_prompt = build_prompt(question, choices, **prompt_kwargs)
    device = next(module.parameters()).device

    inputs = module.processor(
        text=built_prompt,
        images=image,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        output_ids = module.model.generate(  # type: ignore[misc]
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode only the newly generated tokens, skipping the prompt
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]

    answer_letter = (
        module.processor.decode(generated_ids[0], skip_special_tokens=True)
        .strip()
        .upper()
    )
    return answer_letter


@app.command()
def predict(
    checkpoint: Path = typer.Argument(
        ..., help="Path to the trained .ckpt checkpoint file."
    ),
    question: str = typer.Option(..., "--question", "-q", help="Question text."),
    choices: str = typer.Option(
        ...,
        "--choices",
        "-c",
        help=(
            "Comma-separated answer choices, " "e.g. 'oxygen,carbon dioxide,nitrogen'."
        ),
    ),
    image_path: Path = typer.Option(
        None, "--image", "-i", help="Path to the input image file."
    ),
    hint: str = typer.Option(
        "", "--hint", "-h", help="Optional hint text. Skipped if empty."
    ),
    lecture: str = typer.Option(
        "", "--lecture", "-l", help="Optional lecture text. Skipped if empty."
    ),
    max_new_tokens: int = typer.Option(
        10, help="Maximum number of tokens to generate."
    ),
) -> None:
    r"""Predict the answer to a ScienceQA question from a checkpoint.

    Loads the specified checkpoint, constructs the prompt from the question
    and choices, runs a single greedy inference pass, and prints the result.
    Optional fields (hint, lecture) are forwarded to build_prompt only when
    provided, matching the columns retained during preprocessing.

    Example:
        python -m project_name.predict checkpoints/best.ckpt
            --question "What gas do plants absorb during photosynthesis?" \\
            --choices "oxygen,carbon dioxide,nitrogen" \\
            --image data/sample.png \\
            --hint "Plants need sunlight to grow."
    """
    choices_list = [choice.strip() for choice in choices.split(",")]
    image: Image.Image | None = None
    if image_path is not None:
        image = Image.open(image_path).convert("RGB")
        log.info("Loaded image from %s", image_path)
    else:
        log.warning("No image provided, running text-only inference.")

    # Only forward optional fields that were actually provided,
    # matching whatever columns survived --drop-cols during preprocessing.
    prompt_kwargs: dict[str, str] = {}
    if hint:
        prompt_kwargs["hint"] = hint
    if lecture:
        prompt_kwargs["lecture"] = lecture

    module = load_checkpoint(checkpoint)
    log.info(
        "Running inference | question: %s | choices: %s | prompt_kwargs: %s",
        question,
        choices_list,
        list(prompt_kwargs.keys()) or "<none>",
    )
    answer_letter = predict_single(
        module,
        image=image,
        question=question,
        choices=choices_list,
        max_new_tokens=max_new_tokens,
        **prompt_kwargs,
    )

    log.info("Predicted answer: %s", answer_letter)
    typer.echo(answer_letter)


if __name__ == "__main__":
    app()

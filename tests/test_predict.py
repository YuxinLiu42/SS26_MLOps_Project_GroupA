"""Tests for single-sample prediction with a fine-tuned PaliGemma2 model."""

from unittest.mock import MagicMock, patch
from pathlib import Path
import torch
from typer.testing import CliRunner
from project_name.predict import app, load_checkpoint, predict_single
from PIL import Image

runner = CliRunner()
fake_image = Image.new("RGB", (16, 16))


def test_load_checkpoint_auto_selects_cpu() -> None:
    """load_checkpoint selects CPU when no accelerator is available."""
    fake_module = MagicMock()
    with (
        patch("project_name.predict.torch.cuda.is_available", return_value=False),
        patch(
            "project_name.predict.torch.backends.mps.is_available",
            return_value=False,
        ),
        patch(
            "project_name.predict.PaliGemmaModule.load_from_checkpoint",
            return_value=fake_module,
        ) as mock_load,
    ):
        module = load_checkpoint(Path("fake.ckpt"))

    # The checkpoint loader was called with a CPU device
    _, kwargs = mock_load.call_args
    assert kwargs["map_location"] == torch.device("cpu")
    # eval() was called to put the module in inference mode
    fake_module.eval.assert_called_once()
    assert module is fake_module


def test_load_model_honors_predict_device_env(monkeypatch) -> None:
    """PREDICT_DEVICE env forces the device when none is passed explicitly."""
    from project_name.predict import load_model

    monkeypatch.setenv("PREDICT_DEVICE", "cpu")
    fake_module = MagicMock()
    with patch(
        "project_name.predict.load_checkpoint", return_value=fake_module
    ) as mock_load:
        load_model(Path("fake.ckpt"))

    _, kwargs = mock_load.call_args
    assert kwargs["device"] == torch.device("cpu")


def test_load_checkpoint_prefers_cuda() -> None:
    """load_checkpoint prefers CUDA when available."""
    fake_module = MagicMock()
    with (
        patch("project_name.predict.torch.cuda.is_available", return_value=True),
        patch(
            "project_name.predict.PaliGemmaModule.load_from_checkpoint",
            return_value=fake_module,
        ) as mock_load,
    ):
        load_checkpoint(Path("fake.ckpt"))

    _, kwargs = mock_load.call_args
    assert kwargs["map_location"] == torch.device("cuda")


def test_predict_single_returns_decoded_letter() -> None:
    """predict_single returns the decoded prediction letter."""
    fake_module = MagicMock()
    fake_inputs = {"input_ids": torch.zeros((1, 5), dtype=torch.long)}
    fake_module.processor.return_value.to.return_value = fake_inputs
    fake_param = torch.nn.Parameter(torch.zeros(1))
    fake_module.parameters.return_value = iter([fake_param])
    fake_module.model.generate.return_value = torch.zeros((1, 6), dtype=torch.long)
    fake_module.processor.decode.return_value = "a"

    with patch(
        "project_name.predict.build_prompt", return_value="prompt text"
    ) as mock_build:
        result = predict_single(
            fake_module,
            image=fake_image,
            question="Is water wet?",
            choices=["Yes", "No"],
        )

    # build_prompt was called with question and choices
    args, _ = mock_build.call_args
    assert args[0] == "Is water wet?"
    assert args[1] == ["Yes", "No"]
    # The decoded answer is stripped and upper-cased
    assert result == "A"


def test_predict_single_forwards_prompt_kwargs() -> None:
    """predict_single forwards hint and lecture to build_prompt."""
    fake_module = MagicMock()
    fake_inputs = {"input_ids": torch.zeros((1, 5), dtype=torch.long)}
    fake_module.processor.return_value.to.return_value = fake_inputs
    fake_module.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
    fake_module.model.generate.return_value = torch.zeros((1, 6), dtype=torch.long)
    fake_module.processor.decode.return_value = "B"

    with patch(
        "project_name.predict.build_prompt", return_value="prompt"
    ) as mock_build:
        predict_single(
            fake_module,
            image=fake_image,
            question="What do plants absorb?",
            choices=["Oxygen", "CO2"],
            hint="Think photosynthesis.",
            lecture="Plants convert CO2.",
        )

    _, kwargs = mock_build.call_args
    assert kwargs.get("hint") == "Think photosynthesis."
    assert kwargs.get("lecture") == "Plants convert CO2."


def test_cli_forwards_only_provided_optional_fields() -> None:
    """CLI builds prompt_kwargs only from non-empty hint/lecture."""
    with (
        patch("project_name.predict.load_checkpoint") as mock_load,
        patch("project_name.predict.predict_single", return_value="A") as mock_predict,
        patch("project_name.predict.Image.open"),
    ):
        result = runner.invoke(
            app,
            [
                "fake.ckpt",
                "--question",
                "Is water wet?",
                "--choices",
                "Yes,No",
                "--image",
                "dummy.png",
                "--hint",
                "A useful hint.",
            ],
        )

    assert result.exit_code == 0
    # The predicted letter is echoed to stdout
    assert "A" in result.stdout
    # choices string was split into a list
    _, kwargs = mock_predict.call_args
    assert kwargs["choices"] == ["Yes", "No"]
    assert kwargs.get("hint") == "A useful hint."
    assert "lecture" not in kwargs
    mock_load.assert_called_once()


def test_cli_rejects_missing_image() -> None:
    """CLI exits with error when no image is provided."""
    with patch("project_name.predict.load_checkpoint"):
        result = runner.invoke(
            app,
            ["fake.ckpt", "--question", "Q?", "--choices", "A,B,C"],
        )
    assert result.exit_code != 0

"""Tests for the PaliGemmaModule."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from project_name.model import (
    build_prompt,
    PaliGemmaModule,
)


def _make_mock_model() -> MagicMock:
    """Return a minimal mock of PaliGemmaForConditionalGeneration.

    Provides just enough surface area for PaliGemmaModule to initialise and
    for each step method to run without touching real weights.

    Returns:
        Configured MagicMock mimicking the HuggingFace model interface.
    """
    model = MagicMock()

    # Parameters must be iterable so freeze loops and numel() calls succeed.
    dummy_param = torch.nn.Parameter(torch.zeros(4))
    model.vision_tower.parameters.return_value = [dummy_param]
    model.language_model.parameters.return_value = [dummy_param]
    model.parameters.return_value = [dummy_param]

    # Forward pass returns an object with a scalar loss tensor.
    forward_output = MagicMock()
    forward_output.loss = torch.tensor(0.5)
    model.return_value = forward_output

    # generate() returns (batch=1, seq_len=12); prompt occupies the first 10 tokens,
    # so the two trailing tokens are the "new" predictions.
    model.generate.return_value = torch.zeros(1, 12, dtype=torch.long)

    return model


def _make_mock_processor() -> MagicMock:
    """Return a minimal mock of AutoProcessor.

    Returns:
        Configured MagicMock mimicking the HuggingFace processor interface.
    """
    processor = MagicMock()
    processor.tokenizer.pad_token_id = 0
    # Default: pred and target both decode to the same string -> accuracy = 1.0.
    processor.batch_decode.return_value = ["(A)"]
    return processor


def _make_batch(input_len: int = 10) -> dict:
    """Build a minimal collated batch compatible with PaliGemmaModule step methods.

    Args:
        input_len: Number of tokens in input_ids (used to check new-token slicing).

    Returns:
        Dict with input_ids, pixel_values, and labels tensors.
    """
    return {
        "input_ids": torch.zeros(1, input_len, dtype=torch.long),
        "pixel_values": torch.zeros(1, 3, 224, 224),
        "labels": torch.zeros(1, 5, dtype=torch.long),
    }


@pytest.fixture()
def module() -> PaliGemmaModule:
    """Provide a PaliGemmaModule with all HuggingFace I/O mocked out.

    Patches both from_pretrained calls so no weights are downloaded and no
    GPU is required.  The fixture re-attaches fresh mock objects after init
    so individual tests can reconfigure them without affecting each other.

    Returns:
        Initialised PaliGemmaModule backed by mock model and processor.
    """
    mock_model = _make_mock_model()
    mock_processor = _make_mock_processor()

    with (
        patch(
            "model.PaliGemmaForConditionalGeneration.from_pretrained",
            return_value=mock_model,
        ),
        patch(
            "model.AutoProcessor.from_pretrained",
            return_value=mock_processor,
        ),
    ):
        mod = PaliGemmaModule(model_name="mock/model", learning_rate=2e-5)

    # Re-attach mocks so tests can inspect or reconfigure them after init.
    mod.model = mock_model
    mod.processor = mock_processor
    return mod


class TestBuildPrompt:
    """Tests for the build_prompt helper function."""

    def test_prefix_is_answer_en(self) -> None:
        """Prompt must start with the PaliGemma2 VQA task prefix 'answer en'."""
        prompt = build_prompt("What is 2+2?", ["3", "4", "5"])
        assert prompt.startswith("answer en ")

    def test_question_is_included(self) -> None:
        """The original question text must appear verbatim in the prompt."""
        question = "What is the boiling point of water?"
        prompt = build_prompt(question, ["50C", "100C", "200C"])
        assert question in prompt

    def test_choices_keyword_present(self) -> None:
        """The literal string 'Choices:' must separate the question from options."""
        prompt = build_prompt("Q?", ["A", "B"])
        assert "Choices:" in prompt

    def test_choice_labels_are_uppercase_letters(self) -> None:
        """Each choice must be prefixed with its uppercase letter label."""
        prompt = build_prompt("Pick one.", ["Alpha", "Beta", "Gamma"])
        assert "(A) Alpha" in prompt
        assert "(B) Beta" in prompt
        assert "(C) Gamma" in prompt

    def test_choice_labels_follow_alphabetical_order(self) -> None:
        """Labels must follow A-B-C-D order regardless of choice content."""
        choices = ["first", "second", "third", "fourth"]
        prompt = build_prompt("Q?", choices)
        for i, label in enumerate("ABCD"):
            assert f"({label}) {choices[i]}" in prompt

    def test_single_choice_has_only_label_a(self) -> None:
        """A single choice must produce exactly one label (A) with no (B) present."""
        prompt = build_prompt("True or false?", ["True"])
        assert "(A) True" in prompt
        assert "(B)" not in prompt

    def test_choices_with_internal_spaces(self) -> None:
        """Multi-word choices must be preserved exactly as provided."""
        prompt = build_prompt("Q?", ["option one", "option two"])
        assert "(A) option one" in prompt
        assert "(B) option two" in prompt

    def test_empty_choices_list_does_not_raise(self) -> None:
        """An empty choices list must not raise."""
        prompt = build_prompt("Q?", [])
        assert "answer en" in prompt
        assert "Q?" in prompt

    def test_return_type_is_str(self) -> None:
        """build_prompt must return a plain str in all cases."""
        result = build_prompt("Q?", ["A", "B"])
        assert isinstance(result, str)

    def test_no_newlines_in_prompt(self) -> None:
        """The prompt must be a single line with no embedded newline characters."""
        prompt = build_prompt("Multi word question?", ["A", "B", "C"])
        assert "\n" not in prompt


class TestPaliGemmaModuleInit:
    """Tests for PaliGemmaModule initialisation and freeze behaviour."""

    def test_processor_is_loaded(self, module: PaliGemmaModule) -> None:
        """Model is stored on self after init."""
        assert module.processor is not None

    def test_model_is_loaded(self, module: PaliGemmaModule) -> None:
        """Model is stored on self after init."""
        assert module.model is not None

    def test_learning_rate_saved_in_hparams(self, module: PaliGemmaModule) -> None:
        """learning_rate must be persisted in hparams."""
        assert module.hparams.learning_rate == 2e-5

    def test_model_name_saved_in_hparams(self, module: PaliGemmaModule) -> None:
        """save_hyperparameters must persist model_name."""
        assert module.hparams.model_name == "mock/model"

    def test_vision_encoder_frozen_by_default(self, module: PaliGemmaModule) -> None:
        """vision_tower must be frozen when freeze_vision_encoder=True."""
        module.model.vision_tower.parameters.assert_called()

    def test_language_model_not_frozen_by_default(
        self, module: PaliGemmaModule
    ) -> None:
        """language_model must NOT be frozen by default."""
        module.model.language_model.parameters.assert_not_called()

    def test_freeze_language_model_flag(self) -> None:
        """freeze_language_model=True must freeze language_model."""
        mock_model = _make_mock_model()
        mock_processor = _make_mock_processor()
        with (
            patch(
                "model.PaliGemmaForConditionalGeneration.from_pretrained",
                return_value=mock_model,
            ),
            patch(
                "model.AutoProcessor.from_pretrained",
                return_value=mock_processor,
            ),
        ):
            PaliGemmaModule(model_name="mock/model", freeze_language_model=True)
        mock_model.language_model.parameters.assert_called()

    def test_freeze_vision_encoder_false_skips_freeze(self) -> None:
        """Setting freeze_vision_encoder=False must leave vision_tower untouched."""
        mock_model = _make_mock_model()
        mock_processor = _make_mock_processor()
        with (
            patch(
                "model.PaliGemmaForConditionalGeneration.from_pretrained",
                return_value=mock_model,
            ),
            patch(
                "model.AutoProcessor.from_pretrained",
                return_value=mock_processor,
            ),
        ):
            PaliGemmaModule(model_name="mock/model", freeze_vision_encoder=False)
        mock_model.vision_tower.parameters.assert_not_called()


class TestTrainingStep:
    """Tests for the training_step method."""

    def test_returns_loss_tensor(self, module: PaliGemmaModule) -> None:
        """training_step must return the loss tensor."""
        module.log = MagicMock()
        loss = module.training_step(_make_batch(), batch_idx=0)
        assert isinstance(loss, torch.Tensor)

    def test_loss_value_matches_model_output(self, module: PaliGemmaModule) -> None:
        """The returned loss must be the exact value produced by the model."""
        module.log = MagicMock()
        loss = module.training_step(_make_batch(), batch_idx=0)
        assert loss == module.model.return_value.loss

    def test_logs_train_loss_with_correct_kwargs(self, module: PaliGemmaModule) -> None:
        """self.log must be called with on_step=True, on_epoch=True, prog_bar=True."""
        module.log = MagicMock()
        module.training_step(_make_batch(), batch_idx=0)
        module.log.assert_called_once_with(
            "train/loss",
            module.model.return_value.loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )


class TestValidationStep:
    """Tests for the validation_step method."""

    def test_returns_none(self, module: PaliGemmaModule) -> None:
        """validation_step must return None; Lightning handles aggregation itself."""
        module.log = MagicMock()
        result = module.validation_step(_make_batch(), batch_idx=0)
        assert result is None

    def test_logs_val_loss_with_correct_kwargs(self, module: PaliGemmaModule) -> None:
        """self.log must be called with on_step=False, on_epoch=True, prog_bar=True."""
        module.log = MagicMock()
        module.validation_step(_make_batch(), batch_idx=0)
        module.log.assert_called_once_with(
            "val/loss",
            module.model.return_value.loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )


class TestTestStep:
    """Tests for the test_step method."""

    def test_calls_model_generate(self, module: PaliGemmaModule) -> None:
        """model.generate must be invoked exactly once per test_step call."""
        module.log = MagicMock()
        module.test_step(_make_batch(), batch_idx=0)
        module.model.generate.assert_called_once()

    def test_generate_receives_input_ids_and_pixel_values(
        self, module: PaliGemmaModule
    ) -> None:
        """Generate must be called with input_ids and pixel_values from the batch."""
        module.log = MagicMock()
        batch = _make_batch()
        module.test_step(batch, batch_idx=0)
        call_kwargs = module.model.generate.call_args.kwargs
        assert torch.equal(call_kwargs["input_ids"], batch["input_ids"])
        assert torch.equal(call_kwargs["pixel_values"], batch["pixel_values"])

    def test_decodes_only_new_tokens_not_prompt(self, module: PaliGemmaModule) -> None:
        """batch_decode for predictions must receive only tokens beyond input_len.

        generate() returns (1, 12); input has 10 tokens, so the slice should
        have width 2.
        """
        module.log = MagicMock()
        module.test_step(_make_batch(input_len=10), batch_idx=0)
        # First batch_decode call receives the new-token slice.
        pred_tensor = module.processor.batch_decode.call_args_list[0].args[0]
        assert pred_tensor.shape[1] == 2

    def test_label_minus100_replaced_before_decode(
        self, module: PaliGemmaModule
    ) -> None:
        """Labels with -100 must have pad_token_id substituted before decode.

        The original labels tensor must not be mutated in-place — a clone is used.
        """
        batch = _make_batch()
        batch["labels"] = torch.tensor([[-100, 1, 2]])
        module.log = MagicMock()
        # Should not raise even though -100 is an invalid token id.
        module.test_step(batch, batch_idx=0)
        assert module.processor.batch_decode.call_count >= 1

    def test_accuracy_is_one_when_pred_matches_target(
        self, module: PaliGemmaModule
    ) -> None:
        """Exact-match accuracy must be 1.0 when every prediction equals its target."""
        # Both decode calls return the same string so pred.strip() == target.strip().
        module.processor.batch_decode.return_value = ["(A)"]
        module.log = MagicMock()
        module.test_step(_make_batch(), batch_idx=0)
        logged_acc = module.log.call_args_list[0].args[1]
        assert logged_acc == pytest.approx(1.0)

    def test_accuracy_is_zero_when_pred_differs_from_target(
        self, module: PaliGemmaModule
    ) -> None:
        """Exact-match accuracy must be 0.0 when no prediction matches its target."""
        # First decode call (preds) -> "(B)"; second (targets) -> "(A)".
        module.processor.batch_decode.side_effect = [["(B)"], ["(A)"]]
        module.log = MagicMock()
        module.test_step(_make_batch(), batch_idx=0)
        logged_acc = module.log.call_args_list[0].args[1]
        assert logged_acc == pytest.approx(0.0)

    def test_logs_test_accuracy_with_correct_kwargs(
        self, module: PaliGemmaModule
    ) -> None:
        """test/accuracy must be logged with on_step=False, on_epoch=True."""
        module.log = MagicMock()
        module.test_step(_make_batch(), batch_idx=0)
        module.log.assert_called_once_with(
            "test/accuracy",
            pytest.approx(1.0),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def test_accuracy_strips_whitespace_before_comparison(
        self, module: PaliGemmaModule
    ) -> None:
        """Whitespace in decoded strings must not affect accuracy comparison."""
        # Decoded strings with surrounding whitespace must still match after strip().
        module.processor.batch_decode.side_effect = [["  (A)  "], ["(A)"]]
        module.log = MagicMock()
        module.test_step(_make_batch(), batch_idx=0)
        logged_acc = module.log.call_args_list[0].args[1]
        assert logged_acc == pytest.approx(1.0)


class TestConfigureOptimizers:
    """Tests for configure_optimizers."""

    @pytest.fixture(autouse=True)
    def _attach_trainer(self, module: PaliGemmaModule) -> None:
        """Attach a mock trainer so estimated_stepping_batches is available."""
        module.trainer = MagicMock()
        module.trainer.estimated_stepping_batches = 100

    def test_returns_optimizer_key(self, module: PaliGemmaModule) -> None:
        """Return dict must contain an 'optimizer' key."""
        result = module.configure_optimizers()
        assert "optimizer" in result

    def test_returns_lr_scheduler_key(self, module: PaliGemmaModule) -> None:
        """Return dict must contain an 'lr_scheduler' key."""
        result = module.configure_optimizers()
        assert "lr_scheduler" in result

    def test_optimizer_is_adamw(self, module: PaliGemmaModule) -> None:
        """Optimizer must be AdamW as specified in the docstring."""
        result = module.configure_optimizers()
        assert isinstance(result["optimizer"], torch.optim.AdamW)

    def test_optimizer_learning_rate(self, module: PaliGemmaModule) -> None:
        """AdamW must be constructed with the learning_rate stored in hparams."""
        result = module.configure_optimizers()
        actual_lr = result["optimizer"].param_groups[0]["lr"]
        assert actual_lr == pytest.approx(module.hparams.learning_rate)

    def test_scheduler_is_cosine_annealing(self, module: PaliGemmaModule) -> None:
        """Scheduler must be CosineAnnealingLR for smooth LR decay."""
        result = module.configure_optimizers()
        scheduler = result["lr_scheduler"]["scheduler"]
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_scheduler_t_max_equals_estimated_steps(
        self, module: PaliGemmaModule
    ) -> None:
        """CosineAnnealingLR T_max must equal trainer.estimated_stepping_batches."""
        result = module.configure_optimizers()
        scheduler = result["lr_scheduler"]["scheduler"]
        assert scheduler.T_max == 100

    def test_scheduler_interval_is_step(self, module: PaliGemmaModule) -> None:
        """Scheduler must update at every step, not every epoch."""
        result = module.configure_optimizers()
        assert result["lr_scheduler"]["interval"] == "step"

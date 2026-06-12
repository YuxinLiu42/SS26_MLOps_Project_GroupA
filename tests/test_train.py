"""Tests for training entry-point helpers."""

import math

import pytest
from omegaconf import DictConfig, OmegaConf

from project_name.train import resolve_learning_rate


def _make_cfg(
    learning_rate: float | None = None,
    base_learning_rate: float | None = 1e-4,
    reference_effective_batch: int = 16,
    batch_size: int = 4,
    accumulate_grad_batches: int = 4,
) -> DictConfig:
    """Return a minimal config covering the keys resolve_learning_rate reads.

    Args:
        learning_rate: Explicit lr; None means "derive via sqrt scaling".
        base_learning_rate: Base lr defined at the reference effective batch.
        reference_effective_batch: Effective batch the base lr is defined at.
        batch_size: Per-device batch size.
        accumulate_grad_batches: Gradient accumulation steps.

    Returns:
        DictConfig mirroring the model/data/trainer groups of the train config.
    """
    return OmegaConf.create(
        {
            "model": {
                "learning_rate": learning_rate,
                "base_learning_rate": base_learning_rate,
                "reference_effective_batch": reference_effective_batch,
            },
            "data": {"batch_size": batch_size},
            "trainer": {"accumulate_grad_batches": accumulate_grad_batches},
        }
    )


class TestResolveLearningRate:
    """Tests for the sqrt batch-size scaling rule."""

    def test_reference_batch_returns_base_lr(self) -> None:
        """At the reference effective batch the derived lr equals base_lr."""
        cfg = _make_cfg(batch_size=4, accumulate_grad_batches=4)
        assert resolve_learning_rate(cfg) == pytest.approx(1e-4)

    def test_smaller_batch_scales_down_by_sqrt(self) -> None:
        """Halving the effective batch must scale the lr by sqrt(1/2)."""
        cfg = _make_cfg(batch_size=4, accumulate_grad_batches=2)
        assert resolve_learning_rate(cfg) == pytest.approx(1e-4 * math.sqrt(0.5))

    def test_larger_batch_scales_up_by_sqrt(self) -> None:
        """Doubling the effective batch must scale the lr by sqrt(2)."""
        cfg = _make_cfg(batch_size=4, accumulate_grad_batches=8)
        assert resolve_learning_rate(cfg) == pytest.approx(1e-4 * math.sqrt(2.0))

    def test_explicit_learning_rate_bypasses_scaling(self) -> None:
        """An explicit learning_rate must win over the derived value."""
        cfg = _make_cfg(learning_rate=3e-4, accumulate_grad_batches=2)
        assert resolve_learning_rate(cfg) == pytest.approx(3e-4)

    def test_returns_plain_float(self) -> None:
        """The resolved lr must be a plain float, safe to assign into the cfg."""
        cfg = _make_cfg(batch_size=2, accumulate_grad_batches=8)
        assert isinstance(resolve_learning_rate(cfg), float)

    def test_raises_when_both_unset(self) -> None:
        """Neither learning_rate nor base_learning_rate set must raise."""
        cfg = _make_cfg(base_learning_rate=None)
        with pytest.raises(ValueError, match="base_learning_rate"):
            resolve_learning_rate(cfg)

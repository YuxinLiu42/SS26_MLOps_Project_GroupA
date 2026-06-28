"""Unit tests for the pruning utility in optimize.py (CPU, self-contained).

``prune_linear_layers`` operates on any ``nn.Module``, so these run against a
tiny in-test network — no model files, no GPU, no network. The CUDA-only paths
(merge_and_unload, generate, the prune-sweep loop) are exercised on Vertex.
"""

import pytest
import torch
from typer.testing import CliRunner

from scipali.models.optimize import _mask_pruned_grads, app, prune_linear_layers


class _TinyNet(torch.nn.Module):
    """A few Linear layers plus a non-Linear param, standing in for a real model."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(16, 32)  # non-Linear: must stay untouched
        self.fc1 = torch.nn.Linear(32, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.head = torch.nn.Linear(64, 8)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.head(self.fc2(self.fc1(self.embed(idx).mean(1))))


def _linears(model: torch.nn.Module) -> list[torch.nn.Linear]:
    return [m for m in model.modules() if isinstance(m, torch.nn.Linear)]


@pytest.mark.parametrize("amount", [0.3, 0.5, 0.7])
def test_prune_reaches_target_sparsity(amount: float) -> None:
    """Global pruning zeros ~`amount` of the Linear weights and bakes it in.

    Guards the global-threshold logic: a single cutoff must still land the
    *actual* sparsity within 1% of the target at every level.
    """
    model = _TinyNet()
    achieved = prune_linear_layers(model, amount=amount)
    assert abs(achieved - amount) < 0.01, f"sparsity off: {achieved} vs {amount}"

    # prune.remove ran: no reparam buffers left, zeros live in the dense weight.
    linears = _linears(model)
    assert all(not hasattr(m, "weight_orig") for m in linears)
    zeros = sum(int((m.weight == 0).sum()) for m in linears)
    total = sum(m.weight.numel() for m in linears)
    assert abs(zeros / total - amount) < 0.01, (
        f"sparsity off: {zeros / total} vs {amount}"
    )


@pytest.mark.parametrize("amount", [0.1, 0.9, 0.99])
def test_prune_extreme_sparsity(amount: float) -> None:
    """Extreme low/high sparsity still lands within 1% and the net still forwards.

    Guards the histogram cutoff-bin edges (near the start/end of the |w| range)
    and very-high sparsity where almost everything is zeroed.
    """
    model = _TinyNet()
    achieved = prune_linear_layers(model, amount=amount)
    assert abs(achieved - amount) < 0.01, f"sparsity off: {achieved} vs {amount}"
    out = model(torch.randint(0, 16, (2, 4)))
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


def test_prune_leaves_non_linear_params_untouched() -> None:
    """Only nn.Linear weights are pruned — embeddings etc. are left alone."""
    model = _TinyNet()
    before = model.embed.weight.detach().clone()
    prune_linear_layers(model, amount=0.5)
    assert torch.equal(model.embed.weight, before)


def test_pruned_linear_still_forwards() -> None:
    """A pruned net remains usable — guards against a broken prune.remove."""
    model = _TinyNet()
    prune_linear_layers(model, amount=0.5)
    out = model(torch.randint(0, 16, (2, 4)))
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


def test_prune_zero_is_noop() -> None:
    """amount=0 applies no pruning (only incidental zeros from init)."""
    model = _TinyNet()
    assert prune_linear_layers(model, amount=0.0) < 0.01


def test_grad_masking_preserves_sparsity() -> None:
    """Zeroing the gradient at pruned weights keeps the model sparse."""
    torch.manual_seed(0)
    model = _TinyNet()
    achieved = prune_linear_layers(model, amount=0.5)  # dense, zeros baked in
    assert abs(achieved - 0.5) < 0.01
    _mask_pruned_grads(model)
    kept = {id(m): (m.weight != 0).clone() for m in _linears(model)}

    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1.0)
    for _ in range(5):
        opt.zero_grad()
        model(torch.randint(0, 16, (4, 4))).sum().backward()
        opt.step()

    linears = _linears(model)
    # still ~50% sparse, and every originally-pruned position is still exactly 0
    zeros = sum(int((m.weight == 0).sum()) for m in linears)
    total = sum(m.weight.numel() for m in linears)
    assert abs(zeros / total - 0.5) < 0.01
    for m in linears:
        assert torch.all(m.weight[~kept[id(m)]] == 0)


def test_cli_exposes_commands() -> None:
    """run_optimize.sh invokes these exact subcommands for keeping the names stable."""
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "benchmark" in result.output
    assert "prune-sweep" in result.output
    assert "prune-finetune" in result.output

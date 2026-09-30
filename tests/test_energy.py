"""The Hutchinson energy map estimates Yan et al.'s Eq. 19 exactly in expectation.

``E_l(t, t') = ||d h_final,t' / d h_l,t||_F^2``. The exact block comes from the
already-tested identity-cotangent path, restricted to one source and one target
position; the estimator must converge to it and must respect causality.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from jsteer.jacobian import energy_map_for_prompt, jacobian_for_prompt
from tests.tiny_model import TinyDecoder

PROMPT = "the quick brown fox jumps over the lazy dog and keeps on running"
LAYERS = [1, 2]
SKIP_FIRST = 4


class _CausalMixBlock(nn.Module):
    """A residual block that reads a causal running mean over positions.

    The stock tiny decoder is position-wise, so every off-diagonal energy is
    exactly zero and a test on it validates only the diagonal. This mixes
    earlier positions into later ones, as attention does, so ``t < t'`` cells
    carry real energy and the estimator is tested where it matters.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.linear.weight.mul_(0.3)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        counts = torch.arange(1, hidden.shape[1] + 1, device=hidden.device)
        causal_mean = hidden.cumsum(dim=1) / counts[None, :, None]
        return hidden + torch.tanh(self.linear(causal_mean))


@pytest.fixture(scope="module")
def model() -> TinyDecoder:
    torch.manual_seed(0)
    decoder = TinyDecoder(n_layers=5, d_model=8, seed=0)
    decoder.layers = nn.ModuleList(_CausalMixBlock(8) for _ in decoder.layers)
    return decoder


def _exact(model, layer: int, t: int, t_prime: int) -> float:
    block = jacobian_for_prompt(
        model,
        PROMPT,
        [layer],
        source_positions=[t],
        target_positions=[t_prime],
        skip_first=SKIP_FIRST,
    )[layer]
    return float(block.pow(2).sum())


def test_causality_is_exact(model: TinyDecoder) -> None:
    energies, targets, seq_len = energy_map_for_prompt(
        model, PROMPT, LAYERS, target_positions=[6, 10], n_probe=16,
        skip_first=SKIP_FIRST,
    )
    for layer in LAYERS:
        for j, t_prime in enumerate(targets):
            assert torch.all(energies[layer][t_prime + 1 :, j] == 0), (layer, t_prime)


def test_converges_to_exact_frobenius(model: TinyDecoder) -> None:
    t_prime = 10
    energies, targets, _ = energy_map_for_prompt(
        model, PROMPT, LAYERS, target_positions=[t_prime], n_probe=20000,
        skip_first=SKIP_FIRST, seed=3,
    )
    assert targets == [t_prime]
    checked = 0
    for layer in LAYERS:
        for t in range(SKIP_FIRST, t_prime + 1):
            exact = _exact(model, layer, t, t_prime)
            estimate = float(energies[layer][t, 0])
            if exact < 1e-10:
                assert estimate < 1e-8
                continue
            # Hutchinson on a PSD trace: relative sd ~ sqrt(2 / n_probe) ~ 1%.
            assert estimate == pytest.approx(exact, rel=0.06), (layer, t)
            checked += 1
    assert checked > 0, "no nonzero cells -- the test proved nothing"
    # And specifically off the diagonal, which is the component under study.
    assert _exact(model, LAYERS[0], SKIP_FIRST, t_prime) > 1e-6


def test_components_sum_to_the_standard_pullback(model: TinyDecoder) -> None:
    """``diag + off`` is the fitting estimator's own pullback, not a variant."""
    from jsteer.jacobian import component_pullbacks_for_prompt, pullback_for_prompt

    torch.manual_seed(2)
    cotangents = torch.randn(4, model.d_model)
    parts = component_pullbacks_for_prompt(
        model, PROMPT, LAYERS, cotangents, dim_batch=4, skip_first=SKIP_FIRST
    )
    reference = pullback_for_prompt(
        model, PROMPT, LAYERS, cotangents, dim_batch=4, skip_first=SKIP_FIRST
    )
    for layer in LAYERS:
        torch.testing.assert_close(
            parts[layer]["total"], reference[layer], rtol=1e-4, atol=1e-6
        )
        torch.testing.assert_close(
            parts[layer]["diag"] + parts[layer]["off"],
            parts[layer]["total"],
            rtol=1e-5,
            atol=1e-7,
        )
        # The split has to be non-trivial, or the test passes on a bug that
        # puts everything in one component.
        assert parts[layer]["diag"].norm() > 1e-6
        assert parts[layer]["off"].norm() > 1e-6


def test_horizon_buckets_sum_to_the_standard_pullback(model: TinyDecoder) -> None:
    """Binning the pullback by ``d = t' - t`` must partition it, not lose mass.

    Every ``(t, t')`` pair with ``t <= t'`` lands in exactly one bucket, so the
    buckets must add up to ``total`` -- which is itself already tested against
    ``pullback_for_prompt``. A bucket boundary that double-counted or dropped a
    horizon would show up nowhere else.
    """
    from jsteer.jacobian import HORIZON_BUCKETS, horizon_pullbacks_for_prompt

    torch.manual_seed(0)
    cotangents = torch.randn(3, model.d_model)
    parts, counts = horizon_pullbacks_for_prompt(
        model, PROMPT, LAYERS, cotangents, skip_first=SKIP_FIRST
    )
    names = [n for n in parts[LAYERS[0]] if n != "total"]
    assert len(names) == len(HORIZON_BUCKETS)

    for layer in LAYERS:
        stacked = torch.stack([parts[layer][n] for n in names]).sum(dim=0)
        torch.testing.assert_close(stacked, parts[layer]["total"], rtol=2e-4, atol=2e-5)

    # The d=0 bucket must hold exactly one pair per target position, and the
    # partition must account for every causal pair.
    n_valid = len([p for p in range(len(model.tokenizer.encode(PROMPT, add_special_tokens=False))) if p >= SKIP_FIRST])
    assert counts["h0"] == n_valid
    assert sum(counts.values()) == n_valid * (n_valid + 1) // 2


def test_horizon_zero_matches_the_diagonal_component(model: TinyDecoder) -> None:
    """The ``d = 0`` bucket is Yan et al.'s diagonal term, by construction."""
    from jsteer.jacobian import (
        component_pullbacks_for_prompt,
        horizon_pullbacks_for_prompt,
    )

    torch.manual_seed(0)
    cotangents = torch.randn(3, model.d_model)
    comp = component_pullbacks_for_prompt(
        model, PROMPT, LAYERS, cotangents, skip_first=SKIP_FIRST
    )
    horizon, _ = horizon_pullbacks_for_prompt(
        model, PROMPT, LAYERS, cotangents, skip_first=SKIP_FIRST
    )
    for layer in LAYERS:
        torch.testing.assert_close(
            horizon[layer]["h0"], comp[layer]["diag"], rtol=1e-5, atol=1e-6
        )

"""The correctness claims the rest of the project rests on.

1. Our full ``J_x`` is *bit-for-bit* the one ``jlens.fitting`` computes -- if it
   were not, we would be studying a different object from the lens.
2. The batched pullback ``J_x^T c`` equals the explicit matrix product, so the
   cheap path and the expensive path agree.
"""

from __future__ import annotations

import jlens.fitting
import pytest
import torch

from jsteer.jacobian import jacobian_for_prompt, pullback_for_prompt
from tests.tiny_model import TinyDecoder

PROMPT = (
    "the quick brown fox jumps over the lazy dog and keeps on running past the gate"
)
SOURCE_LAYERS = [1, 2]
SKIP_FIRST = 4


@pytest.fixture(scope="module")
def model() -> TinyDecoder:
    return TinyDecoder(n_layers=5, d_model=8, seed=0)


def test_matches_jlens_reference(model: TinyDecoder) -> None:
    """Identity cotangents reproduce ``jlens.fitting.jacobian_for_prompt``."""
    reference, ref_seq_len, ref_valid = jlens.fitting.jacobian_for_prompt(
        model, PROMPT, SOURCE_LAYERS, dim_batch=3, skip_first=SKIP_FIRST
    )
    ours = jacobian_for_prompt(
        model, PROMPT, SOURCE_LAYERS, dim_batch=3, skip_first=SKIP_FIRST
    )
    assert (ours.seq_len, ours.n_valid_positions) == (ref_seq_len, ref_valid)
    for layer in SOURCE_LAYERS:
        # Same graph, same reduction, same order of operations -> exact.
        assert torch.equal(ours[layer], reference[layer]), f"layer {layer}"


def test_pullback_matches_explicit_jacobian(model: TinyDecoder) -> None:
    """``pullback(c) == J_x^T c``, the identity the batching relies on."""
    torch.manual_seed(1)
    cotangents = torch.randn(5, model.d_model)

    J = jacobian_for_prompt(
        model, PROMPT, SOURCE_LAYERS, dim_batch=3, skip_first=SKIP_FIRST
    )
    g = pullback_for_prompt(
        model, PROMPT, SOURCE_LAYERS, cotangents, dim_batch=3, skip_first=SKIP_FIRST
    )
    for layer in SOURCE_LAYERS:
        expected = cotangents @ J[layer]  # [K, d] @ [d, d] == (J^T c)^T
        torch.testing.assert_close(g[layer], expected, rtol=1e-5, atol=1e-6)


def test_pullback_is_linear(model: TinyDecoder) -> None:
    """Pulling back a sum equals summing the pullbacks."""
    torch.manual_seed(2)
    a, b = torch.randn(1, model.d_model), torch.randn(1, model.d_model)
    stacked = torch.cat([a, b, a + 3 * b])
    g = pullback_for_prompt(
        model, PROMPT, [2], stacked, dim_batch=2, skip_first=SKIP_FIRST
    )[2]
    torch.testing.assert_close(g[2], g[0] + 3 * g[1], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dim_batch", [1, 2, 3, 8, 64])
def test_batching_is_invariant(model: TinyDecoder, dim_batch: int) -> None:
    """``dim_batch`` is a memory knob only; it must not change the answer.

    Includes ``dim_batch`` larger than K, which the batch must clamp to.
    """
    torch.manual_seed(3)
    cotangents = torch.randn(7, model.d_model)
    reference = pullback_for_prompt(
        model, PROMPT, [2], cotangents, dim_batch=1, skip_first=SKIP_FIRST
    )[2]
    got = pullback_for_prompt(
        model, PROMPT, [2], cotangents, dim_batch=dim_batch, skip_first=SKIP_FIRST
    )[2]
    torch.testing.assert_close(got, reference, rtol=1e-5, atol=1e-6)


def test_backward_pass_count(model: TinyDecoder) -> None:
    """The cost claim: K cotangents cost ceil(K / dim_batch) passes."""
    cotangents = torch.randn(10, model.d_model)
    assert (
        pullback_for_prompt(
            model, PROMPT, [2], cotangents, dim_batch=4, skip_first=SKIP_FIRST
        ).n_backward_passes
        == 3
    )
    # ...against d_model passes for the full matrix at the same dim_batch.
    assert (
        jacobian_for_prompt(
            model, PROMPT, [2], dim_batch=4, skip_first=SKIP_FIRST
        ).n_backward_passes
        == 2
    )


def test_rejects_bad_cotangent_shape(model: TinyDecoder) -> None:
    with pytest.raises(ValueError, match="d_model"):
        pullback_for_prompt(model, PROMPT, [2], torch.randn(3, model.d_model + 1))
    with pytest.raises(ValueError, match="empty"):
        pullback_for_prompt(model, PROMPT, [2], torch.zeros(0, model.d_model))


def test_source_must_precede_target(model: TinyDecoder) -> None:
    with pytest.raises(ValueError, match="must all be <"):
        pullback_for_prompt(
            model, PROMPT, [3], torch.randn(1, model.d_model), target_layer=2
        )


def test_explicit_positions_work_on_short_prompts(model: TinyDecoder) -> None:
    """The fitting convention is unusable on the paper's steering prompts.

    ``skip_first=16`` needs >17 tokens; "The capital of France is the city of"
    is 8 under Qwen3. Naming the positions explicitly is the way out.
    """
    short = "hi there"
    with pytest.raises(ValueError, match="prompt too short"):
        pullback_for_prompt(model, short, [2], torch.randn(1, model.d_model))
    got = pullback_for_prompt(
        model, short, [2], torch.randn(1, model.d_model), source_positions=[-1]
    )
    assert got.n_valid_positions == 1


def test_default_positions_match_fitting_convention(model: TinyDecoder) -> None:
    """``source_positions=None`` reproduces ``valid_position_mask``."""
    seq_len = model.encode(PROMPT).shape[1]
    explicit = list(range(SKIP_FIRST, seq_len - 1))
    cotangents = torch.randn(3, model.d_model)
    default = pullback_for_prompt(
        model, PROMPT, [2], cotangents, dim_batch=3, skip_first=SKIP_FIRST
    )
    named = pullback_for_prompt(
        model, PROMPT, [2], cotangents, dim_batch=3, source_positions=explicit
    )
    assert default.n_valid_positions == named.n_valid_positions == len(explicit)
    torch.testing.assert_close(default[2], named[2], rtol=1e-6, atol=1e-7)


def test_source_and_target_positions_are_independent(model: TinyDecoder) -> None:
    """Single source position, all downstream targets -- the steering-shaped ask."""
    seq_len = model.encode(PROMPT).shape[1]
    cotangents = torch.randn(2, model.d_model)
    got = pullback_for_prompt(
        model,
        PROMPT,
        [2],
        cotangents,
        source_positions=[5],
        target_positions=list(range(5, seq_len)),
    )
    assert got.n_valid_positions == 1
    # Causality: a source position later than every target contributes nothing.
    causal_zero = pullback_for_prompt(
        model, PROMPT, [2], cotangents, source_positions=[-1], target_positions=[0, 1]
    )[2]
    torch.testing.assert_close(
        causal_zero, torch.zeros_like(causal_zero), atol=1e-7, rtol=0
    )


def test_rejects_out_of_range_positions(model: TinyDecoder) -> None:
    with pytest.raises(ValueError, match="out of range"):
        pullback_for_prompt(
            model, PROMPT, [2], torch.randn(1, model.d_model), source_positions=[9999]
        )

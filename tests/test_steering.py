"""The write path: does the intervention do exactly what it claims, and nothing else?

The causal arm is the only part of this project that mutates a forward pass, so
its failure modes are silent -- a hook left registered, an off-by-one in the
position index, an edit that is secretly a no-op -- and every one of them would
show up as a steering result rather than as an error.
"""

from __future__ import annotations

import pytest
import torch

from jsteer.steering import (
    ResidualEdit,
    additive_edit,
    grade,
    mean_residual_norms,
    next_token_logits,
    steered,
    swap_edit,
)
from tests.tiny_model import TinyDecoder

PROMPT = "the quick brown fox jumps over the lazy dog"
LAYERS = [1, 2, 3]


@pytest.fixture(scope="module")
def model() -> TinyDecoder:
    return TinyDecoder(n_layers=5, d_model=8, seed=0)


@pytest.fixture(scope="module")
def clean(model: TinyDecoder) -> torch.Tensor:
    return next_token_logits(model, PROMPT)


def test_no_edit_is_identity(model: TinyDecoder, clean: torch.Tensor) -> None:
    assert torch.equal(next_token_logits(model, PROMPT, edit=None), clean)
    empty = ResidualEdit({})
    assert torch.equal(next_token_logits(model, PROMPT, edit=empty), clean)


def test_zero_strength_is_identity(model: TinyDecoder, clean: torch.Tensor) -> None:
    """Strength 0 is the paper's control arm; it must be exactly the clean run."""
    directions = {l: torch.randn(model.d_model) for l in LAYERS}
    norms = mean_residual_norms(model, PROMPT, LAYERS)
    edit = additive_edit(directions, norms, strength=0.0)
    torch.testing.assert_close(next_token_logits(model, PROMPT, edit=edit), clean)


def test_edit_changes_the_output(model: TinyDecoder, clean: torch.Tensor) -> None:
    directions = {l: torch.randn(model.d_model) for l in LAYERS}
    norms = mean_residual_norms(model, PROMPT, LAYERS)
    edit = additive_edit(directions, norms, strength=1.0)
    assert not torch.allclose(next_token_logits(model, PROMPT, edit=edit), clean)


def test_hooks_are_removed(model: TinyDecoder, clean: torch.Tensor) -> None:
    """A leaked hook would contaminate every later trial in a sweep."""
    before = {id(block): len(block._forward_hooks) for block in model.layers}
    edit = ResidualEdit({l: torch.ones(model.d_model) for l in LAYERS})
    with steered(model, edit):
        assert any(len(model.layers[l]._forward_hooks) for l in LAYERS)
    assert {id(b): len(b._forward_hooks) for b in model.layers} == before
    assert torch.equal(next_token_logits(model, PROMPT), clean)


def test_hooks_are_removed_on_exception(model: TinyDecoder) -> None:
    before = {id(block): len(block._forward_hooks) for block in model.layers}
    edit = ResidualEdit({l: torch.ones(model.d_model) for l in LAYERS})
    with pytest.raises(RuntimeError), steered(model, edit):
        raise RuntimeError("boom")
    assert {id(b): len(b._forward_hooks) for b in model.layers} == before


def test_edit_at_last_position_only(model: TinyDecoder, clean: torch.Tensor) -> None:
    """Causality check: an edit at the final token cannot change earlier ones.

    This is what pins down that ``positions`` indexes the sequence axis and not
    something else -- if the index were wrong, an edit "at the last position"
    would leak backwards, which attention masking makes impossible.
    """
    seq_len = model.encode(PROMPT).shape[1]
    edit = ResidualEdit(
        {l: torch.randn(model.d_model) * 5 for l in LAYERS}, positions=[seq_len - 1]
    )
    assert not torch.allclose(next_token_logits(model, PROMPT, edit=edit), clean)
    earlier = next_token_logits(model, PROMPT, edit=edit, position=seq_len - 2)
    torch.testing.assert_close(
        earlier, next_token_logits(model, PROMPT, position=seq_len - 2)
    )


def test_negative_positions_wrap(model: TinyDecoder) -> None:
    seq_len = model.encode(PROMPT).shape[1]
    direction = {l: torch.randn(model.d_model) for l in LAYERS}
    a = ResidualEdit(direction, positions=[-1])
    b = ResidualEdit(direction, positions=[seq_len - 1])
    torch.testing.assert_close(
        next_token_logits(model, PROMPT, edit=a),
        next_token_logits(model, PROMPT, edit=b),
    )


def test_additive_magnitude_follows_the_recipe(model: TinyDecoder) -> None:
    """||delta|| == layer mean residual norm * strength, for any input scale."""
    norms = mean_residual_norms(model, PROMPT, LAYERS)
    directions = {l: torch.randn(model.d_model) * 137.0 for l in LAYERS}
    edit = additive_edit(directions, norms, strength=2.5)
    for layer in LAYERS:
        assert edit.vectors[layer].norm().item() == pytest.approx(
            norms[layer] * 2.5, rel=1e-5
        )


def test_swap_with_identical_directions_is_a_noop(
    model: TinyDecoder, clean: torch.Tensor
) -> None:
    """Swapping a token for itself removes and re-adds the same coordinate."""
    g = {l: torch.randn(model.d_model) for l in LAYERS}
    for mode in ("replace", "exchange"):
        edit = swap_edit(model, PROMPT, g, g, mode=mode)
        torch.testing.assert_close(
            next_token_logits(model, PROMPT, edit=edit), clean, rtol=1e-4, atol=1e-5
        )


def test_swap_is_per_position(model: TinyDecoder) -> None:
    """The swap delta depends on the residual, so it differs across positions."""
    source = {l: torch.randn(model.d_model) for l in LAYERS}
    target = {l: torch.randn(model.d_model) for l in LAYERS}
    edit = swap_edit(model, PROMPT, source, target)
    for layer in LAYERS:
        delta = edit.vectors[layer]
        assert delta.ndim == 2, "swap must produce a per-position delta"
        assert not torch.allclose(delta[0], delta[-1])


def test_swap_rejects_unknown_mode(model: TinyDecoder) -> None:
    g = {l: torch.randn(model.d_model) for l in LAYERS}
    with pytest.raises(ValueError, match="mode"):
        swap_edit(model, PROMPT, g, g, mode="project")


def test_clamp_exactly_swaps_the_lens_coordinates(model: TinyDecoder) -> None:
    """The paper's property: ``c = V^dagger h`` is exchanged, on unequal norms.

    Deliberately asserts the *pseudoinverse coordinates*, not the raw
    projections ``<g_y, h>``. Those are not the same thing when
    ``||v_s|| != ||v_t||``, and asserting the wrong one is what let a
    mis-ordered operator pass review: sigma must act on ``(V^T V)^-1 V^T h``,
    not on ``V^T h``.
    """
    from jlens.hooks import ActivationRecorder

    torch.manual_seed(5)
    layer = LAYERS[0]
    g_src = torch.randn(model.d_model)
    # Deliberately correlated with g_src, and not unit norm.
    g_tgt = 0.7 * g_src + 0.6 * torch.randn(model.d_model)
    source = {layer: g_src}
    target = {layer: g_tgt}
    V = torch.stack([g_src, g_tgt], dim=1)
    gram = V.T @ V

    def coordinates(edit) -> torch.Tensor:
        # Steerer first: hooks run in registration order and the edit must
        # land before the recorder reads. See next_token_logits.
        with (
            steered(model, edit),
            ActivationRecorder(model.layers, at=[layer]) as recorder,
            torch.no_grad(),
        ):
            model.forward(model.encode(PROMPT))
            h = recorder.activations[layer][0].detach().float()
        return torch.linalg.solve(gram, (h @ V).T).T

    before = coordinates(None)
    after = coordinates(swap_edit(model, PROMPT, source, target, mode="clamp"))
    torch.testing.assert_close(after, before.flip(-1), rtol=1e-3, atol=1e-3)


def test_clamp_is_minimum_norm(model: TinyDecoder) -> None:
    """The delta lies in the span of the two read directions, nothing else."""
    torch.manual_seed(6)
    layer = LAYERS[0]
    g_src = torch.randn(model.d_model)
    g_tgt = torch.randn(model.d_model)
    edit = swap_edit(model, PROMPT, {layer: g_src}, {layer: g_tgt}, mode="clamp")
    basis = torch.linalg.qr(torch.stack([g_src, g_tgt], dim=1)).Q
    delta = edit.vectors[layer]
    residual = delta - (delta @ basis) @ basis.T
    assert residual.norm() < 1e-4 * delta.norm().clamp_min(1e-12)


def test_clamp_with_identical_directions_is_a_noop(
    model: TinyDecoder, clean: torch.Tensor
) -> None:
    g = {l: torch.randn(model.d_model) for l in LAYERS}
    edit = swap_edit(model, PROMPT, g, g, mode="clamp")
    torch.testing.assert_close(
        next_token_logits(model, PROMPT, edit=edit), clean, rtol=1e-3, atol=1e-4
    )


def test_grade_reports_hit_and_rank(model: TinyDecoder, clean: torch.Tensor) -> None:
    top = int(clean.argmax().item())
    hit = grade(model, clean, target_id=top, source_id=top)
    assert hit.hit and hit.target_rank == 0 and hit.margin == pytest.approx(0.0)

    worst = int(clean.argmin().item())
    miss = grade(model, clean, target_id=worst, source_id=top)
    assert not miss.hit
    assert miss.target_rank == len(clean) - 1
    assert miss.margin < 0


def test_grade_excludes_the_injected_token(
    model: TinyDecoder, clean: torch.Tensor
) -> None:
    """The echo case: the injected word wins the argmax, the answer is second.

    Real behaviour, not a hypothetical -- additive steering toward ``Canada``
    on Qwen3-1.7B made ``" Canada"`` the greedy token while ``Ottawa`` sat at
    rank 13. ``hit`` must stay false (upstream's criterion is upstream's) and
    ``hit_excluding`` must catch it.
    """
    top, second = clean.topk(2).indices.tolist()
    outcome = grade(model, clean, target_id=second, exclude_ids=[top])
    assert not outcome.hit
    assert outcome.hit_excluding
    assert outcome.greedy_excluding == model.tokenizer.decode([second])


def test_grade_without_exclusions_agrees_with_itself(
    model: TinyDecoder, clean: torch.Tensor
) -> None:
    top = int(clean.argmax().item())
    outcome = grade(model, clean, target_id=top)
    assert outcome.hit and outcome.hit_excluding


def test_kl_from_clean_is_zero_for_an_unchanged_distribution(
    model: TinyDecoder, clean: torch.Tensor
) -> None:
    outcome = grade(model, clean, target_id=0, clean_logits=clean)
    assert outcome.kl_from_clean == pytest.approx(0.0, abs=1e-6)


def test_kl_from_clean_grows_with_disruption(
    model: TinyDecoder, clean: torch.Tensor
) -> None:
    """Separates 'undershooting' from 'brute force' -- the distinction A.13 needs.

    A big edit in the wrong direction improves the target's rank while wrecking
    the rest of the distribution; rank alone cannot see that, KL can.
    """
    mild = grade(
        model, clean + 0.1 * torch.randn_like(clean), target_id=0, clean_logits=clean
    )
    wild = grade(
        model, clean + 5.0 * torch.randn_like(clean), target_id=0, clean_logits=clean
    )
    assert 0 < mild.kl_from_clean < wild.kl_from_clean


def test_kl_defaults_to_zero_without_clean_logits(
    model: TinyDecoder, clean: torch.Tensor
) -> None:
    assert grade(model, clean, target_id=0).kl_from_clean == 0.0


def test_swap_clamp_is_invariant_to_UNIFORM_rescaling() -> None:
    """Scaling BOTH swap vectors by the same factor leaves the edit unchanged.

    c = V^+ h scales inversely with V, so V(sigma(c) - c) is fixed. This is the
    half of the invariance that holds, and it is why C4 can compare partial
    averages of very different overall lengths at one strength.

    See the companion test for the half that does NOT hold.
    """
    import torch

    torch.manual_seed(0)
    d, layers = 16, [0, 1]
    h = {ll: torch.randn(1, 3, d) for ll in layers}
    src = {ll: torch.randn(d) for ll in layers}
    tgt = {ll: torch.randn(d) for ll in layers}

    def edit_for(scale: float):
        return _clamp_vectors(
            {ll: src[ll] * scale for ll in layers},
            {ll: tgt[ll] * scale for ll in layers},
            h,
            layers,
        )

    a, b = edit_for(1.0), edit_for(7.5)
    for ll in layers:
        assert torch.allclose(a[ll], b[ll], atol=1e-4), ll


def test_swap_clamp_is_NOT_invariant_to_independent_rescaling() -> None:
    """Rescaling the two columns *differently* does change the edit.

    With V' = V diag(d_s, d_t) we get c' = diag(1/d_s, 1/d_t) c, and

        V'(sigma(c') - c') = V[(d_s/d_t) c_t - c_s, (d_t/d_s) c_s - c_t]

    against the original V[c_t - c_s, c_s - c_t]: equal only when d_s = d_t.
    The permutation sigma commutes with (V^T V)^-1 only for equal-norm columns
    -- structurally the same reason the operator-ordering bug earlier in this
    project was a bug.

    Why this is pinned: C4 compares partial averages across n at a single
    strength, justified by the uniform invariance above. That justification is
    only complete if ||v_s|| and ||v_t|| shrink at the SAME rate as n grows.
    There is no reason they must, so C4 measures the ratio rather than assuming
    it -- and this test is here so nobody upgrades "invariant to scale" into
    "invariant to the relative scale of the two columns".
    """
    import torch

    torch.manual_seed(0)
    d, layers = 16, [0]
    h = {0: torch.randn(1, 3, d)}
    src = {0: torch.randn(d)}
    tgt = {0: torch.randn(d)}

    ref = _clamp_vectors(src, tgt, h, layers)[0]
    skewed = _clamp_vectors({0: src[0] * 7.5}, tgt, h, layers)[0]
    assert not torch.allclose(ref, skewed, atol=1e-3)
    # and the difference is large relative to the edit itself, not a rounding wobble
    assert (ref - skewed).abs().max() > ref.abs().max()


def _clamp_vectors(src, tgt, h, layers):
    """The clamp branch of swap_edit, applied directly to a synthetic h."""
    import torch

    out = {}
    for ll in layers:
        V = torch.stack([src[ll], tgt[ll]], dim=1)
        gram = V.T @ V
        c = torch.linalg.solve(
            gram + 1e-6 * torch.eye(2, dtype=gram.dtype), (h[ll] @ V).transpose(-1, -2)
        ).transpose(-1, -2)
        out[ll] = (c.flip(-1) - c) @ V.T
    return out


def test_off_target_kl_ignores_the_intended_swap_but_plain_kl_does_not() -> None:
    """The whole reason off_target_kl exists: a perfect edit must score zero.

    Construct the ideal intervention -- probability moved from the source
    answer to the target answer, every other token untouched. That is exactly
    what steering is *trying* to do, so a control variable that calls it
    "damage" is measuring the effect it is supposed to be holding constant.

    ``kl_from_clean`` charges for it. ``off_target_kl`` does not, because the
    two answer tokens are merged into one category before the divergence is
    taken. Pinning both halves matters: a version that returned zero for
    everything would also pass the first assertion.
    """
    model = TinyDecoder(n_layers=2, d_model=8, seed=3)
    source, target = 5, 9
    clean = torch.full((32,), -10.0)
    clean[source] = 2.0
    clean[target] = 0.0
    # Same distribution with the two answer logits exchanged: mass moves within
    # the concept pair and nowhere else.
    steered_logits = clean.clone()
    steered_logits[source], steered_logits[target] = 0.0, 2.0

    outcome = grade(
        model,
        steered_logits,
        target_id=target,
        source_id=source,
        clean_logits=clean,
    )
    assert outcome.off_target_kl < 1e-6, (
        "a pure within-concept swap must register as zero collateral, "
        f"got {outcome.off_target_kl}"
    )
    assert outcome.kl_from_clean > 0.1, (
        "the plain KL is supposed to charge for the intended change -- if it "
        "does not, this test is no longer demonstrating the difference"
    )
    # Mass stayed inside the pair, so preservation is ~1.
    assert abs(outcome.concept_mass_ratio - 1.0) < 1e-4

    # And genuine collateral must still be caught: dump mass on an unrelated
    # token and off_target_kl has to move.
    noisy = steered_logits.clone()
    noisy[17] = 5.0
    dirty = grade(model, noisy, target_id=target, source_id=source, clean_logits=clean)
    assert dirty.off_target_kl > 1.0, (
        f"off-target damage went unmeasured: {dirty.off_target_kl}"
    )
    assert dirty.concept_mass_ratio < 0.5, (
        "mass leaked out of the concept pair and preservation did not fall: "
        f"{dirty.concept_mass_ratio}"
    )


def test_answer_matches_reads_past_the_leading_space_the_model_actually_emits():
    """The failure that made the whole numbers category ungradeable.

    Qwen3-8B answers "Two times five equals" with " 10": a bare space, then the
    digits. First-token grading therefore sees only the space -- scoring a miss
    if the space is rejected, and scoring *everything* a hit if it is accepted.
    Matching a few generated characters is the only rule that separates them.
    """
    from jsteer.steering import answer_matches

    assert answer_matches(" 10", "ten", digit="10")
    assert answer_matches(" ten", "ten", digit="10")
    assert answer_matches(" Ten,", "ten", digit="10")
    # The first_letter template elicits a quoted letter.
    assert answer_matches(" 't'", "t")

    # And it must still tell the near-misses apart, which is precisely what
    # first-token digit grading could not do: 10, 14 and 18 all start '1'.
    assert not answer_matches(" 14", "ten", digit="10")
    assert not answer_matches(" 18", "ten", digit="10")
    assert not answer_matches(" fourteen", "ten", digit="10")
    assert answer_matches(" 14", "fourteen", digit="14")

    # A bare space carries no answer and must never match.
    assert not answer_matches(" ", "ten", digit="10")
    assert not answer_matches("", "ten", digit="10")

    # A prefix match must end at a boundary. Both of these appeared in real
    # generations and both would otherwise be scored as correct.
    assert not answer_matches(" 90,", "nine", digit="9")  # 90 is not 9
    assert not answer_matches(" ninety", "nine", digit="9")
    assert not answer_matches(" 100", "ten", digit="10")
    assert answer_matches(" 9, but", "nine", digit="9")
    assert answer_matches(" 10.", "ten", digit="10")


# --------------------------------------------------------------------------
# C28 - ablation, the workspace paper's causal measure
# --------------------------------------------------------------------------


def test_ablation_removes_the_component_it_names(model: TinyDecoder) -> None:
    """After the edit, the residual's coordinate along ``g`` is gone.

    The whole C28 result is "ablation KL is flat across arms". A no-op edit
    would produce exactly that, so the operator has to be pinned: read the
    residual back at the edited layer and assert the projection is zero there
    and that the orthogonal complement is untouched.
    """
    from jlens.hooks import ActivationRecorder

    from jsteer.steering import ablate_edit

    torch.manual_seed(0)
    layer = 2
    g = torch.randn(8)
    edit, diag = ablate_edit(model, PROMPT, {layer: g})

    with ActivationRecorder(model.layers, at=[layer]) as rec:
        model.forward(model.encode(PROMPT))
        before = rec.activations[layer][0].detach().float()
    with torch.no_grad(), steered(model, edit):
        with ActivationRecorder(model.layers, at=[layer]) as rec:
            model.forward(model.encode(PROMPT))
            after = rec.activations[layer][0].detach().float()

    unit = g / g.norm()
    assert after.float().cpu() @ unit == pytest.approx(torch.zeros(after.shape[0]), abs=1e-4)
    # and it removed only that: the component orthogonal to g is unchanged.
    b, a = before.cpu(), after.float().cpu()
    perp_before = b - (b @ unit)[:, None] * unit[None, :]
    perp_after = a - (a @ unit)[:, None] * unit[None, :]
    assert torch.allclose(perp_before, perp_after, atol=1e-4)
    assert 0.0 < diag["frac_norm"] < 1.0


def test_ablation_of_an_absent_direction_is_nearly_free() -> None:
    """A direction the residual has no coordinate along must barely move the output.

    This is the floor the ``_rand`` control arm is supposed to establish: if
    ablating a direction the model is not using still moved the logits, a flat
    KL across real arms would be uninformative. Needs a model wider than the
    prompt is long, or the residuals span everything and no such direction
    exists.
    """
    from jlens.hooks import ActivationRecorder

    from jsteer.steering import ablate_edit

    wide = TinyDecoder(n_layers=4, d_model=48, seed=3)
    prompt = "the quick brown"
    layer = 2
    with torch.no_grad(), ActivationRecorder(wide.layers, at=[layer]) as rec:
        wide.forward(wide.encode(prompt))
        h = rec.activations[layer][0].detach().float().cpu()
    assert h.shape[0] < h.shape[1], "prompt must be shorter than d_model"

    torch.manual_seed(1)
    q, _ = torch.linalg.qr(h.T)  # columns span the residual positions
    z = torch.randn(48)
    z = z - q @ (q.T @ z)
    assert z.norm() > 1e-3

    edit, diag = ablate_edit(wide, prompt, {layer: z})
    assert diag["frac_norm"] == pytest.approx(0.0, abs=1e-5)
    assert torch.allclose(
        next_token_logits(wide, prompt, edit=edit),
        next_token_logits(wide, prompt),
        atol=1e-5,
    )

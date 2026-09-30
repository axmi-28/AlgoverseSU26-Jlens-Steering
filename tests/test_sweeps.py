"""Sharding and merging -- the two places a fanned-out run can silently lose data.

Neither is checkable after the fact: a shard that drops prompts and a merge that
double-counts them both produce a plausible-looking results file.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from jsteer.sweeps import CausalSpec, JacobianSweepSpec, take_shard

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    """Import a ``scripts/NN_*.py`` module by path (they are not a package)."""
    path = next(REPO_ROOT.glob(f"scripts/{name}*.py"))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("n_shards", [1, 2, 3, 5, 8])
def test_shards_partition_exactly(n_shards: int) -> None:
    """Every item lands in exactly one shard -- no gaps, no duplicates."""
    items = list(range(97))  # deliberately coprime-ish with the shard counts
    shards = [take_shard(items, i, n_shards) for i in range(n_shards)]
    flat = [x for shard in shards for x in shard]
    assert sorted(flat) == items
    assert len(flat) == len(set(flat))


@pytest.mark.parametrize("n_shards", [2, 4, 8])
def test_shards_are_balanced(n_shards: int) -> None:
    """Sizes differ by at most one, so no container is the long pole."""
    sizes = [len(take_shard(list(range(97)), i, n_shards)) for i in range(n_shards)]
    assert max(sizes) - min(sizes) <= 1


def test_striding_mixes_expensive_prompts() -> None:
    """The reason for a stride: eval prompts and 128-token controls must mix.

    A contiguous slice would give one shard every wikitext control, and the
    backward pass scales with sequence length -- that shard would run ~16x
    longer than its siblings and set the wall time for the whole fan-out.
    """
    prompts = ["eval"] * 64 + ["fit"] * 32
    for i in range(8):
        shard = take_shard(prompts, i, 8)
        assert "eval" in shard and "fit" in shard


def test_shard_out_of_range_is_rejected() -> None:
    for bad in (-1, 4):
        with pytest.raises(ValueError, match="out of range"):
            take_shard([1, 2, 3], bad, 4)


def test_spec_tags_are_unique_per_shard() -> None:
    """Shard-local filenames: two shards must never write the same path."""
    tags = {JacobianSweepSpec(shard=i, n_shards=4).tag for i in range(4)}
    assert len(tags) == 4
    causal = {CausalSpec(shard=i, n_shards=4).tag for i in range(4)}
    assert len(causal) == 4


def test_merged_name_strips_the_shard_suffix() -> None:
    merge = load_script("07_merge")
    assert (
        merge.merged_name(Path("scalars_qwen3-8b_all_shard3of8.json"))
        == "scalars_qwen3-8b_all.json"
    )
    assert (
        merge.merged_name(Path("steering_qwen3-8b_swap_shard0of1.json"))
        == "steering_qwen3-8b_swap.json"
    )


merge = load_script("07_merge")


def test_merge_json_concatenates(tmp_path: Path) -> None:
    paths = []
    for i in range(3):
        path = tmp_path / f"r_shard{i}of3.json"
        path.write_text(f'[{{"shard": {i}}}]')
        paths.append(path)
    out = tmp_path / "r.json"
    n, bad = merge.merge_json_records(paths, out)
    assert (n, bad) == (3, [])
    import json

    assert sorted(r["shard"] for r in json.loads(out.read_text())) == [0, 1, 2]


def test_merge_refuses_to_write_a_partial_merge(tmp_path: Path) -> None:
    """A 0-byte shard must block the merge, not vanish from it.

    `modal volume get` on a directory writes 0-byte files for entries that
    download fine individually -- observed on 5 of 8 shards, exit code 0. If
    merging skipped them quietly the result would be a normal-looking file
    short by a shard nobody counted.
    """
    good = tmp_path / "r_shard0of2.json"
    good.write_text('[{"shard": 0}]')
    truncated = tmp_path / "r_shard1of2.json"
    truncated.write_bytes(b"")

    out = tmp_path / "r.json"
    n, bad = merge.merge_json_records([good, truncated], out)
    assert len(bad) == 1 and "r_shard1of2.json" in bad[0]
    assert not out.exists(), "must not write a merge that is missing a shard"


def test_group_by_target_refuses_to_mix_runs(tmp_path: Path) -> None:
    """A 3-prompt probe and an 80-prompt run strip to the same merged name."""
    probe = tmp_path / "scalars_x_all_shard0of1.json"
    real = [tmp_path / f"scalars_x_all_shard{i}of8.json" for i in range(8)]
    for path in [probe, *real]:
        path.write_text("[]")

    groups, problems = merge.group_by_target([probe, *real])
    assert not groups, "ambiguous target must not be merged"
    assert len(problems) == 1 and "[1, 8]" in problems[0]

    groups, problems = merge.group_by_target([probe, *real], shards=8)
    assert problems == []
    assert len(groups["scalars_x_all.json"]) == 8


def test_merge_group_means_pools_by_count(tmp_path: Path) -> None:
    """Uneven shards: a mean-of-means would be wrong, a pooled sum/n is right."""
    merge = load_script("07_merge")
    a = torch.ones(4, 4, dtype=torch.float64) * 2.0  # 10 prompts averaging 2
    b = torch.ones(4, 4, dtype=torch.float64) * 8.0  # 2 prompts averaging 8
    torch.save({"eval|3": {"sum": a * 10, "n": 10}}, tmp_path / "g_shard0of2.pt")
    torch.save({"eval|3": {"sum": b * 2, "n": 2}}, tmp_path / "g_shard1of2.pt")

    out = tmp_path / "g.pt"
    counts = merge.merge_group_means(
        [tmp_path / "g_shard0of2.pt", tmp_path / "g_shard1of2.pt"], out
    )
    assert counts == {"eval|3": 12}
    blob = torch.load(out, weights_only=False)
    expected = (2.0 * 10 + 8.0 * 2) / 12  # 3.0, not the mean-of-means 5.0
    torch.testing.assert_close(
        blob["mean"]["eval|3"], torch.full((4, 4), expected), rtol=1e-6, atol=1e-6
    )
    assert blob["n"]["eval|3"] == 12


def test_merge_vectors_unions_and_checks_headers(tmp_path: Path) -> None:
    merge = load_script("07_merge")
    header = {
        "targets": ["France"],
        "layers": [9],
        "averaged": {},
        "config": "x",
        "target_ids": [1],
        "positions": "all",
    }
    torch.save({"local": {"a|9": torch.ones(4)}, **header}, tmp_path / "v_shard0of2.pt")
    torch.save(
        {"local": {"b|9": torch.zeros(4)}, **header}, tmp_path / "v_shard1of2.pt"
    )
    paths = [tmp_path / "v_shard0of2.pt", tmp_path / "v_shard1of2.pt"]
    assert merge.merge_vectors(paths, tmp_path / "v.pt") == 2

    torch.save(
        {"local": {"c|9": torch.zeros(4)}, **{**header, "layers": [11]}}, paths[1]
    )
    with pytest.raises(ValueError, match="disagrees"):
        merge.merge_vectors(paths, tmp_path / "v2.pt")


def test_resume_recomputes_when_the_digest_misses_a_layer(tmp_path) -> None:
    """A 3-layer probe's digest must not satisfy a 12-layer sweep.

    Regression: both write ``digests/{model}_{positions}_k{k}/<prompt>.pt``,
    so a bare exists() check let a probe silently delete a prompt from a later
    sweep -- it was absent from the scalars and from the running mean, and the
    run still reported success.
    """
    import torch

    from jsteer.sweeps import _covers

    path = tmp_path / "countries_capital_France.pt"
    torch.save({24: {}, 36: {}, 48: {}}, path)

    assert _covers(path, [24, 36, 48])
    assert _covers(path, [24, 48])
    assert not _covers(path, [24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57])


def test_resume_recomputes_an_unreadable_digest(tmp_path) -> None:
    from jsteer.sweeps import _covers

    path = tmp_path / "truncated.pt"
    path.write_bytes(b"not a torch file")
    assert not _covers(path, [24])


def test_linear_response_identity_holds_at_the_last_position() -> None:
    """The identity C5 rests on: perturb h_l[last], predict via J_x[last,last]^T u.

    Two things are pinned, both of which can break silently.

    **The identity itself.** With source and target positions both the readout
    token, the response has no leading factor: perturbing h_l[last] by delta
    changes <u, h_final[last]> by <g, delta> to first order. Relative error is
    then O(alpha), because the error is O(alpha^2) against a response that is
    O(alpha) -- so halving the dose halves the relative error, and a constant
    relative error would mean a mis-scaled prediction rather than curvature.

    **The scale.** Relative error below 1e-3 at alpha=0.1 is what rules out a
    mis-scaled prediction. That is the guard that matters, because the
    estimator averages over source positions but *sums* the cotangent over
    target positions: under ``positions="all"`` the same comparison carries a
    factor of ``seq_len``, which would read as "the local Jacobian is biased"
    rather than "the harness is". That factor cannot be exercised here -- the
    tiny model has no attention, so its positions are independent and
    perturbing one never moves another -- so it is guarded by the scale
    assertion and by ``LinearResponseSpec.positions`` being pinned, not by a
    direct test.

    Doses stop at 1e-1 deliberately. The measured quantity is a difference of
    two forward passes, so below some dose the response falls under the
    arithmetic's own resolution and relative error climbs again as alpha falls.
    Measured on this model in fp32 the turnaround is near alpha=0.03 (rel err
    4.5e-6 there, rising to 1.5e-5 at 1e-2 and flat below). The real sweep runs
    fp32 for this reason, and its analysis has to treat the left end of the
    alpha grid as floor-limited rather than as evidence.
    """
    from jsteer.jacobian import pullback_for_prompt
    from jsteer.steering import ResidualEdit
    from jsteer.sweeps import _final_residual
    from tests.tiny_model import TinyDecoder

    model = TinyDecoder(n_layers=4, d_model=8, seed=1)
    prompt = "steering"
    layer = 1
    input_ids = model.encode(prompt)
    seq_len = input_ids.shape[1]
    last = seq_len - 1

    cotangents = torch.eye(model.d_model)[:3]
    g = pullback_for_prompt(
        model,
        prompt,
        [layer],
        cotangents,
        source_positions=[-1],
        target_positions=[-1],
    )[layer]

    h_clean = _final_residual(model, input_ids, None)
    s_clean = cotangents @ h_clean.float()

    direction = torch.zeros(model.d_model)
    direction[2] = 1.0

    errors = {}
    for alpha in (1.0, 0.1):
        delta = direction * alpha
        h = _final_residual(
            model, input_ids, ResidualEdit({layer: delta}, positions=[last])
        )
        actual = cotangents @ h.float() - s_clean
        errors[alpha] = float((actual - g @ delta).norm() / actual.norm())

    assert errors[0.1] < 1e-3, f"prediction is mis-scaled, not curved: {errors}"
    assert errors[0.1] < errors[1.0] / 5, f"relative error is not O(alpha): {errors}"


def test_woodbury_preconditioner_matches_the_direct_inverse() -> None:
    """C7's rank-k solve must equal forming (R^T M R + lambda I)^-1 outright.

    The ladder's whole claim rests on applying a metric correctly, and the fast
    path is an algebraic identity that is easy to get subtly wrong -- a
    transposed basis or a lambda on the wrong side still returns a
    plausible-looking vector of the right shape and norm. Checking it against
    the literal d x d inverse is the only way that fails loudly.

    Also pins the limiting behaviour: as tau grows the preconditioner must
    collapse toward the identity, i.e. back to the raw pullback. That is what
    makes tau a continuous knob from "no preconditioning" to "full whitening"
    rather than a binary, and a sign error would break it.
    """
    from jsteer.sweeps import _damped_inverse_apply

    torch.manual_seed(0)
    d, k = 32, 6
    basis = torch.linalg.qr(torch.randn(d, k))[0].T  # [k, d], orthonormal rows
    root = torch.randn(k, k)
    middle = root @ root.T + 0.5 * torch.eye(k)  # symmetric positive definite
    g = torch.randn(d)

    for tau in (0.05, 1.0):
        lam = tau * float(torch.diagonal(middle).mean())
        direct = torch.linalg.solve(basis.T @ middle @ basis + lam * torch.eye(d), g)
        fast = _damped_inverse_apply(g, basis, middle, tau)
        assert torch.allclose(direct, fast, atol=1e-4), (
            f"tau={tau}: Woodbury form disagrees with the direct inverse, "
            f"max diff {float((direct - fast).abs().max())}"
        )

    # Large tau -> the metric is swamped by the damping -> direction unchanged.
    huge = _damped_inverse_apply(g, basis, middle, tau=1e6)
    cos = float(huge @ g / (huge.norm() * g.norm()))
    assert cos > 0.999, f"heavy damping must return the raw direction, cos={cos}"


# --------------------------------------------------------------------------
# the domain panel
# --------------------------------------------------------------------------


def test_corpus_tag_changes_with_every_input_that_changes_the_documents() -> None:
    """The repo's recurring failure: a cache path that omits one of its inputs.

    Each mutation below produces a genuinely different set of documents, so
    each must produce a different tag. If any two collided, the second run
    would read the first's digests under the same prompt keys, report success,
    and report numbers computed on the wrong corpus.
    """
    from jsteer.domains import DEFAULT_DOMAINS, DomainSpec, corpus_tag

    base = list(DEFAULT_DOMAINS)
    reference = corpus_tag(base, 16, 128)

    assert corpus_tag(base, 16, 128) == reference  # stable across calls
    assert corpus_tag(base, 32, 128) != reference  # more prompts per group
    assert corpus_tag(base, 16, 256) != reference  # different length filter
    assert corpus_tag(base[:3], 16, 128) != reference  # a group dropped

    swapped = (
        base[:2]
        + [DomainSpec("openwebmath", "someone-else/math", None, "train")]
        + base[3:]
    )
    assert corpus_tag(swapped, 16, 128) != reference  # same name, other dataset

    # The two wikitext halves differ only by stride offset, and that alone must
    # move the tag -- otherwise the null and the thing it calibrates could be
    # served from one cache.
    unsplit = [
        DomainSpec("wikitext_a", "Salesforce/wikitext", "wikitext-103-raw-v1", "train")
    ] + base[1:]
    assert corpus_tag(unsplit, 16, 128) != reference


def test_domain_run_does_not_share_a_namespace_with_the_original_sweep() -> None:
    """A domain run must not write over ``group_means_qwen3-8b_all_shard0of8``."""
    from jsteer.sweeps import JacobianSweepSpec

    plain = JacobianSweepSpec(
        config_name="qwen3-8b", positions="all", shard=0, n_shards=8
    )
    domain = JacobianSweepSpec(
        config_name="qwen3-8b",
        positions="all",
        domains=["wikitext_a", "wikitext_b", "openwebmath", "ultrachat"],
        n_per=16,
        shard=0,
        n_shards=8,
    )
    assert plain.namespace != domain.namespace
    assert plain.tag != domain.tag
    # The original sweep's namespace is unchanged, so its existing digests and
    # pooled sums stay addressable.
    assert plain.namespace == "qwen3-8b_all"
    # And two domain panels that differ only in size stay apart.
    smaller = JacobianSweepSpec(
        config_name="qwen3-8b",
        positions="all",
        domains=["wikitext_a", "wikitext_b", "openwebmath", "ultrachat"],
        n_per=8,
        shard=0,
        n_shards=8,
    )
    assert smaller.namespace != domain.namespace


def test_domain_prompt_set_interleaves_groups_and_defers_to_the_lens_convention(
    monkeypatch,
) -> None:
    """Balanced shards, and the fitting position convention on every prompt."""
    from jsteer import run as run_module
    from jsteer.domains import DEFAULT_DOMAINS
    from jsteer.run import build_domain_prompt_set
    from jsteer.sweeps import take_shard

    n_per = 4
    fake = {
        spec.name: [f"{spec.name} document {i}" for i in range(n_per)]
        for spec in DEFAULT_DOMAINS
    }
    monkeypatch.setattr(run_module, "load_domain_corpora", lambda *a, **k: fake)

    prompts = build_domain_prompt_set(
        list(DEFAULT_DOMAINS), n_per=n_per, tokenizer=None
    )
    assert len(prompts) == n_per * len(DEFAULT_DOMAINS)

    # Every prompt uses the lens's own position convention: the panel is
    # compared against a J_bar estimated that way, so any other convention
    # would fold a convention effect into the domain effect.
    assert all(p.fit_positions for p in prompts)

    # Contiguous blocks, so a stride shard is balanced across corpora rather
    # than one shard owning an entire group -- whose loss would delete a whole
    # domain's running sum. The interleaved layout fails exactly here: it puts
    # every prompt of a shard in one corpus whenever the shard count is a
    # multiple of the group count, which 8-over-4 is.
    for n_shards in (2, 4, 8):
        for shard in range(n_shards):
            got = take_shard(prompts, shard, n_shards)
            counts = {s.name: 0 for s in DEFAULT_DOMAINS}
            for prompt in got:
                counts[prompt.group] += 1
            assert max(counts.values()) - min(counts.values()) <= 1, (
                f"{n_shards} shards, shard {shard} is unbalanced: {counts}"
            )


def test_running_means_cover_every_group_in_the_prompt_set() -> None:
    """The group set comes from the prompts, not from a hardcoded pair."""
    from jsteer.domains import DEFAULT_DOMAINS
    from jsteer.run import EvalPrompt

    prompts = [
        EvalPrompt(group=spec.name, key=f"{spec.name}/0", text="x", fit_positions=True)
        for spec in DEFAULT_DOMAINS
    ]
    groups = sorted({p.group for p in prompts})
    assert groups == sorted(s.name for s in DEFAULT_DOMAINS)
    assert "eval" not in groups and "fit" not in groups


# --------------------------------------------------------------------------
# C29 - Park et al.'s causal inner product
# --------------------------------------------------------------------------


def test_unembedding_covariance_matches_the_direct_computation() -> None:
    """The chunked accumulator must equal ``torch.cov`` on the whole matrix.

    C29's whole point is that the transform is *Park's*, so the matrix being
    inverted has to be the covariance of the unembedding rows over the
    vocabulary and not, say, an uncentred second moment. Chunking is the only
    reason this is not a one-liner, and chunking is where the mean goes wrong.
    """
    import torch

    from jsteer.sweeps import _unembedding_covariance
    from tests.tiny_model import TinyDecoder

    model = TinyDecoder(n_layers=2, d_model=8, seed=0)
    cov, mu = _unembedding_covariance(model, chunk=7)

    from jsteer.loading import unembedding_matrix

    W = unembedding_matrix(model).double()
    expect_mu = W.mean(dim=0)
    centred = W - expect_mu
    expect = (centred.T @ centred) / W.shape[0]

    assert torch.allclose(mu, expect_mu, atol=1e-10)
    assert torch.allclose(cov, expect, atol=1e-10)
    # chunk size must not matter
    again, _ = _unembedding_covariance(model, chunk=3)
    assert torch.allclose(cov, again, atol=1e-10)


def test_whitening_is_a_real_rotation_not_a_rescale() -> None:
    """``Cov^-1 v`` must move ``v`` off its own line, or C29 tests nothing.

    If the unembedding covariance were near-isotropic, Park's transform would
    be a scalar multiple and a null result would be vacuous -- the sweep would
    "reproduce our numbers" only because it had changed nothing. The sweep
    reports ``cos_with_untransformed`` for exactly this reason; this pins that
    the quantity is meaningful on a matrix with real anisotropy.
    """
    import torch

    torch.manual_seed(0)
    d = 16
    basis = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))[0]
    spectrum = torch.logspace(0, 3, d, dtype=torch.float64)
    cov = (basis * spectrum) @ basis.T

    evals, evecs = torch.linalg.eigh(cov)
    lam = 1e-3 * float(evals.mean())
    inverse = (evecs * (evals + lam).pow(-1.0)) @ evecs.T

    v = torch.randn(d, dtype=torch.float64)
    w = inverse @ v
    cos = torch.nn.functional.cosine_similarity(v[None], w[None]).item()
    assert abs(cos) < 0.95, f"transform is nearly a rescale (cos={cos:.3f})"
    # and with the ridge taken to zero it is a genuine inverse. At the sweep's
    # own ridge it is NOT, by a wide margin, on a spectrum this stretched --
    # which is why the sweep reports the condition number and runs a second,
    # heavily damped arm rather than trusting one choice of lambda.
    exact = (evecs * evals.pow(-1.0)) @ evecs.T
    assert torch.allclose(exact @ cov, torch.eye(d, dtype=torch.float64), atol=1e-8)
    damped_err = float((inverse @ cov - torch.eye(d, dtype=torch.float64)).abs().max())
    assert damped_err > 1e-3, "ridge should visibly bias a stretched spectrum"

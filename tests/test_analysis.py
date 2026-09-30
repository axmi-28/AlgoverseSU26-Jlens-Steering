"""The statistics, checked against cases whose answers are known in advance.

These functions are how a cloud of vectors turns into a claim, so each one is
pinned at the two ends of its range rather than merely exercised.
"""

from __future__ import annotations

import pytest
import torch

from jsteer.analysis import (
    cloud_stats,
    energy_in_subspace,
    label_permutation_test,
    leave_one_out_influence,
    rank_correlation,
    subspace_alignment,
)


def orthonormal(d: int, k: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.linalg.qr(torch.randn(d, k)).Q.T


def test_subspace_alignment_anchors() -> None:
    A = orthonormal(64, 8, 0)
    assert subspace_alignment(A, A) == pytest.approx(1.0, abs=1e-5)
    # A subspace and its orthogonal complement's first 8 directions: overlap 0.
    full = torch.linalg.qr(torch.randn(64, 64)).Q.T
    assert subspace_alignment(full[:8], full[8:16]) == pytest.approx(0.0, abs=1e-5)


def test_subspace_alignment_is_symmetric_and_basis_free() -> None:
    A, B = orthonormal(64, 8, 1), orthonormal(64, 8, 2)
    forward = subspace_alignment(A, B)
    assert forward == pytest.approx(subspace_alignment(B, A), abs=1e-5)
    # Rotating within the subspace must not move the number.
    rotation = torch.linalg.qr(torch.randn(8, 8)).Q
    assert subspace_alignment(rotation @ A, B) == pytest.approx(forward, abs=1e-5)


def test_subspace_alignment_random_baseline_is_k_over_d() -> None:
    """The floor an observed value has to be read against; it is not 0."""
    d, k = 512, 16
    values = [
        subspace_alignment(orthonormal(d, k, s), orthonormal(d, k, s + 100))
        for s in range(8)
    ]
    assert sum(values) / len(values) == pytest.approx(k / d, rel=0.4)


def test_subspace_alignment_rejects_rank_mismatch() -> None:
    with pytest.raises(ValueError, match="rank mismatch"):
        subspace_alignment(orthonormal(32, 4, 0), orthonormal(32, 5, 1))


def test_energy_in_subspace_anchors() -> None:
    basis = orthonormal(32, 4, 0)
    inside = basis[:1] * 3.0
    assert energy_in_subspace(inside, basis).item() == pytest.approx(1.0, abs=1e-5)
    full = torch.linalg.qr(torch.randn(32, 32)).Q.T
    outside = full[10:11]
    assert energy_in_subspace(outside, full[:4]).item() == pytest.approx(0.0, abs=1e-5)


def test_cloud_stats_separates_bias_from_variance() -> None:
    """A cloud centered on the reference is variance; a displaced one is bias."""
    torch.manual_seed(0)
    reference = torch.zeros(64)
    reference[0] = 1.0
    noise = torch.randn(200, 64) * 0.3

    unbiased = cloud_stats(reference[None, :] + noise, reference)
    assert unbiased["cos_mean_to_ref"] > 0.95  # center lands on the reference
    assert unbiased["cos_to_ref_mean"] < unbiased["cos_mean_to_ref"]  # but spread

    offset = torch.zeros(64)
    offset[1] = 1.0
    biased = cloud_stats(reference[None, :] + offset[None, :] + noise, reference)
    assert biased["cos_mean_to_ref"] < 0.8


def test_cloud_stats_detects_sign_inconsistency() -> None:
    reference = torch.zeros(16)
    reference[0] = 1.0
    flipped = torch.cat([reference[None, :]] * 7 + [-reference[None, :]] * 3)
    stats = cloud_stats(flipped, reference)
    assert stats["cos_to_ref_frac_negative"] == pytest.approx(0.3)


def test_permutation_test_finds_real_structure() -> None:
    torch.manual_seed(0)
    d = 32
    a, b = torch.zeros(d), torch.zeros(d)
    a[0], b[1] = 1.0, 1.0
    vectors = torch.cat(
        [a[None, :] + torch.randn(8, d) * 0.1, b[None, :] + torch.randn(8, d) * 0.1]
    )
    result = label_permutation_test(vectors, ["a"] * 8 + ["b"] * 8, n_shuffles=500)
    assert result["statistic"] > 0.5
    assert result["p_value"] < 0.01


def test_permutation_test_p_is_never_zero() -> None:
    """+1 correction: with 500 shuffles the smallest reportable p is 1/501."""
    torch.manual_seed(0)
    vectors = torch.randn(10, 8)
    result = label_permutation_test(vectors, ["a"] * 5 + ["b"] * 5, n_shuffles=500)
    assert result["p_value"] >= 1 / 501


def test_permutation_test_on_unstructured_labels() -> None:
    torch.manual_seed(3)
    vectors = torch.randn(24, 16)
    result = label_permutation_test(vectors, (["a", "b", "c"] * 8), n_shuffles=500)
    assert 0.02 < result["p_value"] < 0.98


def test_leave_one_out_finds_the_dominating_prompt() -> None:
    """'Dominated by rare prompts' has to be detectable, not just describable."""
    torch.manual_seed(0)
    vectors = torch.randn(20, 32) * 0.1
    vectors[7] = torch.randn(32) * 50
    influence = leave_one_out_influence(vectors)
    assert int(influence.argmax().item()) == 7
    assert influence[7] > 5 * influence.median()


def test_rank_correlation_anchors() -> None:
    x = torch.arange(20.0)
    assert rank_correlation(x, x) == pytest.approx(1.0, abs=1e-5)
    assert rank_correlation(x, -x) == pytest.approx(-1.0, abs=1e-5)
    assert rank_correlation(x, x**3) == pytest.approx(
        1.0, abs=1e-5
    )  # monotone, not linear


def test_rank_correlation_averages_ties_and_is_order_independent() -> None:
    """Tied outcomes must not be broken by input order.

    The C2 outcome is a median target rank, and at a working strength half the
    prompts tie at rank 0. Plain argsort ranks them by whatever order the
    caller built its list in, so the same correlation computed two ways
    disagreed. Both properties are pinned: correct tie handling, and
    invariance to a permutation of the inputs.
    """
    import torch

    from jsteer.analysis import average_ranks, rank_correlation

    ties = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
    assert average_ranks(ties).tolist() == [1.0, 1.0, 1.0, 3.0, 4.0]

    x = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    y = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0])
    base = rank_correlation(x, y)

    perm = torch.tensor([5, 0, 7, 2, 6, 1, 4, 3])
    assert abs(rank_correlation(x[perm], y[perm]) - base) < 1e-6


def test_linear_cka_is_one_for_a_rescaled_copy_and_low_for_noise() -> None:
    """CKA's point here is scale invariance, which Frobenius distance lacks."""
    import torch

    from jsteer.analysis import linear_cka
    from jsteer.metrics import relative_frobenius

    torch.manual_seed(0)
    A = torch.randn(64, 64)

    assert linear_cka(A, A) == pytest.approx(1.0, abs=1e-5)
    # A pure gain change leaves the map's structure alone; CKA sees that and
    # relative Frobenius does not. That is exactly the confound this measure
    # exists to sidestep when two domain averages differ in overall magnitude.
    assert linear_cka(A, 7.0 * A) == pytest.approx(1.0, abs=1e-5)
    assert relative_frobenius(A, 7.0 * A) > 0.8

    # The chance floor is emphatically NOT zero: two independent isotropic
    # matrices score ~0.5, converging to exactly 0.5 as d grows. A structured
    # pair scores far lower. So CKA is only readable against a null measured on
    # the same kind of matrix -- assert the ordering, never an absolute cutoff.
    unrelated = linear_cka(A, torch.randn(64, 64))
    assert 0.4 < unrelated < 0.6

    perturbed = linear_cka(A, A + 0.05 * torch.randn(64, 64))
    assert perturbed > 0.9
    assert perturbed > unrelated

    # Structured matrices drive the floor down, which is why the floor cannot
    # be hardcoded and the wikitext half-split has to supply it empirically.
    def structured(seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        d = 256
        U = torch.linalg.qr(torch.randn(d, d)).Q
        V = torch.linalg.qr(torch.randn(d, d)).Q
        return U @ torch.diag(torch.exp(-torch.arange(d).float() / 32)) @ V.T

    assert linear_cka(structured(1), structured(2)) < 0.3

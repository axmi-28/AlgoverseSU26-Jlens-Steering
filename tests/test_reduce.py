"""The reduction is lossy on purpose; these fix *what* it is allowed to lose.

Every ``J_x`` is thrown away immediately after being digested, so a bug here is
unrecoverable without re-renting the GPU.
"""

from __future__ import annotations

import pytest
import torch

from jsteer.reduce import (
    RunningMean,
    digest_bytes,
    digest_jacobian,
    participation_ratio,
)


@pytest.fixture
def matrices() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    J_x = torch.randn(16, 16)
    J_bar = torch.randn(16, 16)
    cotangents = torch.randn(4, 16)
    return J_x, J_bar, cotangents


def test_full_rank_digest_reconstructs_the_matrix(matrices) -> None:
    """At k = d the digest is an exact factorization, so nothing is lost yet."""
    J_x, J_bar, cotangents = matrices
    digest = digest_jacobian(J_x, J_bar, cotangents, layer=3, k=16)
    reconstructed = digest.left.T @ torch.diag(digest.singular_values) @ digest.right
    torch.testing.assert_close(reconstructed, J_x, rtol=1e-4, atol=1e-5)


def test_bases_are_orthonormal(matrices) -> None:
    """``subspace_alignment`` assumes it; SVD guarantees it; assert it anyway."""
    J_x, J_bar, cotangents = matrices
    digest = digest_jacobian(J_x, J_bar, cotangents, layer=0, k=8)
    for basis in (digest.left, digest.right):
        torch.testing.assert_close(basis @ basis.T, torch.eye(8), rtol=1e-4, atol=1e-5)


def test_pullbacks_match_the_explicit_product(matrices) -> None:
    """The digest's ``g_x`` must be the same object script 05 computes."""
    J_x, J_bar, cotangents = matrices
    digest = digest_jacobian(J_x, J_bar, cotangents, layer=0, k=4)
    torch.testing.assert_close(digest.pullbacks, cotangents @ J_x)


def test_truncation_keeps_the_leading_directions(matrices) -> None:
    J_x, J_bar, cotangents = matrices
    full = digest_jacobian(J_x, J_bar, cotangents, layer=0, k=16)
    partial = digest_jacobian(J_x, J_bar, cotangents, layer=0, k=5)
    torch.testing.assert_close(partial.singular_values, full.singular_values[:5])
    assert (partial.singular_values.diff() <= 1e-5).all(), "must be descending"


def test_identical_matrices_have_zero_relative_frobenius(matrices) -> None:
    J_x, _, cotangents = matrices
    digest = digest_jacobian(J_x, J_x, cotangents, layer=0, k=4)
    assert digest.scalars["rel_frobenius"] == pytest.approx(0.0, abs=1e-6)
    assert digest.scalars["norm_ratio"] == pytest.approx(1.0, rel=1e-6)


def test_j_bar_may_be_omitted(matrices) -> None:
    J_x, _, cotangents = matrices
    digest = digest_jacobian(J_x, None, cotangents, layer=0, k=4)
    assert "rel_frobenius" not in digest.scalars
    assert digest.scalars["fro_norm"] > 0


def test_participation_ratio_bounds() -> None:
    """1 for a rank-1 spectrum, n for a flat one -- the two anchors it is read against."""
    assert participation_ratio(torch.tensor([5.0, 0.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert participation_ratio(torch.ones(7)) == pytest.approx(7.0)


def test_running_mean_matches_the_batch_mean() -> None:
    torch.manual_seed(1)
    tensors = [torch.randn(6, 6) for _ in range(50)]
    accumulator = RunningMean()
    for tensor in tensors:
        accumulator.add(tensor)
    assert accumulator.n == 50
    torch.testing.assert_close(
        accumulator.mean, torch.stack(tensors).mean(0), rtol=1e-6, atol=1e-6
    )


def test_running_mean_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        _ = RunningMean().mean


def test_digest_bytes_matches_the_tensors(matrices) -> None:
    """The budget the run scripts print has to be the budget they consume."""
    J_x, J_bar, cotangents = matrices
    digest = digest_jacobian(J_x, J_bar, cotangents, layer=0, k=8)
    assert digest.nbytes() == digest_bytes(16, 8, 4)


def test_running_mean_resume_is_exact() -> None:
    """A shard re-run after a timeout must not lose the prompts it skipped.

    The sweep rewrites its group-means file wholesale at the end, so without
    seeding from the saved state a resumed shard would replace its own output
    with a mean over only the prompts of the final attempt — a number that
    looks fine and is wrong.
    """
    torch.manual_seed(2)
    tensors = [torch.randn(5, 5) for _ in range(12)]

    first = RunningMean()
    for tensor in tensors[:7]:
        first.add(tensor)
    state = first.state()

    resumed = RunningMean()
    resumed.load_state(state)
    for tensor in tensors[7:]:
        resumed.add(tensor)

    assert resumed.n == 12
    torch.testing.assert_close(
        resumed.mean, torch.stack(tensors).mean(0), rtol=1e-6, atol=1e-6
    )


def test_running_mean_state_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        RunningMean().state()


def test_running_mean_load_state_accumulates() -> None:
    """Two shard states pool into one exact mean, as the merge step relies on."""
    a, b = RunningMean(), RunningMean()
    for _ in range(3):
        a.add(torch.full((2, 2), 2.0))
    for _ in range(1):
        b.add(torch.full((2, 2), 10.0))
    pooled = RunningMean()
    pooled.load_state(a.state())
    pooled.load_state(b.state())
    assert pooled.n == 4
    torch.testing.assert_close(pooled.mean, torch.full((2, 2), 4.0))

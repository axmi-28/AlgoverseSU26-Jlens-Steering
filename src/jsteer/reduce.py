"""Reducing ``J_x`` on the fly, because it cannot be stored.

Research question 1 wants the *distribution* of prompt-local Jacobians, which
means many of them. One ``J_x`` on Qwen3-8B is ``4096 x 4096`` fp32 = 67 MB per
layer, and the workspace band is 26 layers, so a single prompt is ~1.7 GB. At
64 prompts that is 112 GB of matrices to answer questions that are all about a
handful of scalars and a low-dimensional subspace.

So no ``J_x`` is ever written to disk. Each one is materialized, reduced by
:func:`digest_jacobian` while it is still in memory, and dropped. The digest is
~2 MB per (prompt, layer) at ``k=64``, which is 3.5 GB for the whole sweep --
large but ordinary, and re-analysable without touching a GPU again.

What survives the reduction, and why
------------------------------------
- **Top-k singular vectors and values.** Left vectors span the output side:
  which directions in the final-layer basis this prompt's Jacobian can write
  into at all, hence which ``u_y`` read stably. Right vectors span the input
  side: where in the residual stream a steering vector has to be written to
  land anywhere. Both are needed, and they answer different halves of the
  question. Storing the top-k basis rather than the matrix is also what makes
  the numbers comparable to A-LQR, which reports subspace alignment.
- **Pullbacks of the target cotangents.** Computing them here rather than in a
  second pass guarantees the ``g_x`` used for research question 2 comes from
  exactly the same ``J_x`` (same position convention, same truncation) as the
  research-question-1 geometry.
- **Scalars against ``J_bar``.** Relative Frobenius is kept, but as a
  *secondary* measure: it is dominated by bulk directions nothing reads, so a
  large value need not mean the lens is wrong for any token anyone cares about.
  The per-token cosine in the pullback block is the primary quantity.
- **A running sum.** ``J_bar`` is itself a mean of per-prompt Jacobians over
  the fitting corpus, so ``J_bar^T u = mean_x(J_x^T u)`` over *fitting*
  prompts, exactly. That identity turns wikitext into a calibrated null: any
  displacement of the cloud's center there is finite-sample error only (the
  published fits early-stopped at ~450-480 prompts), so displacement measured
  on eval prompts is genuine distribution shift rather than an artifact of the
  estimator. Keeping a running per-group mean is what lets that comparison be
  made.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from jsteer.metrics import identity_distance, relative_frobenius

logger = logging.getLogger(__name__)


@dataclass
class JacobianDigest:
    """Everything kept from one ``(prompt, layer)`` Jacobian.

    Attributes:
        layer: Source layer.
        singular_values: ``[k]``, descending.
        left: ``[k, d_model]``, rows are left singular vectors (output side).
        right: ``[k, d_model]``, rows are right singular vectors (input side).
        pullbacks: ``[K, d_model]``, ``J_x^T c`` for the run's cotangents.
        scalars: Named scalars -- see :func:`digest_jacobian`.
    """

    layer: int
    singular_values: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    pullbacks: torch.Tensor
    scalars: dict[str, float]

    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.singular_values, self.left, self.right, self.pullbacks)
        )


def participation_ratio(values: torch.Tensor) -> float:
    """``(sum s^2)^2 / sum s^4`` -- effective rank of a spectrum.

    Reported alongside the top-k basis because a subspace-alignment number is
    only interpretable next to how many directions actually carry weight: two
    Jacobians can agree perfectly on their top-8 and disagree everywhere else
    if the spectrum is flat.
    """
    s2 = values.float() ** 2
    return float((s2.sum() ** 2 / (s2**2).sum().clamp_min(1e-30)).item())


def digest_jacobian(
    J_x: torch.Tensor,
    J_bar: torch.Tensor | None,
    cotangents: torch.Tensor,
    *,
    layer: int,
    k: int = 64,
    lowrank: bool = False,
) -> JacobianDigest:
    """Reduce one prompt-local Jacobian to a storable digest.

    Args:
        J_x: ``[d_model, d_model]``, row ``i`` is ``dh_target[i] / dh_l``, the
            ``jlens`` convention.
        J_bar: The lens's averaged Jacobian at the same layer, or ``None`` to
            skip the comparison scalars.
        cotangents: ``[K, d_model]`` unembedding rows to pull back.
        layer: Recorded on the digest.
        k: How many singular directions to keep.
        lowrank: Use randomized SVD for the top-k instead of a full one. An
            exact 4096x4096 SVD is ~5 s on CPU, which is ~3 h across a 96-prompt
            x 26-layer sweep; the randomized version is a small multiple of a
            matmul. The cost is that it introduces sampling error into
            ``singular_values`` and into the very subspaces the alignment
            statistic is computed from, so it is off by default and
            :func:`lowrank_agreement` exists to bound the error on real
            matrices before a sweep relies on it. ``participation_ratio`` and
            ``spectral_norm`` are computed from the truncated spectrum under
            this flag and are not comparable with the exact ones.
    """
    J_x = J_x.float()
    if lowrank:
        # Oversample and take power iterations: the spectra here decay fast
        # (participation ratio 12-290 on qwen3-1.7b), which is the regime
        # randomized SVD is accurate in.
        U, S, V = torch.svd_lowrank(J_x, q=min(2 * k + 16, J_x.shape[0]), niter=4)
        Vh = V.T
    else:
        U, S, Vh = torch.linalg.svd(J_x, full_matrices=False)
    k = min(k, S.shape[0])

    scalars: dict[str, float] = {
        "fro_norm": float(J_x.norm().item()),
        "identity_distance": identity_distance(J_x),
        "participation_ratio": participation_ratio(S),
        "spectral_norm": float(S[0].item()),
    }
    if J_bar is not None:
        J_bar = J_bar.float()
        scalars["rel_frobenius"] = relative_frobenius(J_x, J_bar)
        scalars["norm_ratio"] = float(
            (J_x.norm() / J_bar.norm().clamp_min(1e-12)).item()
        )

    return JacobianDigest(
        layer=layer,
        singular_values=S[:k].clone(),
        left=U[:, :k].T.contiguous().clone(),
        right=Vh[:k].contiguous().clone(),
        pullbacks=(cotangents.float() @ J_x).clone(),
        scalars=scalars,
    )


class RunningMean:
    """Streaming mean of same-shaped tensors, fp64 to keep the sum honest.

    Used to rebuild a group's average Jacobian without holding the group's
    Jacobians. fp64 because the fitting corpus is ~460 prompts and the entries
    are O(1): an fp32 running sum loses low bits exactly where the interesting
    displacement lives.
    """

    def __init__(self) -> None:
        self.total: torch.Tensor | None = None
        self.n = 0

    def add(self, tensor: torch.Tensor) -> None:
        contribution = tensor.detach().to(torch.float64).cpu()
        if self.total is None:
            self.total = contribution.clone()
        else:
            self.total += contribution
        self.n += 1

    def load_state(self, state: dict) -> None:
        """Seed from a previously saved ``{"sum", "n"}``, for an exact resume.

        A shard that is re-run after a timeout skips the prompts whose outputs
        already landed. Its running mean must still cover them, or the pooled
        mean would silently be an average over only the prompts of the final
        attempt -- a number that looks fine and is wrong.
        """
        contribution = state["sum"].to(torch.float64)
        self.total = (
            contribution.clone() if self.total is None else self.total + contribution
        )
        self.n += int(state["n"])

    def state(self) -> dict:
        """``{"sum", "n"}`` -- what gets written, so merging can pool exactly."""
        if self.total is None:
            raise ValueError("RunningMean is empty")
        return {"sum": self.total, "n": self.n}

    @property
    def mean(self) -> torch.Tensor:
        if self.total is None:
            raise ValueError("RunningMean is empty")
        return (self.total / self.n).float()


def digest_bytes(d_model: int, k: int, n_cotangents: int) -> int:
    """Disk cost of one digest, for the budgeting the run scripts print."""
    return 4 * (k + 2 * k * d_model + n_cotangents * d_model)


def lowrank_agreement(J: torch.Tensor, k: int = 64) -> dict[str, float]:
    """How closely randomized SVD reproduces the exact top-k on this matrix.

    Run this on a handful of real ``J_x`` before enabling ``lowrank`` on a
    sweep. ``subspace_alignment`` near 1 and a small relative error on the
    singular values means the approximation is not what the results are
    measuring; anything else means pay for the exact SVD.
    """
    from jsteer.analysis import subspace_alignment

    exact = digest_jacobian(J, None, torch.zeros(1, J.shape[1]), layer=0, k=k)
    approx = digest_jacobian(
        J, None, torch.zeros(1, J.shape[1]), layer=0, k=k, lowrank=True
    )
    return {
        "left_alignment": subspace_alignment(exact.left, approx.left),
        "right_alignment": subspace_alignment(exact.right, approx.right),
        "singular_value_rel_error": float(
            (
                (approx.singular_values - exact.singular_values).norm()
                / exact.singular_values.norm()
            ).item()
        ),
    }

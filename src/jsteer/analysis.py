"""Turning digests into the statistics the two research questions ask for.

Nothing here touches a model, so it runs on a laptop against the artifacts a
GPU run leaves behind.

The organizing idea is **bias against variance**. Take the cloud of prompt-local
objects and ask where its center sits relative to the lens's averaged object:

- Center on ``J_bar`` / ``g_bar``, cloud wide: *variance*. The averaged
  direction is right on average and steering failures are bad luck.
- Center displaced: *bias*. The averaged direction is systematically wrong for
  the eval distribution, and failures are predictable rather than unlucky --
  the stronger and more useful result.

Whether a displacement is real rests on a calibration that comes free from the
estimator. ``J_bar`` is a mean of per-prompt Jacobians over the fitting corpus,
so on *fitting* prompts the cloud's center is ``J_bar`` by construction and any
measured displacement is finite-sample error. Wikitext is therefore not an
arbitrary control but a null with a known answer, which is what licenses
reading a displacement on eval prompts as distribution shift.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def subspace_alignment(A: torch.Tensor, B: torch.Tensor) -> float:
    """Overlap of two subspaces in ``[0, 1]``, given spanning row-sets.

    ``A``, ``B``: ``[k, d]``, rows spanning each subspace (the ``left`` / ``right``
    blocks of a :class:`~jsteer.reduce.JacobianDigest`, which are already
    orthonormal). Returns ``||Q_A^T Q_B||_F^2 / k`` -- the mean squared cosine
    of the principal angles, 1 for identical subspaces and ``k/d`` in
    expectation for random ones.

    This is the quantity A-LQR reports (~0.8 early and late, ~0.5 mid-network on
    Gemma-2-2B across 50 Jacobians per layer), so it is the one number here with
    an external point of comparison. Note the random-chance floor is not 0: at
    ``k=64``, ``d=4096`` it is 0.016, so an observed 0.5 is far from chance and
    far from agreement at the same time.
    """
    if A.shape[0] != B.shape[0]:
        raise ValueError(f"rank mismatch: {A.shape[0]} vs {B.shape[0]}")
    Qa = torch.linalg.qr(A.float().T).Q
    Qb = torch.linalg.qr(B.float().T).Q
    return float(((Qa.T @ Qb).norm() ** 2 / A.shape[0]).item())


def linear_cka(A: torch.Tensor, B: torch.Tensor) -> float:
    """Linear CKA between two same-shaped matrices, in ``[0, 1]``.

    The J-lens paper uses CKA for J-space geometry -- "For each pair of layers,
    we compute the similarity of the J-space's geometry. We do so using
    centered kernel alignment" -- and reads its block structure as evidence
    about where the lens works. Using it here makes a cross-*domain* comparison
    at a fixed layer commensurable with the paper's cross-*layer* one, though
    the two are not the same measurement and the numbers should not be pooled.

    Its value here is that it is invariant to isotropic rescaling of either
    matrix, which relative Frobenius is not: two Jacobians can differ mostly in
    overall gain, which inflates a Frobenius distance while leaving the map's
    structure -- the thing a lens reads through -- unchanged.

    Rows are treated as samples and columns as features, and both are
    column-centered, which is what makes this the *centered* kernel alignment
    rather than a bare cosine between flattened matrices.

    **Its chance floor is not zero, and not even a constant.** Measured on
    square ``d x d`` matrices: two independent isotropic Gaussians score 0.489
    at ``d=16`` and converge to exactly 0.500 by ``d=1024``, while two
    independent matrices with a fast-decaying spectrum -- the regime a real
    Jacobian is in -- score 0.126 at ``d=256`` and 0.032 at ``d=1024``. So the
    floor depends on the spectrum of the matrices being compared, and a CKA of
    0.5 between two Jacobians could mean "no shared structure at all" or
    "substantial shared structure", depending on how quickly their singular
    values decay. Never read one of these numbers on its own: read it against
    the null pair measured on the same kind of matrix, which is the entire
    reason the domain panel carries two disjoint wikitext halves.
    """
    if A.shape != B.shape:
        raise ValueError(f"shape mismatch: {tuple(A.shape)} vs {tuple(B.shape)}")
    X = A.float() - A.float().mean(dim=0, keepdim=True)
    Y = B.float() - B.float().mean(dim=0, keepdim=True)
    # ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F), computed without forming the
    # d x d Gram matrices any larger than they already are.
    cross = (Y.T @ X).norm() ** 2
    denom = (X.T @ X).norm() * (Y.T @ Y).norm()
    return float((cross / denom.clamp_min(1e-30)).item())


def energy_in_subspace(vectors: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Fraction of each vector's squared norm lying in ``basis``'s span.

    ``vectors``: ``[K, d]``; ``basis``: ``[k, d]`` orthonormal rows. This is the
    bridge from research question 1 to research question 2: a token whose
    unembedding row lies mostly *outside* the stably-shared left subspace has
    no reason for its pullback to be stable either, which is a prediction about
    which target tokens steer reliably that can be checked against the causal
    outcomes rather than assumed.
    """
    Q = torch.linalg.qr(basis.float().T).Q
    projected = vectors.float() @ Q
    total = vectors.float().norm(dim=-1).clamp_min(1e-12) ** 2
    return (projected.norm(dim=-1) ** 2) / total


def cloud_stats(vectors: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """Bias/variance decomposition of a cloud of directions about a reference.

    ``vectors``: ``[n, d]`` prompt-local directions. ``reference``: ``[d]``, the
    averaged direction the lens would use.

    Returns:
        ``cos_mean_to_ref``   -- angle between the cloud's *center* and the
            reference. This is the bias term: near 1 means the averaged
            direction points where the eval prompts point on average.
        ``cos_to_ref_mean`` / ``cos_to_ref_frac_negative`` -- the per-prompt
            spread about the reference. Sign-inconsistency is the interesting
            tail: a prompt with negative cosine is one the averaged direction
            steers the wrong way, not merely weakly.
        ``cos_to_center_mean`` -- spread about the cloud's own center, i.e. how
            much of the disagreement survives after removing the bias.
        ``norm_ratio_*``      -- the length half of the comparison, which the
            cosine deliberately discards; a direction can be perfectly aligned
            and still under- or over-shoot.
        ``participation_ratio`` -- effective number of directions in the cloud.
            Low values mean the spread is dominated by a few rare prompts, one
            of the dispositions research question 2 asks about by name.
    """
    vectors = vectors.float()
    reference = reference.float()
    center = vectors.mean(dim=0)

    def cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a * b).sum(-1) / (
            a.norm(dim=-1).clamp_min(1e-12) * b.norm(dim=-1).clamp_min(1e-12)
        )

    to_ref = cos(vectors, reference[None, :])
    to_center = cos(vectors, center[None, :])
    norms = vectors.norm(dim=-1)
    ratio = norms / reference.norm().clamp_min(1e-12)

    # Effective rank of the cloud itself, via the Gram spectrum of the
    # mean-removed directions.
    centered = vectors - center
    s = torch.linalg.svdvals(centered) ** 2
    pr = float((s.sum() ** 2 / (s**2).sum().clamp_min(1e-30)).item())

    return {
        "n": int(vectors.shape[0]),
        "cos_mean_to_ref": float(cos(center[None, :], reference[None, :]).item()),
        "cos_to_ref_mean": float(to_ref.mean().item()),
        "cos_to_ref_median": float(to_ref.median().item()),
        "cos_to_ref_p10": float(to_ref.quantile(0.10).item()),
        "cos_to_ref_frac_negative": float((to_ref < 0).float().mean().item()),
        "cos_to_center_mean": float(to_center.mean().item()),
        "norm_ratio_median": float(ratio.median().item()),
        "norm_ratio_iqr": float((ratio.quantile(0.75) - ratio.quantile(0.25)).item()),
        "center_norm_ratio": float(
            (center.norm() / reference.norm().clamp_min(1e-12)).item()
        ),
        "participation_ratio": pr,
    }


def _mean_within_between(
    vectors: torch.Tensor, labels: list, generator: torch.Generator | None = None
) -> float:
    """Mean within-group cosine minus mean between-group cosine."""
    unit = vectors.float()
    unit = unit / unit.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    gram = unit @ unit.T
    n = len(labels)
    same = torch.tensor(
        [[labels[i] == labels[j] for j in range(n)] for i in range(n)], dtype=torch.bool
    )
    off = ~torch.eye(n, dtype=torch.bool)
    within = gram[same & off]
    between = gram[~same & off]
    if within.numel() == 0 or between.numel() == 0:
        return float("nan")
    return float((within.mean() - between.mean()).item())


def label_permutation_test(
    vectors: torch.Tensor,
    labels: list,
    *,
    n_shuffles: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Does a *known* label explain the cloud's structure better than chance?

    Preferred over unsupervised clustering here because the candidate
    structures are named in advance -- category, template, argument -- and
    because the design is small: the crossed 4-arg x 4-template grid only exists
    *within* a category (categories share no templates and no args), so a
    within-category test has n=16. Asking k-means to discover a partition at
    that size would mostly measure the clustering algorithm.

    The statistic is mean within-group cosine minus mean between-group cosine,
    compared against its distribution under ``n_shuffles`` label permutations.
    Run the test inside each category and pool the *statistic* across the four;
    never pool the vectors, since a between-category contrast would be
    confounded with the templates and args differing too.
    """
    observed = _mean_within_between(vectors, labels)
    generator = torch.Generator().manual_seed(seed)
    null = []
    for _ in range(n_shuffles):
        permutation = torch.randperm(len(labels), generator=generator).tolist()
        null.append(_mean_within_between(vectors, [labels[i] for i in permutation]))
    null_tensor = torch.tensor(null)
    # +1 in both terms: the observed value is one draw from the null under H0,
    # so this cannot report p=0 off 1000 shuffles.
    p_value = float(((null_tensor >= observed).sum() + 1).item() / (n_shuffles + 1))
    return {
        "statistic": observed,
        "null_mean": float(null_tensor.mean().item()),
        "null_std": float(null_tensor.std().item()),
        "p_value": p_value,
        "n": len(labels),
    }


def leave_one_out_influence(vectors: torch.Tensor) -> torch.Tensor:
    """How far the cloud's center moves when each prompt is removed, ``[n]``.

    Normalized by the center's norm. This is the direct test of "dominated by
    rare prompts": if a handful of prompts each move the mean by a large
    fraction, the averaged direction is an artifact of those prompts rather
    than a summary of the distribution.
    """
    vectors = vectors.float()
    n = vectors.shape[0]
    total = vectors.sum(dim=0)
    center = total / n
    loo = (total[None, :] - vectors) / (n - 1)
    return (loo - center[None, :]).norm(dim=-1) / center.norm().clamp_min(1e-12)


def average_ranks(v: torch.Tensor) -> torch.Tensor:
    """Ranks with ties given their mean rank -- the standard Spearman treatment.

    This is not a detail here. The steerability outcome is a *median target
    rank*, and at a working strength 20-30 of the 64 prompts sit at rank 0
    together. A plain ``argsort`` breaks those ties arbitrarily, by input
    order, which makes the resulting correlation both wrong and dependent on
    the order the caller happened to build its lists in -- two call sites
    computing "the same" number disagreed by up to 0.1, which is how this was
    found.
    """
    v = v.float()
    n = len(v)
    order = v.argsort()
    ordered = v[order]
    out = torch.empty(n, dtype=torch.float32)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ordered[j + 1] == ordered[i]:
            j += 1
        out[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return out


def rank_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """Spearman rho, for relating a geometric predictor to a steerability score.

    Spearman rather than Pearson because the causal scores are bounded and the
    geometric predictors are not, and no linear relationship is claimed -- the
    hypothesis is only monotone (prompts further from the average steer worse).

    Ties are averaged; see :func:`average_ranks` for why that matters here.
    """
    ranks = average_ranks
    rx, ry = ranks(x), ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = (rx.norm() * ry.norm()).clamp_min(1e-12)
    return float(((rx * ry).sum() / denominator).item())

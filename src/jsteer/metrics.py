"""Geometry between prompt-local and averaged Jacobians.

These are the quantities research question 1 is phrased in: how far ``J_x`` sits
from ``J_bar``, and how far ``g_x = J_x^T u_y`` sits from ``g_bar = J_bar^T u_y``.
"""

from __future__ import annotations

import torch


def cosine(a: torch.Tensor, b: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """Cosine similarity, fp32, safe on zero vectors (returns 0)."""
    a, b = a.float(), b.float()
    na = a.norm(dim=dim, keepdim=True).clamp_min(1e-12)
    nb = b.norm(dim=dim, keepdim=True).clamp_min(1e-12)
    return ((a / na) * (b / nb)).sum(dim=dim)


def relative_frobenius(a: torch.Tensor, b: torch.Tensor) -> float:
    """``||a - b||_F / ||b||_F``."""
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def identity_distance(J: torch.Tensor) -> float:
    """``||J - I||_F / sqrt(d)``.

    This reproduces the ``identity_distance`` column in the pre-fitted lenses'
    published ``*_convergence.csv``, which is how far the running-mean Jacobian
    sits from the identity -- a proxy for how much work the transport is doing
    beyond a plain logit lens.

    The definition was not documented anywhere; it was recovered by matching
    the published final values, and the column is evaluated at the *last*
    source layer (the one just below the target), not aggregated over layers:

        qwen3-1.7b  L26  ours 0.524536  published 0.52469
        gemma-2-2b  L24  ours 0.478107  published 0.478237

    (the residual gap is the fp16 storage of the published lens against the
    fp32 running sum it was computed from). :func:`lens_identity_distance`
    applies the layer convention; :func:`identity_distance_candidates` keeps
    the alternatives that were ruled out.
    """
    d = J.shape[-1]
    eye = torch.eye(d, dtype=torch.float32, device=J.device)
    return ((J.float() - eye).norm() / (d**0.5)).item()


def identity_distance_candidates(J: torch.Tensor) -> dict[str, float]:
    """Plausible definitions of ``identity_distance``, for pinning it down.

    Neuronpedia's ``fit_lens.py`` is not public, so the column's definition is
    inferred by matching the published final values. Kept around so the match
    can be re-checked on a new model rather than taken on faith.
    """
    J = J.float()
    d = J.shape[-1]
    eye = torch.eye(d, dtype=torch.float32, device=J.device)
    diff = J - eye
    return {
        "fro_over_sqrt_d": (diff.norm() / (d**0.5)).item(),
        "fro_over_fro_eye": (diff.norm() / eye.norm()).item(),
        "fro_over_fro_J": (diff.norm() / J.norm().clamp_min(1e-12)).item(),
        "fro": diff.norm().item(),
        "mean_abs": diff.abs().mean().item(),
        "one_minus_mean_diag": (1.0 - J.diagonal().mean()).item(),
    }


def jacobian_summary(J_x: torch.Tensor, J_bar: torch.Tensor) -> dict[str, float]:
    """How one prompt-local Jacobian sits relative to the average."""
    J_x, J_bar = J_x.float(), J_bar.float()
    return {
        "rel_frobenius": relative_frobenius(J_x, J_bar),
        "cos_flat": cosine(J_x.flatten(), J_bar.flatten()).item(),
        "norm_ratio": (J_x.norm() / J_bar.norm().clamp_min(1e-12)).item(),
        "identity_distance_local": identity_distance(J_x),
        "identity_distance_avg": identity_distance(J_bar),
    }


def pullback_summary(g_x: torch.Tensor, g_bar: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-cotangent comparison of local and averaged pullbacks.

    ``g_x``, ``g_bar``: ``[K, d_model]``. The ``cos`` column is the headline
    quantity for research question 2 -- if ``cos(g_x, g_bar)`` is near 1 the
    averaged write direction is locally faithful; if it scatters or goes
    negative, the averaged direction is the wrong direction for that prompt.
    """
    g_x, g_bar = g_x.float(), g_bar.float()
    return {
        "cos": cosine(g_x, g_bar),
        "norm_local": g_x.norm(dim=-1),
        "norm_avg": g_bar.norm(dim=-1),
        "norm_ratio": g_x.norm(dim=-1) / g_bar.norm(dim=-1).clamp_min(1e-12),
        "rel_error": (g_x - g_bar).norm(dim=-1) / g_bar.norm(dim=-1).clamp_min(1e-12),
    }


def lens_identity_distance(lens) -> float:
    """``identity_distance`` for a fitted lens, using the published convention.

    Evaluated at ``lens.source_layers[-1]`` -- see :func:`identity_distance`.
    Compare against ``config.fit.final_identity_distance``.
    """
    return identity_distance(lens.jacobians[lens.source_layers[-1]])

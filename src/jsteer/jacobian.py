"""Prompt-local Jacobians and their pullbacks.

The lens uses an average Jacobian ``J_bar = E_x[J_x]``. This project is about
``J_x`` itself, so the primitive here is the *pullback*

    g_x = J_x^T c

for an arbitrary cotangent ``c`` in the target-layer basis, computed without
ever materializing ``J_x``.

Why that matters. ``jlens.fitting.jacobian_for_prompt`` builds ``J_x`` by
injecting one-hot cotangents, ``d_model`` of them, ``dim_batch`` per backward
pass. For Qwen3-8B that is 4096 one-hots = 32 backward passes and a 67 MB fp32
matrix *per layer*. But almost every question we want to ask is of the form
"where does the unembedding row for token ``y`` pull back to?", which needs
only ``J_x^T u_y`` -- one column of numbers, not the whole matrix. Batching
``K`` cotangents along the batch axis costs ``ceil(K / dim_batch)`` backward
passes, so ~100 target tokens is a single pass and 1.6 MB. Full ``J_x`` stays
available for small subsamples.

The estimator. ``jlens`` defines, over the valid position set ``V``,

    J_x[i, :] = mean_{p in V}  d( sum_{p' in V} h_target[p', i] ) / d h_l[p, :]

so, by linearity, for any cotangent ``c``:

    (J_x^T c)[j] = sum_i c[i] J_x[i, j]
                 = mean_{p in V} d( sum_{p' in V} <c, h_target[p', :]> ) / d h_l[p, j]

i.e. exactly the same backward pass with the one-hot replaced by ``c``. That
identity is what :func:`jacobian_for_prompt` (cotangents = I) and the
``test_pullback_matches_explicit_jacobian`` test check.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from jlens.fitting import (
    SKIP_FIRST_N_POSITIONS,
    _check_layer_indices,
    valid_position_mask,
)
from jlens.hooks import ActivationRecorder

logger = logging.getLogger(__name__)


def resolve_positions(
    seq_len: int,
    positions: Sequence[int] | None,
    *,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    what: str = "positions",
) -> torch.Tensor:
    """Resolve a position set, defaulting to the lens's fitting convention.

    ``None`` reproduces ``jlens.fitting.valid_position_mask``: positions
    ``[skip_first, seq_len - 1)``, dropping attention-sink positions and the
    final position (which has no next-token target).

    That default is only usable on prompts longer than ``skip_first + 1``
    tokens. The lenses were fitted on 128-token wikitext windows, but the
    paper's steering prompt sets are much shorter -- "The capital of France is
    the city of" is 8 tokens under Qwen3 -- so on eval prompts the convention
    has to be stated explicitly rather than inherited. Which choice is right is
    a research question, not a default: the fitting estimator averages ``J_x``
    over source positions, whereas the steering interventions write at
    particular positions, so "the prompt-local Jacobian" is only well-defined
    once a position set is named.

    Negative indices count from the end, so ``[-1]`` is the last token (the one
    the next-token prediction is read at).
    """
    if positions is None:
        return valid_position_mask(seq_len, skip_first=skip_first).nonzero(
            as_tuple=True
        )[0]
    resolved = sorted({p + seq_len if p < 0 else p for p in positions})
    if not resolved:
        raise ValueError(f"{what} is empty")
    if resolved[0] < 0 or resolved[-1] >= seq_len:
        raise ValueError(
            f"{what} {sorted(positions)} out of range for seq_len={seq_len}"
        )
    return torch.tensor(resolved, dtype=torch.long)


@dataclass
class PullbackResult:
    """Output of :func:`pullback_for_prompt`.

    Attributes:
        pullbacks: ``{source_layer: Tensor[K, d_model]}``, fp32 on CPU. Row
            ``k`` is ``J_x[layer]^T c_k``.
        seq_len: Tokenized length of the (truncated) prompt.
        n_valid_positions: Size of the position set the estimator averages over.
        n_backward_passes: ``ceil(K / dim_batch)``.
        target_layer: Resolved target layer index.
    """

    pullbacks: dict[int, torch.Tensor]
    seq_len: int
    n_valid_positions: int
    n_backward_passes: int
    target_layer: int

    def __getitem__(self, layer: int) -> torch.Tensor:
        return self.pullbacks[layer]


def pullback_for_prompt(
    model,
    prompt: str,
    source_layers: Sequence[int],
    cotangents: torch.Tensor,
    *,
    target_layer: int | None = None,
    source_positions: Sequence[int] | None = None,
    target_positions: Sequence[int] | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
) -> PullbackResult:
    """Pull ``K`` cotangents back through the prompt-local Jacobian.

    Args:
        model: A ``jlens`` ``LensModel``.
        prompt: Input text.
        source_layers: Layers ``l`` to pull back to.
        cotangents: ``[K, d_model]`` in the *target-layer residual* basis. For
            the J-Lens steering direction of token ``y`` this is
            ``jsteer.loading.unembedding_rows(model, [y])``.
        target_layer: Defaults to the final layer, matching the fitted lenses
            (their ``config.yaml`` records ``target_layer: null``).
        source_positions: Positions ``p`` the Jacobian is averaged over.
            ``None`` uses the fitting convention (see
            :func:`resolve_positions`), which requires a prompt longer than
            ``skip_first + 1`` tokens. Negative indices count from the end, so
            ``[-1]`` gives the Jacobian at the final token alone.
        target_positions: Positions ``p'`` the cotangent is injected at, summed
            over. ``None`` matches ``source_positions``, as the fitting
            estimator does.
        dim_batch: Cotangents per backward pass; the prompt is replicated this
            many times along the batch axis, so this trades memory for passes.
        max_seq_len: Truncate the prompt to this many tokens. The pre-fitted
            lenses used 128; changing it changes the position set and so
            changes ``J_x``.
        skip_first: Leading positions excluded from the average (attention
            sinks). The pre-fitted lenses used 16.

    Returns:
        A :class:`PullbackResult`.

    Raises:
        ValueError: If ``cotangents`` is the wrong shape or the prompt is too
            short to leave any valid positions.
    """
    if cotangents.ndim != 2 or cotangents.shape[1] != model.d_model:
        raise ValueError(
            f"cotangents must be [K, d_model={model.d_model}], "
            f"got {tuple(cotangents.shape)}"
        )
    n_cotangents = cotangents.shape[0]
    if n_cotangents == 0:
        raise ValueError("cotangents is empty")

    source_layers, target_layer = _check_layer_indices(
        source_layers, target_layer, model.n_layers
    )

    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    source_index = resolve_positions(
        seq_len, source_positions, skip_first=skip_first, what="source_positions"
    )
    target_index = (
        source_index
        if target_positions is None
        else resolve_positions(
            seq_len, target_positions, skip_first=skip_first, what="target_positions"
        )
    )
    n_valid_positions = len(source_index)

    batch = min(dim_batch, n_cotangents)
    n_passes = math.ceil(n_cotangents / batch)
    out = {
        layer: torch.zeros(n_cotangents, model.d_model, dtype=torch.float32)
        for layer in source_layers
    }

    with (
        ActivationRecorder(
            model.layers,
            at=[*source_layers, target_layer],
            start_graph_at=min(source_layers),
        ) as recorder,
        torch.enable_grad(),
    ):
        # One forward on the prompt replicated `batch` times; the retained
        # graph is reused by every backward below. Same shape as the jlens
        # fitting loop -- only the cotangents differ.
        model.forward(input_ids.expand(batch, -1))
        target_activation = recorder.activations[target_layer]
        source_activations = [recorder.activations[layer] for layer in source_layers]

        device = target_activation.device
        target_index = target_index.to(device)
        # Cast once: the backward runs in the model's dtype regardless, so
        # carrying fp32 cotangents in would only hide where precision is lost.
        cotangents_dev = cotangents.to(device=device, dtype=target_activation.dtype)
        cotangent_buffer = torch.zeros_like(target_activation)

        for pass_idx in range(n_passes):
            start = pass_idx * batch
            stop = min(start + batch, n_cotangents)
            n_this_pass = stop - start

            # c_k at every valid target position, for batch element k-start.
            cotangent_buffer.zero_()
            cotangent_buffer[:n_this_pass, target_index, :] = cotangents_dev[
                start:stop, None, :
            ]

            grads = torch.autograd.grad(
                outputs=target_activation,
                inputs=source_activations,
                grad_outputs=cotangent_buffer,
                retain_graph=(pass_idx < n_passes - 1),
            )
            for layer, grad in zip(source_layers, grads, strict=True):
                positions_on_device = source_index.to(grad.device, non_blocking=True)
                rows = grad[:n_this_pass, positions_on_device, :].float().mean(dim=1)
                out[layer][start:stop, :] = rows.cpu()
            del grads

    return PullbackResult(
        pullbacks=out,
        seq_len=seq_len,
        n_valid_positions=n_valid_positions,
        n_backward_passes=n_passes,
        target_layer=target_layer,
    )


def jacobian_for_prompt(
    model,
    prompt: str,
    source_layers: Sequence[int],
    *,
    target_layer: int | None = None,
    source_positions: Sequence[int] | None = None,
    target_positions: Sequence[int] | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
) -> PullbackResult:
    """Full ``J_x`` as the identity-cotangent case of :func:`pullback_for_prompt`.

    Returns ``[d_model, d_model]`` per layer, row ``i`` being ``J_x[i, :]`` --
    the same convention and the same numbers as
    ``jlens.fitting.jacobian_for_prompt`` (asserted in ``tests/``). Prefer
    :func:`pullback_for_prompt` unless the whole matrix is genuinely needed:
    this costs ``ceil(d_model / dim_batch)`` backward passes.
    """
    identity = torch.eye(model.d_model, dtype=torch.float32)
    return pullback_for_prompt(
        model,
        prompt,
        source_layers,
        identity,
        target_layer=target_layer,
        source_positions=source_positions,
        target_positions=target_positions,
        dim_batch=dim_batch,
        max_seq_len=max_seq_len,
        skip_first=skip_first,
    )


def averaged_pullback(lens, cotangents: torch.Tensor, layer: int) -> torch.Tensor:
    """``g_bar = J_bar^T c`` from a fitted lens, shape ``[K, d_model]``.

    The baseline every prompt-local ``g_x`` gets compared against.
    """
    J_bar = lens.jacobians[layer]
    return cotangents.float() @ J_bar.float()


def steering_direction(
    pullback: torch.Tensor, *, residual_norm: float, strength: float = 1.0
) -> torch.Tensor:
    """Turn a pullback into the paper's write vector.

    The paper's recipe (``data/experiments/README.md``, verbal-introspection):
    "the unit-normalized transpose row for that token, scaled by the layer's
    mean residual norm times a strength scalar". Treating the covector's
    components as an activation-space vector is exactly the step this project
    is questioning -- this function implements the convention so that we
    measure the thing the causal experiments actually do.
    """
    unit = pullback / pullback.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return unit * (residual_norm * strength)


def energy_map_for_prompt(
    model,
    prompt: str,
    source_layers: Sequence[int],
    *,
    target_positions: Sequence[int],
    n_probe: int = 8,
    target_layer: int | None = None,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    seed: int = 0,
) -> tuple[dict[int, torch.Tensor], list[int], int]:
    """Yan et al.'s Jacobian energy ``E_l(t, t') = ||d h_final,t' / d h_l,t||_F^2``.

    Their Sec. 3.3 splits the J-lens's behaviour by where this energy sits: mass
    on the diagonal ``t ~ t'`` is short-horizon next-token prediction, while
    horizontal and vertical stripes are the sparse-concept positions. The split
    is the paper's, the estimator below is not -- they do not say how they
    compute a ``d x d`` block norm at every cell.

    Computing each block exactly costs ``d_model`` backward passes, which is
    hopeless for a full map. But the norm is a trace, so Hutchinson applies:
    for ``v`` with identity covariance, ``E_v[||A^T v||^2] = tr(A A^T) =
    ||A||_F^2``. One backward pass from a cotangent placed at a single ``t'``
    returns ``v^T d h_final,t' / d h_l,t`` for *every* source position ``t`` at
    once, so ``n_probe`` passes estimate an entire row of the map -- every
    source position and every requested layer -- rather than one cell.

    The estimate is unbiased but noisy at ``n_probe = 8``; it is used only for
    the aggregate offset profile, where averaging over positions and documents
    is what carries the signal, never for a single cell.

    Returns:
        ``(energies, targets, seq_len)``. ``energies`` maps each layer to a
        ``[seq_len, n_target]`` tensor indexed by absolute source position and
        by position within ``targets``. Cells with ``t > t'`` are exactly zero
        by causality and are left in, so the caller can assert on them.
    """
    source_layers, target_layer = _check_layer_indices(
        source_layers, target_layer, model.n_layers
    )
    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    targets = [t for t in target_positions if skip_first <= t < seq_len]
    if not targets:
        raise ValueError(f"no target positions in [{skip_first}, {seq_len})")

    generator = torch.Generator().manual_seed(seed)
    out = {
        layer: torch.zeros(seq_len, len(targets), dtype=torch.float32)
        for layer in source_layers
    }

    with (
        ActivationRecorder(
            model.layers,
            at=[*source_layers, target_layer],
            start_graph_at=min(source_layers),
        ) as recorder,
        torch.enable_grad(),
    ):
        model.forward(input_ids.expand(n_probe, -1))
        target_activation = recorder.activations[target_layer]
        source_activations = [recorder.activations[layer] for layer in source_layers]
        device, dtype = target_activation.device, target_activation.dtype

        # Gaussian, not Rademacher: both have identity covariance and so both
        # are unbiased here, but the probes are shared across every t' below
        # and Rademacher's +-1 entries make that reuse visibly correlated.
        probes = torch.randn(
            n_probe, model.d_model, generator=generator, dtype=torch.float32
        ).to(device=device, dtype=dtype)
        buffer = torch.zeros_like(target_activation)

        for j, t_prime in enumerate(targets):
            buffer.zero_()
            buffer[:, t_prime, :] = probes
            grads = torch.autograd.grad(
                outputs=target_activation,
                inputs=source_activations,
                grad_outputs=buffer,
                retain_graph=(j < len(targets) - 1),
            )
            for layer, grad in zip(source_layers, grads, strict=True):
                # [n_probe, seq_len, d] -> mean over probes of the squared norm.
                out[layer][:, j] = grad.float().pow(2).sum(-1).mean(0).cpu()
            del grads

    return out, targets, seq_len


#: Horizon buckets for :func:`horizon_pullbacks_for_prompt`, as inclusive
#: ``(lo, hi)`` ranges over ``d = t' - t``. ``(0, 0)`` is Yan et al.'s diagonal;
#: everything above it is the off-diagonal term resolved by distance. Widths
#: grow because large ``d`` is rarer inside a 128-token window, so equal-width
#: bins would leave the far buckets estimated from a handful of pairs.
HORIZON_BUCKETS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 4),
    (5, 8),
    (9, 16),
    (17, 1 << 30),
)


def horizon_pullbacks_for_prompt(
    model,
    prompt: str,
    source_layers: Sequence[int],
    cotangents: torch.Tensor,
    *,
    target_layer: int | None = None,
    dim_batch: int = 16,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    t_stride: int = 1,
    buckets: Sequence[tuple[int, int]] = HORIZON_BUCKETS,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[str, int]]:
    """Resolve the pullback by *distance* ``d = t' - t``, not just diag/off.

    :func:`component_pullbacks_for_prompt` splits the averaged pullback into
    ``t' = t`` and ``t' > t``. This bins the second half by how far ahead the
    influenced position is, giving one write direction per horizon:

        J_bar^(d) = E[ d h_final,t+d / d h_l,t ],   v_y^(d) = J_bar^(d)^T u_y

    The question it exists to answer is whether those are different directions
    at all. If ``v^(1)`` and ``v^(16)`` are near-collinear, the off-diagonal
    term is horizon-agnostic, the only special horizon is 0, and there is no
    positional addressing to look for. If they separate, writing may be
    targetable in time as well as in content.

    Costs exactly what the diag/off split costs -- one backward pass per target
    position either way -- since a pass already yields the whole column over
    ``t`` and this only bins that column differently.

    Returns:
        ``({layer: {bucket_name: [K, d_model], ..., "total": [K, d_model]}},
        {bucket_name: n_pairs})``. Buckets are normalized like ``total`` (by
        the number of target positions), so they sum to it; the pair counts are
        returned separately because a bucket's direction is only as trustworthy
        as the number of ``(t, t')`` pairs behind it.
    """
    if cotangents.ndim != 2 or cotangents.shape[1] != model.d_model:
        raise ValueError(
            f"cotangents must be [K, d_model={model.d_model}], "
            f"got {tuple(cotangents.shape)}"
        )
    source_layers, target_layer = _check_layer_indices(
        source_layers, target_layer, model.n_layers
    )
    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    valid = resolve_positions(seq_len, None, skip_first=skip_first)
    targets = valid[::t_stride]
    n_valid = len(targets)
    n_cotangents = cotangents.shape[0]
    batch = min(dim_batch, n_cotangents)

    names = [f"h{lo}" if lo == hi else f"h{lo}_{hi}" for lo, hi in buckets]
    names = [
        n if b[1] < (1 << 30) else f"h{b[0]}plus" for n, b in zip(names, buckets, strict=True)
    ]
    out = {
        layer: {
            name: torch.zeros(n_cotangents, model.d_model, dtype=torch.float32)
            for name in [*names, "total"]
        }
        for layer in source_layers
    }
    pair_counts = dict.fromkeys(names, 0)

    with (
        ActivationRecorder(
            model.layers,
            at=[*source_layers, target_layer],
            start_graph_at=min(source_layers),
        ) as recorder,
        torch.enable_grad(),
    ):
        model.forward(input_ids.expand(batch, -1))
        target_activation = recorder.activations[target_layer]
        source_activations = [recorder.activations[layer] for layer in source_layers]
        device = target_activation.device
        valid_on_device = valid.to(device)
        cotangents_dev = cotangents.to(device=device, dtype=target_activation.dtype)
        buffer = torch.zeros_like(target_activation)

        # Which valid source positions fall in each bucket, per target. Built
        # once per t' and reused across cotangent batches and layers.
        selections: dict[int, list[torch.Tensor]] = {}
        for t_prime in targets.tolist():
            d = t_prime - valid_on_device
            sel = []
            for lo, hi in buckets:
                mask = (d >= lo) & (d <= hi)
                sel.append(valid_on_device[mask])
            selections[t_prime] = sel

        n_batches = math.ceil(n_cotangents / batch)
        for pass_idx in range(n_batches):
            start = pass_idx * batch
            stop = min(start + batch, n_cotangents)
            n_this = stop - start
            for j, t_prime in enumerate(targets.tolist()):
                buffer.zero_()
                buffer[:n_this, t_prime, :] = cotangents_dev[start:stop, None, :][:, 0]
                last = pass_idx == n_batches - 1 and j == n_valid - 1
                grads = torch.autograd.grad(
                    outputs=target_activation,
                    inputs=source_activations,
                    grad_outputs=buffer,
                    retain_graph=not last,
                )
                for layer, grad in zip(source_layers, grads, strict=True):
                    rows = grad[:n_this].float()
                    out[layer]["total"][start:stop] += (
                        rows[:, valid_on_device, :].sum(dim=1).cpu()
                    )
                    for name, sel in zip(names, selections[t_prime], strict=True):
                        if sel.numel():
                            out[layer][name][start:stop] += (
                                rows[:, sel, :].sum(dim=1).cpu()
                            )
                del grads
        # Counted once, not once per cotangent batch or layer.
        for t_prime in targets.tolist():
            for name, sel in zip(names, selections[t_prime], strict=True):
                pair_counts[name] += int(sel.numel())

    for layer in source_layers:
        for name in [*names, "total"]:
            out[layer][name] /= n_valid
    return out, pair_counts


def component_pullbacks_for_prompt(
    model,
    prompt: str,
    source_layers: Sequence[int],
    cotangents: torch.Tensor,
    *,
    target_layer: int | None = None,
    dim_batch: int = 16,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    t_stride: int = 1,
) -> dict[int, dict[str, torch.Tensor]]:
    """Split ``J_x^T c`` into Yan et al.'s two components (Eq. 20).

    The fitting estimator averages over source positions the sum over target
    positions, ``mean_t sum_t' d h_final,t' / d h_l,t``. Eq. 20 splits that sum
    by where ``t'`` sits relative to ``t``:

    * ``diag`` -- the ``t' = t`` term alone. The output at a position given its
      own activation: short-horizon, next-token prediction.
    * ``off`` -- everything with ``t' > t``. Influence on what gets emitted
      *later*, which is where a concept broadcast or a re-mention would live.

    Their Table 1 filters pairs this way and scores READOUT. This returns the
    two components as write directions, so the same split can be steered with,
    which is the arm their paper does not have.

    One backward pass per target position gives that position's whole column
    over ``t``, so the cost is ``n_valid`` passes per cotangent batch (~111 on
    a 128-token window) rather than one -- still far below the ``d_model``
    passes a full Jacobian needs.

    Returns:
        ``{layer: {"diag": [K, d], "off": [K, d], "total": [K, d]}}``, where
        ``total`` reproduces :func:`pullback_for_prompt` and equals
        ``diag + off`` up to floating point.
    """
    if cotangents.ndim != 2 or cotangents.shape[1] != model.d_model:
        raise ValueError(
            f"cotangents must be [K, d_model={model.d_model}], "
            f"got {tuple(cotangents.shape)}"
        )
    source_layers, target_layer = _check_layer_indices(
        source_layers, target_layer, model.n_layers
    )
    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    valid = resolve_positions(seq_len, None, skip_first=skip_first)
    # Subsampling target positions costs one backward pass each and leaves both
    # components unbiased estimates of their own means. It matters because the
    # pass count is (n_targets x cotangent batches), and a 70-word target list
    # makes the full 111 targets hours rather than minutes. Safe here because
    # ``swap_edit`` unit-normalises the read directions, so the overall scale
    # these share is never used.
    targets = valid[::t_stride]
    n_valid = len(targets)
    n_cotangents = cotangents.shape[0]
    batch = min(dim_batch, n_cotangents)

    out = {
        layer: {
            part: torch.zeros(n_cotangents, model.d_model, dtype=torch.float32)
            for part in ("diag", "total")
        }
        for layer in source_layers
    }

    with (
        ActivationRecorder(
            model.layers,
            at=[*source_layers, target_layer],
            start_graph_at=min(source_layers),
        ) as recorder,
        torch.enable_grad(),
    ):
        model.forward(input_ids.expand(batch, -1))
        target_activation = recorder.activations[target_layer]
        source_activations = [recorder.activations[layer] for layer in source_layers]
        device = target_activation.device
        valid_on_device = valid.to(device)
        cotangents_dev = cotangents.to(device=device, dtype=target_activation.dtype)
        buffer = torch.zeros_like(target_activation)

        n_batches = math.ceil(n_cotangents / batch)
        for pass_idx in range(n_batches):
            start = pass_idx * batch
            stop = min(start + batch, n_cotangents)
            n_this = stop - start
            for j, t_prime in enumerate(targets.tolist()):
                buffer.zero_()
                buffer[:n_this, t_prime, :] = cotangents_dev[start:stop, None, :][:, 0]
                last = pass_idx == n_batches - 1 and j == n_valid - 1
                grads = torch.autograd.grad(
                    outputs=target_activation,
                    inputs=source_activations,
                    grad_outputs=buffer,
                    retain_graph=not last,
                )
                for layer, grad in zip(source_layers, grads, strict=True):
                    rows = grad[:n_this].float()
                    # Causality zeroes t > t', so summing the valid set is the
                    # same as summing t <= t'.
                    out[layer]["total"][start:stop] += (
                        rows[:, valid_on_device, :].sum(dim=1).cpu()
                    )
                    out[layer]["diag"][start:stop] += rows[:, t_prime, :].cpu()
                del grads

    for layer in source_layers:
        for part in ("diag", "total"):
            out[layer][part] /= n_valid
        out[layer]["off"] = out[layer]["total"] - out[layer]["diag"]
    return out

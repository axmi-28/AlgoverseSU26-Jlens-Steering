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

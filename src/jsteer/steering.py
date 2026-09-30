"""The causal arm: writing a direction into the residual stream and grading it.

Everything else in this package *reads* the Jacobian. This module is the only
place that writes, and it exists so "how does this vary with causal effects of
steering?" has an answer computed under the same conventions as the geometry.

Two interventions, and the difference matters
---------------------------------------------
Upstream's headline flexible-generalization number (76/192 at ordinary
strength, 101/192 at double) comes from a **swap**, not from additive steering.
The convention section of ``data/experiments/README.md`` defines it as
"clamping a lens coordinate replaces one token's direction with another's at
every band layer at the specified positions". That is a rank-2 edit: it removes
one direction *and* adds another, so it exercises the Jacobian in both its read
role (which component to cancel) and its write role (what to inject).

This project is about whether a read direction is a valid *write* direction, so
a rank-2 operator confounds exactly the two things we want separated. The
design here is therefore:

- :func:`swap_edit` reproduces the paper's operator, used only as a
  **validation gate** -- if we cannot recover ~76/192 we have the wrong recipe
  and no downstream causal number is trustworthy.
- :func:`additive_edit` is the **dependent variable**: pure rank-1 addition
  under the convention upstream states for verbal-introspection ("the
  unit-normalized transpose row for that token, scaled by the layer's mean
  residual norm times a strength scalar"), added at every band layer at every
  prompt position.

The gap between them is itself a finding: a trial where the swap succeeds and
the addition fails isolates the contribution of *removal*.

Inferred, not inherited
-----------------------
Upstream ships prompt sets, not intervention code -- ``jlens`` packages only
``jlens/*``, and nothing in it mentions swapping. Both operators below are
reconstructed from the README's prose. :func:`swap_edit` accordingly takes a
``mode``: "replace" reads "replaces one token's direction with another's"
literally (project out the source coordinate, re-inject its magnitude along the
target), while "exchange" swaps the two coordinates symmetrically. Which one
reproduces 76/192 is an empirical question, which is what the gate is for.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from jlens.hooks import ActivationRecorder

logger = logging.getLogger(__name__)


@dataclass
class ResidualEdit:
    """An additive edit to apply to the residual stream during a forward pass.

    Attributes:
        vectors: ``{layer: Tensor}`` added to that block's output. Either
            ``[d_model]`` (one vector broadcast over every selected position,
            the additive case) or ``[n_positions, d_model]`` (a per-position
            delta, which is what a swap produces).
        positions: Token positions to add at; ``None`` means every position,
            which is the paper's convention ("applied at every prompt
            position").
    """

    vectors: dict[int, torch.Tensor]
    positions: Sequence[int] | None = None

    def scaled(self, factor: float) -> ResidualEdit:
        return ResidualEdit(
            {layer: v * factor for layer, v in self.vectors.items()}, self.positions
        )


@contextmanager
def steered(model, edit: ResidualEdit | None):
    """Register forward hooks that add ``edit`` to each layer's output.

    Edits compose down the stack by construction: adding at layer ``l`` changes
    the input to ``l+1``, and the band edit is applied at every band layer, so
    the model sees the perturbation propagate rather than a single splice.
    """
    if edit is None or not edit.vectors:
        yield
        return

    handles = []

    def make_hook(vector: torch.Tensor):
        def hook(module, inputs, output):
            tensor = output if torch.is_tensor(output) else output[0]
            delta = vector.to(device=tensor.device, dtype=tensor.dtype)
            if edit.positions is None:
                edited = tensor + delta
            else:
                edited = tensor.clone()
                index = torch.tensor(
                    [p % tensor.shape[1] for p in edit.positions],
                    device=tensor.device,
                    dtype=torch.long,
                )
                edited[:, index, :] = edited[:, index, :] + delta
            if torch.is_tensor(output):
                return edited
            return (edited, *output[1:])

        return hook

    try:
        for layer, vector in edit.vectors.items():
            handles.append(model.layers[layer].register_forward_hook(make_hook(vector)))
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def next_token_logits(
    model,
    prompt: str,
    *,
    edit: ResidualEdit | None = None,
    position: int = -1,
    max_seq_len: int = 128,
) -> torch.Tensor:
    """Logits at ``position`` under ``edit``, shape ``[vocab]``, fp32 on CPU.

    Reads the final *block* output and calls ``model.unembed`` rather than
    taking the text module's ``last_hidden_state``, which is already
    normalized; this is the path ``JacobianLens.apply`` uses, so the steered
    and lens readouts stay comparable.
    """
    input_ids = model.encode(prompt, max_length=max_seq_len)
    final_layer = model.n_layers - 1
    # Registration order is execution order, and a hook that returns a value
    # replaces the output for every hook after it. So the steerer must be
    # registered FIRST or the recorder reads the residual from before the edit
    # -- which is invisible here (the band sits below the final layer) right up
    # until someone steers the layer being read.
    with (
        steered(model, edit),
        ActivationRecorder(model.layers, at=[final_layer]) as recorder,
    ):
        model.forward(input_ids)
        residual = recorder.activations[final_layer][0, position].detach()
    return model.unembed(residual).float().cpu()


#: Leading characters stripped before matching a generated continuation.
#: The model answers "Two times five equals" with " 10" and the first-letter
#: template with " 't'", so the informative characters sit behind a space and
#: sometimes a quote.
_LEAD = " \t\n'\"\u2018\u2019\u201c\u201d:=.,-*"


def answer_matches(text: str, answer: str, *, digit: str | None = None) -> bool:
    """Does a generated continuation begin with ``answer`` (or its digit form)?

    First-token grading cannot see these answers at all. Qwen3-8B emits a bare
    space as its greedy *first* token for 11 of the 16 numbers prompts -- the
    digit follows in the second -- so every first-token rule scores them as
    misses, and any rule loose enough to accept the space scores everything as
    a hit. Reading a few generated tokens is the only formulation that
    distinguishes "the model said ten" from "the model said fourteen".
    """
    cleaned = text.lstrip(_LEAD).lower()
    forms = [answer.lower()] + ([digit] if digit else [])
    for form in forms:
        if not form or not cleaned.startswith(form):
            continue
        # The match must end at a boundary. Without this, " 90," scores as
        # "nine" (because "90" starts with "9") and " ninety" scores as "nine"
        # -- both seen in real output, and both wrong in the direction that
        # manufactures hits.
        rest = cleaned[len(form) :]
        if not rest or not rest[0].isalnum():
            return True
    return False


@torch.no_grad()
def greedy_continuation(
    model,
    prompt: str,
    *,
    edit: ResidualEdit | None = None,
    n_tokens: int = 4,
    max_seq_len: int = 128,
) -> str:
    """Greedily decode ``n_tokens`` under ``edit`` and return the decoded text.

    The edit is pinned to the *prompt* positions. A swap's delta has one row
    per prompt position, so letting it default to "every position" would
    broadcast-fail as soon as a token is appended -- and semantically the
    intervention is on the prompt: we perturb what the model is thinking about
    and then read what it goes on to say, without continuing to steer its own
    output.
    """
    input_ids = model.encode(prompt, max_length=max_seq_len)
    if edit is not None and edit.positions is None:
        edit = ResidualEdit(edit.vectors, positions=list(range(input_ids.shape[1])))
    final_layer = model.n_layers - 1

    produced: list[int] = []
    for _ in range(n_tokens):
        with (
            steered(model, edit),
            ActivationRecorder(model.layers, at=[final_layer]) as recorder,
        ):
            model.forward(input_ids)
            residual = recorder.activations[final_layer][0, -1].detach()
        token = int(model.unembed(residual).float().argmax().item())
        produced.append(token)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[token]], device=input_ids.device)], dim=1
        )
    return model.tokenizer.decode(produced)


@torch.no_grad()
def mean_residual_norms(
    model, prompt: str, layers: Sequence[int], *, max_seq_len: int = 128
) -> dict[int, float]:
    """Per-layer mean ``||h||_2`` over the prompt's positions.

    This is the scale the paper's steering recipe multiplies by. Upstream says
    "the layer's mean residual norm" without saying over what; measuring it on
    the clean forward pass for the prompt being steered is the self-calibrating
    reading, and keeps the strength scalar comparable across prompts whose
    residual scales differ.
    """
    input_ids = model.encode(prompt, max_length=max_seq_len)
    with ActivationRecorder(model.layers, at=list(layers)) as recorder:
        model.forward(input_ids)
        return {
            layer: recorder.activations[layer][0].float().norm(dim=-1).mean().item()
            for layer in layers
        }


def _unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm().clamp_min(1e-12)


def additive_edit(
    directions: dict[int, torch.Tensor],
    residual_norms: dict[int, float],
    *,
    strength: float = 1.0,
    positions: Sequence[int] | None = None,
) -> ResidualEdit:
    """Rank-1 steering: unit direction x layer mean residual norm x strength.

    Args:
        directions: ``{layer: g}``. The pullback ``J^T u_y``, local or averaged
            -- which one is the independent variable of the whole experiment.
        residual_norms: From :func:`mean_residual_norms`.
        strength: The paper's strength scalar; 0 is the control.
        positions: ``None`` for every position (the paper's convention).
    """
    return ResidualEdit(
        {
            layer: _unit(g.float()) * (residual_norms[layer] * strength)
            for layer, g in directions.items()
        },
        positions,
    )


@torch.no_grad()
def swap_edit(
    model,
    prompt: str,
    source_directions: dict[int, torch.Tensor],
    target_directions: dict[int, torch.Tensor],
    *,
    mode: str = "replace",
    strength: float = 1.0,
    positions: Sequence[int] | None = None,
    max_seq_len: int = 128,
) -> ResidualEdit:
    """The paper's rank-2 swap, reconstructed. Validation gate only.

        Reads the clean residual at each layer, measures the prompt's coordinate
        along the (unit) source and target read directions, and builds the additive
        delta that clamps them.

    ``mode="clamp"`` is the exact one: it solves for the minimum-norm ``delta``
        that makes the lens read the two tokens' coordinates swapped, via a 2x2
        solve against the Gram matrix of the two read directions. ``replace``
        (``h - a*g_src + a*g_tgt``) and ``exchange`` are the cheaper readings that
        treat the directions as orthonormal; they differ from ``clamp`` by exactly
        the off-diagonal Gram term, which on real pullbacks is not small.
        Because the coordinate is per-position, so is the delta -- unlike
        :func:`additive_edit`, this edit is prompt- and position-dependent by
        construction, which is one more reason it cannot serve as the clean test of
        a *fixed* write direction.
    """
    if mode not in ("replace", "exchange", "clamp"):
        raise ValueError(f"mode must be 'replace', 'exchange' or 'clamp', got {mode!r}")

    layers = sorted(source_directions)
    input_ids = model.encode(prompt, max_length=max_seq_len)
    with ActivationRecorder(model.layers, at=layers) as recorder:
        model.forward(input_ids)
        residuals = {l: recorder.activations[l][0].detach().float() for l in layers}

    seq_len = input_ids.shape[1]
    index = (
        list(range(seq_len)) if positions is None else [p % seq_len for p in positions]
    )

    vectors: dict[int, torch.Tensor] = {}
    for layer in layers:
        g_src_raw = source_directions[layer].float().cpu()
        g_tgt_raw = target_directions[layer].float().cpu()
        h = residuals[layer].cpu()[index]  # [n_pos, d_model]

        if mode == "clamp":
            # The paper's operator, verbatim (J-Lens paper, S2.5): form
            # V = [v_s v_t], read the lens coordinates c = V^dagger h, and set
            #     h_patched = h + V (sigma(c) - c)
            # where sigma swaps the two entries of c and alpha scales the
            # update. The component of h orthogonal to span{v_s, v_t} is
            # unchanged.
            #
            # The ORDER matters and is easy to get backwards. sigma acts on the
            # pseudoinverse coordinates c = (V^T V)^-1 V^T h, NOT on the raw
            # projections V^T h. Those differ: sigma and (V^T V)^-1 commute
            # only when ||v_s|| == ||v_t||, which pullbacks do not satisfy. An
            # earlier version here exchanged V^T h and then un-Grammed, which
            # is a subtly mis-rotated delta -- and doubling a mis-rotated delta
            # amplifies the error instead of completing the exchange, which is
            # the shape of an alpha=1 -> alpha=2 regression.
            #
            # Note the consequence: it is the *coordinates* that end up
            # exchanged, not the lens readouts <g_y, h>. After the edit
            # V^T h' = (V^T V) sigma(c), which equals sigma(V^T h) only in the
            # equal-norm case.
            V = torch.stack([g_src_raw, g_tgt_raw], dim=1)  # [d, 2]
            gram = V.T @ V  # [2, 2]
            c = torch.linalg.solve(
                gram + 1e-6 * torch.eye(2, dtype=gram.dtype), (h @ V).T
            ).T  # [n_pos, 2], the lens coordinates V^dagger h
            vectors[layer] = strength * ((c.flip(-1) - c) @ V.T)  # [n_pos, d]
            continue

        g_src = _unit(g_src_raw)
        g_tgt = _unit(g_tgt_raw)
        a = h @ g_src  # [n_pos]
        if mode == "replace":
            delta = strength * (a[:, None] * (g_tgt - g_src)[None, :])
        else:
            b = h @ g_tgt
            delta = strength * (
                (b - a)[:, None] * g_src[None, :] + (a - b)[:, None] * g_tgt[None, :]
            )
        # A per-position delta: ResidualEdit broadcasts one vector over the
        # selected positions, so carry the full [n_pos, d_model] block instead.
        vectors[layer] = delta
    return ResidualEdit(vectors, index)


@dataclass
class TrialOutcome:
    """What one steered forward pass produced.

    Three levels of evidence, deliberately kept separate:

    ``hit`` is upstream's criterion -- greedy next token equals the target
    answer. ``hit_excluding`` is the same after masking the injected token
    itself out of the argmax. That mask is not a convenience: a first
    end-to-end run on Qwen3-1.7B (``notes/findings.md``) had additive steering
    toward ``Canada`` turn the greedy token into ``" Canada"`` rather than
    ``" Ottawa"`` -- the injected word echoed straight through -- while
    ``Ottawa``'s rank still improved from 1659 to 13. Scoring only ``hit``
    would have recorded that as a total failure and thrown away a large real
    effect; scoring only ``hit_excluding`` would quietly change upstream's
    criterion. Both are stored.

    ``margin`` (target answer logit minus source answer logit) and
    ``target_rank`` are the graded signals. They still move when the greedy
    token does not, which is what makes a per-prompt steerability score
    possible instead of a 0/1 label on 192 trials.
    """

    greedy_token: str
    greedy_id: int
    hit: bool
    margin: float
    target_logit: float
    source_logit: float
    target_rank: int
    hit_excluding: bool = False
    greedy_excluding: str = ""
    #: KL(steered || clean) over the full next-token distribution, in nats.
    #:
    #: This is what separates the paper's own explanation of its failures from
    #: the alternative it never rules out. A.13 attributes misses to the edit
    #: moving "in the right direction but not far enough" -- but a *large* edit
    #: in the wrong direction also improves the target's rank while scrambling
    #: everything else, and rank alone cannot tell the two apart. KL measures
    #: total disruption, so:
    #:   low KL  + rank improves -> right direction, undershooting
    #:   high KL + rank improves -> brute force, the distribution is wrecked
    #: The paper reports no KL at either strength, so this is additional to it.
    kl_from_clean: float = 0.0
    #: KL(clean || steered) over the *concept-merged* distribution, in nats.
    #:
    #: ``kl_from_clean`` has a defect as a control variable: it counts the
    #: intended change as damage. Steering that cleanly moves probability from
    #: the source answer to the target answer and touches nothing else still
    #: scores a large KL, so matching two arms on it partly matches them on the
    #: very effect being compared -- and it penalises whichever arm produces
    #: more on-target movement per unit strength, which is exactly the local
    #: direction. Every matched-KL comparison in this project inherits that
    #: bias.
    #:
    #: This is FishBack's fix (arXiv 2605.17231 S5.3, crediting Park et al.'s
    #: concept-decomposed evaluation), verbatim: the vocabulary is partitioned
    #: into concept-relevant groups plus the rest, and the off-target
    #: distribution merges each group into one category,
    #:     P^Z(z_i) = P(y_i^0) + P(y_i^1),  P^Z(y) = P(y) for y in Y_rest
    #: so that "a method that cleanly shifts P^W without affecting P^Z achieves
    #: D_KL^off = 0". Here the single group is {target answer variants} union
    #: {source answer variants}: swapping probability between those two is the
    #: intended edit and is invisible to this metric.
    #:
    #: Two conventions worth stating because they are not interchangeable. The
    #: direction is KL(original || steered), following FishBack, and is the
    #: opposite of ``kl_from_clean``'s KL(steered || clean). And the injected
    #: argument's own token stays in Y_rest, so the model echoing " Canada"
    #: instead of answering "Ottawa" counts as collateral -- which is right,
    #: since that echo is not the intended output.
    off_target_kl: float = 0.0
    #: Steered probability mass on the concept group over the clean mass on it.
    #:
    #: FishBack's secondary metric (counterfactual mass preservation). ~1 means
    #: the edit moved probability *within* the concept pair; << 1 means it
    #: drained the pair and leaked the mass to unrelated tokens, which is a
    #: failure mode that leaves ``margin`` looking healthy.
    concept_mass_ratio: float = 1.0
    extras: dict = field(default_factory=dict)


def grade(
    model,
    logits: torch.Tensor,
    *,
    target_id: int | Sequence[int],
    source_id: int | Sequence[int] | None = None,
    exclude_ids: Sequence[int] = (),
    clean_logits: torch.Tensor | None = None,
    concept_ids: Sequence[int] | None = None,
) -> TrialOutcome:
    """Score a next-token distribution against the target answer token.

    Args:
        target_id: The answer for the *injected* argument -- the token upstream
            grades against. A sequence accepts any surface form of the same
            word (see :func:`jsteer.loading.answer_variant_ids`); the best of
            them sets the logit, rank and margin.
        source_id: The answer the unsteered prompt gives, for the margin.
        exclude_ids: Masked out of the argmax for ``hit_excluding``; pass the
            injected argument's own token id.
        clean_logits: The unsteered logits for the same prompt. Supply them to
            get ``kl_from_clean``; without them it stays 0.
        concept_ids: The concept-relevant token group -- normally every target
            and source answer variant. Supply it (with ``clean_logits``) to get
            ``off_target_kl`` and ``concept_mass_ratio``; without it they stay
            at their neutral defaults. Defaults to ``targets + sources`` when
            not given, which is the right group for every caller in this
            project; pass it explicitly only to widen or narrow the group.
    """
    targets = [target_id] if isinstance(target_id, int) else list(target_id)
    sources = (
        []
        if source_id is None
        else ([source_id] if isinstance(source_id, int) else list(source_id))
    )
    greedy_id = int(logits.argmax().item())
    target_logit = float(max(logits[i].item() for i in targets))
    source_logit = float(max(logits[i].item() for i in sources)) if sources else 0.0
    rank = int((logits > target_logit).sum().item())

    masked = logits.clone()
    for token_id in exclude_ids:
        masked[token_id] = float("-inf")
    greedy_excluding_id = int(masked.argmax().item())

    kl = 0.0
    off_target = 0.0
    mass_ratio = 1.0
    if clean_logits is not None:
        steered_log = torch.log_softmax(logits.float(), dim=-1)
        clean_log = torch.log_softmax(clean_logits.float(), dim=-1)
        kl = float((steered_log.exp() * (steered_log - clean_log)).sum().item())
        group = sorted(
            set(concept_ids if concept_ids is not None else targets + sources)
        )
        if group:
            off_target, mass_ratio = _off_target(clean_log, steered_log, group)

    return TrialOutcome(
        greedy_token=model.tokenizer.decode([greedy_id]),
        greedy_id=greedy_id,
        hit=greedy_id in targets,
        margin=target_logit - source_logit,
        target_logit=target_logit,
        source_logit=source_logit,
        target_rank=rank,
        hit_excluding=greedy_excluding_id in targets,
        greedy_excluding=model.tokenizer.decode([greedy_excluding_id]),
        kl_from_clean=kl,
        off_target_kl=off_target,
        concept_mass_ratio=mass_ratio,
    )


def _off_target(
    clean_log: torch.Tensor, steered_log: torch.Tensor, group: Sequence[int]
) -> tuple[float, float]:
    """FishBack's D_KL^off and counterfactual mass preservation.

    Collapses ``group`` into a single category in both distributions and takes
    KL(clean || steered) over the result, so probability moved *within* the
    group is invisible. Returns (off-target KL in nats, steered group mass over
    clean group mass).
    """
    clean = clean_log.exp()
    steered = steered_log.exp()
    mask = torch.zeros_like(clean, dtype=torch.bool)
    mask[torch.tensor(list(group), dtype=torch.long)] = True

    clean_group = clean[mask].sum()
    steered_group = steered[mask].sum()
    p = torch.cat([clean[~mask], clean_group.reshape(1)])
    q = torch.cat([steered[~mask], steered_group.reshape(1)])
    # Renormalize: softmax is already normalized, but the merge is exact only
    # up to fp error, and a q entry of exactly 0 would give an infinite KL for
    # a token the model simply never predicts.
    p = p / p.sum()
    q = (q / q.sum()).clamp_min(1e-30)
    off = float((p * (p.log() - q.log())).sum().item())
    return off, float((steered_group / clean_group.clamp_min(1e-30)).item())


@torch.no_grad()
def ablate_edit(
    model,
    prompt: str,
    directions: dict[int, torch.Tensor],
    *,
    strength: float = 1.0,
    positions: Sequence[int] | None = None,
    max_seq_len: int = 128,
) -> tuple[ResidualEdit, dict[str, float]]:
    """Project a direction OUT of the residual stream -- the paper's causal measure.

    The workspace paper's App. A.6 lens comparison and its App. A.7 recipe
    ablation both score a direction by *removing* it and measuring the induced
    output KL: a direction the model is actually using should, when deleted,
    change what the model says. Higher KL = more causally important. That is
    the **opposite** sign convention to the one this repo uses elsewhere, where
    KL is a collateral-damage cost. Keep the two apart.

    The edit is ``h -> h - <h, g_hat> g_hat`` at every selected position of
    every named layer, expressed as the equivalent additive per-position delta
    so it rides the existing :class:`ResidualEdit` machinery. ``strength=1``
    removes the component exactly; the parameter exists so a partial ablation
    can be dosed like any other arm.

    Like :func:`swap_edit`, the coefficients are read off the CLEAN forward
    pass and then applied together, rather than recomputed layer by layer as
    the perturbation propagates. That is the same convention the swap arms use,
    which is the point -- the two measures have to see the same operator for
    their disagreement to mean anything.

    Returns the edit and a small diagnostic dict, because "ablation changed
    nothing" is ambiguous between "the direction is not used" and "there was
    nothing there to remove": ``frac_norm`` is the mean fraction of the
    residual's norm that the projection took out, and ``abs_cos`` the mean
    |cos| between the residual and the direction.
    """
    layers = sorted(directions)
    input_ids = model.encode(prompt, max_length=max_seq_len)
    with ActivationRecorder(model.layers, at=layers) as recorder:
        model.forward(input_ids)
        residuals = {l: recorder.activations[l][0].detach().float() for l in layers}

    seq_len = input_ids.shape[1]
    index = (
        list(range(seq_len)) if positions is None else [p % seq_len for p in positions]
    )

    vectors: dict[int, torch.Tensor] = {}
    fracs: list[float] = []
    coss: list[float] = []
    for layer in layers:
        g = _unit(directions[layer].float().cpu())
        h = residuals[layer].cpu()[index]  # [n_pos, d_model]
        coeff = h @ g  # [n_pos]
        removed = coeff[:, None] * g[None, :]
        vectors[layer] = -strength * removed
        hn = h.norm(dim=-1).clamp_min(1e-12)
        fracs.append(float((removed.norm(dim=-1) / hn).mean()))
        coss.append(float((coeff.abs() / hn).mean()))
    return ResidualEdit(vectors, index), {
        "frac_norm": float(sum(fracs) / len(fracs)),
        "abs_cos": float(sum(coss) / len(coss)),
    }

"""Plumbing shared by the run scripts: prompt sets, targets, positions.

Kept in one place because the three sweeps have to agree on it exactly. A
``g_x`` from the research-question-2 sweep is only comparable with a ``J_x``
from the research-question-1 sweep if both used the same position convention
and the same truncation, and a causal outcome is only attributable to a
direction if the direction was written where the Jacobian was measured.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from jsteer.config import ModelConfig
from jsteer.corpus import load_fitting_prompts
from jsteer.data import BasePrompt, all_args, flexible_generalization_prompts
from jsteer.domains import DomainSpec, load_domain_corpora
from jsteer.loading import single_token_id, unembedding_rows

logger = logging.getLogger(__name__)

POSITION_MODES = ("all", "last", "arg", "fit")


@dataclass(frozen=True)
class EvalPrompt:
    """A prompt in a sweep, tagged with the group it belongs to.

    ``group`` is ``"eval"`` or ``"fit"`` in the two original sweeps, or a
    corpus name in the domain sweep. The distinction carries the whole
    calibration argument: on ``fit`` prompts the lens's ``J_bar`` is the cloud's
    center by construction, so whatever displacement is measured there is the
    finite-sample floor for the same measurement on ``eval``.

    ``fit_positions`` says to defer to the lens's own position convention
    (``skip_first``, drop the final position) instead of an explicit position
    list. It is a field rather than a test on ``group`` because the domain
    sweep has four corpus groups that all need the fitting convention and none
    of which is called ``"fit"`` -- and inferring it from the group name would
    silently hand a corpus prompt the eval convention, changing ``J_x`` without
    changing anything visible in the output.
    """

    group: str
    key: str
    text: str
    base: BasePrompt | None = None
    fit_positions: bool = False


def build_prompt_set(
    config: ModelConfig,
    *,
    n_fit: int,
    tokenizer=None,
    cache_path=None,
) -> list[EvalPrompt]:
    """The 64 flexible-generalization prompts plus ``n_fit`` wikitext controls."""
    prompts = [
        EvalPrompt(group="eval", key=base.key, text=base.prompt, base=base)
        for base in flexible_generalization_prompts()
    ]
    if n_fit:
        fitting = load_fitting_prompts(
            config.fit, n_fit, tokenizer=tokenizer, cache_path=cache_path
        )
        prompts += [
            EvalPrompt(group="fit", key=f"wikitext/{i}", text=text, fit_positions=True)
            for i, text in enumerate(fitting)
        ]
    return prompts


def build_domain_prompt_set(
    specs: Sequence[DomainSpec],
    *,
    n_per: int,
    tokenizer,
    max_chars: int = 2000,
    min_tokens: int = 128,
    cache_path=None,
) -> list[EvalPrompt]:
    """The domain panel: ``n_per`` prompts from each corpus, one flat list.

    Every prompt carries ``fit_positions=True``. The panel exists to be
    compared against the published ``J_bar``, and ``J_bar`` was estimated under
    the lens's own convention -- averaging over ``[skip_first, seq_len - 1)``.
    Measuring a domain average under any other convention would make the
    difference from ``J_bar`` a sum of a domain effect and a convention effect,
    with no way to separate them after the fact.

    Groups are laid out in contiguous blocks, one corpus after another, which
    is what makes a *stride* shard balanced. The reasoning is not obvious and
    it is worth writing down, because the intuitive layout is the broken one:
    with blocks, shard ``s`` of ``S`` takes indices ``s, s+S, s+2S, ...``, and
    since each block is ``n_per`` long it lands in every block equally. Under
    the interleaved layout the group of index ``i`` is ``i % n_groups``, so
    whenever ``S`` is a multiple of ``n_groups`` -- 8 shards over 4 corpora,
    the configuration actually used -- every index in a shard has the same
    residue and each shard becomes a single corpus. A shard is the unit of the
    running-mean partial sum, so that failure mode puts an entire domain's
    average in one container: lose it and the group is gone, and the pooled
    result still looks like a normal average over fewer prompts.
    """
    corpora = load_domain_corpora(
        list(specs),
        tokenizer=tokenizer,
        n_per=n_per,
        max_chars=max_chars,
        min_tokens=min_tokens,
        cache_path=cache_path,
    )
    return [
        EvalPrompt(
            group=spec.name,
            key=f"{spec.name}/{i}",
            text=corpora[spec.name][i],
            fit_positions=True,
        )
        for spec in specs
        for i in range(n_per)
    ]


@dataclass(frozen=True)
class TargetTokens:
    """The cotangents ``u_y``, and the labels they belong to.

    These are the 16 **argument** words, not the answer words. Upstream injects
    the arg ("Canada") and grades the answer ("Ottawa"), so the Jacobian is only
    ever asked about args; answers exist for the grader and never become a
    cotangent. That also keeps the budget small enough to pull every target back
    through every prompt, which is what makes relevance a measured variable
    rather than a design choice: "Canada" is the arg of one prompt and foreign
    to another, and both cells are in the same fully-crossed table.
    """

    words: list[str]
    token_ids: list[int]
    cotangents: torch.Tensor  # [K, d_model]

    def index_of(self, word: str) -> int:
        return self.words.index(word)


def resolve_targets(model, words: Sequence[str] | None = None) -> TargetTokens:
    """Unembedding rows for the target words, dropping any that are multi-token.

    Raises if fewer than two survive: a sweep over one target cannot say
    anything about the distribution across targets.
    """
    words = list(words) if words is not None else all_args()
    kept_words, kept_ids = [], []
    for word in words:
        try:
            kept_ids.append(single_token_id(model, word))
            kept_words.append(word)
        except ValueError as exc:
            logger.warning("dropping target %r: %s", word, exc)
    if len(kept_words) < 2:
        raise ValueError(
            f"only {len(kept_words)} single-token targets out of {len(words)}"
        )
    return TargetTokens(kept_words, kept_ids, unembedding_rows(model, kept_ids))


def positions_for(
    mode: str,
    model,
    prompt: EvalPrompt,
    *,
    max_seq_len: int,
) -> list[int] | None:
    """Resolve a position convention to an explicit index list.

    ``fit`` returns ``None``, deferring to the lens's own convention
    (``skip_first=16``, drop the final position) -- correct for the wikitext
    controls and impossible for the eval prompts, which are ~8 tokens long.
    ``all`` is the convention the steering interventions use ("applied at every
    prompt position"), and is the default for that reason. ``last`` is the
    single position the next-token prediction is read at. ``arg`` is the span
    the swapped word occupies.

    These give genuinely different ``J_x``, and which one best predicts steering
    success is part of research question 1 rather than a detail to settle by
    fiat -- the fitting estimator *averages* over source positions while the
    intervention *writes* at particular ones, so "the prompt-local Jacobian" is
    not well-defined until a position set is named.
    """
    if mode == "fit":
        return None
    seq_len = model.encode(prompt.text, max_length=max_seq_len).shape[1]
    if mode == "all":
        return list(range(seq_len))
    if mode == "last":
        return [seq_len - 1]
    if mode == "arg":
        span = arg_span(model, prompt, max_seq_len=max_seq_len)
        if span is None:
            logger.warning("no arg span for %s; falling back to all", prompt.key)
            return list(range(seq_len))
        return list(range(*span))
    raise ValueError(f"unknown position mode {mode!r}; expected {POSITION_MODES}")


def arg_span(model, prompt: EvalPrompt, *, max_seq_len: int) -> tuple[int, int] | None:
    """``[start, stop)`` token positions covering the argument word, or ``None``.

    Located by character offsets rather than by decoding, because the argument
    is not always its own token span. ``"A group of {arg}s is called a"`` puts
    the plural ``s`` inside the same token as the noun, so ``lion`` tokenizes
    into a piece that decodes to ``lions`` -- an exact round-trip check rejects
    all four ``animals/group`` prompts, and a prefix-length subtraction is
    wrong wherever the tokenizer merges across the template boundary.

    Taking every token whose character span *overlaps* the argument's is the
    only rule that is right in both cases. It over-covers by design: a token
    carrying both ``lion`` and its plural marker is included, since the
    Jacobian at that position is where the argument lives regardless of what
    else shares the token.
    """
    if prompt.base is None:
        return None
    text = prompt.text
    char_start = text.find(prompt.base.arg)
    if char_start < 0:
        return None
    char_stop = char_start + len(prompt.base.arg)

    offsets = _offsets(model, text, max_seq_len)
    if offsets is not None:
        covering = [
            i
            for i, (begin, end) in enumerate(offsets)
            if end > char_start and begin < char_stop
        ]
        if covering:
            return (covering[0], covering[-1] + 1)

    # Slow tokenizer, or no offsets: rebuild the mapping by decoding prefixes.
    input_ids = model.encode(text, max_length=max_seq_len)[0].tolist()
    cursor, span = 0, []
    for i in range(len(input_ids)):
        piece = model.tokenizer.decode(input_ids[i : i + 1])
        begin, end = cursor, cursor + len(piece)
        if end > char_start and begin < char_stop:
            span.append(i)
        cursor = end
    return (span[0], span[-1] + 1) if span else None


def _offsets(model, text: str, max_seq_len: int) -> list[tuple[int, int]] | None:
    """Per-token character spans from a fast tokenizer, or ``None``.

    The offsets have to line up with what :meth:`encode` produced, so this
    re-encodes under the same truncation and drops any special tokens the
    tokenizer prepends (offsets for those are ``(0, 0)``).
    """
    tokenizer = model.tokenizer
    if not getattr(tokenizer, "is_fast", False):
        return None
    try:
        encoded = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors=None,
        )
    except (TypeError, NotImplementedError):
        return None
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    n_expected = model.encode(text, max_length=max_seq_len).shape[1]
    if len(offsets) == n_expected:
        return offsets
    # encode() adds a BOS the offset call did not; pad the front to realign.
    if len(offsets) == n_expected - 1:
        return [(0, 0), *offsets]
    logger.warning(
        "offset mapping is %d tokens, encode gave %d", len(offsets), n_expected
    )
    return None

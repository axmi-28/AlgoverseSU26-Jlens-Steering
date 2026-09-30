"""The three GPU sweeps, as importable functions.

Everything here was originally inline in ``scripts/04``, ``05`` and ``06``. It
moved so that the Modal entrypoints and the command line call the *same* code:
the sweep that costs money is the one that cannot be run locally, so a second
implementation of it is a second implementation that nobody checks.

Three properties every sweep has, all of them there because of how Modal runs:

**Sharded.** A sweep takes ``shard`` / ``n_shards`` and processes
``prompts[shard::n_shards]``. The stride is deliberate. The prompt set mixes
8-token eval prompts with 128-token wikitext controls, and the backward pass
scales with sequence length, so a contiguous slice would hand one container
every expensive prompt and leave it running ~16x longer than its siblings --
and a fan-out is only as fast as its slowest shard.

**Idempotent.** Per-prompt outputs are skipped if already present. A container
that times out or is preempted resumes instead of restarting, which is what
makes ``retries`` on the Modal function a real safety net rather than a way to
pay twice.

**Shard-local in its writes.** No sweep writes a file another shard also writes;
merging is a separate, cheap, local step (``scripts/07_merge_shards.py``).
Concurrent writers to one path on a shared volume is the classic way to end a
long run with a corrupt file and no way to tell.
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import random
import re
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from jlens.hooks import ActivationRecorder

from jsteer.config import REPO_ROOT, ModelConfig, load_config
from jsteer.data import (
    all_args,
    category_args,
    flexible_generalization_prompts,
    flexible_generalization_trials,
    heldout_trials,
    probe_swap_items,
)
from jsteer.jacobian import averaged_pullback, jacobian_for_prompt, pullback_for_prompt
from jsteer.loading import (
    answer_variant_ids,
    first_token_id,
    load_lens,
    load_model,
    single_token_id,
    unembedding_matrix,
    unembedding_rows,
)
from jsteer.metrics import pullback_summary
from jsteer.reduce import RunningMean, digest_bytes, digest_jacobian
from jsteer.run import (
    EvalPrompt,
    arg_span,
    build_domain_prompt_set,
    build_prompt_set,
    positions_for,
    resolve_targets,
)
from jsteer.steering import (
    ResidualEdit,
    _unit,
    ablate_edit,
    additive_edit,
    answer_matches,
    grade,
    greedy_continuation,
    mean_residual_norms,
    next_token_logits,
    steered,
    swap_edit,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# specs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepSpec:
    """What every sweep needs. Plain data, so Modal can ship it to a container.

    Attributes:
        config_name: A name under ``configs/``.
        positions: Position convention -- see :func:`jsteer.run.positions_for`.
            Which one is right is an open research question, not a default.
        layers: Band layers to run; ``None`` means the config's band intersected
            with the lens's fitted layers.
        n_fit: Wikitext control prompts. These are the calibration arm: ``J_bar``
            is a mean over the fitting corpus, so on these prompts the cloud's
            center is ``J_bar`` by construction and any measured displacement is
            the finite-sample floor.
        shard / n_shards: This container's stride through the prompt list.
        limit: Truncate the prompt list before sharding (smoke tests).
        out_dir: Where results go; ``None`` means ``results/`` in the repo.
        corpus_cache: Where the materialized wikitext control prompts live.
            Every shard must read the *same* list -- it strides a list it
            builds itself, so if two shards ever disagreed about which document
            is wikitext prompt 5, their outputs would not be poolable. Pointing
            this at one file on a shared volume makes that structural rather
            than a property of the dataset stream staying deterministic, and
            saves re-streaming wikitext once per container.
        overwrite: Redo prompts whose output already exists.
    """

    config_name: str = "qwen3-8b"
    positions: str = "all"
    layers: list[int] | None = None
    n_fit: int = 32
    dim_batch: int | None = None
    shard: int = 0
    n_shards: int = 1
    limit: int | None = None
    out_dir: str | None = None
    corpus_cache: str | None = None
    overwrite: bool = False
    device: str | None = None

    def resolve_out(self, leaf: str) -> Path:
        root = Path(self.out_dir) if self.out_dir else REPO_ROOT / "results"
        path = root / leaf
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def namespace(self) -> str:
        """Everything about a run that changes its artifacts, minus the shard.

        Split out of :attr:`tag` so subclasses can widen it. Any input that
        changes what a cached file *contains* has to appear here, or a later
        run with different inputs reads the earlier one's artifacts, reports
        success, and produces numbers from the wrong data.
        """
        return f"{self.config_name}_{self.positions}"

    @property
    def tag(self) -> str:
        return f"{self.namespace}_shard{self.shard}of{self.n_shards}"


@dataclass(frozen=True)
class JacobianSweepSpec(SweepSpec):
    """Research question 1. The expensive one.

    Also the domain sweep, when ``domains`` is set: same machinery, a different
    prompt set. Instead of "eval prompts vs wikitext controls" the groups
    become corpora, and the per-group running mean -- which already existed to
    rebuild a group average exactly -- becomes ``J_bar_domain``.
    """

    #: Corpus names from :mod:`jsteer.domains`, or ``None`` for the original
    #: eval-vs-wikitext prompt set. Setting this replaces the prompt set
    #: entirely; ``n_fit`` is then unused.
    domains: list[str] | None = None
    #: Prompts drawn from each corpus. Every group gets the same count, so any
    #: between-group comparison is at matched N and the wikitext half-split is
    #: a null at exactly the N of the domains it calibrates.
    n_per: int = 16
    #: Minimum (and truncated) token length for every domain prompt.
    min_tokens: int = 128

    k: int = 64
    layer_chunk: int = 26
    lowrank: bool = False
    svd_device: str | None = None
    #: Cap on ``dim_batch * seq_len`` for a single backward pass.
    #:
    #: The retained graph scales with the product, not with ``dim_batch``
    #: alone, and the prompt set mixes 8-token eval prompts with 128-token
    #: wikitext controls. A fixed ``dim_batch=128`` is right for the former and
    #: 16x too large for the latter -- measured, it OOMs an 80 GB A100 at
    #: 79.05 GiB. Capping the product keeps one setting correct for both.
    #:
    #: Note this trades memory for passes, not for total work: the cost of a
    #: prompt is ``d_model * seq_len * n_layers`` however it is batched, so a
    #: 128-token control costs ~16x an 8-token eval prompt either way.
    max_batch_tokens: int = 2048

    @property
    def domain_specs(self) -> list:
        from jsteer.domains import DOMAINS_BY_NAME

        return [DOMAINS_BY_NAME[name] for name in self.domains or []]

    @property
    def namespace(self) -> str:
        """Domain runs get their own namespace, keyed on the whole panel.

        Without this a domain run writes ``group_means_qwen3-8b_all_shard0of8``
        -- the *same* path the eval-vs-wikitext sweep already wrote -- and
        silently overwrites it, or is pooled together with it. The corpus tag
        hashes every input that changes which documents are read, so changing
        a dataset or ``n_per`` moves the artifacts rather than reusing them.
        The position convention is dropped from the name because every domain
        prompt uses the lens's own convention by construction.
        """
        if not self.domains:
            return super().namespace
        from jsteer.domains import corpus_tag

        tag = corpus_tag(self.domain_specs, self.n_per, self.min_tokens)
        return f"{self.config_name}_{tag}"


@dataclass(frozen=True)
class PullbackSweepSpec(SweepSpec):
    """Research question 2. One backward pass per prompt."""

    n_fit: int = 100


@dataclass(frozen=True)
class DomainPullbackSpec(JacobianSweepSpec):
    """Per-corpus write directions without ever forming a Jacobian.

    The causal arm consumes exactly one field of a pooled digest -- the
    ``[n_args, d_model]`` block of target pullbacks -- and never the top-k
    bases or the scalars beside them. Those pullbacks are reachable far more
    cheaply than by averaging Jacobians, because averaging and pulling back
    commute::

        mean_x (J_x^T u)  ==  (mean_x J_x)^T u

    The left side is one backward pass per prompt (~4 s on an 8B at 128
    tokens); the right side is ``d_model`` of them (~53 s). For a study whose
    whole design is "fit the same pipeline on many corpora", that 13x is the
    difference between five replicates and one.

    What this deliberately gives up: the SVD bases, hence every geometric
    measure in C8 (subspace alignment, CKA, effective rank). Those need the
    matrix itself. Use :class:`JacobianSweepSpec` when the question is
    geometry, and this when the question is steering.
    """

    #: Distinguishes these digests from the full-Jacobian ones for the same
    #: panel. Without it a cheap run and an expensive run over identical
    #: corpora write one filename, and whichever ran last silently defines what
    #: the causal arm steers with -- the cache-key bug this repo keeps hitting.
    #: Cotangent words the pullbacks are taken against. ``None`` uses the
    #: upstream 16 args. A held-out eval has different arguments, and the
    #: stored pullbacks are indexed BY argument -- so a digest fitted against
    #: the wrong list cannot steer that eval, and would index the wrong row
    #: rather than fail. Hence it is in the namespace below.
    target_words: list[str] | None = None

    @property
    def namespace(self) -> str:
        base = f"{super().namespace}_pb"
        if not self.target_words:
            return base
        import hashlib

        tag = hashlib.sha256("|".join(self.target_words).encode()).hexdigest()[:8]
        return f"{base}_t{tag}"


@dataclass(frozen=True)
class AverageNSpec:
    """C4 - how many prompts does the average need before it steers?

    The falsification test for the denoising hypothesis. If J_bar's causal
    advantage over a single prompt's g_x is variance reduction, then averaging
    g_x over n prompts should climb smoothly from the n=1 result toward the
    full lens as n grows. If instead it stays flat and jumps, or plateaus well
    short of the lens, the averaged direction is not a denoised g_x -- it is a
    point in the space that no individual prompt resembles.

    Design notes that matter for reading the output:

    * **The pool is wikitext, not the eval set.** J_bar is a mean over the
      lens's *fitting* distribution, so only a wikitext average converges to
      it. An average over eval prompts converges somewhere else -- that
      displacement is the RQ1 bias result, and conflating the two would make
      the curve converge to the wrong asymptote.
    * **The swap operator is magnitude-invariant** (scaling V leaves
      V(sigma(c) - c) unchanged, since c = V^+ h scales inversely), so a single
      strength is fair across every n despite partial averages having very
      different norms. This is why C4 uses swap and not additive: it removes
      the dose confound by construction rather than by matching.
    * **Replicates.** For small n a single random subset is itself noisy, so
      each n is run over several disjoint subsets and reported as a spread.
    * **Two controls at n=1.** ``local_matched`` is the trial prompt's own g_x
      (the C1 arm). ``local_mismatched`` is a *different* eval prompt's g_x. If
      those two score the same, prompt-specificity was never doing any work and
      the whole effect is sample size.
    """

    config_name: str = "qwen3.6-27b"
    layers: list[int] | None = None
    #: Subset sizes to average over. 0 is reserved for the lens's own J_bar.
    ns: list[int] = field(default_factory=lambda: [1, 2, 5, 10, 25, 50, 100])
    #: Wikitext prompts to draw subsets from.
    n_pool: int = 100
    #: Independent disjoint subsets per n, capped by what the pool allows.
    replicates: int = 5
    strength: float = 0.5
    swap_mode: str = "clamp"
    seed: int = 0
    label: str = ""
    dim_batch: int | None = None
    strict_answers: bool = False
    shard: int = 0
    n_shards: int = 1
    limit: int | None = None
    out_dir: str | None = None
    corpus_cache: str | None = None
    overwrite: bool = False
    device: str | None = None
    #: Component/domain digests whose per-word pullbacks give extra arms to
    #: ``workspace_loading``. Unused by the averaging sweeps.
    directions_paths: list[str] = field(default_factory=list)
    #: Groups to take from them. Empty takes all.
    arms: list[str] = field(default_factory=list)
    positions: str = "all"
    n_fit: int = 0

    def resolve_out(self, leaf: str) -> Path:
        root = Path(self.out_dir) if self.out_dir else REPO_ROOT / "results"
        path = root / leaf
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def tag(self) -> str:
        suffix = f"_{self.label}" if self.label else ""
        return f"{self.config_name}_avgn{suffix}_shard{self.shard}of{self.n_shards}"


@dataclass(frozen=True)
class ReadoutSpec(SweepSpec):
    """Does a domain-averaged J_bar *read* better, or only *write* better?

    The whole corpus result so far is about writing: swap steering. But the
    paper defines J-space by READOUT -- the directions along which a concept
    becomes verbalizable -- so "gsm8k gives a better lens" and "gsm8k gives a
    better actuator" are different claims and the causal experiments cannot
    tell them apart.

    Readout here is the paper's own quantity, ``unembed(J_bar h)``: push the
    clean residual through the averaged Jacobian, read the result through the
    unembedding, and ask where the prompt's own argument lands. Concretely
    ``logits[y] = W_U[y] . (J_bar h)``, which for a single concept is the
    familiar ``<J_bar^T u_y, h>`` -- the same lens vector the steering arm
    writes with, scored as a reader instead.

    Two properties make this eval immune to the grading failures that dogged
    the causal side: the metric is a RANK over the whole vocabulary, so there
    is no string matching and no surface-form ambiguity; and rank is invariant
    to positive rescaling of J_bar, so corpora whose maps differ in magnitude
    are compared fairly without any dose matching.

    Needs the full mean matrices (``group_means_full_*.pt``), not the rank-k
    digests: a digest cannot produce full-vocabulary logits.
    """

    #: ``group_means_full_*.pt`` files holding the corpora to score.
    means_paths: list[str] = field(default_factory=list)
    #: Corpus groups to score, alongside the published lens. Empty scores all.
    groups: list[str] = field(default_factory=list)
    #: Ranks recorded for pass@k; k=1 is exact top-1 recovery.
    ks: list[int] = field(default_factory=lambda: [1, 5, 10, 50])
    #: Score each prompt's target against a *different* prompt's activations.
    #: The prior control. A readout score is meant to say that ``J_bar``
    #: transports THIS h to THIS token; but a lens whose vector for " seven"
    #: simply has a large norm ranks " seven" highly under any h at all, and
    #: the two are indistinguishable from the matched run alone. Under the
    #: derangement below, h no longer contains the answer, so whatever score
    #: survives is the lens's standing token prior and has to be subtracted
    #: from the matched score before any corpus is called a better reader.
    #: Nonzero is the permutation seed; 0 disables.
    permute_acts: int = 0

    @property
    def namespace(self) -> str:
        """Widened: ``SweepSpec.namespace`` is model and position convention
        only, so two runs over different corpora or different layers would
        write the same file and the second would silently serve the first.
        """
        import hashlib

        parts = ",".join(sorted(self.groups)) or "all"
        if self.layers:
            parts += ";" + ",".join(str(x) for x in sorted(self.layers))
        tag = hashlib.sha256(parts.encode()).hexdigest()[:8]
        base = f"{super().namespace}_g{tag}"
        return f"{base}_perm{self.permute_acts}" if self.permute_acts else base


@dataclass(frozen=True)
class CausalSpec:
    """The causal arm. Not a prompt sweep -- it shards over the 192 trials."""

    config_name: str = "qwen3-8b"
    mode: str = "swap"
    strengths: list[float] = field(default_factory=lambda: [1.0, 2.0])
    directions: list[str] = field(default_factory=lambda: ["averaged"])
    swap_mode: str = "clamp"
    layers: list[int] | None = None
    #: Appended to the output filename. Needed whenever a grid varies something
    #: the tag does not already encode -- the band, above all. Without it two
    #: band configurations write the same path and the second silently wins.
    label: str = ""
    dim_batch: int | None = None
    strict_answers: bool = False
    #: Greedy tokens to decode after the prompt for string-matched grading.
    #: 0 disables it and keeps first-token grading only. Four is enough for
    #: " twenty-five" and for the quoted single letter the first_letter
    #: template elicits.
    gen_tokens: int = 4
    #: Pooled ``group_mean_digests_*.pt`` files whose per-group pullbacks
    #: supply extra arms. A domain arm needs no new Jacobian work: the pooled
    #: digest already stores ``J_bar_D^T u_y`` for all 16 args at every layer,
    #: which is exactly the write direction, so naming a group here is enough.
    directions_paths: list[str] = field(default_factory=list)
    #: Name of a file under ``data/heldout``; empty uses the upstream
    #: flexible-generalization set.
    trial_set: str = ""
    shard: int = 0
    n_shards: int = 1
    limit: int | None = None
    out_dir: str | None = None
    overwrite: bool = False
    device: str | None = None

    def resolve_out(self, leaf: str) -> Path:
        root = Path(self.out_dir) if self.out_dir else REPO_ROOT / "results"
        path = root / leaf
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def tag(self) -> str:
        suffix = f"_{self.label}" if self.label else ""
        return (
            f"{self.config_name}_{self.mode}{suffix}_shard{self.shard}of{self.n_shards}"
        )


# --------------------------------------------------------------------------
# shared plumbing
# --------------------------------------------------------------------------


def take_shard(items: list, shard: int, n_shards: int) -> list:
    """``items[shard::n_shards]`` -- a stride, not a contiguous block.

    See the module docstring: contiguous slicing puts all the long wikitext
    prompts in one shard and makes the fan-out as slow as its worst container.
    """
    if not 0 <= shard < n_shards:
        raise ValueError(f"shard {shard} out of range for n_shards={n_shards}")
    return items[shard::n_shards]


def resolve_layers(config: ModelConfig, lens, layers: list[int] | None) -> list[int]:
    """Requested layers, or the band intersected with what the lens was fitted at."""
    if layers:
        unknown = sorted(set(layers) - set(lens.source_layers))
        if unknown:
            raise ValueError(f"layers {unknown} are not fitted in this lens")
        return sorted(layers)
    return [l for l in config.band if l in lens.source_layers]


def _setup(spec) -> tuple[ModelConfig, Any, Any, list[int], int]:
    config = load_config(spec.config_name)
    # Specs that need a precision other than the config's declare `dtype`;
    # everything else inherits it. See jsteer.loading.load_model.
    model = load_model(config, device=spec.device, dtype=getattr(spec, "dtype", None))
    lens = load_lens(config)
    layers = resolve_layers(config, lens, spec.layers)
    return config, model, lens, layers, spec.dim_batch or config.dim_batch


def _prompt_shard(spec: SweepSpec, config: ModelConfig, model) -> list[EvalPrompt]:
    if getattr(spec, "domains", None):
        # Named for the panel, not for the model: the panel is what the file
        # holds, and two models can share one materialized corpus only if they
        # tokenize it the same way -- which they do not, so the model name
        # stays in the path as well.
        cache = (
            Path(spec.corpus_cache)
            if spec.corpus_cache
            else REPO_ROOT / "data" / "corpora" / f"{spec.namespace}.json"
        )
        prompts = build_domain_prompt_set(
            spec.domain_specs,
            n_per=spec.n_per,
            tokenizer=model.tokenizer,
            max_chars=config.fit.max_chars,
            min_tokens=spec.min_tokens,
            cache_path=cache,
        )
        if spec.limit:
            prompts = prompts[: spec.limit]
        return take_shard(prompts, spec.shard, spec.n_shards)

    cache = (
        Path(spec.corpus_cache)
        if spec.corpus_cache
        else REPO_ROOT / "data" / "corpora" / f"fit_{config.name}.json"
    )
    prompts = build_prompt_set(
        config, n_fit=spec.n_fit, tokenizer=model.tokenizer, cache_path=cache
    )
    if spec.limit:
        prompts = prompts[: spec.limit]
    return take_shard(prompts, spec.shard, spec.n_shards)


def _covers(digest_path: Path, layers: list[int]) -> bool:
    """Does an existing digest hold *every* layer this run wants?

    Existence alone is not enough. The digest directory is namespaced by model,
    position convention and k -- but NOT by the layer grid, because the grid is
    a per-run argument. So a 3-layer probe and a 12-layer sweep over the same
    model write the same filename, and a bare ``exists()`` check makes the
    sweep skip that prompt entirely: it vanishes from the scalars *and* from
    the running mean, and the run reports success.

    That is exactly how ``countries/capital/France`` went missing from the
    first qwen3.6-27b RQ1 sweep -- 63 eval prompts pooled where 64 were
    expected. Checking coverage turns a silent skip into a recompute.
    """
    try:
        have = set(torch.load(digest_path, map_location="cpu", weights_only=False))
    except Exception:  # truncated or half-written -- recompute it
        return False
    return set(layers) <= have


def _digit_form(answer: str) -> str | None:
    """The digit spelling of a number-word answer, for string matching.

    Unlike the first-token path this is safe for multi-digit answers: matching
    "10" against generated text cannot be confused with "14", which is exactly
    the ambiguity that makes first-token digit grading unusable.
    """
    from jsteer.loading import _NUMBER_WORDS

    value = _NUMBER_WORDS.get(answer.lower())
    return None if value is None else str(value)


def _relevance(prompt: EvalPrompt, word: str, args_by_category: dict) -> str:
    """Is this target the prompt's own arg, an in-category swap target, or foreign?

    The column that turns "does relevance matter?" into a measured variable.
    Every one of the 16 targets is pulled back through every prompt, so all
    three cells are populated by construction.
    """
    if prompt.base is None:
        return "control"
    if word == prompt.base.arg:
        return "self"
    return (
        "swap_target" if word in args_by_category[prompt.base.category] else "foreign"
    )


# --------------------------------------------------------------------------
# research question 1
# --------------------------------------------------------------------------


def run_jacobian_sweep(spec: JacobianSweepSpec) -> dict:
    """Full ``J_x`` per (prompt, layer), reduced to a digest and dropped.

    Returns a small summary dict; the bulk lands under ``out_dir/rq1``.
    """
    config, model, lens, layers, dim_batch = _setup(spec)
    svd_device = spec.svd_device or str(next(model.layers.parameters()).device)
    targets = resolve_targets(model)
    prompts = _prompt_shard(spec, config, model)

    out = spec.resolve_out("rq1")
    # Namespaced by run, NOT bare prompt key. A digest depends on the model,
    # the position convention and k, none of which appear in the prompt key --
    # so a flat directory silently hands a qwen3-1.7b smoke-test digest to a
    # qwen3-8b run and the idempotency check reports it as already done. That
    # happened; the run "completed" in 0.0 s having computed nothing.
    digest_dir = out / "digests" / f"{spec.namespace}_k{spec.k}"
    digest_dir.mkdir(parents=True, exist_ok=True)

    logger.info("%s | %s", config, lens)
    logger.info(
        "shard %d/%d: %d prompts x %d layers, k=%d, positions=%s, svd on %s",
        spec.shard,
        spec.n_shards,
        len(prompts),
        len(layers),
        spec.k,
        spec.positions,
        svd_device,
    )
    logger.info(
        "%.1f MB per digest, %.2f GB for this shard",
        digest_bytes(config.d_model, spec.k, len(targets.words)) / 1e6,
        len(prompts)
        * len(layers)
        * digest_bytes(config.d_model, spec.k, len(targets.words))
        / 1e9,
    )

    # Groups come from the prompt set, not from a literal. The domain sweep has
    # four corpus groups and no "eval"/"fit" at all, and a hardcoded pair would
    # KeyError on the first prompt -- or worse, if it were written defensively,
    # drop every corpus mean while the run reported success.
    means = {
        (group, layer): RunningMean()
        for group in sorted({p.group for p in prompts})
        for layer in layers
    }
    scalars: list[dict] = []
    skipped = 0
    start = time.time()

    # Resume state. Both files are rewritten wholesale at the end, so a shard
    # that skips every prompt would otherwise replace its own previous output
    # with an empty list and a mean over nothing -- a silent data loss that
    # looks exactly like a successful run.
    scalars_path = out / f"scalars_{spec.tag}.json"
    means_path = out / f"group_means_{spec.tag}.pt"
    if not spec.overwrite:
        if scalars_path.exists():
            scalars.extend(json.loads(scalars_path.read_text()))
        if means_path.exists():
            prior = torch.load(means_path, map_location="cpu", weights_only=False)
            for key, state in prior.items():
                group, layer = key.split("|")
                index = (group, int(layer))
                if index in means:
                    means[index].load_state(state)

    for i, prompt in enumerate(prompts):
        digest_path = digest_dir / f"{prompt.key.replace('/', '_')}.pt"
        if digest_path.exists() and not spec.overwrite and _covers(digest_path, layers):
            # Idempotent resume. The running mean is NOT reconstructed from the
            # digest -- a digest is truncated to rank k, so adding it back would
            # mix a low-rank approximation into a quantity that must be an exact
            # mean. The prior sum loaded above covers these prompts instead.
            skipped += 1
            continue

        positions = positions_for(
            spec.positions, model, prompt, max_seq_len=config.fit.max_seq_len
        )
        seq_len = model.encode(prompt.text, max_length=config.fit.max_seq_len).shape[1]
        prompt_batch = max(1, min(dim_batch, spec.max_batch_tokens // max(seq_len, 1)))
        if prompt_batch < dim_batch:
            logger.info(
                "    %s is %d tokens -> dim_batch %d (from %d)",
                prompt.key,
                seq_len,
                prompt_batch,
                dim_batch,
            )
        digests = {}
        for begin in range(0, len(layers), spec.layer_chunk):
            chunk = layers[begin : begin + spec.layer_chunk]
            result = jacobian_for_prompt(
                model,
                prompt.text,
                chunk,
                target_layer=config.fit.target_layer,
                source_positions=None if prompt.fit_positions else positions,
                dim_batch=prompt_batch,
                max_seq_len=config.fit.max_seq_len,
                skip_first=config.fit.skip_first,
            )
            for layer in chunk:
                J_x = result[layer]
                means[(prompt.group, layer)].add(J_x)
                digest = digest_jacobian(
                    J_x.to(svd_device),
                    lens.jacobians[layer].to(svd_device),
                    targets.cotangents.to(svd_device),
                    layer=layer,
                    k=spec.k,
                    lowrank=spec.lowrank,
                )
                digests[layer] = digest
                scalars.append(
                    {
                        "prompt": prompt.key,
                        "group": prompt.group,
                        "category": prompt.base.category if prompt.base else None,
                        "func": prompt.base.func if prompt.base else None,
                        "arg": prompt.base.arg if prompt.base else None,
                        "layer": layer,
                        "seq_len": result.seq_len,
                        "n_valid_positions": result.n_valid_positions,
                        **digest.scalars,
                    }
                )
            del result

        # Written last, so a half-finished prompt never looks complete to a
        # resumed run.
        torch.save(
            {
                layer: {
                    "singular_values": d.singular_values.cpu(),
                    "left": d.left.cpu(),
                    "right": d.right.cpu(),
                    "pullbacks": d.pullbacks.cpu(),
                    "scalars": d.scalars,
                }
                for layer, d in digests.items()
            },
            digest_path,
        )
        elapsed = time.time() - start
        done = i + 1 - skipped
        logger.info(
            "  %3d/%d  %-28s %5.1fs  (eta %.0f min)",
            i + 1,
            len(prompts),
            prompt.key,
            elapsed,
            (elapsed / max(done, 1)) * (len(prompts) - i - 1) / 60,
        )

    torch.save(
        {f"{g}|{l}": m.state() for (g, l), m in means.items() if m.n}, means_path
    )
    scalars_path.write_text(json.dumps(scalars))
    summary = {
        "shard": spec.shard,
        "n_shards": spec.n_shards,
        "n_prompts": len(prompts),
        "n_skipped": skipped,
        "n_records": len(scalars),
        "layers": layers,
        "elapsed_s": round(time.time() - start, 1),
    }
    logger.info("shard done: %s", summary)
    return summary


# --------------------------------------------------------------------------
# research question 2
# --------------------------------------------------------------------------


def run_probe_readout(spec: ReadoutSpec) -> dict:
    """Readout of a LATENT intermediate the prompt never states.

    The complement of :func:`run_readout`, and the stronger test. There the
    concept being recovered sits in the prompt, so a lens can score well by
    representing its input sharply; recovering a token that is present is the
    easy case and is not what J-space is supposed to be about.

    Here the target is ``probe-swap``'s ``intermediate``: for "the language
    spoken in the country where the Amazon River ends", the intermediate is
    *Brazil*, which appears nowhere in the prompt and exists only as something
    the model computes on the way to the answer. If J_bar surfaces it, that is
    broadcasting in the sense the paper means.

    Three things are scored at once so the result is interpretable:

    * the **intermediate** (Brazil) -- the latent concept;
    * the **answer** (Portuguese) -- what the model is about to say, which any
      competent map should show and which therefore calibrates the other two;
    * a **logit-lens baseline** (``J = I``, i.e. no transport at all). Without
      it "gsm8k reads the intermediate at rank 12" is uninterpretable: the
      question is whether J_bar beats simply unembedding the residual, and on
      the last positions of a prompt the raw residual is already dominated by
      the upcoming answer.
    """
    config, model, lens, layers, _ = _setup(spec)
    max_seq_len = config.fit.max_seq_len
    items = probe_swap_items()
    # Scored on the accelerator: 35k full-vocabulary products of a
    # [151936, 4096] matrix took 78 minutes on CPU for one 4-lens run, and the
    # multi-corpus comparison needs several times that.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W_U = unembedding_matrix(model).float().to(device)
    d_model = config.d_model

    banks: dict[str, dict[int, torch.Tensor]] = {
        "published": {ll: lens.jacobians[ll].float() for ll in layers},
        # The no-transport control. Not a corpus -- the point of comparison
        # that says whether J_bar is doing anything at all.
        "logit_lens": {ll: torch.eye(d_model) for ll in layers},
    }
    for raw in spec.means_paths:
        blob = torch.load(raw, map_location="cpu", weights_only=False)
        for key, mean in blob["means"].items():
            group, layer_str = key.split("|")
            layer = int(layer_str)
            if spec.groups and group not in spec.groups:
                continue
            if layer in layers:
                banks.setdefault(group, {})[layer] = mean.float()
        del blob
    missing = [g for g in spec.groups if g not in banks]
    if missing:
        raise ValueError(f"no full means for {missing}; loaded {sorted(banks)}")
    banks = {g: {ll: J.to(device) for ll, J in pl.items()} for g, pl in banks.items()}

    records: list[dict] = []
    skipped: dict[str, int] = {}
    start = time.time()

    # ---- pass 1: one forward per item, activations kept ------------------
    gathered: list[dict] = []
    for item in items:
        # first_token_id, not single_token_id: "Portuguese" and many
        # intermediates are multi-token, and requiring single tokens would
        # discard most of the eval. The rank is then the rank of the word's
        # FIRST token, which is the same convention the causal grader uses.
        try:
            ids = {
                "intermediate": first_token_id(model, item.intermediate),
                "answer": first_token_id(model, item.answer),
            }
        except ValueError as exc:
            skipped[str(exc)[:50]] = skipped.get(str(exc)[:50], 0) + 1
            continue

        input_ids = model.encode(item.prompt, max_length=max_seq_len)
        seq_len = input_ids.shape[1]
        # The last eight positions: the intermediate has to be resolved before
        # the answer can be produced, so this is where it must appear if it
        # appears anywhere. Reading the whole prompt would mostly measure the
        # tokens that are literally present.
        positions = list(range(max(0, seq_len - 8), seq_len))

        with ActivationRecorder(model.layers, at=list(layers)) as recorder:
            model.forward(input_ids)
            acts = {ll: recorder.activations[ll][0].float().cpu() for ll in layers}
        gathered.append(
            {"item": item, "ids": ids, "seq_len": seq_len, "acts": acts}
        )

    # ---- pass 2: score, optionally against another item's activations ----
    # See ``ReadoutSpec.permute_acts``: under the derangement the residual no
    # longer contains this item's intermediate, so any surviving score is the
    # lens's standing prior over those tokens rather than transport.
    donors = _derange(len(gathered), spec.permute_acts)
    for i, entry in enumerate(gathered):
        item, ids = entry["item"], entry["ids"]
        donor = gathered[donors[i]]
        acts, seq_len = donor["acts"], donor["seq_len"]
        positions = list(range(max(0, seq_len - 8), seq_len))

        for group, per_layer in banks.items():
            for ll in layers:
                J = per_layer[ll]
                for pos in positions:
                    logits = W_U @ (J @ acts[ll][pos].to(device))
                    top = torch.topk(logits, 5)
                    row = {
                        "name": item.name,
                        "category": item.category,
                        "lens": group,
                        "layer": ll,
                        "from_end": seq_len - 1 - pos,
                        "top5": [
                            model.tokenizer.decode([t]) for t in top.indices.tolist()
                        ],
                    }
                    for what, tid in ids.items():
                        row[f"rank_{what}"] = int((logits > logits[tid]).sum())
                    records.append(row)
        if (i + 1) % 20 == 0:
            logger.info("  %3d/%d  %.0fs", i + 1, len(gathered), time.time() - start)

    out = spec.resolve_out("readout")
    path = out / f"probe_readout_{spec.tag}.json"
    path.write_text(json.dumps(records))
    summary = {}
    for group in sorted(banks):
        sub = [r for r in records if r["lens"] == group]
        summary[group] = {
            what: {
                "pass@10": round(
                    sum(r[f"rank_{what}"] < 10 for r in sub) / max(len(sub), 1), 4
                ),
                "median": statistics.median([r[f"rank_{what}"] for r in sub])
                if sub
                else None,
            }
            for what in ("intermediate", "answer")
        }
    logger.info("probe readout: %s", summary)
    return {
        "n_items": len({r["name"] for r in records}),
        "n_records": len(records),
        "skipped": skipped,
        "lenses": sorted(banks),
        "summary": summary,
        "path": str(path),
        "elapsed_s": round(time.time() - start, 1),
    }


def _derange(n: int, seed: int) -> list[int]:
    """Index map for the prior control: who supplies each item's activations.

    ``seed == 0`` is the identity -- the matched run, every item read through
    its own activations. Otherwise a *derangement*: a permutation with no fixed
    point, so no item is accidentally scored against itself and the control is
    uncontaminated. Sattolo's algorithm produces one directly (a single
    n-cycle) rather than rejection-sampling a plain shuffle, which for small n
    retries often and, worse, would silently leave fixed points if anyone
    replaced it with ``random.shuffle``.
    """
    if seed == 0:
        return list(range(n))
    if n < 2:
        raise ValueError(f"cannot derange {n} items")
    order = list(range(n))
    rng = random.Random(seed)
    for i in range(n - 1, 0, -1):
        j = rng.randrange(i)  # strictly below i: Sattolo, not Fisher-Yates
        order[i], order[j] = order[j], order[i]
    return order


def run_readout(spec: ReadoutSpec) -> dict:
    """Rank the prompt's own argument under each corpus's averaged Jacobian.

    One forward pass per prompt; everything after is two matmuls per (lens,
    layer, position). No backward passes at all -- the Jacobians were computed
    once and pooled, and this only reads them.
    """
    config, model, lens, layers, _ = _setup(spec)
    max_seq_len = config.fit.max_seq_len
    prompts = [p for p in build_prompt_set(config, n_fit=0, tokenizer=model.tokenizer)]
    # Everything after the forward pass is CPU: W_U is [V, d_model] and the
    # means are d_model^2 each, so keeping them off the accelerator leaves room
    # for the model and costs a few seconds per prompt.
    # Scored on the accelerator: 35k full-vocabulary products of a
    # [151936, 4096] matrix took 78 minutes on CPU for one 4-lens run, and the
    # multi-corpus comparison needs several times that.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W_U = unembedding_matrix(model).float().to(device)

    # ---- assemble the lens bank ------------------------------------------
    # The published lens is always present as the reference every corpus is
    # measured against; without it a table of corpus ranks has no baseline.
    banks: dict[str, dict[int, torch.Tensor]] = {
        "published": {ll: lens.jacobians[ll].float() for ll in layers}
    }
    for raw in spec.means_paths:
        blob = torch.load(raw, map_location="cpu", weights_only=False)
        for key, mean in blob["means"].items():
            group, layer_str = key.split("|")
            layer = int(layer_str)
            if spec.groups and group not in spec.groups:
                continue
            if layer in layers:
                banks.setdefault(group, {})[layer] = mean.float()
        del blob
    missing = [g for g in spec.groups if g not in banks]
    if missing:
        raise ValueError(f"no full means for {missing}; loaded {sorted(banks)}")
    banks = {g: {ll: J.to(device) for ll, J in pl.items()} for g, pl in banks.items()}
    for group, per_layer in banks.items():
        absent = sorted(set(layers) - set(per_layer))
        if absent:
            raise ValueError(f"{group} has no mean at layers {absent}")
    logger.info("lenses: %s over %d layers", sorted(banks), len(layers))

    records: list[dict] = []
    skipped: dict[str, int] = {}
    start = time.time()

    # ---- pass 1: one forward per prompt, activations kept --------------
    # Gathered rather than scored in place so that the prior control can pair
    # prompt i's target with prompt pi(i)'s activations. Seven layers of a
    # ~13-position prompt is ~1.5 MB, so the whole eval set is well under a GB.
    gathered: list[dict] = []
    for prompt in prompts:
        if prompt.base is None:
            continue
        try:
            arg_id = single_token_id(model, prompt.base.arg)
        except ValueError as exc:
            skipped[str(exc)[:50]] = skipped.get(str(exc)[:50], 0) + 1
            continue

        span = arg_span(model, prompt, max_seq_len=max_seq_len)
        input_ids = model.encode(prompt.text, max_length=max_seq_len)
        seq_len = input_ids.shape[1]
        # The paper's convention for workspace loading: the argument's own
        # tokens plus the final position where the answer is read out.
        positions = sorted(set(range(*span) if span else ()) | {seq_len - 1})

        with ActivationRecorder(model.layers, at=list(layers)) as recorder:
            model.forward(input_ids)
            acts = {ll: recorder.activations[ll][0].float().cpu() for ll in layers}
        gathered.append(
            {
                "prompt": prompt,
                "arg_id": arg_id,
                "positions": positions,
                "seq_len": seq_len,
                "acts": acts,
            }
        )

    # ---- pass 2: score, optionally against someone else's activations ----
    donors = _derange(len(gathered), spec.permute_acts)
    for i, item in enumerate(gathered):
        prompt, arg_id = item["prompt"], item["arg_id"]
        donor = gathered[donors[i]]
        acts, seq_len = donor["acts"], donor["seq_len"]
        # Positions are the target prompt's, clipped into the donor's range:
        # the control has to hold the position convention fixed and vary only
        # whose activations are read, or it confounds the two.
        positions = [min(p, seq_len - 1) for p in item["positions"]]
        for group, per_layer in banks.items():
            for ll in layers:
                J = per_layer[ll]
                for pos in positions:
                    h = acts[ll][pos].to(device)
                    logits = W_U @ (J @ h)
                    # Rank of the argument's own token, 0 = top of the vocab.
                    rank = int((logits > logits[arg_id]).sum())
                    top = torch.topk(logits, 5)
                    records.append(
                        {
                            "prompt": prompt.key,
                            "category": prompt.base.category,
                            "func": prompt.base.func,
                            "arg": prompt.base.arg,
                            "lens": group,
                            "layer": ll,
                            "position": pos,
                            "is_last": pos == seq_len - 1,
                            "rank": rank,
                            "logit": float(logits[arg_id]),
                            # Kept for the audit: a readout that "fails" by
                            # returning a near-synonym or a casing variant is a
                            # different fact from one that returns noise, and
                            # only the decoded tokens can tell them apart.
                            "top5": [
                                model.tokenizer.decode([t])
                                for t in top.indices.tolist()
                            ],
                        }
                    )
        if (i + 1) % 16 == 0:
            logger.info("  %3d/%d  %.0fs", i + 1, len(gathered), time.time() - start)

    out = spec.resolve_out("readout")
    path = out / f"readout_{spec.tag}.json"
    path.write_text(json.dumps(records))

    summary = {}
    for group in sorted(banks):
        sub = [r["rank"] for r in records if r["lens"] == group]
        summary[group] = {
            f"pass@{k}": round(sum(r < k for r in sub) / max(len(sub), 1), 4)
            for k in spec.ks
        }
        summary[group]["median_rank"] = statistics.median(sub) if sub else None
    logger.info("readout: %s", summary)
    return {
        "n_prompts": len({r["prompt"] for r in records}),
        "n_records": len(records),
        "skipped": skipped,
        "layers": layers,
        "lenses": sorted(banks),
        "summary": summary,
        "path": str(path),
        "elapsed_s": round(time.time() - start, 1),
    }


def run_domain_pullbacks(spec: DomainPullbackSpec) -> dict:
    """Average ``J_x^T u`` within each corpus; write a causal-ready digest.

    Writes ``rq1/group_mean_digests_{namespace}.pt`` in the same shape
    :func:`pool_group_means` produces, but with only the ``pullbacks`` field
    populated -- the sole field the causal arm reads. The geometric fields are
    absent rather than zero-filled, so a reader that wants them fails loudly
    instead of silently comparing zeros.

    Single-shard by construction. The whole point is that this is cheap; at
    ~4 s per prompt a ten-corpus, 32-per replicate panel is ~20 minutes on one
    GPU, and sharding it would add a pooling step and a second cache key for
    no wall-time worth having.
    """
    config, model, lens, layers, dim_batch = _setup(spec)
    targets = resolve_targets(model, spec.target_words)
    prompts = _prompt_shard(spec, config, model)
    out = spec.resolve_out("rq1")
    out.mkdir(parents=True, exist_ok=True)
    if spec.target_words and len(targets.words) != len(spec.target_words):
        # resolve_targets DROPS multi-token words with a warning. Silently
        # steering a 120-trial eval with 14 of its 16 arguments present is a
        # result-shaped artifact, so refuse instead.
        dropped = sorted(set(spec.target_words) - set(targets.words))
        raise RuntimeError(
            f"{len(dropped)} target words are not single-token and were dropped: "
            f"{dropped}. Every argument of an eval must survive tokenization or "
            "its trials silently vanish from the swap."
        )

    groups = sorted({p.group for p in prompts})
    logger.info("%s | %s", config, lens)
    logger.info(
        "%d prompts over %d corpora x %d layers, %d targets",
        len(prompts),
        len(groups),
        len(layers),
        len(targets.words),
    )
    # Counts, not exit codes. A corpus that yielded fewer documents than asked
    # -- a stream that ran dry, a min_tokens filter that bit -- produces a mean
    # over the wrong N and no error at all.
    per_group = collections.Counter(p.group for p in prompts)
    short = {g: n for g, n in per_group.items() if n != spec.n_per}
    if short:
        raise RuntimeError(
            f"expected {spec.n_per} prompts per corpus; got {short}. "
            "A corpus short of documents makes a mean over a different N "
            "than its siblings, which no downstream check would catch."
        )

    means = {(g, layer): RunningMean() for g in groups for layer in layers}
    start = time.time()
    for i, prompt in enumerate(prompts):
        positions = positions_for(
            spec.positions, model, prompt, max_seq_len=config.fit.max_seq_len
        )
        result = pullback_for_prompt(
            model,
            prompt.text,
            layers,
            targets.cotangents,
            target_layer=config.fit.target_layer,
            source_positions=None if prompt.fit_positions else positions,
            dim_batch=dim_batch,
            max_seq_len=config.fit.max_seq_len,
            skip_first=config.fit.skip_first,
        )
        for layer in layers:
            means[(prompt.group, layer)].add(result[layer])
        if (i + 1) % 20 == 0 or i + 1 == len(prompts):
            rate = (time.time() - start) / (i + 1)
            logger.info(
                "  %3d/%d  %.1fs elapsed  %.1fs/prompt  eta %.0fs",
                i + 1,
                len(prompts),
                time.time() - start,
                rate,
                rate * (len(prompts) - i - 1),
            )

    digests = {}
    for (group, layer), mean in means.items():
        if mean.total is None:
            raise RuntimeError(f"no prompts contributed to {group} at layer {layer}")
        digests[f"{group}|{layer}"] = {
            "group": group,
            "layer": layer,
            "n": mean.n,
            "pullbacks": (mean.total / mean.n).float(),
        }
    path = out / f"group_mean_digests_{spec.namespace}.pt"
    torch.save(
        {"digests": digests, "targets": targets.words, "pullback_only": True}, path
    )
    logger.info("wrote %d group-layer digests -> %s", len(digests), path)
    return {
        "n_prompts": len(prompts),
        "per_group": dict(per_group),
        "groups": groups,
        "layers": layers,
        "namespace": spec.namespace,
        "path": str(path),
        "elapsed_s": round(time.time() - start, 1),
    }


def run_pullback_sweep(spec: PullbackSweepSpec) -> dict:
    """``g_x = J_x^T u_y`` for every (prompt, layer, target), against ``g_bar``."""
    config, model, lens, layers, dim_batch = _setup(spec)
    targets = resolve_targets(model)
    prompts = _prompt_shard(spec, config, model)
    out = spec.resolve_out("rq2")

    records_path = out / f"pullbacks_{spec.tag}.json"
    vectors_path = out / f"vectors_{spec.tag}.pt"
    if records_path.exists() and vectors_path.exists() and not spec.overwrite:
        logger.info("shard %d already complete; skipping", spec.shard)
        return {"shard": spec.shard, "skipped": True}

    logger.info("%s | %s", config, lens)
    logger.info(
        "shard %d/%d: %d prompts (%d eval / %d fit) x %d layers x %d targets, positions=%s",
        spec.shard,
        spec.n_shards,
        len(prompts),
        sum(p.group == "eval" for p in prompts),
        sum(p.group == "fit" for p in prompts),
        len(layers),
        len(targets.words),
        spec.positions,
    )

    # g_bar is prompt-independent: one matmul per layer for the whole sweep.
    averaged = {l: averaged_pullback(lens, targets.cotangents, l) for l in layers}
    args_by_category = category_args()

    records: list[dict] = []
    local_store: dict[str, torch.Tensor] = {}
    start = time.time()

    for i, prompt in enumerate(prompts):
        positions = positions_for(
            spec.positions, model, prompt, max_seq_len=config.fit.max_seq_len
        )
        result = pullback_for_prompt(
            model,
            prompt.text,
            layers,
            targets.cotangents,
            target_layer=config.fit.target_layer,
            source_positions=None if prompt.fit_positions else positions,
            dim_batch=dim_batch,
            max_seq_len=config.fit.max_seq_len,
            skip_first=config.fit.skip_first,
        )
        for layer in layers:
            g_local = result[layer]
            summary = pullback_summary(g_local, averaged[layer])
            local_store[f"{prompt.key}|{layer}"] = g_local
            for k, word in enumerate(targets.words):
                records.append(
                    {
                        "prompt": prompt.key,
                        "group": prompt.group,
                        "category": prompt.base.category if prompt.base else None,
                        "func": prompt.base.func if prompt.base else None,
                        "arg": prompt.base.arg if prompt.base else None,
                        "layer": layer,
                        "target": word,
                        "relevance": _relevance(prompt, word, args_by_category),
                        "cos": summary["cos"][k].item(),
                        "norm_local": summary["norm_local"][k].item(),
                        "norm_avg": summary["norm_avg"][k].item(),
                        "norm_ratio": summary["norm_ratio"][k].item(),
                        "rel_error": summary["rel_error"][k].item(),
                    }
                )
        if (i + 1) % 10 == 0 or i + 1 == len(prompts):
            logger.info("  %3d/%d  %.1fs", i + 1, len(prompts), time.time() - start)

    torch.save(
        {
            "local": local_store,
            "averaged": averaged,
            "targets": targets.words,
            "target_ids": targets.token_ids,
            "layers": layers,
            "positions": spec.positions,
            "config": config.name,
        },
        vectors_path,
    )
    records_path.write_text(json.dumps(records))
    summary = {
        "shard": spec.shard,
        "n_prompts": len(prompts),
        "n_records": len(records),
        "elapsed_s": round(time.time() - start, 1),
    }
    logger.info("shard done: %s", summary)
    return summary


# --------------------------------------------------------------------------
# the causal arm
# --------------------------------------------------------------------------


def _subsets(n: int, pool: int, replicates: int, rng) -> list[list[int]]:
    """Up to ``replicates`` disjoint index subsets of size ``n`` from ``pool``.

    Disjoint, not independent draws with replacement: overlapping subsets share
    prompts and would understate the spread across subsets, which is the whole
    point of running replicates at small n.
    """
    if n > pool:
        return []
    order = list(range(pool))
    rng.shuffle(order)
    k = min(replicates, pool // n)
    return [order[i * n : (i + 1) * n] for i in range(k)]


def workspace_loading(spec: AverageNSpec) -> dict:
    """The paper's own per-prompt predictor of swap success, for use as a baseline.

    Definition, J-Lens paper section 3.4, verbatim: "we define a concept's
    **workspace loading** as the cosine similarity between the residual stream
    and that concept's lens vector, averaged over the argument and readout
    positions in the unmodified forward pass. Workspace loading of the source
    argument predicts swap success well."

    Three details that are easy to get wrong:

    * It is the **source** argument -- the concept being swapped *out* -- not
      the target. The outcome is target rank but the predictor is defined on
      the other token.
    * The aggregation named in the paper is over token positions only. Layers
      are not mentioned, which is a genuine ambiguity; we average over the same
      workspace band every other predictor in C2 uses, so the columns are
      comparable. That is a fitted choice and is disclosed as one.
    * The lens vector is J_bar^T u_src (the averaged pullback), and the
      residual stream is the *clean* forward pass. No J-space basis and no
      projection onto the span of lens vectors are involved -- it is the
      simplest predictor in the table, which is part of what makes it a fair
      baseline.

    Reuses AverageNSpec purely for its config/layers/out plumbing; none of the
    averaging fields apply.

    With ``directions_paths`` set, the same quantity is also computed with each
    named arm's OWN lens vector. That is the control the component result
    needs: an arm could steer better simply because its write direction sits
    closer to the residual stream, which is what loading measures. Positions,
    band and the source-argument convention are held identical across arms, so
    only the vector changes.
    """
    config, model, lens, layers, _ = _setup(spec)
    max_seq_len = config.fit.max_seq_len
    prompts = [p for p in build_prompt_set(config, n_fit=0, tokenizer=model.tokenizer)]

    arm_banks: dict[str, dict[int, torch.Tensor]] = {}
    word_index: dict[str, int] = {}
    for raw in spec.directions_paths:
        blob = torch.load(raw, map_location="cpu", weights_only=False)
        word_index = {w: i for i, w in enumerate(blob["targets"])}
        for entry in blob["digests"].values():
            if spec.arms and entry["group"] not in spec.arms:
                continue
            if entry["layer"] in layers:
                arm_banks.setdefault(entry["group"], {})[entry["layer"]] = entry[
                    "pullbacks"
                ].float()
        del blob
    absent = [a for a in spec.arms if a not in arm_banks]
    if absent:
        raise ValueError(f"no arms {absent}; loaded {sorted(arm_banks)}")

    rows: dict[str, dict] = {}
    for prompt in prompts:
        if prompt.base is None:
            continue
        try:
            src_id = single_token_id(model, prompt.base.arg)
        except ValueError:
            continue
        u_src = unembedding_rows(model, [src_id])

        span = arg_span(model, prompt, max_seq_len=max_seq_len)
        input_ids = model.encode(prompt.text, max_length=max_seq_len)
        seq_len = input_ids.shape[1]
        # "the argument and readout positions": the argument's own tokens plus
        # the final position, where the answer is read out.
        positions = sorted(set(range(*span) if span else ()) | {seq_len - 1})

        with ActivationRecorder(model.layers, at=list(layers)) as recorder:
            model.forward(input_ids)
            acts = {ll: recorder.activations[ll][0].float().cpu() for ll in layers}

        per_layer = {}
        for ll in layers:
            # averaged_pullback follows the lens (CPU); the recorder follows the
            # model (GPU). Reduce on CPU -- this is 11 layers x a handful of
            # positions, so there is nothing to gain from keeping it on device.
            v = averaged_pullback(lens, u_src, ll)[0].float().cpu()
            h = acts[ll][positions]
            cos = (h @ v) / (h.norm(dim=-1) * v.norm() + 1e-12)
            per_layer[ll] = float(cos.mean())
        per_arm: dict[str, float] = {}
        for arm, bank in arm_banks.items():
            if prompt.base.arg not in word_index:
                continue
            vals = []
            for ll in layers:
                v = bank[ll][word_index[prompt.base.arg]].float()
                h = acts[ll][positions]
                vals.append(
                    float(
                        ((h @ v) / (h.norm(dim=-1) * v.norm() + 1e-12)).mean()
                    )
                )
            per_arm[arm] = float(statistics.mean(vals))
        rows[prompt.key] = {
            "loading": float(statistics.mean(per_layer.values())),
            "by_arm": per_arm,
            "by_layer": per_layer,
            "n_positions": len(positions),
            "arg": prompt.base.arg,
            "category": prompt.base.category,
        }

    out = spec.resolve_out("c2")
    suffix = ""
    if spec.arms or spec.directions_paths:
        suffix = "_" + hashlib.sha256(
            repr((sorted(spec.arms), sorted(spec.directions_paths))).encode()
        ).hexdigest()[:8]
    path = (
        out / f"workspace_loading_{config.name}_L{layers[0]}-{layers[-1]}{suffix}.json"
    )
    path.write_text(json.dumps(rows))
    by_cat: dict[str, list[float]] = {}
    for r in rows.values():
        by_cat.setdefault(r["category"], []).append(r["loading"])
    logger.info(
        "workspace loading by category: %s",
        {k: round(statistics.mean(v), 4) for k, v in sorted(by_cat.items())},
    )
    return {
        "n_prompts": len(rows),
        "layers": layers,
        "out": str(path),
        "by_category": {k: round(statistics.mean(v), 4) for k, v in by_cat.items()},
    }


def norm_ratio_report(spec: AverageNSpec) -> dict:
    """Does the swap's column-norm ratio drift as n grows?

    C4 compares partial averages at one strength, justified by the swap being
    invariant to *uniform* rescaling of V. That justification is only complete
    if ||v_s|| and ||v_t|| shrink at the same rate with n: the operator is NOT
    invariant to the two columns being rescaled differently (see
    tests/test_steering.py). Nothing guarantees the source and target lens
    vectors converge at the same speed, so measure it instead of assuming it.

    Precompute only -- no steering, no 192-trial eval, a couple of GPU minutes.
    """
    config, model, lens, layers, dim_batch = _setup(spec)
    max_seq_len = config.fit.max_seq_len
    targets = resolve_targets(model)
    index = {word: i for i, word in enumerate(targets.words)}

    cache = (
        Path(spec.corpus_cache)
        if spec.corpus_cache
        else REPO_ROOT / "data" / "corpora" / f"fit_{config.name}.json"
    )
    prompts = build_prompt_set(
        config, n_fit=spec.n_pool, tokenizer=model.tokenizer, cache_path=cache
    )
    pool = [p for p in prompts if p.group == "fit"]

    pool_g = []
    for prompt in pool:
        seq_len = model.encode(prompt.text, max_length=max_seq_len).shape[1]
        result = pullback_for_prompt(
            model,
            prompt.text,
            layers,
            targets.cotangents,
            target_layer=config.fit.target_layer,
            source_positions=list(range(seq_len)),
            dim_batch=dim_batch,
            max_seq_len=max_seq_len,
            skip_first=config.fit.skip_first,
        )
        pool_g.append({ll: result[ll].float().cpu() for ll in layers})

    pairs = [
        (index[t.source_arg], index[t.target_arg])
        for t in flexible_generalization_trials()
        if t.source_arg in index and t.target_arg in index
    ]

    def ratios(bank: dict[int, torch.Tensor]) -> list[float]:
        out = []
        for ll in layers:
            norms = bank[ll].norm(dim=-1)
            out += [float(norms[si] / norms[ti]) for si, ti in pairs if norms[ti] > 0]
        return out

    rng = random.Random(spec.seed)
    rows = {}
    for n in spec.ns:
        vals = []
        for idx in _subsets(n, len(pool_g), spec.replicates, rng):
            bank = {
                ll: torch.stack([pool_g[j][ll] for j in idx]).mean(0) for ll in layers
            }
            vals += ratios(bank)
        if vals:
            rows[str(n)] = {
                "median": round(statistics.median(vals), 4),
                "p10": round(sorted(vals)[len(vals) // 10], 4),
                "p90": round(sorted(vals)[9 * len(vals) // 10], 4),
            }
    lens_bank = {ll: averaged_pullback(lens, targets.cotangents, ll) for ll in layers}
    vals = ratios(lens_bank)
    rows["lens_Jbar"] = {
        "median": round(statistics.median(vals), 4),
        "p10": round(sorted(vals)[len(vals) // 10], 4),
        "p90": round(sorted(vals)[9 * len(vals) // 10], 4),
    }
    logger.info("||v_s||/||v_t|| by n: %s", rows)
    return {"n_pool": len(pool_g), "layers": layers, "ratios": rows}


def run_average_n(spec: AverageNSpec) -> dict:
    """C4: steering success as a function of how many prompts J_bar averages.

    Precomputes the pullbacks of all 16 argument cotangents once per pool
    prompt, then forms partial means. Pullback is linear in the prompt, so the
    mean of the pullbacks equals the pullback through the mean Jacobian -- no
    d_model x d_model matrix ever has to be built or stored.
    """
    config, model, lens, layers, dim_batch = _setup(spec)
    max_seq_len = config.fit.max_seq_len
    out = spec.resolve_out("causal")
    path = out / f"steering_{spec.tag}.json"
    if path.exists() and not spec.overwrite:
        logger.info("shard %d already complete; skipping", spec.shard)
        return {"shard": spec.shard, "skipped": True}

    targets = resolve_targets(model)
    index = {word: i for i, word in enumerate(targets.words)}

    cache = (
        Path(spec.corpus_cache)
        if spec.corpus_cache
        else REPO_ROOT / "data" / "corpora" / f"fit_{config.name}.json"
    )
    all_prompts = build_prompt_set(
        config, n_fit=spec.n_pool, tokenizer=model.tokenizer, cache_path=cache
    )
    pool = [p for p in all_prompts if p.group == "fit"]
    evals = [p for p in all_prompts if p.group == "eval"]
    logger.info(
        "%s | pool=%d wikitext, %d eval prompts, layers L%d-L%d",
        config,
        len(pool),
        len(evals),
        layers[0],
        layers[-1],
    )

    def pullbacks(prompt) -> dict[int, torch.Tensor]:
        seq_len = model.encode(prompt.text, max_length=max_seq_len).shape[1]
        result = pullback_for_prompt(
            model,
            prompt.text,
            layers,
            targets.cotangents,
            target_layer=config.fit.target_layer,
            source_positions=list(range(seq_len)),
            dim_batch=dim_batch,
            max_seq_len=max_seq_len,
            skip_first=config.fit.skip_first,
        )
        return {ll: result[ll].float().cpu() for ll in layers}

    start = time.time()
    pool_g: list[dict[int, torch.Tensor]] = []
    for i, prompt in enumerate(pool):
        pool_g.append(pullbacks(prompt))
        if (i + 1) % 20 == 0:
            logger.info("  pool %d/%d  %.0fs", i + 1, len(pool), time.time() - start)
    eval_g = {p.key: pullbacks(p) for p in evals}
    logger.info("  pullbacks done in %.0fs", time.time() - start)

    rng = random.Random(spec.seed)
    # arm name -> {layer: [n_targets, d]}
    banks: dict[str, dict[int, torch.Tensor]] = {
        "lens_Jbar": {
            ll: averaged_pullback(lens, targets.cotangents, ll) for ll in layers
        }
    }
    subsets: dict[str, tuple[int, int]] = {}
    for n in spec.ns:
        for r, idx in enumerate(_subsets(n, len(pool_g), spec.replicates, rng)):
            name = f"avg_n{n}_r{r}"
            banks[name] = {
                ll: torch.stack([pool_g[j][ll] for j in idx]).mean(0) for ll in layers
            }
            subsets[name] = (n, r)

    trials = flexible_generalization_trials()
    if spec.limit:
        trials = trials[: spec.limit]
    trials = take_shard(trials, spec.shard, spec.n_shards)
    logger.info(
        "shard %d/%d: %d trials x %d arms",
        spec.shard,
        spec.n_shards,
        len(trials),
        len(banks) + 2,
    )

    def answer_id(m, word):
        return (
            [single_token_id(m, word)]
            if spec.strict_answers
            else answer_variant_ids(m, word)
        )

    records: list[dict] = []
    skipped: dict[str, int] = {}
    for i, trial in enumerate(trials):
        try:
            ids = {
                "source_arg": single_token_id(model, trial.source_arg),
                "target_arg": single_token_id(model, trial.target_arg),
                "source_answer": answer_id(model, trial.source_answer),
                "target_answer": answer_id(model, trial.target_answer),
            }
        except ValueError as exc:
            key = str(exc)[:60]
            skipped[key] = skipped.get(key, 0) + 1
            continue
        if trial.source_arg not in index or trial.target_arg not in index:
            skipped["arg not in target bank"] = (
                skipped.get("arg not in target bank", 0) + 1
            )
            continue
        si, ti = index[trial.source_arg], index[trial.target_arg]
        prompt_key = f"{trial.category}/{trial.func}/{trial.source_arg}"
        base = {
            "category": trial.category,
            "func": trial.func,
            "prompt_key": prompt_key,
            "source_arg": trial.source_arg,
            "target_arg": trial.target_arg,
            "target_answer": trial.target_answer,
        }

        clean_logits = next_token_logits(model, trial.prompt, max_seq_len=max_seq_len)
        clean = grade(model, clean_logits, target_id=ids["source_answer"])
        records.append(
            {
                **base,
                "arm": "baseline",
                "n": 0,
                "replicate": -1,
                "strength": 0.0,
                **vars(clean),
            }
        )

        arms = dict(banks)
        # The two n=1 controls. matched = this prompt's own g_x (the C1 arm);
        # mismatched = a different eval prompt's, deterministically chosen so
        # the comparison is reproducible.
        if prompt_key in eval_g:
            arms["local_matched"] = eval_g[prompt_key]
        others = [k for k in eval_g if k != prompt_key]
        if others:
            arms["local_mismatched"] = eval_g[others[i % len(others)]]

        exclude = [ids["target_arg"], ids["source_arg"]]
        for name, g in arms.items():
            edit = swap_edit(
                model,
                trial.prompt,
                {ll: g[ll][si] for ll in layers},
                {ll: g[ll][ti] for ll in layers},
                mode=spec.swap_mode,
                strength=spec.strength,
                max_seq_len=max_seq_len,
            )
            n, r = subsets.get(name, (0, -1))
            records.append(
                {
                    **base,
                    "arm": name,
                    "n": n,
                    "replicate": r,
                    "strength": spec.strength,
                    "baseline_ok": clean.hit,
                    **vars(
                        grade(
                            model,
                            next_token_logits(
                                model,
                                trial.prompt,
                                edit=edit,
                                max_seq_len=max_seq_len,
                            ),
                            target_id=ids["target_answer"],
                            source_id=ids["source_answer"],
                            exclude_ids=exclude,
                            clean_logits=clean_logits,
                        )
                    ),
                }
            )
        if (i + 1) % 10 == 0 or i + 1 == len(trials):
            logger.info("  %3d/%d  %.0fs", i + 1, len(trials), time.time() - start)

    path.write_text(json.dumps(records))
    summary = {
        "shard": spec.shard,
        "n_trials": len(trials),
        "n_records": len(records),
        "n_arms": len(banks) + 2,
        "n_pool": len(pool_g),
        "n_skipped": sum(skipped.values()),
        "skipped_reasons": skipped,
        "elapsed_s": round(time.time() - start, 1),
    }
    logger.info("shard done: %s", summary)
    return summary


def run_causal(spec: CausalSpec) -> dict:
    """Steer, grade, record. Shards over trials rather than prompts."""
    config, model, lens, layers, dim_batch = _setup(spec)
    max_seq_len = config.fit.max_seq_len
    out = spec.resolve_out("causal")

    path = out / f"steering_{spec.tag}.json"
    if path.exists() and not spec.overwrite:
        # Existence is not completeness. The filename carries the model, mode,
        # label and shard -- but NOT the arm list, so a run that adds ten domain
        # arms resolves to the path a one-arm run wrote and skips, reporting
        # success having computed nothing new. That happened here: a Tier-0
        # rerun with eleven arms returned the single-arm file untouched.
        try:
            have = {r["arm"] for r in json.loads(path.read_text())}
        except Exception:  # truncated or half-written -- recompute
            have = set()
        want = set(spec.directions)
        if want <= have:
            logger.info("shard %d already complete; skipping", spec.shard)
            return {"shard": spec.shard, "skipped": True, "arms": sorted(have)}
        logger.info(
            "shard %d exists but lacks arms %s; recomputing",
            spec.shard,
            sorted(want - have),
        )

    trials = (
        heldout_trials(spec.trial_set)
        if getattr(spec, "trial_set", "")
        else flexible_generalization_trials()
    )
    if spec.limit:
        trials = trials[: spec.limit]
    trials = take_shard(trials, spec.shard, spec.n_shards)

    logger.info("%s | %s", config, lens)
    logger.info(
        "shard %d/%d: %d trials, band L%d-L%d, mode=%s, arms=%s",
        spec.shard,
        spec.n_shards,
        len(trials),
        layers[0],
        layers[-1],
        spec.mode,
        spec.directions,
    )

    # Args must be single tokens -- the intervention needs one unembedding row
    # and there is no first-token fallback for a write. Answers are only read
    # by the grader, so by default they accept any surface form's first token:
    # exact-id grading scored countries/currency 0/4 on Qwen3-8B purely because
    # the key writes "Euro" and the model writes " euro". See
    # jsteer.loading.answer_variant_ids.
    def answer_id(m, word):
        return (
            [single_token_id(m, word)]
            if spec.strict_answers
            else answer_variant_ids(m, word)
        )

    # Domain arms, loaded once. ``pullbacks`` is [n_args, d_model] in the
    # order of ``targets``, so a trial indexes it by argument word.
    domain: dict[str, dict[int, torch.Tensor]] = {}
    domain_words: list[str] = []
    for raw in spec.directions_paths:
        blob = torch.load(raw, map_location="cpu", weights_only=False)
        words = list(blob["targets"])
        if domain_words and words != domain_words:
            raise ValueError(
                f"{raw}: target order {words} does not match {domain_words}; "
                "pullback rows would be indexed by the wrong argument"
            )
        domain_words = words
        for key, entry in blob["digests"].items():
            group, layer = key.split("|")
            layer = int(layer)
            if layer in layers:
                domain.setdefault(group, {})[layer] = entry["pullbacks"].float()
    missing = [
        name
        for name in spec.directions
        if name not in ("averaged", "local") and name not in domain
    ]
    if missing:
        raise ValueError(f"no directions for {missing}; loaded {sorted(domain)}")
    for group, per_layer in domain.items():
        absent = sorted(set(layers) - set(per_layer))
        if absent:
            raise ValueError(f"{group} has no pullbacks at layers {absent}")
    if domain:
        logger.info("domain arms: %s over %d layers", sorted(domain), len(layers))
        # Loaded but unrequested is the silent failure: run_causal iterates
        # spec.directions, so a digest full of corpora that nobody named
        # produces a run that exits 0 with only the "averaged" arm in it.
        unused = sorted(set(domain) - set(spec.directions))
        if not set(domain) & set(spec.directions):
            raise ValueError(
                f"loaded {len(domain)} domain groups {unused} but spec.directions "
                f"= {spec.directions} names none of them; every arm must be listed "
                "in directions or it is silently dropped"
            )
        if unused:
            logger.warning("loaded but not steered with: %s", unused)

    # Guard, not decoration. A whitespace token in an answer's accepted set
    # scores every leading-space completion as a hit, which silently turned
    # the whole numbers category into a fake 9/48. Assert it cannot happen
    # rather than trusting that the fix shipped.
    for probe in ("eighteen", "ten", "six", "t"):
        admitted = answer_id(model, probe)
        blank = [i for i in admitted if not model.tokenizer.decode([i]).strip()]
        logger.info("grader %-9s -> %d ids, whitespace=%s", probe, len(admitted), blank)
        if blank:
            raise RuntimeError(
                f"answer_variant_ids({probe!r}) admits whitespace token(s) "
                f"{blank}; grading would count any leading space as a hit"
            )

    records: list[dict] = []
    skipped: dict[str, int] = {}
    start = time.time()

    for i, trial in enumerate(trials):
        try:
            ids = {
                "source_arg": single_token_id(model, trial.source_arg),
                "target_arg": single_token_id(model, trial.target_arg),
                "source_answer": answer_id(model, trial.source_answer),
                "target_answer": answer_id(model, trial.target_answer),
            }
        except ValueError as exc:
            key = str(exc)[:60]
            skipped[key] = skipped.get(key, 0) + 1
            continue

        base = {
            "category": trial.category,
            "func": trial.func,
            # The join key back to the geometry sweeps: J_x is a property of the
            # prompt, a steering outcome is a property of (prompt, target).
            "prompt_key": f"{trial.category}/{trial.func}/{trial.source_arg}",
            "source_arg": trial.source_arg,
            "target_arg": trial.target_arg,
            "target_answer": trial.target_answer,
        }
        norms = mean_residual_norms(
            model, trial.prompt, layers, max_seq_len=max_seq_len
        )

        clean_logits = next_token_logits(model, trial.prompt, max_seq_len=max_seq_len)
        clean = grade(model, clean_logits, target_id=ids["source_answer"])
        clean_text = (
            greedy_continuation(
                model, trial.prompt, n_tokens=spec.gen_tokens, max_seq_len=max_seq_len
            )
            if spec.gen_tokens
            else ""
        )
        records.append(
            {
                **base,
                "arm": "baseline",
                "strength": 0.0,
                **vars(clean),
                "generated": clean_text,
                # Baseline asks whether the model can do the task at all, so it
                # is graded against its *own* answer, not the swap target.
                "hit_generated": bool(clean_text)
                and answer_matches(
                    clean_text,
                    trial.source_answer,
                    digit=_digit_form(trial.source_answer),
                ),
            }
        )

        cotangents = unembedding_rows(model, [ids["source_arg"], ids["target_arg"]])
        arms: dict[str, dict[int, torch.Tensor]] = {
            "averaged": {l: averaged_pullback(lens, cotangents, l) for l in layers}
        }
        if "local" in spec.directions:
            seq_len = model.encode(trial.prompt, max_length=max_seq_len).shape[1]
            result = pullback_for_prompt(
                model,
                trial.prompt,
                layers,
                cotangents,
                target_layer=config.fit.target_layer,
                source_positions=list(range(seq_len)),
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
                skip_first=config.fit.skip_first,
            )
            arms["local"] = {l: result[l] for l in layers}

        # A domain arm is the same swap with V built from that corpus's
        # averaged Jacobian instead of the lens's.
        if domain:
            si = domain_words.index(trial.source_arg)
            ti = domain_words.index(trial.target_arg)
            for group, per_layer in domain.items():
                arms[group] = {
                    l: torch.stack([per_layer[l][si], per_layer[l][ti]]) for l in layers
                }

        exclude = [ids["target_arg"], ids["source_arg"]]

        target_digit = _digit_form(trial.target_answer)

        def _gen(edit, _trial=trial, _digit=target_digit) -> dict:
            """Generated continuation and its string-matched verdict."""
            # getattr: PreconditionSpec shares this code shape but has no
            # gen_tokens field, and C7 grades on logits only.
            if not getattr(spec, "gen_tokens", 0):
                return {"generated": "", "hit_generated": False}
            text = greedy_continuation(
                model,
                _trial.prompt,
                edit=edit,
                n_tokens=getattr(spec, "gen_tokens", 0),
                max_seq_len=max_seq_len,
            )
            return {
                "generated": text,
                "hit_generated": answer_matches(
                    text, _trial.target_answer, digit=_digit
                ),
            }

        for strength in spec.strengths:
            for source in spec.directions:
                g = arms[source]
                if spec.mode in ("swap", "both"):
                    edit = swap_edit(
                        model,
                        trial.prompt,
                        {l: g[l][0] for l in layers},
                        {l: g[l][1] for l in layers},
                        mode=spec.swap_mode,
                        strength=strength,
                        max_seq_len=max_seq_len,
                    )
                    records.append(
                        {
                            **base,
                            "arm": f"swap_{source}",
                            "strength": strength,
                            "baseline_ok": clean.hit,
                            **vars(
                                grade(
                                    model,
                                    next_token_logits(
                                        model,
                                        trial.prompt,
                                        edit=edit,
                                        max_seq_len=max_seq_len,
                                    ),
                                    target_id=ids["target_answer"],
                                    source_id=ids["source_answer"],
                                    exclude_ids=exclude,
                                    clean_logits=clean_logits,
                                )
                            ),
                            **_gen(edit),
                        }
                    )
                if spec.mode in ("additive", "both"):
                    edit = additive_edit(
                        {l: g[l][1] for l in layers}, norms, strength=strength
                    )
                    records.append(
                        {
                            **base,
                            "arm": f"additive_{source}",
                            "strength": strength,
                            "baseline_ok": clean.hit,
                            **vars(
                                grade(
                                    model,
                                    next_token_logits(
                                        model,
                                        trial.prompt,
                                        edit=edit,
                                        max_seq_len=max_seq_len,
                                    ),
                                    target_id=ids["target_answer"],
                                    source_id=ids["source_answer"],
                                    exclude_ids=exclude,
                                    clean_logits=clean_logits,
                                )
                            ),
                            **_gen(edit),
                        }
                    )

        if (i + 1) % 20 == 0 or i + 1 == len(trials):
            logger.info("  %3d/%d  %.0fs", i + 1, len(trials), time.time() - start)

    path.write_text(json.dumps(records))
    summary = {
        "shard": spec.shard,
        "n_trials": len(trials),
        "n_records": len(records),
        "n_skipped": sum(skipped.values()),
        "skipped_reasons": skipped,
        "elapsed_s": round(time.time() - start, 1),
    }
    logger.info("shard done: %s", summary)
    return summary


def pool_group_means(
    config_name: str,
    *,
    positions: str = "all",
    namespace: str | None = None,
    k: int = 64,
    shards: int | None = None,
    out_dir: str | None = None,
    svd_device: str = "cpu",
    save_means: bool = False,
) -> dict:
    """Pool the per-shard running sums into one digested group mean per layer.

    This exists because the raw artifact is too big to move. Each shard stores
    an exact fp64 running *sum* -- a full ``d_model x d_model`` matrix per
    (group, layer) -- so on Qwen3-8B one shard file is 4096^2 x 8 bytes x 2
    groups x 26 layers = **6.98 GB**, and eight shards is 56 GB. Downloading
    that to compute a handful of scalars is absurd, and in practice the bulk
    download silently truncated the files instead.

    So the pooling runs where the data already is, and only the digest comes
    home: top-k bases, the target pullbacks, and the scalars against ``J_bar``
    -- 2.4 MB per (group, layer), ~125 MB for the run.

    The sums are pooled exactly (sum of sums over sum of counts), never as a
    mean of shard means, which would be wrong whenever shards differ in size --
    and they do, whenever one resumes.

    Returns:
        Summary dict; writes ``group_mean_digests_{namespace}.pt``, plus
        ``group_means_full_{namespace}.pt`` when ``save_means`` is set.
    """
    config = load_config(config_name)
    lens = load_lens(config)
    root = Path(out_dir) if out_dir else REPO_ROOT / "results"
    directory = root / "rq1"

    # ``namespace`` is the domain sweep's widened run identity; without it the
    # glob would sweep up an eval-vs-wikitext run's sums alongside a domain
    # run's and pool the two into one meaningless average.
    namespace = namespace or f"{config.name}_{positions}"
    paths = sorted(directory.glob(f"group_means_{namespace}_shard*.pt"))
    if not paths:
        raise FileNotFoundError(f"no shard sums for {namespace!r} under {directory}")

    # Only pool one run. Files from different shard counts are different runs --
    # a 3-prompt probe at shard0of1 next to an 8-shard production run -- and
    # pooling them together double-counts whatever the probe touched. Same
    # failure the merge script guards; it belongs here too.
    import re as _re

    counts = {int(_re.search(r"_shard\d+of(\d+)\.pt$", p.name).group(1)) for p in paths}
    if shards is None and len(counts) > 1:
        raise ValueError(
            f"{len(counts)} separate runs present (shard counts {sorted(counts)}); "
            "pass shards= to choose one"
        )
    chosen = shards if shards is not None else counts.pop()
    paths = [p for p in paths if p.name.endswith(f"of{chosen}.pt")]
    if not paths:
        raise FileNotFoundError(f"no shard sums with {chosen} shards under {directory}")

    totals: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    for path in paths:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        for key, state in blob.items():
            totals[key] = (
                state["sum"] if key not in totals else totals[key] + state["sum"]
            )
            counts[key] = counts.get(key, 0) + int(state["n"])
        del blob

    # The cotangents have to be the same 16 arg rows the per-prompt digests
    # used, or the pooled pullbacks are not comparable with them.
    model = load_model(config)
    targets = resolve_targets(model)

    digests = {}
    full_means: dict[str, torch.Tensor] = {}
    for key, total in totals.items():
        group, layer_str = key.split("|")
        layer = int(layer_str)
        mean = (total / counts[key]).float()
        digest = digest_jacobian(
            mean.to(svd_device),
            lens.jacobians[layer].to(svd_device),
            targets.cotangents.to(svd_device),
            layer=layer,
            k=k,
        )
        digests[key] = {
            "group": group,
            "layer": layer,
            "n": counts[key],
            "singular_values": digest.singular_values.cpu(),
            "left": digest.left.cpu(),
            "right": digest.right.cpu(),
            "pullbacks": digest.pullbacks.cpu(),
            "scalars": digest.scalars,
        }
        if save_means:
            full_means[key] = mean.cpu()

    out = directory / f"group_mean_digests_{namespace}.pt"
    torch.save({"digests": digests, "targets": targets.words, "k": k}, out)
    if save_means:
        # The rank-k digest cannot answer a cross-domain *readout* question:
        # ``lens(h) = unembed(J_bar h)`` needs the whole matrix, not its top-k
        # subspace, so reading one domain's activations through another
        # domain's lens requires the means themselves. fp32 because these get
        # downloaded -- 4096^2 x 4 bytes is 67 MB per (group, layer), which is
        # 1.9 GB for a four-group seven-layer panel and merely large, whereas
        # the fp64 sums it is built from are 10.7 GB per shard.
        means_out = directory / f"group_means_full_{namespace}.pt"
        torch.save({"means": full_means, "counts": counts}, means_out)
        logger.info("wrote %d full mean matrices -> %s", len(full_means), means_out)
    # Surfaced deliberately: a group mean over fewer prompts than the sweep
    # ran is the signature of a resumed shard, whose skipped prompts are in the
    # digests but not in any saved sum. It looks like a normal result.
    per_group: dict[str, set[int]] = {}
    for key, n in counts.items():
        per_group.setdefault(key.split("|")[0], set()).add(n)
    logger.info("prompts per group: %s", {g: sorted(v) for g, v in per_group.items()})

    summary = {
        "n_shards": len(paths),
        "n_keys": len(digests),
        "prompts_per_group": {g: sorted(v) for g, v in per_group.items()},
        "counts": {g: n for g, n in sorted(counts.items())},
        "out": str(out),
        "saved_means": bool(save_means),
    }
    logger.info("pooled %d shard sums -> %s", len(paths), out)
    return summary


# --------------------------------------------------------------------------
# C5 - linear response: which Jacobian is the better local model?
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearResponseSpec:
    """C5 - measure the linearization error of J_x and J_bar against dose.

    Why this exists, and why it comes before any more steering
    ------------------------------------------------------------
    Every causal result so far compares J_x and J_bar as *write* directions and
    finds J_x far worse (12/192 vs 84/192 on swap). Two explanations survive
    the controls already run:

    * **the metric.** g = J^T u is a covector; adding it to an activation is
      only correct under the Euclidean metric, and FishBack S2 says so
      verbatim. J_bar may survive by accident because averaging flattens its
      spectrum.
    * **the validity radius.** J_x is exact at h_x and wrong away from it, with
      error growing like ||d||^2; J_bar is biased at first order but its
      curvature error is the *average* curvature, which is flatter. Then there
      is a crossover dose below which J_x wins and above which it loses, and
      all our steering sits above it.

    Steering success cannot separate these, because it is only measurable at
    doses large enough to move an argmax. Linear response can: it asks how well
    each Jacobian predicts the model's *actual* response, at doses running four
    orders of magnitude below the steering regime.

    It is also, and this matters more, the validation we never ran. If J_x does
    not predict infinitesimal responses better than J_bar does, our Jacobian is
    wrong and every negative causal result downstream of it is uninterpretable.
    At the smallest dose the local prediction error must go to zero. If it does
    not, stop and fix the pipeline.

    The perturbation has to match the estimator
    -------------------------------------------
    J_x is ``d h_final / d h_l`` -- one source layer. Our steering edits *every
    band layer at every position*, which no single-layer Jacobian models, so
    reusing that operator here would measure the wrong thing and blame the
    Jacobian for it. This sweep therefore perturbs **one layer at one
    position**, and uses ``positions="last"`` so that source and target
    position sets are both the readout token. Under that convention the
    estimator identity has no leading factor:

        Delta <u_y, h_final[last]>  ~=  <g_y, delta>,     g_y = J_x[last,last]^T u_y

    With ``positions="all"`` the same identity carries a factor of ``seq_len``
    (the estimator averages over source positions but sums the cotangent over
    target positions), which is a live way to get a plausible-looking wrong
    answer. Hence the convention is pinned here rather than inherited.

    What is predicted, and what is merely reported
    ----------------------------------------------
    The scalar the Jacobian actually models is ``S_y = <u_y, h_final[last]>``,
    the *raw* unembedding projection -- ``unembedding_rows`` deliberately
    ignores the final norm. So ``S_y`` is the honest target of prediction and
    the logit is not: the logit adds an RMSNorm the lens never linearized.
    Both are recorded. Prediction error is scored on ``S``; the logit and the
    KL are there to mark where the steering regime sits on the same axis.

    Arms
    ----
    ``random`` probes are the fair test of "which J is the better local model",
    because they are not derived from either J and so favour neither. The
    ``local_*`` / ``averaged_*`` arms perturb along the directions we actually
    steer with, for each of the prompt's three swap targets, and answer the
    separate question of which direction moves the target logit more at a given
    dose -- that one is RQ3 read literally.

    Attributes:
        alphas: Dose grid. ``delta = alpha * ||h_l[last]|| * unit(dir)``, so
            alpha is in units of the residual norm at the perturbed position --
            the same convention the steering recipe uses, except anchored to
            the last position rather than the prompt mean (both are recorded).
            The grid must reach far below the steering regime or the crossover
            is off the left edge of the plot.
        n_random: Independent random probe directions per (prompt, layer). Two
            is enough to see whether the spread across probes is small relative
            to the local-vs-averaged gap.
    """

    config_name: str = "qwen3-8b"
    layers: list[int] | None = None
    alphas: list[float] = field(
        default_factory=lambda: [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 2.0, 4.0]
    )
    n_random: int = 2
    seed: int = 0
    label: str = ""
    dim_batch: int | None = None
    shard: int = 0
    n_shards: int = 1
    limit: int | None = None
    out_dir: str | None = None
    overwrite: bool = False
    device: str | None = None
    #: Overrides the config. float32 by default and not negotiable in spirit:
    #: this sweep reads a *difference* of two forward passes, and bfloat16
    #: quantizes away the small-alpha end of the grid entirely. Qwen3-8B in
    #: fp32 is ~32 GB of weights, so this needs an 80 GB card.
    dtype: str | None = "float32"
    #: Pinned; see the class docstring. Kept as a field so it lands in the tag.
    positions: str = "last"

    def resolve_out(self, leaf: str) -> Path:
        root = Path(self.out_dir) if self.out_dir else REPO_ROOT / "results"
        path = root / leaf
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def tag(self) -> str:
        # Every input that changes the contents goes in the path. The layer
        # grid and the alpha grid have both silently invalidated a cache in
        # this repo before -- see CLAUDE.md -- so they are hashed in rather
        # than trusted to the label.
        stamp = hashlib.sha256(
            repr(
                (
                    self.layers,
                    self.alphas,
                    self.n_random,
                    self.seed,
                    self.positions,
                    self.dtype,
                )
            ).encode()
        ).hexdigest()[:8]
        suffix = f"_{self.label}" if self.label else ""
        return (
            f"{self.config_name}_linresp{suffix}_{stamp}"
            f"_shard{self.shard}of{self.n_shards}"
        )


@torch.no_grad()
def _final_residual(model, input_ids, edit, *, position: int = -1) -> torch.Tensor:
    """``h_final[position]`` under ``edit``, on the model's device and dtype.

    Reads the final *block* output, not ``last_hidden_state``, so the vector is
    in the same (pre-norm) basis as the Jacobian's target and the raw
    unembedding rows. Returned un-cast because the caller feeds it back into
    ``model.unembed``, which wants the model's own dtype; cast to fp32 on CPU
    only for the cotangent products. The steerer is registered before the
    recorder for the reason given in
    :func:`jsteer.steering.next_token_logits`.
    """
    final_layer = model.n_layers - 1
    with (
        steered(model, edit),
        ActivationRecorder(model.layers, at=[final_layer]) as recorder,
    ):
        model.forward(input_ids)
        return recorder.activations[final_layer][0, position].detach().float().cpu()


def run_linear_response(spec: LinearResponseSpec) -> dict:
    """Perturb one layer at one position; compare predicted to actual response."""
    config, model, lens, layers, dim_batch = _setup(spec)
    max_seq_len = config.fit.max_seq_len
    out = spec.resolve_out("c5")
    path = out / f"linresp_{spec.tag}.pt"
    if path.exists() and not spec.overwrite:
        logger.info("shard %d already complete; skipping", spec.shard)
        return {"shard": spec.shard, "skipped": True, "path": str(path)}

    # Targets are the 16 arguments, not the answers: the intervention writes an
    # argument's direction and the lens vector is that argument's pullback.
    trials = flexible_generalization_trials()
    args = sorted({t.source_arg for t in trials})
    swap_targets: dict[str, list[str]] = {}
    prompt_text: dict[str, str] = {}
    for t in trials:
        key = f"{t.category}/{t.func}/{t.source_arg}"
        swap_targets.setdefault(key, []).append(t.target_arg)
        prompt_text[key] = t.prompt

    keys = sorted(prompt_text)
    if spec.limit:
        keys = keys[: spec.limit]
    keys = take_shard(keys, spec.shard, spec.n_shards)

    arg_ids = {}
    for a in args:
        try:
            arg_ids[a] = single_token_id(model, a)
        except ValueError:
            logger.warning("argument %r is not a single token; dropped", a)
    args = [a for a in args if a in arg_ids]
    cotangents = unembedding_rows(model, [arg_ids[a] for a in args])  # [16, d]
    arg_index = {a: i for i, a in enumerate(args)}

    logger.info("%s | %s", config, lens)
    logger.info(
        "shard %d/%d: %d prompts, layers %s, %d alphas, %d cotangents",
        spec.shard,
        spec.n_shards,
        len(keys),
        layers,
        len(spec.alphas),
        len(args),
    )

    records: list[dict] = []
    start = time.time()

    for i, key in enumerate(keys):
        text = prompt_text[key]
        input_ids = model.encode(text, max_length=max_seq_len)
        seq_len = input_ids.shape[1]
        last = seq_len - 1

        # Clean pass: the reference S, the reference logits, and the residual
        # norms the dose is expressed in.
        h_clean = _final_residual(model, input_ids, None)
        s_clean = cotangents @ h_clean.float().cpu()  # [16]
        logits_clean = model.unembed(h_clean).float().cpu()
        logp_clean = torch.log_softmax(logits_clean, dim=-1)

        with ActivationRecorder(model.layers, at=list(layers)) as recorder:
            model.forward(input_ids)
            band = {
                l: recorder.activations[l][0].detach().float().cpu() for l in layers
            }
        last_norms = {l: float(band[l][last].norm()) for l in layers}
        mean_norms = {l: float(band[l].norm(dim=-1).mean()) for l in layers}

        # Local pullback under the SAME convention the perturbation uses:
        # source and target positions both the readout token.
        local = pullback_for_prompt(
            model,
            text,
            layers,
            cotangents,
            target_layer=config.fit.target_layer,
            source_positions=[-1],
            target_positions=[-1],
            dim_batch=dim_batch,
            max_seq_len=max_seq_len,
        )
        averaged = {l: averaged_pullback(lens, cotangents, l) for l in layers}

        for layer in layers:
            g_local = local[layer]  # [16, d]
            g_avg = averaged[layer]  # [16, d]

            probes: dict[str, torch.Tensor] = {}
            # Seeded per (prompt, layer) so a re-run reproduces the same probes
            # and a resumed shard is comparable to a fresh one.
            stream = int.from_bytes(
                hashlib.sha256(f"{key}|{layer}".encode()).digest()[:4], "big"
            )
            gen = torch.Generator().manual_seed(spec.seed ^ stream)
            for r in range(spec.n_random):
                v = torch.randn(model.d_model, generator=gen, dtype=torch.float32)
                probes[f"random{r}"] = v / v.norm()
            for target in swap_targets[key]:
                if target not in arg_index:
                    continue
                j = arg_index[target]
                probes[f"local_{target}"] = _unit(g_local[j])
                probes[f"averaged_{target}"] = _unit(g_avg[j])

            for name, direction in probes.items():
                for alpha in spec.alphas:
                    delta = direction * (alpha * last_norms[layer])
                    edit = ResidualEdit({layer: delta}, positions=[last])
                    h = _final_residual(model, input_ids, edit)
                    actual = cotangents @ h.float().cpu() - s_clean  # [16]
                    logits = model.unembed(h).float().cpu()
                    logp = torch.log_softmax(logits, dim=-1)
                    records.append(
                        {
                            "prompt_key": key,
                            "layer": layer,
                            "arm": name,
                            "alpha": alpha,
                            "delta_norm": float(delta.norm()),
                            "last_norm": last_norms[layer],
                            "mean_norm": mean_norms[layer],
                            # The three [16] vectors the analysis compares.
                            "actual": actual,
                            "pred_local": g_local @ delta,
                            "pred_averaged": g_avg @ delta,
                            # Context, not prediction targets: the logit carries
                            # the final norm the lens never linearized.
                            "dlogit": (logits - logits_clean)[
                                [arg_ids[a] for a in args]
                            ],
                            "kl": float(
                                torch.sum(logp.exp() * (logp - logp_clean)).clamp_min(0)
                            ),
                            "greedy": int(logits.argmax()),
                        }
                    )

        if (i + 1) % 8 == 0:
            logger.info(
                "  %d/%d prompts, %d records, %.1f min",
                i + 1,
                len(keys),
                len(records),
                (time.time() - start) / 60,
            )

    payload = {
        "records": records,
        "args": args,
        "arg_ids": [arg_ids[a] for a in args],
        "layers": layers,
        "alphas": spec.alphas,
        "config": config.name,
        "positions": spec.positions,
        "spec": repr(spec),
    }
    torch.save(payload, path)
    logger.info("wrote %s (%d records)", path, len(records))
    return {
        "shard": spec.shard,
        "prompts": len(keys),
        "records": len(records),
        "path": str(path),
        "minutes": round((time.time() - start) / 60, 1),
    }


# --------------------------------------------------------------------------
# C7 - the preconditioning ladder: is the raw pullback a type error?
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PreconditionSpec:
    """C7 - does raising the index rescue the prompt-local direction?

    The experiment the project's conclusion depends on
    -------------------------------------------------
    Every steering arm run so far (C1, C3, C4, C6) writes the **raw** pullback
    g = J^T u. FishBack S2 names that the type error outright: "q is a covector
    (dual vector): converting it into a tangent vector to be added to h
    requires raising the index via a metric", and "the Euclidean steering
    delta_h ~ q corresponds to the special case G = I". So our whole negative
    result about prompt-local directions is a result about G = I, and says
    nothing about any other metric.

    This matters because the nearest prior work found the opposite of what we
    would otherwise conclude: FishBack's headline is that *preconditioned local*
    steering beats CAA -- an averaged direction -- at matched concept
    probability. Concluding "averaging is the crux" without running this ladder
    asserts something that paper contradicts, on evidence that does not address
    it.

    The four rungs, and what each one isolates
    ------------------------------------------
    * ``none``   -- raw g. The G = I case, i.e. everything we have run so far.
    * ``sigma``  -- (Sigma_l + lambda I)^-1 g, Sigma_l the activation covariance
      at that layer over the fitting corpus. Plain whitening: prompt-independent
      and by far the cheapest thing that could work. If this rescues local, no
      Fisher is needed and the story is simply "the residual stream is
      anisotropic".
    * ``gn``     -- (J^T J + lambda I)^-1 g. Gauss-Newton; uses the prompt's own
      Jacobian but no output curvature.
    * ``fisher`` -- (J^T H J + lambda I)^-1 g with H the output Fisher, which is
      FishBack's G exactly.

    Running the ladder on the **averaged** direction too is what separates
    "preconditioning helps any direction" from "preconditioning is specifically
    what local was missing". FishBack has no Sigma or J^T J arm at all -- their
    only isolating comparison is against G = I -- so rungs 2 and 3 are the
    ablation that paper should have run.

    Two approximations, both disclosed
    ----------------------------------
    ``gn`` and ``fisher`` are built from the rank-``k`` SVD of J_x cached by the
    RQ1 sweep, so the preconditioner acts as the identity on the discarded
    complement. That is a damped approximation, not the exact inverse. It is
    defensible here because the measured spectra are concentrated -- median
    participation ratio ~14 out of 4096 -- but it is an approximation and the
    ladder cannot distinguish "the Fisher does not help" from "the top 64
    directions are not where the help lives".

    Damping is set per preconditioner as ``lambda = tau * mean(eigenvalue)``, so
    ``tau`` is scale-free and comparable across rungs and layers. Large tau
    drives every rung back to ``none``, which makes tau a continuous knob from
    "no preconditioning" to "full whitening" rather than a binary.

    Directions are unit-normalized after preconditioning, so the dose is matched
    across rungs by construction and only the *direction* differs.
    """

    config_name: str = "qwen3-8b"
    layers: list[int] | None = None
    mode: str = "both"
    strengths: list[float] = field(
        default_factory=lambda: [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 2.0]
    )
    directions: list[str] = field(default_factory=lambda: ["averaged", "local"])
    preconditioners: list[str] = field(
        default_factory=lambda: ["none", "sigma", "gn", "fisher"]
    )
    #: Damping, as a multiple of the mean eigenvalue of the metric.
    tau: float = 0.1
    swap_mode: str = "clamp"
    #: Wikitext prompts the activation covariance is estimated from.
    n_cov: int = 100
    #: k of the cached rank-k SVD; must match the RQ1 digest directory.
    k: int = 64
    digest_positions: str = "all"
    label: str = ""
    dim_batch: int | None = None
    strict_answers: bool = False
    shard: int = 0
    n_shards: int = 1
    limit: int | None = None
    out_dir: str | None = None
    corpus_cache: str | None = None
    overwrite: bool = False
    device: str | None = None
    positions: str = "all"
    n_fit: int = 0

    def resolve_out(self, leaf: str) -> Path:
        root = Path(self.out_dir) if self.out_dir else REPO_ROOT / "results"
        path = root / leaf
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def tag(self) -> str:
        suffix = f"_{self.label}" if self.label else ""
        return (
            f"{self.config_name}_precond{suffix}_tau{self.tau}"
            f"_shard{self.shard}of{self.n_shards}"
        )


@torch.no_grad()
def activation_covariance(
    model, texts: list[str], layers: Sequence[int], *, max_seq_len: int = 128
) -> dict[int, torch.Tensor]:
    """``E[h h^T]`` at each layer over every position of every text, fp32 CPU.

    The metric for the ``sigma`` rung. Estimated on the *fitting* corpus,
    because that is the distribution the lens was built on and the one whose
    anisotropy the lens's directions inherit.
    """
    d = model.d_model
    total = {layer: torch.zeros(d, d, dtype=torch.float64) for layer in layers}
    count = 0
    for text in texts:
        input_ids = model.encode(text, max_length=max_seq_len)
        with ActivationRecorder(model.layers, at=list(layers)) as recorder:
            model.forward(input_ids)
            for layer in layers:
                h = recorder.activations[layer][0].detach().float().cpu().double()
                total[layer] += h.T @ h
        count += input_ids.shape[1]
    return {layer: (total[layer] / count).float() for layer in layers}


def _damped_inverse_apply(
    g: torch.Tensor, basis: torch.Tensor, middle: torch.Tensor, tau: float
) -> torch.Tensor:
    """Apply ``(R^T M R + lambda I)^-1`` to ``g`` for orthonormal-row ``R``.

    ``basis`` is ``R`` with shape ``[k, d]`` and orthonormal rows; ``middle`` is
    the ``[k, k]`` matrix ``M``. Because ``R R^T = I``, Woodbury collapses to

        (lambda I + R^T M R)^-1 = (1 / lambda) [ I - R^T (lambda M^-1 + I)^-1 R ]

    which is exact for the rank-k metric and costs one k x k solve. Written this
    way rather than as a d x d inverse both for speed and because the rank-k
    metric is singular on the complement -- the damping is what makes it
    invertible at all, and this form makes that explicit.
    """
    k = middle.shape[0]
    eye = torch.eye(k, dtype=middle.dtype)
    lam = tau * float(torch.diagonal(middle).mean().clamp_min(1e-12))
    # lambda M^-1 + I, formed as a solve against M to avoid inverting a middle
    # that may be near-singular in its own right.
    inner = torch.linalg.solve(middle + 1e-9 * eye, lam * eye) + eye
    return (g - basis.T @ torch.linalg.solve(inner, basis @ g)) / lam


def _preconditioned(
    g: torch.Tensor,
    kind: str,
    *,
    sigma_eig: tuple[torch.Tensor, torch.Tensor] | None,
    svd: dict | None,
    fisher_middle: torch.Tensor | None,
    tau: float,
) -> torch.Tensor:
    """One rung of the ladder applied to one pullback."""
    if kind == "none":
        return g
    if kind == "sigma":
        evals, evecs = sigma_eig
        lam = tau * float(evals.mean().clamp_min(1e-12))
        return evecs @ ((evecs.T @ g) / (evals + lam))
    basis = svd["right"].float()  # [k, d], orthonormal rows
    s = svd["singular_values"].float()
    if kind == "gn":
        return _damped_inverse_apply(g, basis, torch.diag(s * s), tau)
    if kind == "fisher":
        return _damped_inverse_apply(g, basis, fisher_middle, tau)
    raise ValueError(f"unknown preconditioner {kind!r}")


def _fisher_middle(
    svd: dict, unembed: torch.Tensor, probs: torch.Tensor
) -> torch.Tensor:
    """``diag(s) U^T H U diag(s)``, the k x k core of ``G = J^T H J``.

    ``H = W_U^T (diag(p) - p p^T) W_U`` is the output Fisher in the final
    residual basis. Never materialized: with ``B = U^T W_U^T`` of shape
    ``[k, vocab]``, ``U^T H U = (B * p) B^T - (B p)(B p)^T``, which is k x k.
    """
    left = svd["left"].float()  # [k, d], rows are left singular vectors
    s = svd["singular_values"].float()
    B = left @ unembed.T.float()  # [k, vocab]
    Bp = B @ probs  # [k]
    core = (B * probs) @ B.T - torch.outer(Bp, Bp)
    return torch.diag(s) @ core @ torch.diag(s)


def run_precondition(spec: PreconditionSpec) -> dict:
    """Steer with every (direction, preconditioner) pair and grade the lot."""
    config, model, lens, layers, dim_batch = _setup(spec)
    max_seq_len = config.fit.max_seq_len
    out = spec.resolve_out("causal")
    path = out / f"precond_{spec.tag}.json"
    if path.exists() and not spec.overwrite:
        logger.info("shard %d already complete; skipping", spec.shard)
        return {"shard": spec.shard, "skipped": True}

    root = Path(spec.out_dir) if spec.out_dir else REPO_ROOT / "results"
    digest_dir = (
        root / "rq1" / "digests" / f"{config.name}_{spec.digest_positions}_k{spec.k}"
    )
    needs_svd = {"gn", "fisher"} & set(spec.preconditioners)
    if needs_svd and not digest_dir.is_dir():
        raise FileNotFoundError(
            f"{sorted(needs_svd)} need the cached rank-{spec.k} SVD at {digest_dir}; "
            "run the RQ1 sweep for this model/positions/k first"
        )

    # Sigma is prompt-independent, so it is estimated once per container and
    # cached on the results volume rather than per shard.
    sigma_eig: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    if "sigma" in spec.preconditioners:
        cov_path = (
            root
            / "c7"
            / f"cov_{config.name}_L{layers[0]}-{layers[-1]}_n{spec.n_cov}.pt"
        )
        cov_path.parent.mkdir(parents=True, exist_ok=True)
        if cov_path.exists():
            cov = torch.load(cov_path, map_location="cpu", weights_only=False)
        else:
            texts = [
                p.text
                for p in build_prompt_set(
                    config,
                    n_fit=spec.n_cov,
                    tokenizer=model.tokenizer,
                    cache_path=Path(spec.corpus_cache) if spec.corpus_cache else None,
                )
                if p.group == "fit"
            ]
            logger.info("estimating activation covariance over %d texts", len(texts))
            cov = activation_covariance(model, texts, layers, max_seq_len=max_seq_len)
            torch.save(cov, cov_path)
        for layer in layers:
            evals, evecs = torch.linalg.eigh(cov[layer].double())
            sigma_eig[layer] = (evals.clamp_min(0).float(), evecs.float())

    unembed = unembedding_matrix(model).detach().float().cpu()

    trials = flexible_generalization_trials()
    if spec.limit:
        trials = trials[: spec.limit]
    trials = take_shard(trials, spec.shard, spec.n_shards)

    def answer_id(m, word):
        return (
            [single_token_id(m, word)]
            if spec.strict_answers
            else answer_variant_ids(m, word)
        )

    records: list[dict] = []
    start = time.time()
    for i, trial in enumerate(trials):
        try:
            ids = {
                "source_arg": single_token_id(model, trial.source_arg),
                "target_arg": single_token_id(model, trial.target_arg),
                "source_answer": answer_id(model, trial.source_answer),
                "target_answer": answer_id(model, trial.target_answer),
            }
        except ValueError:
            continue
        prompt_key = f"{trial.category}/{trial.func}/{trial.source_arg}"
        base = {
            "category": trial.category,
            "func": trial.func,
            "prompt_key": prompt_key,
            "source_arg": trial.source_arg,
            "target_arg": trial.target_arg,
            "target_answer": trial.target_answer,
        }
        norms = mean_residual_norms(
            model, trial.prompt, layers, max_seq_len=max_seq_len
        )
        clean_logits = next_token_logits(model, trial.prompt, max_seq_len=max_seq_len)
        clean = grade(model, clean_logits, target_id=ids["source_answer"])
        clean_text = (
            greedy_continuation(
                model, trial.prompt, n_tokens=spec.gen_tokens, max_seq_len=max_seq_len
            )
            if spec.gen_tokens
            else ""
        )
        records.append(
            {
                **base,
                "arm": "baseline",
                "strength": 0.0,
                **vars(clean),
                "generated": clean_text,
                # Baseline asks whether the model can do the task at all, so it
                # is graded against its *own* answer, not the swap target.
                "hit_generated": bool(clean_text)
                and answer_matches(
                    clean_text,
                    trial.source_answer,
                    digit=_digit_form(trial.source_answer),
                ),
            }
        )

        cotangents = unembedding_rows(model, [ids["source_arg"], ids["target_arg"]])
        raw: dict[str, dict[int, torch.Tensor]] = {
            "averaged": {l: averaged_pullback(lens, cotangents, l) for l in layers}
        }
        if "local" in spec.directions:
            seq_len = model.encode(trial.prompt, max_length=max_seq_len).shape[1]
            result = pullback_for_prompt(
                model,
                trial.prompt,
                layers,
                cotangents,
                target_layer=config.fit.target_layer,
                source_positions=list(range(seq_len)),
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
                skip_first=config.fit.skip_first,
            )
            raw["local"] = {l: result[l] for l in layers}

        svd = None
        fisher_middle: dict[int, torch.Tensor] = {}
        if needs_svd:
            digest = torch.load(
                digest_dir / f"{prompt_key.replace('/', '_')}.pt",
                map_location="cpu",
                weights_only=False,
            )
            svd = {l: digest[l] for l in layers}
            if "fisher" in spec.preconditioners:
                probs = torch.softmax(clean_logits.float(), dim=-1)
                for layer in layers:
                    fisher_middle[layer] = _fisher_middle(svd[layer], unembed, probs)

        # Precondition once per (direction, rung); the two cotangent rows are
        # the swap's source and target and both get the same treatment.
        arms: dict[str, dict[int, torch.Tensor]] = {}
        for source in spec.directions:
            for kind in spec.preconditioners:
                arms[f"{source}_{kind}"] = {
                    layer: torch.stack(
                        [
                            _preconditioned(
                                raw[source][layer][row],
                                kind,
                                sigma_eig=sigma_eig.get(layer),
                                svd=svd[layer] if svd else None,
                                fisher_middle=fisher_middle.get(layer),
                                tau=spec.tau,
                            )
                            for row in (0, 1)
                        ]
                    )
                    for layer in layers
                }

        exclude = [ids["target_arg"], ids["source_arg"]]

        target_digit = _digit_form(trial.target_answer)

        def _gen(edit, _trial=trial, _digit=target_digit) -> dict:
            """Generated continuation and its string-matched verdict."""
            # getattr: PreconditionSpec shares this code shape but has no
            # gen_tokens field, and C7 grades on logits only.
            if not getattr(spec, "gen_tokens", 0):
                return {"generated": "", "hit_generated": False}
            text = greedy_continuation(
                model,
                _trial.prompt,
                edit=edit,
                n_tokens=getattr(spec, "gen_tokens", 0),
                max_seq_len=max_seq_len,
            )
            return {
                "generated": text,
                "hit_generated": answer_matches(
                    text, _trial.target_answer, digit=_digit
                ),
            }

        for strength in spec.strengths:
            for arm, g in arms.items():
                if spec.mode in ("swap", "both"):
                    edit = swap_edit(
                        model,
                        trial.prompt,
                        {l: g[l][0] for l in layers},
                        {l: g[l][1] for l in layers},
                        mode=spec.swap_mode,
                        strength=strength,
                        max_seq_len=max_seq_len,
                    )
                    records.append(
                        {
                            **base,
                            "arm": f"swap_{arm}",
                            "strength": strength,
                            "baseline_ok": clean.hit,
                            **vars(
                                grade(
                                    model,
                                    next_token_logits(
                                        model,
                                        trial.prompt,
                                        edit=edit,
                                        max_seq_len=max_seq_len,
                                    ),
                                    target_id=ids["target_answer"],
                                    source_id=ids["source_answer"],
                                    exclude_ids=exclude,
                                    clean_logits=clean_logits,
                                )
                            ),
                            **_gen(edit),
                        }
                    )
                if spec.mode in ("additive", "both"):
                    edit = additive_edit(
                        {l: g[l][1] for l in layers}, norms, strength=strength
                    )
                    records.append(
                        {
                            **base,
                            "arm": f"additive_{arm}",
                            "strength": strength,
                            "baseline_ok": clean.hit,
                            **vars(
                                grade(
                                    model,
                                    next_token_logits(
                                        model,
                                        trial.prompt,
                                        edit=edit,
                                        max_seq_len=max_seq_len,
                                    ),
                                    target_id=ids["target_answer"],
                                    source_id=ids["source_answer"],
                                    exclude_ids=exclude,
                                    clean_logits=clean_logits,
                                )
                            ),
                            **_gen(edit),
                        }
                    )
        if (i + 1) % 8 == 0:
            logger.info(
                "  %d/%d trials, %d records, %.1f min",
                i + 1,
                len(trials),
                len(records),
                (time.time() - start) / 60,
            )

    path.write_text(json.dumps(records))
    logger.info("wrote %s (%d records)", path, len(records))
    return {
        "shard": spec.shard,
        "n_trials": len(trials),
        "n_records": len(records),
        "arms": sorted(arms),
        "minutes": round((time.time() - start) / 60, 1),
    }


def run_lens_prior(spec: ReadoutSpec) -> dict:
    """Per-token-class norm of the lens vectors. The mechanism behind the
    prior control, and the direct test of token-class coverage.

    Readout ranks token ``y`` by ``<J_bar^T u_y, h>``, so a corpus can appear
    to "read numbers better" two ways: by transporting ``h`` onto the number
    direction (the interesting way), or by giving every numeral's lens vector
    ``J_bar^T u_y`` a large norm, which raises its logit under *any* ``h`` (the
    dull way). :attr:`ReadoutSpec.permute_acts` separates them end-to-end;
    this says which token classes the effect lives on.

    Norms are reported as a ratio to a random-vocabulary baseline computed on
    the same matrix, because ``J_bar`` scale is arbitrary -- ranks are
    invariant to rescaling, so only the *relative* inflation of one class over
    the vocabulary at large can move a rank.
    """
    config, model, lens, layers, _ = _setup(spec)
    W_U = unembedding_matrix(model).float().cpu()

    # Classes come from the eval set itself, so "months" means exactly the
    # month tokens the readout eval scores, not a hand-written list that might
    # tokenize differently.
    classes: dict[str, set[int]] = {}
    for prompt in build_prompt_set(config, n_fit=0, tokenizer=model.tokenizer):
        if prompt.base is None:
            continue
        try:
            classes.setdefault(prompt.base.category, set()).add(
                single_token_id(model, prompt.base.arg)
            )
        except ValueError:
            continue
    rng = torch.Generator().manual_seed(0)
    classes["_random"] = set(
        torch.randint(0, W_U.shape[0], (2048,), generator=rng).tolist()
    )
    logger.info("classes: %s", {k: len(v) for k, v in classes.items()})

    banks: dict[str, dict[int, torch.Tensor]] = {
        "published": {ll: lens.jacobians[ll].float() for ll in layers}
    }
    for raw in spec.means_paths:
        blob = torch.load(raw, map_location="cpu", weights_only=False)
        for key, mean in blob["means"].items():
            group, layer_str = key.split("|")
            if spec.groups and group not in spec.groups:
                continue
            if int(layer_str) in layers:
                banks.setdefault(group, {})[int(layer_str)] = mean.float()
        del blob
    missing = [g for g in spec.groups if g not in banks]
    if missing:
        raise ValueError(f"no full means for {missing}; loaded {sorted(banks)}")

    records: list[dict] = []
    for group, per_layer in banks.items():
        for ll in layers:
            J = per_layer[ll]
            norms: dict[str, float] = {}
            for name, ids in classes.items():
                rows = W_U[sorted(ids)]  # [n, d_model]
                # (J^T u) for a batch of u is (U @ J), one matmul.
                norms[name] = float((rows @ J).norm(dim=1).mean())
            base = norms["_random"] or 1.0
            records.append(
                {
                    "lens": group,
                    "layer": ll,
                    "mean_norm": norms,
                    "ratio_to_random": {
                        k: round(v / base, 4) for k, v in norms.items()
                    },
                }
            )
            logger.info(
                "%-12s L%-3d %s",
                group,
                ll,
                {k: round(v / base, 3) for k, v in norms.items() if k != "_random"},
            )

    out = spec.resolve_out("readout")
    path = out / f"lens_prior_{spec.tag}.json"
    path.write_text(json.dumps(records))
    return {
        "n_records": len(records),
        "lenses": sorted(banks),
        "classes": {k: len(v) for k, v in classes.items()},
        "path": str(path),
    }


@dataclass(frozen=True)
class EnergySpec(SweepSpec):
    """Where each corpus puts its Jacobian energy. The mechanism arm.

    The corpus result so far is a black box: a J_bar fitted on GSM8K recovers
    present tokens far better and latent concepts somewhat worse, and none of
    the seven content ablations explained it. Yan et al. (arXiv 2608.25347)
    supply the vocabulary the explanation needs: their Sec. 3.3 splits the
    lens's readout into a short-horizon component, carried by energy on the
    ``t ~ t'`` diagonal, and a sparse-concept component carried by off-diagonal
    stripes -- exactly the two axes the corpus result moves in opposite
    directions.

    So the hypothesis is that GSM8K's *documents* concentrate Jacobian energy
    near the diagonal, and averaging over them yields a J_bar dominated by
    short-horizon structure. That is a property of the corpus, measurable
    before any lens is fitted and independent of every eval in this repo.

    Note this measures the energy of prompt-local Jacobians over corpus
    documents, which is not the same object as the fitted J_bar; the claim
    being tested is that the former predicts the behaviour of the latter.
    """

    #: Corpus names from :mod:`jsteer.domains`. Must exist in one of the panels.
    corpora: list[str] = field(default_factory=lambda: ["gsm8k", "wikitext_a"])
    n_docs: int = 24
    #: Hutchinson probes per target position. The estimator is unbiased at any
    #: value; this trades variance against backward passes.
    n_probe: int = 8
    #: Target positions are every ``t_stride``-th valid position. The full set
    #: would be ~112 backward passes per document per probe batch.
    t_stride: int = 8

    @property
    def namespace(self) -> str:
        tag = hashlib.sha256(
            f"{sorted(self.corpora)}|{self.n_docs}|{self.n_probe}|{self.t_stride}"
            f"|{self.layers}".encode()
        ).hexdigest()[:8]
        return f"{super().namespace}_energy_{tag}"


def run_energy_map(spec: EnergySpec) -> dict:
    """Offset profile of the Jacobian energy, per corpus and layer."""
    from jsteer import domains
    from jsteer.domains import DomainSpec, load_domain_corpora
    from jsteer.jacobian import energy_map_for_prompt

    # Every panel the module defines, not a hand-picked two: the first full
    # run died instantly on "unknown corpora" because the ablations live in
    # their own tuple, and a spawned call's error only surfaces to a client
    # that is still waiting on it.
    known = {
        s.name: s
        for value in vars(domains).values()
        if isinstance(value, tuple) and value and isinstance(value[0], DomainSpec)
        for s in value
    }
    missing = [c for c in spec.corpora if c not in known]
    if missing:
        raise ValueError(f"unknown corpora {missing}; have {sorted(known)}")

    config, model, _, layers, _ = _setup(spec)
    max_seq_len = config.fit.max_seq_len
    skip_first = config.fit.skip_first

    corpora = load_domain_corpora(
        [known[c] for c in spec.corpora],
        tokenizer=model.tokenizer,
        n_per=spec.n_docs,
        min_tokens=max_seq_len,
        cache_path=spec.corpus_cache,
    )
    # The guard this repo keeps relearning: a short group silently becomes a
    # smaller-N average and nothing errors. Check counts, not exit codes.
    for name, texts in corpora.items():
        if len(texts) != spec.n_docs:
            raise ValueError(f"{name}: {len(texts)} documents, wanted {spec.n_docs}")

    records: list[dict] = []
    start = time.time()
    for name in spec.corpora:
        for doc_index, text in enumerate(corpora[name]):
            targets = list(range(skip_first, max_seq_len, spec.t_stride))
            energies, targets, seq_len = energy_map_for_prompt(
                model,
                text,
                layers,
                target_positions=targets,
                n_probe=spec.n_probe,
                max_seq_len=max_seq_len,
                skip_first=skip_first,
                seed=doc_index,
            )
            for layer, matrix in energies.items():
                # Bin by offset d = t' - t. Causality makes d < 0 exactly zero;
                # asserting it here is the cheapest possible check that the
                # cotangent landed at the position we think it did.
                by_offset: dict[int, float] = collections.defaultdict(float)
                leak = 0.0
                for j, t_prime in enumerate(targets):
                    for t in range(skip_first, seq_len):
                        value = float(matrix[t, j])
                        if t > t_prime:
                            leak += value
                        else:
                            by_offset[t_prime - t] += value
                if leak > 1e-6 * (sum(by_offset.values()) or 1.0):
                    raise RuntimeError(
                        f"{name}/{doc_index} L{layer}: {leak:.3g} energy at t>t', "
                        "which causality forbids -- the probe is landing wrong"
                    )
                total = sum(by_offset.values()) or 1.0

                # Offset share alone cannot tell a stripe from a smear: both
                # put mass off the diagonal, but Yan et al.'s sparse-concept
                # claim is specifically that a FEW source positions carry it,
                # which is what their top-10%-by-energy filter presupposes. So
                # also record the participation ratio over source positions --
                # the effective number of t contributing to each t' -- with the
                # diagonal removed, since the diagonal is the other component
                # and would otherwise dominate the count.
                ratios: list[float] = []
                for j, t_prime in enumerate(targets):
                    column = matrix[skip_first : t_prime + 1, j].double()
                    if column.numel() > 1:
                        column = column[:-1]  # drop t == t_prime
                    mass, energy2 = float(column.sum()), float(column.pow(2).sum())
                    if mass > 0 and energy2 > 0:
                        ratios.append(mass * mass / energy2)
                # The other half of Yan et al.'s sparse-concept pattern (Sec.
                # 3.3): "broadcast origins", one SOURCE t feeding many targets.
                # The ratio above is the "integration" half (one target, many
                # sources). Per source row over the sampled targets t' > t, so
                # it is relative across corpora, not an absolute count.
                broadcast: list[float] = []
                for t in range(skip_first, seq_len):
                    row = torch.tensor(
                        [
                            float(matrix[t, j])
                            for j, t_prime in enumerate(targets)
                            if t_prime > t
                        ],
                        dtype=torch.float64,
                    )
                    mass, energy2 = float(row.sum()), float(row.pow(2).sum())
                    if row.numel() > 1 and mass > 0 and energy2 > 0:
                        broadcast.append(mass * mass / energy2 / row.numel())
                records.append(
                    {
                        "corpus": name,
                        "doc": doc_index,
                        "layer": layer,
                        "seq_len": seq_len,
                        "total_energy": total,
                        # Effective fraction of later targets each source feeds.
                        "broadcast_pr_frac": round(statistics.mean(broadcast), 4)
                        if broadcast
                        else None,
                        # Cumulative share within d, the short-horizon measure.
                        "share": {
                            str(d): round(
                                sum(v for k, v in by_offset.items() if k <= d) / total,
                                5,
                            )
                            for d in (0, 1, 2, 4, 8, 16, 32)
                        },
                        # Effective number of off-diagonal source positions.
                        "offdiag_pr": round(statistics.mean(ratios), 4)
                        if ratios
                        else None,
                        # Same thing as a fraction of the positions available,
                        # so documents of different length stay comparable.
                        "offdiag_pr_frac": round(
                            statistics.mean(
                                r / max(t - skip_first, 1)
                                for r, t in zip(ratios, targets, strict=False)
                            ),
                            4,
                        )
                        if ratios
                        else None,
                    }
                )
        logger.info("%-12s done  %.0fs", name, time.time() - start)

    out = spec.resolve_out("energy")
    path = out / f"energy_{spec.tag}.json"
    path.write_text(json.dumps(records))

    summary: dict[str, dict] = {}
    for name in spec.corpora:
        for layer in layers:
            sub = [r for r in records if r["corpus"] == name and r["layer"] == layer]
            summary[f"{name}|{layer}"] = {
                **{
                    d: round(statistics.mean(r["share"][d] for r in sub), 4)
                    for d in ("0", "2", "8")
                },
                "pr": round(
                    statistics.mean(
                        r["offdiag_pr"] for r in sub if r["offdiag_pr"] is not None
                    ),
                    2,
                ),
                "pr_frac": round(
                    statistics.mean(
                        r["offdiag_pr_frac"]
                        for r in sub
                        if r["offdiag_pr_frac"] is not None
                    ),
                    4,
                ),
            }
    logger.info("energy shares: %s", summary)
    return {
        "n_records": len(records),
        "corpora": spec.corpora,
        "layers": layers,
        "summary": summary,
        "path": str(path),
        "elapsed_s": round(time.time() - start, 1),
    }


@dataclass(frozen=True)
class ComponentSpec(DomainPullbackSpec):
    """C18 - steer with each half of Yan et al.'s Eq. 20 decomposition.

    Their split of the averaged Jacobian into a ``t' = t`` diagonal term
    (short-horizon, next-token) and a ``t' > t`` term (influence on what is
    emitted later) is only ever evaluated as a *readout* filter, in their
    Table 1. Writing with each half separately is the arm they do not have,
    and it is the one this project is placed to run.

    It is also a second, independent test of the re-mention account from C17.
    If a corpus helps because it teaches "a token here will be said again
    later", the useful signal lives in the ``t' > t`` term, and the diagonal
    should steer badly however good the corpus is. The sharper prediction is
    about wikitext: if its off-diagonal term is simply swamped by diagonal
    mass, then *filtering* wikitext should recover part of GSM8K's advantage
    with no change of corpus at all -- which would make the finding about
    which position pairs are averaged rather than about subject matter.

    Emits three arms per corpus -- ``_full``, ``_diag``, ``_off`` -- in the
    causal arm's own digest format, so the existing swap eval steers them with
    no new evaluation code. ``_full`` is not redundant: it must reproduce the
    plain domain digest, and it is the within-run baseline the halves are read
    against.

    Costs ``n_valid`` backward passes per prompt (~111) rather than one, so a
    two-corpus 32-document panel is ~45 minutes, not ~4.
    """

    #: Average over every ``t_stride``-th target position. 1 is the exact
    #: estimator; larger values trade variance for passes and are needed once
    #: the target-word list is long.
    t_stride: int = 1

    @property
    def namespace(self) -> str:
        base = f"{super().namespace}_comp"
        return base if self.t_stride == 1 else f"{base}_s{self.t_stride}"


def run_component_pullbacks(spec: ComponentSpec) -> dict:
    """Average each Eq. 20 component's write directions within each corpus."""
    from jsteer.jacobian import component_pullbacks_for_prompt

    config, model, lens, layers, dim_batch = _setup(spec)
    targets = resolve_targets(model, spec.target_words)
    prompts = _prompt_shard(spec, config, model)
    out = spec.resolve_out("rq1")
    out.mkdir(parents=True, exist_ok=True)

    if spec.target_words and len(targets.words) != len(spec.target_words):
        dropped = sorted(set(spec.target_words) - set(targets.words))
        raise RuntimeError(
            f"{len(dropped)} target words are not single-token and were dropped: "
            f"{dropped}. Every argument of an eval must survive tokenization."
        )
    # The primitive uses the lens's own position convention throughout, which
    # is right for corpus documents and wrong for short eval prompts. Refuse
    # rather than silently average over a different position set.
    stray = [p.key for p in prompts if not p.fit_positions]
    if stray:
        raise RuntimeError(f"{len(stray)} prompts need explicit positions: {stray[:3]}")

    groups = sorted({p.group for p in prompts})
    per_group = collections.Counter(p.group for p in prompts)
    short = {g: n for g, n in per_group.items() if n != spec.n_per}
    if short:
        raise RuntimeError(f"expected {spec.n_per} prompts per corpus; got {short}.")

    parts_wanted = ("full", "diag", "off")
    means = {
        (f"{g}_{part}", layer): RunningMean()
        for g in groups
        for part in parts_wanted
        for layer in layers
    }
    logger.info(
        "%d prompts over %d corpora x %d layers, %d targets, %d arms",
        len(prompts),
        len(groups),
        len(layers),
        len(targets.words),
        len(means),
    )
    start = time.time()
    for i, prompt in enumerate(prompts):
        parts = component_pullbacks_for_prompt(
            model,
            prompt.text,
            layers,
            targets.cotangents,
            target_layer=config.fit.target_layer,
            dim_batch=dim_batch,
            max_seq_len=config.fit.max_seq_len,
            skip_first=config.fit.skip_first,
            t_stride=spec.t_stride,
        )
        for layer in layers:
            means[(f"{prompt.group}_full", layer)].add(parts[layer]["total"])
            means[(f"{prompt.group}_diag", layer)].add(parts[layer]["diag"])
            means[(f"{prompt.group}_off", layer)].add(parts[layer]["off"])
        if (i + 1) % 5 == 0 or i + 1 == len(prompts):
            rate = (time.time() - start) / (i + 1)
            logger.info(
                "  %3d/%d  %.1fs/prompt  eta %.0fs",
                i + 1,
                len(prompts),
                rate,
                rate * (len(prompts) - i - 1),
            )

    digests = {}
    for (group, layer), mean in means.items():
        if mean.total is None:
            raise RuntimeError(f"no prompts contributed to {group} at layer {layer}")
        digests[f"{group}|{layer}"] = {
            "group": group,
            "layer": layer,
            "n": mean.n,
            "pullbacks": (mean.total / mean.n).float(),
        }
    path = out / f"group_mean_digests_{spec.namespace}.pt"
    torch.save(
        {"digests": digests, "targets": targets.words, "pullback_only": True}, path
    )
    # Reported so the halves can be read as magnitudes, not only as ranks: a
    # component that steers worse because its write vector is shorter is a
    # different fact from one whose direction is wrong.
    norms = {
        group: round(
            float(
                torch.stack(
                    [digests[f"{group}|{ll}"]["pullbacks"].norm(dim=1).mean() for ll in layers]
                ).mean()
            ),
            4,
        )
        for group in sorted({d["group"] for d in digests.values()})
    }
    logger.info("mean write-vector norms: %s", norms)
    return {
        "n_prompts": len(prompts),
        "per_group": dict(per_group),
        "arms": sorted({d["group"] for d in digests.values()}),
        "layers": layers,
        "mean_norms": norms,
        "namespace": spec.namespace,
        "path": str(path),
        "elapsed_s": round(time.time() - start, 1),
    }


def run_direction_geometry(spec: AverageNSpec) -> dict:
    """C21 - how much of each write direction is just the unembedding row?

    The C18 result -- dropping the ``t' = t`` term of Eq. 20 roughly doubles
    swap success at *lower* KL and half the collateral -- has no mechanism
    attached to it yet. This is the cheapest candidate, and if it holds it
    explains the whole panel at once.

    The residual stream is a sum, so the ``t' = t`` block of
    ``d h_L,t / d h_l,t`` carries an identity path straight through. If that
    path dominates the diagonal term, then ``J_diag^T u_y ~ u_y`` up to scale:
    the short-horizon half of the J-lens write direction is approximately
    *logit-lens steering*. Under that reading the full lens writes a blend of
    "say y next" and "be in a state that leads to y", and ``_off`` is the
    second one alone -- which would account for the lower KL (it is not
    shoving mass onto one token), the halved collateral, the lower ``said``
    rate on the two-hop eval, and ``logit_lens`` being the worst arm there.

    Per arm, layer and target word:

    * ``cos_self`` -- cos(v_y, u_y), the quantity above, signed.
    * ``cos_null`` -- |cos| against unembedding rows of random other tokens.
      The rows are not orthogonal and d is large, so ``cos_self`` means
      nothing without the level it has to clear.
    * ``norm`` -- ||v_y||, so "points elsewhere" stays distinguishable from
      "same direction, shorter vector".

    Plus cross-arm cosines (full/diag, full/off, diag/off), which say how much
    of the full lens's direction the diagonal was supplying in the first place.

    Reuses ``AverageNSpec`` for plumbing only; it needs ``directions_paths``.
    Pure linear algebra over a cached digest -- no prompts, no backward passes.
    """
    config, model, lens, layers, _ = _setup(spec)

    arm_banks: dict[str, dict[int, torch.Tensor]] = {}
    words: list[str] = []
    for raw in spec.directions_paths:
        blob = torch.load(raw, map_location="cpu", weights_only=False)
        if words and list(blob["targets"]) != words:
            raise RuntimeError(
                f"{raw} has a different target-word list; refusing to index "
                "two digests by one word order"
            )
        words = list(blob["targets"])
        for entry in blob["digests"].values():
            if spec.arms and entry["group"] not in spec.arms:
                continue
            if entry["layer"] in layers:
                arm_banks.setdefault(entry["group"], {})[entry["layer"]] = entry[
                    "pullbacks"
                ].float()
        del blob
    if not arm_banks:
        raise ValueError("no arms loaded; directions_paths is required here")
    absent = [a for a in spec.arms if a not in arm_banks]
    if absent:
        raise ValueError(f"no arms {absent}; loaded {sorted(arm_banks)}")

    # The recurring bug class: check counts, not exit codes. A digest missing
    # a layer would otherwise produce a table that silently averages over a
    # different band for one arm than for another.
    for arm, bank in sorted(arm_banks.items()):
        missing = [ll for ll in layers if ll not in bank]
        if missing:
            raise RuntimeError(f"arm {arm!r} is missing layers {missing}")
        bad = {ll: tuple(bank[ll].shape) for ll in layers if bank[ll].shape[0] != len(words)}
        if bad:
            raise RuntimeError(f"arm {arm!r} has {bad} rows against {len(words)} words")

    token_ids = [single_token_id(model, w) for w in words]
    U = unembedding_rows(model, token_ids)
    Un = U / U.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    # The published lens gets a row too: it is the arm every C18 table is read
    # against, and its pullback is built here rather than loaded so that the
    # cotangent convention is identical to the digests' by construction.
    arm_banks["published"] = {
        ll: averaged_pullback(lens, U, ll).float() for ll in layers
    }

    # Null level. Sampled from the whole vocabulary rather than from the 16
    # targets, which are a semantically narrow set and would understate it.
    rng = torch.Generator().manual_seed(spec.seed)
    vocab = int(unembedding_matrix(model).shape[0])
    null_ids = torch.randint(0, vocab, (256,), generator=rng).tolist()
    N = unembedding_rows(model, null_ids)
    Nn = N / N.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    records: list[dict] = []
    for arm in sorted(arm_banks):
        for ll in layers:
            V = arm_banks[arm][ll]
            Vn = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            cos_self = (Vn * Un).sum(-1)
            cos_null = (Vn @ Nn.T).abs().mean(-1)
            norms = V.norm(dim=-1)
            for i, word in enumerate(words):
                records.append(
                    {
                        "arm": arm,
                        "layer": ll,
                        "word": word,
                        "cos_self": round(float(cos_self[i]), 5),
                        "cos_null": round(float(cos_null[i]), 5),
                        "norm": round(float(norms[i]), 5),
                    }
                )

    # Cross-arm: how much of the full direction did the diagonal supply?
    cross: list[dict] = []
    corpora = sorted(
        {a.rsplit("_", 1)[0] for a in arm_banks if a.rsplit("_", 1)[-1] in ("full", "diag", "off")}
    )
    for corpus in corpora:
        for left, right in (("full", "diag"), ("full", "off"), ("diag", "off")):
            a, b = f"{corpus}_{left}", f"{corpus}_{right}"
            if a not in arm_banks or b not in arm_banks:
                continue
            for ll in layers:
                A = arm_banks[a][ll]
                B = arm_banks[b][ll]
                cos = (A * B).sum(-1) / (A.norm(dim=-1) * B.norm(dim=-1)).clamp_min(1e-12)
                for i, word in enumerate(words):
                    cross.append(
                        {
                            "corpus": corpus,
                            "pair": f"{left}_vs_{right}",
                            "layer": ll,
                            "word": word,
                            "cos": round(float(cos[i]), 5),
                        }
                    )

    out = spec.resolve_out("geometry")
    leaf = f"geometry_{spec.config_name}{'_' + spec.label if spec.label else ''}.json"
    path = out / leaf
    path.write_text(json.dumps({"records": records, "cross": cross}, indent=1))
    logger.info("%d records, %d cross rows -> %s", len(records), len(cross), path)

    # Logged so the run is readable from the Modal console without fetching.
    for arm in sorted(arm_banks):
        rows = [r for r in records if r["arm"] == arm]
        logger.info(
            "  %-20s cos_self %+.4f   null %.4f   norm %.4f",
            arm,
            statistics.mean(r["cos_self"] for r in rows),
            statistics.mean(r["cos_null"] for r in rows),
            statistics.mean(r["norm"] for r in rows),
        )

    return {
        "arms": sorted(arm_banks),
        "layers": layers,
        "n_words": len(words),
        "n_records": len(records),
        "path": str(path),
    }


@torch.no_grad()
def _continuation_nll(model, prompt: str, text: str, *, max_seq_len: int) -> float | None:
    """Mean NLL of ``text`` given ``prompt``, under the UNSTEERED model.

    The steered model is not the judge of its own output: we ask how surprising
    the continuation is to the *clean* model, which is what "did the
    intervention push it off-distribution" means. Teacher-forced, so it does
    not re-decode.
    """
    p_ids = model.encode(prompt, max_length=max_seq_len)
    c_ids = model.tokenizer.encode(text, add_special_tokens=False)
    if not c_ids:
        return None
    ids = torch.cat(
        [p_ids, torch.tensor([c_ids], device=p_ids.device, dtype=p_ids.dtype)], dim=1
    )
    final_layer = model.n_layers - 1
    with ActivationRecorder(model.layers, at=[final_layer]) as recorder:
        model.forward(ids)
        h = recorder.activations[final_layer][0].detach()
    logprobs = torch.log_softmax(model.unembed(h).float(), dim=-1)
    n_p = p_ids.shape[1]
    targets = ids[0, n_p:]
    # Position t's logits predict token t+1, so the continuation's first token
    # is predicted from the prompt's last position.
    picked = logprobs[n_p - 1 : -1].gather(1, targets[:, None]).squeeze(1)
    return float(-picked.mean())


def _degeneracy(model, text: str) -> dict:
    """Repetition statistics of a continuation, on token ids not characters."""
    ids = model.tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return {"n_tokens": 0, "distinct1": None, "rep4": None, "max_run": 0}
    grams = [tuple(ids[i : i + 4]) for i in range(len(ids) - 3)]
    seen: set = set()
    repeats = 0
    for g in grams:
        if g in seen:
            repeats += 1
        seen.add(g)
    run = best = 1
    for a, b in zip(ids, ids[1:], strict=False):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return {
        "n_tokens": len(ids),
        "distinct1": round(len(set(ids)) / len(ids), 4),
        "rep4": round(repeats / len(grams), 4) if grams else None,
        "max_run": best,
    }


def run_component_readout(spec: AverageNSpec) -> dict:
    """C24 - does dropping the diagonal term cost anything on the READ side?

    The measurement the read/write asymmetry currently rests on two other
    papers for. Yan et al. masked the diagonal and reported that readout
    degraded (their SHL and ICR both decline); we mask it and report that
    writing roughly doubles. Those are different papers, different models and
    different metrics, so the asymmetry is presently an inference across
    studies rather than an observation. This measures both sides of it on one
    lens, one model and one prompt set.

    **This is a restricted-vocabulary readout, and that limitation is real.**
    The paper's readout is ``unembed(J_bar h)`` scored as a rank over the whole
    vocabulary, which needs the full mean matrices; the component sweep only
    ever materialized per-word pullbacks, because the full matrices would cost
    ``d_model`` backward passes per position pair instead of 16. So the rank
    here is among the 16 argument words rather than among 150k tokens. That
    still answers the comparative question -- every arm is scored on the
    identical 16-way choice, so "does removing the diagonal hurt reading"
    is answerable -- but the numbers are not comparable to C14's pass@k and
    must not be quoted as if they were.

    Scored at the argument and readout positions of the clean forward pass,
    over the same band the arms were fitted at, exactly as C14 and the loading
    control do.
    """
    config, model, lens, layers, _ = _setup(spec)
    max_seq_len = config.fit.max_seq_len
    prompts = list(build_prompt_set(config, n_fit=0, tokenizer=model.tokenizer))

    banks: dict[str, dict[int, torch.Tensor]] = {}
    words: list[str] = []
    for raw in spec.directions_paths:
        blob = torch.load(raw, map_location="cpu", weights_only=False)
        if words and list(blob["targets"]) != words:
            raise RuntimeError(f"{raw} disagrees on the target-word list")
        words = list(blob["targets"])
        for entry in blob["digests"].values():
            if spec.arms and entry["group"] not in spec.arms:
                continue
            if entry["layer"] in layers:
                banks.setdefault(entry["group"], {})[entry["layer"]] = entry[
                    "pullbacks"
                ].float()
        del blob
    absent = [a for a in spec.arms if a not in banks]
    if absent:
        raise ValueError(f"no arms {absent}; loaded {sorted(banks)}")

    token_ids = [single_token_id(model, w) for w in words]
    banks["published"] = {
        ll: averaged_pullback(lens, unembedding_rows(model, token_ids), ll).float()
        for ll in layers
    }
    for arm, bank in sorted(banks.items()):
        missing = [ll for ll in layers if ll not in bank]
        if missing:
            raise RuntimeError(f"arm {arm!r} is missing layers {missing}")

    word_index = {w: i for i, w in enumerate(words)}
    records: list[dict] = []
    for prompt in prompts:
        if prompt.base is None or prompt.base.arg not in word_index:
            continue
        gold = word_index[prompt.base.arg]
        span = arg_span(model, prompt, max_seq_len=max_seq_len)
        input_ids = model.encode(prompt.text, max_length=max_seq_len)
        seq_len = input_ids.shape[1]
        positions = sorted(set(range(*span) if span else ()) | {seq_len - 1})

        with ActivationRecorder(model.layers, at=list(layers)) as recorder:
            model.forward(input_ids)
            acts = {ll: recorder.activations[ll][0].float().cpu() for ll in layers}

        for arm, bank in banks.items():
            for ll in layers:
                # <v_y, h> for all 16 candidates, averaged over the positions.
                scores = (acts[ll][positions] @ bank[ll].T).mean(dim=0)
                rank = int((scores > scores[gold]).sum())
                records.append(
                    {
                        "arm": arm,
                        "layer": ll,
                        "key": prompt.key,
                        "category": prompt.base.category,
                        "word": prompt.base.arg,
                        "rank": rank,
                        "top1": rank == 0,
                        "rr": round(1.0 / (rank + 1), 5),
                    }
                )

    out = spec.resolve_out("readout")
    leaf = f"component_readout_{spec.config_name}{'_' + spec.label if spec.label else ''}.json"
    path = out / leaf
    path.write_text(json.dumps(records, indent=1))

    n_prompts = len({r["key"] for r in records})
    expected = n_prompts * len(banks) * len(layers)
    if len(records) != expected:
        raise RuntimeError(f"{len(records)} records, expected {expected}")
    logger.info("%d prompts x %d arms x %d layers", n_prompts, len(banks), len(layers))
    for arm in sorted(banks):
        rows = [r for r in records if r["arm"] == arm]
        logger.info(
            "  %-24s top1 %.3f  MRR %.3f",
            arm,
            statistics.mean(r["top1"] for r in rows),
            statistics.mean(r["rr"] for r in rows),
        )
    return {
        "arms": sorted(banks),
        "layers": layers,
        "n_prompts": n_prompts,
        "n_candidates": len(words),
        "n_records": len(records),
        "path": str(path),
    }


def run_fluency(spec: CausalSpec) -> dict:
    """C23 - does dropping the diagonal term damage generation?

    The reviewer's first question, and the one this project currently cannot
    answer. Every existing metric -- swap success, KL, collateral -- is read
    off a single graded token, so none of them sees whether the text stays
    coherent. There is a specific reason to expect damage: Yan et al. describe
    the ``t' = t`` term as the component where a position "primarily preserves
    or prepares the next token", and the ``_off`` arm deletes it. The J-Lens
    paper separately reports that J-space ablation "tends to impair the
    coherence of responses".

    Both outcomes are publishable and they are different papers. If fluency
    holds, the short-horizon component is pure cost for writing. If it drops,
    there is a trade-off curve to characterize. Publishing a doubled success
    rate with no fluency measurement is the one option that is not available.

    Same swap as the causal arm, at the same layers and strength, but decoding
    ``gen_tokens`` rather than four and scoring the text three ways:

    * ``nll`` -- mean negative log-likelihood of the continuation under the
      **clean** model. The steered model does not grade its own output.
    * ``rep4`` -- share of repeated 4-grams; ``max_run`` -- longest run of one
      repeated token. These catch the classic steering failure that perplexity
      can miss, where text is locally fluent and globally stuck.
    * ``distinct1`` -- type/token ratio.

    ``baseline`` is the unsteered continuation of the same prompt, so every
    arm is read as a delta against the model's own text rather than against an
    absolute scale.
    """
    config, model, lens, layers, dim_batch = _setup(spec)
    max_seq_len = config.fit.max_seq_len
    out = spec.resolve_out("fluency")

    trials = flexible_generalization_trials()
    if spec.limit:
        trials = trials[: spec.limit]
    trials = take_shard(trials, spec.shard, spec.n_shards)

    domain: dict[str, dict[int, torch.Tensor]] = {}
    domain_words: list[str] = []
    for raw in spec.directions_paths:
        blob = torch.load(raw, map_location="cpu", weights_only=False)
        words = list(blob["targets"])
        if domain_words and words != domain_words:
            raise ValueError(f"{raw}: target order disagrees with {domain_words}")
        domain_words = words
        for key, entry in blob["digests"].items():
            group, layer = key.split("|")
            if int(layer) in layers:
                domain.setdefault(group, {})[int(layer)] = entry["pullbacks"].float()
        del blob
    missing = [d for d in spec.directions if d != "averaged" and d not in domain]
    if missing:
        raise ValueError(f"no directions for {missing}; loaded {sorted(domain)}")

    logger.info(
        "shard %d/%d: %d trials x %d arms x %d strengths, %d tokens each",
        spec.shard,
        spec.n_shards,
        len(trials),
        len(spec.directions),
        len(spec.strengths),
        spec.gen_tokens,
    )
    records: list[dict] = []
    start = time.time()
    for i, trial in enumerate(trials):
        try:
            ids = {
                "source_arg": single_token_id(model, trial.source_arg),
                "target_arg": single_token_id(model, trial.target_arg),
            }
        except ValueError:
            continue
        # SwapTrial has no id field; (category, func, source, target) is the
        # unique key of a trial in this set and is what the record is joined on.
        key = f"{trial.category}/{trial.func}/{trial.source_arg}->{trial.target_arg}"
        base = {
            "key": key,
            "category": trial.category,
            "func": trial.func,
            "prompt": trial.prompt,
            "source_arg": trial.source_arg,
            "target_arg": trial.target_arg,
        }

        clean_text = greedy_continuation(
            model, trial.prompt, n_tokens=spec.gen_tokens, max_seq_len=max_seq_len
        )
        records.append(
            {
                **base,
                "arm": "baseline",
                "strength": 0.0,
                "generated": clean_text,
                "nll": _continuation_nll(
                    model, trial.prompt, clean_text, max_seq_len=max_seq_len
                ),
                **_degeneracy(model, clean_text),
            }
        )

        cotangents = unembedding_rows(model, [ids["source_arg"], ids["target_arg"]])
        arms: dict[str, dict[int, torch.Tensor]] = {
            "averaged": {ll: averaged_pullback(lens, cotangents, ll) for ll in layers}
        }
        if domain:
            si = domain_words.index(trial.source_arg)
            ti = domain_words.index(trial.target_arg)
            for group, per_layer in domain.items():
                arms[group] = {
                    ll: torch.stack([per_layer[ll][si], per_layer[ll][ti]])
                    for ll in layers
                }

        for strength in spec.strengths:
            for source in spec.directions:
                g = arms[source]
                edit = swap_edit(
                    model,
                    trial.prompt,
                    {ll: g[ll][0] for ll in layers},
                    {ll: g[ll][1] for ll in layers},
                    mode=spec.swap_mode,
                    strength=strength,
                    max_seq_len=max_seq_len,
                )
                text = greedy_continuation(
                    model,
                    trial.prompt,
                    edit=edit,
                    n_tokens=spec.gen_tokens,
                    max_seq_len=max_seq_len,
                )
                records.append(
                    {
                        **base,
                        "arm": source,
                        "strength": strength,
                        "generated": text,
                        "nll": _continuation_nll(
                            model, trial.prompt, text, max_seq_len=max_seq_len
                        ),
                        **_degeneracy(model, text),
                    }
                )
        if (i + 1) % 5 == 0 or i + 1 == len(trials):
            rate = (time.time() - start) / (i + 1)
            logger.info(
                "  %3d/%d  %.1fs/trial  eta %.0fs",
                i + 1,
                len(trials),
                rate,
                rate * (len(trials) - i - 1),
            )

    path = out / f"fluency_{spec.tag}.json"
    path.write_text(json.dumps(records, indent=1))

    expected = len({r["key"] for r in records}) * (
        1 + len(spec.directions) * len(spec.strengths)
    )
    if len(records) != expected:
        raise RuntimeError(f"{len(records)} records, expected {expected}")
    for arm in ["baseline", *spec.directions]:
        rows = [r for r in records if r["arm"] == arm and r["nll"] is not None]
        if rows:
            logger.info(
                "  %-24s nll %.3f  rep4 %.3f  max_run %.1f",
                arm,
                statistics.mean(r["nll"] for r in rows),
                statistics.mean(r["rep4"] for r in rows if r["rep4"] is not None),
                statistics.mean(r["max_run"] for r in rows),
            )
    return {
        "shard": spec.shard,
        "n_records": len(records),
        "n_trials": len({r["key"] for r in records}),
        "arms": ["baseline", *spec.directions],
        "path": str(path),
        "elapsed_s": round(time.time() - start, 1),
    }


@torch.no_grad()
def _all_position_logprobs(model, input_ids, edit=None) -> torch.Tensor:
    """``[T, vocab]`` log-probs for every position of one sequence."""
    final_layer = model.n_layers - 1
    with (
        steered(model, edit),
        ActivationRecorder(model.layers, at=[final_layer]) as recorder,
    ):
        model.forward(input_ids)
        h = recorder.activations[final_layer][0].detach()
    return torch.log_softmax(model.unembed(h).float(), dim=-1)


def run_horizon_effect(spec: CausalSpec) -> dict:
    """C27 - does writing at horizon d move the output d tokens later?

    C26 established that per-horizon write directions are distinct and
    smoothly ordered. Distinct is necessary but not sufficient: they could
    encode "influence on nearby syntax" versus "influence on document topic"
    with no timing content at all. This is the experiment that discriminates.

    Write ``v_y^(d)`` additively across the band at the prompt's positions, and
    measure how much the log-probability of ``y`` rises at each *subsequent*
    output position.

    The design point that makes the matrix well defined: the continuation is
    **teacher-forced to the model's own clean greedy continuation** and held
    fixed across every arm. Letting each arm decode its own text would change
    the context and make position ``k`` mean something different per arm, so
    the comparison would not be between positions at all. Here one forward pass
    per arm yields the whole row, and the edit stays pinned to the prompt
    positions while the measurement moves forward -- which is exactly "write
    here, look d tokens later".

    Reports ``delta[d][k] = log p_steered(y at k) - log p_clean(y at k)``.
    A diagonal band means J-space is addressable in time. Flat rows differing
    only in scale mean horizon sets direction but not timing.
    """
    config, model, lens, layers, _ = _setup(spec)
    max_seq_len = config.fit.max_seq_len
    out = spec.resolve_out("horizon")

    trials = flexible_generalization_trials()
    if spec.limit:
        trials = trials[: spec.limit]
    trials = take_shard(trials, spec.shard, spec.n_shards)

    banks: dict[str, dict[int, torch.Tensor]] = {}
    words: list[str] = []
    for raw in spec.directions_paths:
        blob = torch.load(raw, map_location="cpu", weights_only=False)
        if words and list(blob["targets"]) != words:
            raise ValueError(f"{raw} disagrees on the target-word list")
        words = list(blob["targets"])
        for key, entry in blob["digests"].items():
            group, layer = key.split("|")
            if int(layer) in layers:
                banks.setdefault(group, {})[int(layer)] = entry["pullbacks"].float()
        del blob
    missing = [d for d in spec.directions if d not in banks]
    if missing:
        raise ValueError(f"no directions for {missing}; loaded {sorted(banks)}")
    for arm in spec.directions:
        absent = sorted(set(layers) - set(banks[arm]))
        if absent:
            raise ValueError(f"{arm} has no pullbacks at layers {absent}")

    n_tokens = spec.gen_tokens
    logger.info(
        "shard %d/%d: %d trials x %d arms, %d positions",
        spec.shard,
        spec.n_shards,
        len(trials),
        len(spec.directions),
        n_tokens,
    )
    records: list[dict] = []
    start = time.time()
    for i, trial in enumerate(trials):
        try:
            y = single_token_id(model, trial.target_arg)
        except ValueError:
            continue
        if trial.target_arg not in words:
            continue
        wi = words.index(trial.target_arg)

        clean_text = greedy_continuation(
            model, trial.prompt, n_tokens=n_tokens, max_seq_len=max_seq_len
        )
        prompt_ids = model.encode(trial.prompt, max_length=max_seq_len)
        n_p = prompt_ids.shape[1]
        cont = model.tokenizer.encode(clean_text, add_special_tokens=False)[:n_tokens]
        if len(cont) < n_tokens:
            continue
        full = torch.cat(
            [prompt_ids, torch.tensor([cont], device=prompt_ids.device)], dim=1
        )
        norms = mean_residual_norms(model, trial.prompt, layers, max_seq_len=max_seq_len)
        clean_lp = _all_position_logprobs(model, full)

        for arm in spec.directions:
            edit = additive_edit(
                {ll: banks[arm][ll][wi] for ll in layers},
                norms,
                strength=spec.strengths[0],
                positions=list(range(n_p)),
            )
            lp = _all_position_logprobs(model, full, edit)
            for k in range(n_tokens):
                pos = n_p - 1 + k
                records.append(
                    {
                        "key": f"{trial.category}/{trial.func}/{trial.source_arg}->{trial.target_arg}",
                        "category": trial.category,
                        "arm": arm,
                        "k": k,
                        "delta": round(float(lp[pos, y] - clean_lp[pos, y]), 5),
                    }
                )
        if (i + 1) % 10 == 0 or i + 1 == len(trials):
            rate = (time.time() - start) / (i + 1)
            logger.info("  %3d/%d  %.2fs/trial  eta %.0fs", i + 1, len(trials), rate, rate * (len(trials) - i - 1))

    path = out / f"horizon_effect_{spec.tag}.json"
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=1))

    n_trials = len({r["key"] for r in records})
    expected = n_trials * len(spec.directions) * n_tokens
    if len(records) != expected:
        raise RuntimeError(f"{len(records)} records, expected {expected}")
    for arm in spec.directions:
        per_k = [
            statistics.mean(r["delta"] for r in records if r["arm"] == arm and r["k"] == k)
            for k in range(n_tokens)
        ]
        logger.info("  %-10s %s", arm, " ".join(f"{v:+.2f}" for v in per_k))
    return {
        "shard": spec.shard,
        "n_trials": n_trials,
        "n_records": len(records),
        "arms": list(spec.directions),
        "n_tokens": n_tokens,
        "path": str(path),
        "elapsed_s": round(time.time() - start, 1),
    }


@dataclass(frozen=True)
class HorizonSpec(ComponentSpec):
    """C26 - resolve the off-diagonal term by distance, and ask if it varies.

    The gating check for horizon-targeted writing. ``_off`` lumps every
    ``d = t' - t >= 1`` together and works well, which is equally consistent
    with "the only special horizon is 0" and with "horizons carry distinct,
    addressable directions". Those differ in an immediately measurable way: if
    ``v^(1)`` and ``v^(16)`` are near-collinear there is nothing to address.

    Costs what the component sweep costs -- the backward passes are the same,
    only the binning differs -- so this is deliberately run on a small panel.
    """

    @property
    def namespace(self) -> str:
        return f"{super().namespace}_hz"


def run_horizon_pullbacks(spec: HorizonSpec) -> dict:
    """Average per-horizon write directions, and report how distinct they are."""
    from jsteer.jacobian import HORIZON_BUCKETS, horizon_pullbacks_for_prompt

    config, model, lens, layers, dim_batch = _setup(spec)
    targets = resolve_targets(model, spec.target_words)
    prompts = _prompt_shard(spec, config, model)
    out = spec.resolve_out("rq1")

    stray = [p.key for p in prompts if not p.fit_positions]
    if stray:
        raise RuntimeError(f"{len(stray)} prompts need explicit positions: {stray[:3]}")
    groups = sorted({p.group for p in prompts})
    per_group = collections.Counter(p.group for p in prompts)
    short = {g: n for g, n in per_group.items() if n != spec.n_per}
    if short:
        raise RuntimeError(f"expected {spec.n_per} prompts per corpus; got {short}.")

    names: list[str] = []
    means: dict[tuple[str, int], RunningMean] = {}
    pair_totals: collections.Counter = collections.Counter()
    logger.info(
        "%d prompts over %d corpora x %d layers, %d targets, %d buckets",
        len(prompts),
        len(groups),
        len(layers),
        len(targets.words),
        len(HORIZON_BUCKETS),
    )
    start = time.time()
    for i, prompt in enumerate(prompts):
        parts, counts = horizon_pullbacks_for_prompt(
            model,
            prompt.text,
            layers,
            targets.cotangents,
            target_layer=config.fit.target_layer,
            dim_batch=dim_batch,
            max_seq_len=config.fit.max_seq_len,
            skip_first=config.fit.skip_first,
            t_stride=spec.t_stride,
        )
        if not names:
            names = [n for n in parts[layers[0]] if n != "total"]
        pair_totals.update(counts)
        for layer in layers:
            for name in [*names, "total"]:
                means.setdefault(
                    (f"{prompt.group}_{name}", layer), RunningMean()
                ).add(parts[layer][name])
        if (i + 1) % 5 == 0 or i + 1 == len(prompts):
            rate = (time.time() - start) / (i + 1)
            logger.info(
                "  %3d/%d  %.1fs/prompt  eta %.0fs",
                i + 1,
                len(prompts),
                rate,
                rate * (len(prompts) - i - 1),
            )

    digests = {}
    for (group, layer), mean in means.items():
        if mean.total is None:
            raise RuntimeError(f"no prompts contributed to {group} at layer {layer}")
        digests[f"{group}|{layer}"] = {
            "group": group,
            "layer": layer,
            "n": mean.n,
            "pullbacks": (mean.total / mean.n).float(),
        }
    path = out / f"group_mean_digests_{spec.namespace}.pt"
    torch.save({"digests": digests, "targets": targets.words, "pullback_only": True}, path)

    expected = len(groups) * (len(names) + 1) * len(layers)
    if len(digests) != expected:
        raise RuntimeError(f"wrote {len(digests)} digests, expected {expected}")

    # THE GATING CHECK. Pairwise cosine between horizon buckets, per corpus,
    # averaged over words and layers. Near-1 everywhere means the off-diagonal
    # term is horizon-agnostic and there is no positional addressing to find.
    cosines: list[dict] = []
    for group in groups:
        for a in names:
            for b in names:
                if a >= b:
                    continue
                vals = []
                for layer in layers:
                    A = digests[f"{group}_{a}|{layer}"]["pullbacks"]
                    B = digests[f"{group}_{b}|{layer}"]["pullbacks"]
                    vals.append(
                        float(
                            (
                                (A * B).sum(-1)
                                / (A.norm(dim=-1) * B.norm(dim=-1)).clamp_min(1e-12)
                            ).mean()
                        )
                    )
                cosines.append(
                    {
                        "corpus": group,
                        "a": a,
                        "b": b,
                        "cos": round(statistics.mean(vals), 4),
                        "by_layer": [round(v, 4) for v in vals],
                    }
                )
    norms = {
        f"{g}_{n}": round(
            float(
                torch.stack(
                    [digests[f"{g}_{n}|{ll}"]["pullbacks"].norm(dim=1).mean() for ll in layers]
                ).mean()
            ),
            4,
        )
        for g in groups
        for n in names
    }
    geo = spec.resolve_out("geometry") / f"horizon_{spec.namespace}.json"
    geo.write_text(
        json.dumps(
            {"cosines": cosines, "norms": norms, "pair_counts": dict(pair_totals)},
            indent=1,
        )
    )
    logger.info("pair counts: %s", dict(pair_totals))
    logger.info("mean norms: %s", norms)
    for row in cosines:
        logger.info("  cos[%s] %s vs %s = %+.4f", row["corpus"], row["a"], row["b"], row["cos"])

    return {
        "n_prompts": len(prompts),
        "buckets": names,
        "layers": layers,
        "pair_counts": dict(pair_totals),
        "norms": norms,
        "namespace": spec.namespace,
        "path": str(path),
        "geometry_path": str(geo),
        "elapsed_s": round(time.time() - start, 1),
    }


def run_beta_family(spec: AverageNSpec) -> dict:
    """C25 - the unembedding correction as a one-parameter family, and its closed form.

    C22 removed exactly 100% of the ``u_y`` component of the write direction
    and recovered 64-84% of what horizon filtering achieves. Nothing says 100%
    is the optimum. Define

        v_y(beta) = v_y - beta * <v_y, u_hat_y> * u_hat_y

    so ``beta = 0`` is the standard lens and ``beta = 1`` is C22's ``_perp``.
    ``beta > 1`` over-corrects: it drives the direction to *anti*-align with
    the token's unembedding row, actively suppressing the "say y next" channel
    while leaving the state-change channel intact. The two-hop leakage numbers
    are the reason to think that might help -- ``_off`` cut leakage from 0.38
    to 0.09, and if leakage is the failure mode then more suppression may buy
    more second-hop success.

    **The closed form.** Separately, if the diagonal term is approximately
    ``alpha * I``, then the expensive horizon-filtered fit has a cheap
    surrogate: subtract a single multiple of ``u_y`` from the standard lens.
    Estimated per layer from the pullbacks alone,

        alpha_l = sum_y <v_y, u_y> / sum_y ||u_y||^2

    which is one scalar shared by every word, as against beta's per-word
    coefficient. The ``_galpha`` arm applies it. This matters practically:
    ``_off`` costs ~111 backward passes per prompt because it needs per-target
    -position pullbacks, while ``_galpha`` costs nothing and applies to the
    published lens as shipped, with no refit.

    Also reports, for the corpora that have one, ``cos(v(beta), v_off)`` -- how
    close each cheap arm gets to the expensive estimator's actual direction.

    Pure linear algebra over cached digests. Emits a causal-ready digest.
    """
    config, model, lens, layers, _ = _setup(spec)

    banks: dict[str, dict[int, torch.Tensor]] = {}
    words: list[str] = []
    for raw in spec.directions_paths:
        blob = torch.load(raw, map_location="cpu", weights_only=False)
        if words and list(blob["targets"]) != words:
            raise RuntimeError(f"{raw} disagrees on the target-word list")
        words = list(blob["targets"])
        for entry in blob["digests"].values():
            if entry["layer"] in layers:
                banks.setdefault(entry["group"], {})[entry["layer"]] = entry[
                    "pullbacks"
                ].float()
        del blob

    # With no component digest the only base is the published lens, which needs
    # no fit -- so this runs on ANY model that has a lens, which is how the
    # correction gets tested on a second model without refitting anything.
    if not words:
        words = all_args()
        logger.info("no directions_paths: published lens only, %d args", len(words))

    token_ids = [single_token_id(model, w) for w in words]
    U = unembedding_rows(model, token_ids)
    Un = U / U.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    banks["published_full"] = {
        ll: averaged_pullback(lens, U, ll).float() for ll in layers
    }

    # ``ns`` is reused verbatim as the beta grid; it is a list of numbers on
    # AverageNSpec and nothing here interprets it as a subset size.
    betas = [float(b) for b in (spec.ns or [0.0, 0.5, 1.0, 2.0, 3.0])]
    bases = [b for b in ("gsm8k_full", "wikitext_a_full", "published_full") if b in banks]
    if not bases:
        raise ValueError(f"no base arms; loaded {sorted(banks)}")

    digests: dict[str, dict] = {}
    geometry: list[dict] = []
    alphas: dict[str, dict[int, float]] = {}
    for base in bases:
        prefix = base[: -len("_full")]
        off = banks.get(f"{prefix}_off")
        for ll in layers:
            V = banks[base][ll]
            if V.shape[0] != len(words):
                raise RuntimeError(f"{base}@{ll}: {V.shape} for {len(words)} words")
            coeff = (V * Un).sum(dim=-1, keepdim=True)

            # One scalar per layer, shared by every word: the least-squares
            # identity coefficient, read off the pullbacks.
            alpha = float((V * U).sum() / (U * U).sum())
            alphas.setdefault(prefix, {})[ll] = round(alpha, 5)

            variants = {f"b{b:g}": V - b * coeff * Un for b in betas}
            variants["galpha"] = V - alpha * U

            for tag, P in variants.items():
                name = f"{prefix}_{tag}"
                digests[f"{name}|{ll}"] = {
                    "group": name,
                    "layer": ll,
                    "n": 1,
                    "pullbacks": P,
                }
                row = {
                    "arm": name,
                    "layer": ll,
                    "norm_kept": round(
                        float((P.norm(dim=-1) / V.norm(dim=-1)).mean()), 4
                    ),
                    "cos_u": round(
                        float(
                            (
                                (P * Un).sum(-1)
                                / P.norm(dim=-1).clamp_min(1e-12)
                            ).mean()
                        ),
                        4,
                    ),
                }
                if off is not None:
                    O = off[ll]
                    row["cos_off"] = round(
                        float(
                            (
                                (P * O).sum(-1)
                                / (P.norm(dim=-1) * O.norm(dim=-1)).clamp_min(1e-12)
                            ).mean()
                        ),
                        4,
                    )
                geometry.append(row)

    out = spec.resolve_out("rq1")
    tag = spec.label or "beta"
    path = out / f"group_mean_digests_{spec.config_name}_{tag}.pt"
    torch.save({"digests": digests, "targets": words, "pullback_only": True}, path)

    expected = len(bases) * (len(betas) + 1) * len(layers)
    if len(digests) != expected:
        raise RuntimeError(f"wrote {len(digests)}, expected {expected}")

    logger.info("betas %s over %d bases -> %s", betas, len(bases), path)
    for prefix, per_layer in sorted(alphas.items()):
        logger.info("  alpha[%s] = %s", prefix, per_layer)
    for arm in sorted({g["arm"] for g in geometry}):
        rows = [g for g in geometry if g["arm"] == arm]
        co = [g["cos_off"] for g in rows if "cos_off" in g]
        logger.info(
            "  %-24s norm %.3f  cos_u %+.3f%s",
            arm,
            statistics.mean(g["norm_kept"] for g in rows),
            statistics.mean(g["cos_u"] for g in rows),
            f"  cos_off {statistics.mean(co):+.3f}" if co else "",
        )

    geo_path = spec.resolve_out("geometry") / f"beta_family_{spec.config_name}_{tag}.json"
    geo_path.write_text(json.dumps({"geometry": geometry, "alphas": alphas}, indent=1))

    return {
        "arms": sorted({g["arm"] for g in geometry}),
        "betas": betas,
        "layers": layers,
        "alphas": alphas,
        "n_entries": len(digests),
        "namespace": f"{spec.config_name}_{tag}",
        "path": str(path),
        "geometry_path": str(geo_path),
    }


def run_project_out(spec: AverageNSpec) -> dict:
    """C22 - remove the unembedding row from the FULL direction, keep everything else.

    The control C21 asks for. C21 found that the ``t' = t`` half of a J-lens
    write direction is approximately the token's own unembedding row
    (cos 0.60-0.66 against a 0.012 null), and that alignment with ``u_y``
    predicts steering failure across arms at rho = -0.95. But ``_off``'s low
    alignment is partly algebraic -- ``full = diag + off``, so subtracting a
    term that points along ``u_y`` must lower the cosine -- and the C18 swap
    result therefore cannot yet distinguish two accounts:

    * **u_y contamination.** The write direction fails because it carries the
      unembedding row, which makes the model *say* y rather than *be in a
      state that produces* y. Horizon filtering works only because the
      diagonal is where that contamination lives.
    * **the off-diagonal carries something more.** Long-horizon influence is a
      qualitatively different object, and removing ``u_y`` alone is not
      enough.

    These differ in a directly testable way. Project ``u_y`` out of the FULL
    direction by Gram-Schmidt, leaving the diagonal otherwise intact:

        v_perp = v_full - <v_full, u_hat_y> u_hat_y

    If ``_full_perp`` recovers ``_off``'s gain, the first account holds and
    the horizon framing is the route rather than the operative fact. If it
    does not, the second holds and the paper's claim is about horizons.

    Emits, per corpus:

    * ``{c}_full_perp``  -- the test arm.
    * ``{c}_full_rperp`` -- the **control**: an equal-magnitude component
      removed along a *random other token's* unembedding row instead of the
      target's. Without it, "removing a component helped" is confounded with
      "removing this particular component helped", since any projection
      shortens the vector and changes its direction.
    * ``{c}_off_perp``   -- ``_off`` with ``u_y`` also removed. Should barely
      move if ``_off`` is already free of it; a large change would say C21's
      reading of the cosines is wrong.

    The published lens gets ``published_perp`` / ``published_rperp`` too, since
    C21 found it as ``u_y``-contaminated as the diagonal arms (+0.576).

    Writes a digest in the causal arm's own format, so ``domain_causal --arms``
    steers these unchanged. The swap operator unit-normalizes its read
    directions, so the norm lost to the projection does not confound the dose.

    Pure linear algebra over a cached digest: no prompts, no backward passes.
    """
    config, model, lens, layers, _ = _setup(spec)

    banks: dict[str, dict[int, torch.Tensor]] = {}
    words: list[str] = []
    for raw in spec.directions_paths:
        blob = torch.load(raw, map_location="cpu", weights_only=False)
        if words and list(blob["targets"]) != words:
            raise RuntimeError(f"{raw} disagrees on the target-word list")
        words = list(blob["targets"])
        for entry in blob["digests"].values():
            if spec.arms and entry["group"] not in spec.arms:
                continue
            if entry["layer"] in layers:
                banks.setdefault(entry["group"], {})[entry["layer"]] = entry[
                    "pullbacks"
                ].float()
        del blob
    if not banks:
        raise ValueError("no arms loaded; directions_paths is required here")
    absent = [a for a in spec.arms if a not in banks]
    if absent:
        raise ValueError(f"no arms {absent}; loaded {sorted(banks)}")

    token_ids = [single_token_id(model, w) for w in words]
    U = unembedding_rows(model, token_ids)
    Un = U / U.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    banks["published"] = {
        ll: averaged_pullback(lens, U, ll).float() for ll in layers
    }

    # The control's direction: each word's component removed along a DIFFERENT
    # token's unembedding row. A derangement rather than fresh random ids, so
    # the removed directions come from the same distribution as the real ones
    # and no word is paired with itself.
    donors = _derange(len(words), spec.seed or 1)
    Rn = Un[donors]

    def project_out(V: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        coeff = (V * basis).sum(dim=-1, keepdim=True)
        return V - coeff * basis

    digests: dict[str, dict] = {}
    report: dict[str, dict[str, float]] = {}
    for arm, bank in sorted(banks.items()):
        missing = [ll for ll in layers if ll not in bank]
        if missing:
            raise RuntimeError(f"arm {arm!r} is missing layers {missing}")
        for suffix, basis in (("perp", Un), ("rperp", Rn)):
            name = f"{arm}_{suffix}"
            kept = []
            for ll in layers:
                V = bank[ll]
                if V.shape[0] != len(words):
                    raise RuntimeError(f"{arm}@{ll} has {V.shape} for {len(words)} words")
                P = project_out(V, basis)
                kept.append(float((P.norm(dim=-1) / V.norm(dim=-1)).mean()))
                digests[f"{name}|{ll}"] = {
                    "group": name,
                    "layer": ll,
                    "n": 1,
                    "pullbacks": P,
                }
            report[name] = {"norm_kept": round(statistics.mean(kept), 4)}

    out = spec.resolve_out("rq1")
    tag = spec.label or "perp"
    path = out / f"group_mean_digests_{spec.config_name}_{tag}.pt"
    torch.save({"digests": digests, "targets": words, "pullback_only": True}, path)

    expected = len(banks) * 2 * len(layers)
    if len(digests) != expected:
        raise RuntimeError(f"wrote {len(digests)} entries, expected {expected}")
    logger.info("%d arms x %d layers -> %s", len(banks) * 2, len(layers), path)
    for name, stats in sorted(report.items()):
        logger.info("  %-28s keeps %.3f of its norm", name, stats["norm_kept"])

    return {
        "arms": sorted(report),
        "layers": layers,
        "n_words": len(words),
        "n_entries": len(digests),
        "norm_kept": report,
        "namespace": f"{spec.config_name}_{tag}",
        "path": str(path),
    }


@dataclass(frozen=True)
class TwoHopSpec(ReadoutSpec):
    """C19 - write a latent INTERMEDIATE and require the second hop to follow.

    Every causal arm in this project so far swaps a token that is present in
    the prompt, and grades the token that comes next. That is a write on the
    short-horizon channel: the thing written is the thing read out. Yan et
    al.'s other component -- the sparse concept -- has only ever been scored as
    a readout, and the corpus result (C14/C17) says the two channels come
    apart.

    So: take probe-swap's two-hop items, whose ``intermediate`` (Brazil) never
    appears in the prompt, and swap it for ``swap_to`` (Mexico). Success is not
    that the model says "Mexico" -- that would mean the write landed on the
    surface, the short-horizon outcome. Success is that it says
    ``swap_answer`` (Spanish), which requires the model to have *used* the
    swapped intermediate in a computation it performs afterwards.

    The two failure modes are therefore recorded separately, and the
    interesting cell is "said swap_to but not swap_answer": a write that
    reaches the output without reaching the computation.

    Needs no new fit. ``swap_edit`` clamps per position from the clean
    residual's own coordinate along the read direction, so the write lands
    wherever the intermediate is loaded without anyone naming a position; and
    the write direction for any token is ``u_y^T J_bar``, which the stored full
    means give for arbitrary words, not only the 16 fitted arguments.
    """

    strengths: list[float] = field(default_factory=lambda: [1.0, 2.0, 4.0])
    swap_mode: str = "clamp"
    gen_tokens: int = 6
    #: Component digests (``run_component_pullbacks``) whose per-word pullbacks
    #: supply extra arms. This is how the Eq. 20 halves reach this eval: their
    #: write directions exist only per fitted word, not as a d x d matrix, so
    #: they cannot come from ``means_paths``.
    directions_paths: list[str] = field(default_factory=list)
    #: Groups to take from those digests. Empty takes all.
    arms: list[str] = field(default_factory=list)

    @property
    def namespace(self) -> str:
        stamp = hashlib.sha256(
            repr(
                (
                    self.strengths,
                    self.swap_mode,
                    self.gen_tokens,
                    sorted(self.arms),
                    sorted(self.directions_paths),
                )
            ).encode()
        ).hexdigest()[:6]
        return f"{super().namespace}_twohop{stamp}"


def run_two_hop(spec: TwoHopSpec) -> dict:
    """Swap the latent intermediate; grade the downstream answer."""
    config, model, lens, layers, _ = _setup(spec)
    max_seq_len = config.fit.max_seq_len
    items = probe_swap_items()
    d_model = config.d_model

    banks: dict[str, dict[int, torch.Tensor]] = {
        "published": {ll: lens.jacobians[ll].float() for ll in layers},
        # J = I: write the unembedding row itself. The control that says
        # whether transporting the direction through J_bar buys anything.
        "logit_lens": {ll: torch.eye(d_model) for ll in layers},
    }
    for raw in spec.means_paths:
        blob = torch.load(raw, map_location="cpu", weights_only=False)
        for key, mean in blob["means"].items():
            group, layer_str = key.split("|")
            if spec.groups and group not in spec.groups:
                continue
            if int(layer_str) in layers:
                banks.setdefault(group, {})[int(layer_str)] = mean.float()
        del blob
    missing = [g for g in spec.groups if g not in banks]
    if missing:
        raise ValueError(f"no full means for {missing}; loaded {sorted(banks)}")

    # Component arms: per-word pullbacks rather than a matrix, so an item is
    # only steerable by them if BOTH its words were fitted.
    word_banks: dict[str, dict[int, torch.Tensor]] = {}
    word_index: dict[str, int] = {}
    for raw in spec.directions_paths:
        blob = torch.load(raw, map_location="cpu", weights_only=False)
        word_index = {w: i for i, w in enumerate(blob["targets"])}
        for entry in blob["digests"].values():
            if spec.arms and entry["group"] not in spec.arms:
                continue
            if entry["layer"] in layers:
                word_banks.setdefault(entry["group"], {})[entry["layer"]] = entry[
                    "pullbacks"
                ].float()
        del blob
    absent = [a for a in spec.arms if a not in word_banks]
    if absent:
        raise ValueError(f"no component arms {absent}; loaded {sorted(word_banks)}")

    records: list[dict] = []
    skipped: dict[str, int] = {}
    start = time.time()
    for i, item in enumerate(items):
        try:
            # first_token_id, not single_token_id: most intermediates are
            # multi-token, and the causal grader uses the same convention.
            src_id = first_token_id(model, item.intermediate)
            tgt_id = first_token_id(model, item.swap_to)
        except ValueError as exc:
            skipped[str(exc)[:50]] = skipped.get(str(exc)[:50], 0) + 1
            continue

        clean_text = greedy_continuation(
            model, item.prompt, n_tokens=spec.gen_tokens, max_seq_len=max_seq_len
        )
        baseline_ok = answer_matches(clean_text, item.answer)
        base = {
            "name": item.name,
            "category": item.category,
            "intermediate": item.intermediate,
            "swap_to": item.swap_to,
            "answer": item.answer,
            "swap_answer": item.swap_answer,
            "baseline_ok": baseline_ok,
        }
        records.append(
            {**base, "arm": "baseline", "strength": 0.0, "generated": clean_text,
             "hit_swap_answer": False, "hit_answer": baseline_ok,
             "said_swap_to": bool(re.search(item.swap_to, clean_text, re.I))}
        )

        cotangents = unembedding_rows(model, [src_id, tgt_id])
        directions: dict[str, dict[int, torch.Tensor]] = {
            group: {ll: cotangents.float() @ per_layer[ll] for ll in layers}
            for group, per_layer in banks.items()
        }
        if word_banks:
            if item.intermediate in word_index and item.swap_to in word_index:
                si, ti = word_index[item.intermediate], word_index[item.swap_to]
                for group, per_layer in word_banks.items():
                    directions[group] = {
                        ll: torch.stack([per_layer[ll][si], per_layer[ll][ti]])
                        for ll in layers
                    }
            else:
                skipped["word not fitted for component arms"] = (
                    skipped.get("word not fitted for component arms", 0) + 1
                )
        for strength in spec.strengths:
            for group, per_layer in directions.items():
                g = per_layer
                edit = swap_edit(
                    model,
                    item.prompt,
                    {ll: g[ll][0] for ll in layers},
                    {ll: g[ll][1] for ll in layers},
                    mode=spec.swap_mode,
                    strength=strength,
                    max_seq_len=max_seq_len,
                )
                text = greedy_continuation(
                    model, item.prompt, edit=edit, n_tokens=spec.gen_tokens,
                    max_seq_len=max_seq_len,
                )
                records.append(
                    {
                        **base,
                        "arm": group,
                        "strength": strength,
                        "generated": text,
                        # The whole point of the eval: did the second hop run?
                        "hit_swap_answer": answer_matches(text, item.swap_answer),
                        "hit_answer": answer_matches(text, item.answer),
                        # ...or did the write only reach the surface?
                        "said_swap_to": bool(re.search(item.swap_to, text, re.I)),
                    }
                )
        if (i + 1) % 10 == 0:
            logger.info("  %3d/%d  %.0fs", i + 1, len(items), time.time() - start)

    out = spec.resolve_out("causal")
    path = out / f"two_hop_{spec.tag}.json"
    path.write_text(json.dumps(records))

    usable = {r["name"] for r in records if r["baseline_ok"]}
    summary: dict[str, dict] = {}
    for arm in sorted({r["arm"] for r in records if r["arm"] != "baseline"}):
        for strength in spec.strengths:
            sub = [
                r for r in records
                if r["arm"] == arm and r["strength"] == strength and r["name"] in usable
            ]
            if not sub:
                continue
            summary[f"{arm}@{strength}"] = {
                "n": len(sub),
                "swap_answer": round(sum(r["hit_swap_answer"] for r in sub) / len(sub), 4),
                "kept_answer": round(sum(r["hit_answer"] for r in sub) / len(sub), 4),
                "said_swap_to": round(sum(r["said_swap_to"] for r in sub) / len(sub), 4),
            }
    logger.info("two-hop: %s", summary)
    return {
        "n_items": len({r["name"] for r in records}),
        "n_usable": len(usable),
        "skipped": skipped,
        "arms": sorted({r["arm"] for r in records if r["arm"] != "baseline"}),
        "summary": summary,
        "path": str(path),
        "elapsed_s": round(time.time() - start, 1),
    }


def run_ablation_effect(spec: CausalSpec) -> dict:
    """C28 - score our arms on the WORKSPACE PAPER's causal measure, not ours.

    This is the experiment that decides whether the paper has a story, so the
    reasoning is worth writing down in full.

    The workspace paper (App. A.7) already tested the recipe choice this
    project is built on. Its main text says so verbatim: "We examine several
    variants of the Jacobian lens methodology (e.g. computing only present and
    not future token effects, ...); our qualitative results are robust to these
    choices." Yan et al. Sec. 5.1 masks the same diagonal for readout. So
    "nobody split t'=t from t'>t" is not available to us, and neither is "they
    only did readout" -- A.7's Fig. 58 is a causal eval.

    Our swap numbers nonetheless separate the arms enormously (non-math /105:
    diag 16, full 35, off 60). Two measures, same knob, opposite verdicts. The
    only honest resolutions are that one of us is wrong, or that the two
    measures ask different questions. This run tests the second.

    **Their question** (:func:`jsteer.steering.ablate_edit`): delete the
    direction from the residual stream and see how much the output moves.
    Higher KL = the direction was more load-bearing. It scores *presence*.

    **Our question** (``run_causal``): write the direction in and see whether
    the model's answer follows. It scores *control*.

    The thesis predicts a dissociation with a specific shape: ablation KL
    roughly flat across ``_full`` / ``_diag`` / ``_off`` -- reproducing A.7's
    null under A.7's own measure -- while ``_diag`` is at least as ablatable as
    ``_off`` despite steering a third as well. The diagonal is the part of the
    direction the model genuinely uses to *emit* the token, which is exactly
    why deleting it registers and why writing it does not control.

    Guards against the two ways this could report a fake null:

    * ``frac_norm`` and ``abs_cos`` per arm. "Ablation changed nothing" is
      ambiguous between "the direction is not used" and "there was nothing
      there to remove"; a flat KL over arms that removed very different
      fractions of the residual is a different claim from a flat KL over arms
      that removed the same fraction.
    * a ``_rand`` control arm per corpus -- ablate along a *different* token's
      unembedding row, deranged so no word draws itself. This is the floor:
      any real arm must beat it, or the measure is not measuring anything.

    Records, per (trial, arm): full-vocab KL(clean || ablated) at the final
    prompt position (their metric), and the change in log p of the answer the
    source argument implies (a concept-specific reading of the same edit, which
    should go DOWN if the direction carries the concept).

    Shards over trials like every causal run. One clean forward pass and one
    ablated forward pass per arm; no backward passes and no generation.
    """
    config, model, lens, layers, _ = _setup(spec)
    max_seq_len = config.fit.max_seq_len
    out = spec.resolve_out("causal")

    path = out / f"ablation_{spec.tag}.json"
    if path.exists() and not spec.overwrite:
        try:
            have = {r["arm"] for r in json.loads(path.read_text())}
        except Exception:
            have = set()
        if set(spec.directions) <= have:
            logger.info("shard %d already complete; skipping", spec.shard)
            return {"shard": spec.shard, "skipped": True, "arms": sorted(have)}

    trials = (
        heldout_trials(spec.trial_set)
        if getattr(spec, "trial_set", "")
        else flexible_generalization_trials()
    )
    if spec.limit:
        trials = trials[: spec.limit]
    trials = take_shard(trials, spec.shard, spec.n_shards)

    # Arm banks, in the same digest format every other causal sweep reads.
    domain: dict[str, dict[int, torch.Tensor]] = {}
    words: list[str] = []
    for raw in spec.directions_paths:
        blob = torch.load(raw, map_location="cpu", weights_only=False)
        if words and list(blob["targets"]) != words:
            raise ValueError(f"{raw} disagrees on the target-word list")
        words = list(blob["targets"])
        for key, entry in blob["digests"].items():
            group, layer = key.split("|")
            if int(layer) in layers:
                domain.setdefault(group, {})[int(layer)] = entry["pullbacks"].float()
        del blob
    if not words:
        raise ValueError("directions_paths is required: this sweep steers nothing else")

    index = {w: i for i, w in enumerate(words)}
    token_ids = [single_token_id(model, w) for w in words]
    U = unembedding_rows(model, token_ids)

    # ``published`` is the upstream lens, built the same way run_project_out
    # builds it, so the published arm is scored by identical code.
    domain["published"] = {ll: averaged_pullback(lens, U, ll).float() for ll in layers}
    # The floor. A derangement rather than fresh randoms: the donor rows come
    # from the same distribution as the real ones and no word draws itself.
    donors = _derange(len(words), 1)  # CausalSpec carries no seed; fixed and reproducible
    domain["rand_uy"] = {ll: U[donors].float().clone() for ll in layers}
    # The logit-lens arm, ablated: u_y itself as a direction.
    domain["uy"] = {ll: U.float().clone() for ll in layers}

    missing = [a for a in spec.directions if a not in domain]
    if missing:
        raise ValueError(f"no directions for {missing}; loaded {sorted(domain)}")
    for group in spec.directions:
        absent = sorted(set(layers) - set(domain[group]))
        if absent:
            raise ValueError(f"{group} has no pullbacks at layers {absent}")

    logger.info("%s | %s", config, lens)
    logger.info(
        "shard %d/%d: %d trials, %d arms, band L%d-L%d",
        spec.shard,
        spec.n_shards,
        len(trials),
        len(spec.directions),
        layers[0],
        layers[-1],
    )

    records: list[dict] = []
    skipped: dict[str, int] = {}
    start = time.time()
    for trial in trials:
        if trial.source_arg not in index:
            skipped["arg not in digest"] = skipped.get("arg not in digest", 0) + 1
            continue
        try:
            answer_ids = answer_variant_ids(model, trial.source_answer)
        except ValueError as exc:
            key = str(exc)[:60]
            skipped[key] = skipped.get(key, 0) + 1
            continue

        row = index[trial.source_arg]
        clean_log = torch.log_softmax(
            next_token_logits(model, trial.prompt, max_seq_len=max_seq_len), dim=-1
        )
        clean_answer = float(torch.logsumexp(clean_log[answer_ids], dim=0))

        base = {
            "category": trial.category,
            "func": trial.func,
            "prompt_key": f"{trial.category}/{trial.func}/{trial.source_arg}",
            "source_arg": trial.source_arg,
            "source_answer": trial.source_answer,
        }

        for arm in spec.directions:
            directions = {ll: domain[arm][ll][row] for ll in layers}
            edit, diag = ablate_edit(
                model, trial.prompt, directions, max_seq_len=max_seq_len
            )
            with steered(model, edit):
                abl_log = torch.log_softmax(
                    next_token_logits(model, trial.prompt, max_seq_len=max_seq_len),
                    dim=-1,
                )
            p = clean_log.exp()
            kl = float((p * (clean_log - abl_log)).sum())
            records.append(
                {
                    **base,
                    "arm": arm,
                    "kl": kl,
                    "dlogp_answer": float(
                        torch.logsumexp(abl_log[answer_ids], dim=0)
                    )
                    - clean_answer,
                    **diag,
                }
            )

    if skipped:
        logger.warning("skipped: %s", skipped)
    path.write_text(json.dumps(records))
    logger.info("%d records -> %s", len(records), path)

    per_arm = collections.defaultdict(list)
    for r in records:
        per_arm[r["arm"]].append(r)
    summary = {
        arm: {
            "kl": round(statistics.mean(x["kl"] for x in rows), 4),
            "dlogp": round(statistics.mean(x["dlogp_answer"] for x in rows), 4),
            "frac_norm": round(statistics.mean(x["frac_norm"] for x in rows), 4),
        }
        for arm, rows in sorted(per_arm.items())
    }
    for arm, stats in summary.items():
        logger.info("  %-24s %s", arm, stats)

    return {
        "shard": spec.shard,
        "n_trials": len(records) // max(len(spec.directions), 1),
        "n_records": len(records),
        "arms": sorted(per_arm),
        "skipped": skipped,
        "summary": summary,
        "path": str(path),
        "elapsed_s": round(time.time() - start, 1),
    }


def _unembedding_covariance(
    model, *, device: str = "cpu", chunk: int = 8192
) -> tuple[torch.Tensor, torch.Tensor]:
    """``Cov(gamma)`` over the WHOLE vocabulary's unembedding rows, plus its mean.

    Park et al. 2023 (arXiv:2311.03658) Thm 3.4 takes the expectation over the
    vocabulary, so this is every row of ``W_U``, not the 16 argument rows. The
    Gram is accumulated in chunks because ``[151936, 4096]`` in fp32 is ~2.5 GB
    and the outer product must not be materialised.
    """
    W = unembedding_matrix(model)
    n, d = W.shape
    mu = torch.zeros(d, dtype=torch.float64)
    for i in range(0, n, chunk):
        mu += W[i : i + chunk].to(device=device, dtype=torch.float64).sum(dim=0).cpu()
    mu /= n
    cov = torch.zeros(d, d, dtype=torch.float64)
    mu_dev = mu.to(device=device)
    for i in range(0, n, chunk):
        block = W[i : i + chunk].to(device=device, dtype=torch.float64) - mu_dev
        cov += (block.T @ block).cpu()
    cov /= n
    return cov, mu


def run_park_whiten(spec: AverageNSpec) -> dict:
    """C29 - the Park et al. baseline, the one experiment that could sink the paper.

    Park et al. 2023 (arXiv:2311.03658) formalise exactly the read-direction /
    write-direction problem this project is about. Their Thm 3.2 says the
    unembedding representation (what you probe with) and the embedding
    representation (what you steer with) are the same object under a *causal
    inner product*; Thm 3.4 gives that inner product explicitly as
    ``<g, g'>_C = g^T Cov(gamma)^-1 g'``, the covariance taken over the
    vocabulary's unembedding rows. The induced map from a read direction to a
    write direction is therefore ``g -> Cov(gamma)^-1 g``.

    Our beta=1 / alpha_l correction removes a **rank-1** piece: the target's own
    unembedding row. Park's whitening rescales the whole unembedding subspace
    and is blind to which token is being steered toward. Those are different
    operations, but they are in the same family and a reviewer will ask. If
    Park whitening recovers our gain, the contribution collapses to
    "Park 2023, applied to the J-lens" and we should say so. If it does not,
    the claim sharpens from "remove a subspace" to "remove *this* rank-1
    direction", and the random-donor control from C22 (20 -> 20) already has
    the right shape to support that.

    Arms emitted, per loaded arm ``{a}``:

    * ``{a}_park``   -- ``Cov^-1 v``, Thm 3.4's Riesz map, light damping.
    * ``{a}_park2``  -- the same at 100x the damping. Park's ``Cov^-1`` is
      ill-conditioned on a 151k x 4096 unembedding, and a negative result that
      only holds at one ridge is not a negative result.
    * ``{a}_parkh``  -- ``Cov^-1/2 v``, the symmetric-whitening reading of the
      same theorem, since the literature states it both ways.

    plus three arms that use no Jacobian at all and are the honest baselines
    this paper has been missing:

    * ``uy``       -- steer along the raw unembedding row. The logit-lens
      write direction, i.e. the degenerate extreme of the contamination axis.
      Published precedent that it works at all: Bushnaq et al.'s VPD edit
      replaces a subcomponent's write vector with ``-a u_o/||u_o||``.
    * ``uy_park``  -- ``Cov^-1 u_y``: **Park's method, run as specified**.
    * ``uy_parkh`` -- ``Cov^-1/2 u_y``.

    The swap operator unit-normalises its read directions, so none of these
    transformations changes the dose; only the direction moves.

    Pure linear algebra over a cached digest plus one pass over ``W_U``. No
    prompts, no backward passes.
    """
    config, model, lens, layers, _ = _setup(spec)

    banks: dict[str, dict[int, torch.Tensor]] = {}
    words: list[str] = []
    for raw in spec.directions_paths:
        blob = torch.load(raw, map_location="cpu", weights_only=False)
        if words and list(blob["targets"]) != words:
            raise RuntimeError(f"{raw} disagrees on the target-word list")
        words = list(blob["targets"])
        for entry in blob["digests"].values():
            if spec.arms and entry["group"] not in spec.arms:
                continue
            if entry["layer"] in layers:
                banks.setdefault(entry["group"], {})[entry["layer"]] = entry[
                    "pullbacks"
                ].float()
        del blob
    if not banks:
        raise ValueError("no arms loaded; directions_paths is required here")
    absent = [a for a in spec.arms if a not in banks]
    if absent:
        raise ValueError(f"no arms {absent}; loaded {sorted(banks)}")

    token_ids = [single_token_id(model, w) for w in words]
    U = unembedding_rows(model, token_ids)
    banks["published"] = {
        ll: averaged_pullback(lens, U, ll).float() for ll in layers
    }

    device = spec.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cov, _ = _unembedding_covariance(model, device=device)
    evals, evecs = torch.linalg.eigh(cov)
    evals = evals.clamp_min(0)
    scale = float(evals.mean())
    spectrum = {
        "d": int(cov.shape[0]),
        "eig_max": float(evals[-1]),
        "eig_min": float(evals[0]),
        "eig_mean": scale,
        # The number that justifies damping at all: a raw inverse of this
        # matrix amplifies the bottom of the spectrum by this factor.
        "condition": float(evals[-1] / evals[0].clamp_min(1e-30)),
    }
    logger.info("Cov(gamma): %s", spectrum)

    def spd_power(power: float, ridge: float) -> torch.Tensor:
        lam = ridge * scale
        return (evecs * (evals + lam).pow(power)) @ evecs.T

    transforms = {
        "park": spd_power(-1.0, 1e-3),
        "park2": spd_power(-1.0, 1e-1),
        "parkh": spd_power(-0.5, 1e-3),
    }

    # The Jacobian-free arms. ``uy`` is constant over layers by construction --
    # the unembedding row does not depend on where you inject it -- which is
    # itself part of what makes it a weak write direction.
    banks["uy"] = {ll: U.float().clone() for ll in layers}

    # ``gamma`` is Park's object, and getting this right is the difference
    # between testing their method and testing a formula on something it was
    # never defined over. Their gamma_bar_W is a CONCEPT representation
    # estimated from COUNTERFACTUAL PAIRS -- a difference of unembeddings, with
    # sign meaningful (king->queen opposite to woman->man) -- not a single
    # word's row. Theorems 3.2/3.4 say nothing about a bare u_y. So build the
    # concept direction the dataset actually affords: within each category, the
    # argument's row against the mean of the other three arguments' rows. That
    # is a counterfactual contrast in unembedding space, which is the space
    # Park's covariance is taken over.
    cat_of: dict[str, str] = {}
    for category, members in category_args().items():
        for w in members:
            cat_of[w] = category
    G = torch.zeros_like(U.float())
    for i, w in enumerate(words):
        siblings = [
            j for j, v in enumerate(words)
            if v != w and cat_of.get(v) == cat_of.get(w)
        ]
        if not siblings:
            raise RuntimeError(f"{w!r} has no same-category counterfactual partner")
        G[i] = U[i].float() - U[siblings].float().mean(dim=0)
    banks["gamma"] = {ll: G.clone() for ll in layers}

    digests: dict[str, dict] = {}
    report: dict[str, dict[str, float]] = {}
    for arm, bank in sorted(banks.items()):
        missing = [ll for ll in layers if ll not in bank]
        if missing:
            raise RuntimeError(f"arm {arm!r} is missing layers {missing}")
        for suffix, T in transforms.items():
            name = f"{arm}_{suffix}"
            turned = []
            for ll in layers:
                V = bank[ll]
                if V.shape[0] != len(words):
                    raise RuntimeError(
                        f"{arm}@{ll} has {V.shape} for {len(words)} words"
                    )
                P = (V.double() @ T.T).float()
                # How far the transform actually rotated the direction. A
                # cosine of ~1 would mean the whitening is a no-op and any null
                # result is uninformative; report it rather than assume it.
                cos = torch.nn.functional.cosine_similarity(P, V, dim=-1)
                turned.append(float(cos.mean()))
                digests[f"{name}|{ll}"] = {
                    "group": name,
                    "layer": ll,
                    "n": 1,
                    "pullbacks": P,
                }
            report[name] = {"cos_with_untransformed": round(statistics.mean(turned), 4)}
        # Carry the untransformed arm through too, so the causal run can score
        # `uy` itself without a second digest file.
        for ll in layers:
            digests[f"{arm}|{ll}"] = {
                "group": arm,
                "layer": ll,
                "n": 1,
                "pullbacks": bank[ll],
            }
        report[arm] = {"cos_with_untransformed": 1.0}

    out = spec.resolve_out("rq1")
    tag = spec.label or "park"
    path = out / f"group_mean_digests_{spec.config_name}_{tag}.pt"
    torch.save({"digests": digests, "targets": words, "pullback_only": True}, path)

    expected = len(banks) * (len(transforms) + 1) * len(layers)
    if len(digests) != expected:
        raise RuntimeError(f"wrote {len(digests)} entries, expected {expected}")
    logger.info("%d arms x %d layers -> %s", len(report), len(layers), path)
    for name, stats in sorted(report.items()):
        logger.info("  %-28s cos vs untransformed %.3f", name, stats["cos_with_untransformed"])

    return {
        "arms": sorted(report),
        "layers": layers,
        "n_words": len(words),
        "n_entries": len(digests),
        "spectrum": spectrum,
        "rotation": report,
        "namespace": f"{spec.config_name}_{tag}",
        "path": str(path),
    }


def run_diffmean(spec: AverageNSpec) -> dict:
    """C30 - the supervised baseline the steering audience will ask for.

    Every comparison so far has been J-lens against J-lens. DiffMean (the
    mass-mean shift behind CAA, and the direction ITI actually steers along
    after selecting heads with a probe) is the standard supervised construction
    and it is the obvious thing a reviewer will want beaten -- or, more
    honestly, will want to see us *lose* to by a stated margin, since it is
    fitted per concept on contrastive data while a J-lens direction is read off
    an unsupervised average and costs nothing per concept.

    Construction, matched to the dataset's own structure. The 64 base prompts
    are 4 categories x 4 templates x 4 args, so a contrast pair that differs
    *only* in the concept is available directly: same category, same template,
    different arg. For each arg ``y``::

        v_y(l) = mean_t [ h_l(t, y) - mean_{y' != y} h_l(t, y') ]

    over the fitting templates ``t``, with ``h`` read either at the last prompt
    position (``_last``, the usual CAA convention) or averaged over the
    argument's own token span (``_arg``, which is where this dataset puts the
    concept).

    **The leakage control is the point of the template split.** Fitting on all
    four templates would make this an in-distribution ceiling rather than a
    baseline. ``spec.n_fit`` templates per category are used for the fit (2 by
    default, in sorted order); the other two never enter it, so the causal run
    downstream can be scored separately on trials whose template was seen and
    trials whose template was not. Both numbers belong in the paper: the seen
    half is the ceiling, the unseen half is the fair comparison.

    ``diffmean_shuf`` is the specificity control -- the same vectors, assigned
    to the wrong args by a derangement. A supervised direction that steers as
    well when mislabelled is measuring prompt position, not concept.

    One forward pass per base prompt; no backward passes.
    """
    config, model, lens, layers, _ = _setup(spec)
    max_seq_len = config.fit.max_seq_len

    bases = flexible_generalization_prompts()
    by_category: dict[str, list] = collections.defaultdict(list)
    for base in bases:
        by_category[base.category].append(base)

    # Template names are scoped to their category ("capital" exists only under
    # countries), so the split has to be taken WITHIN each category. Splitting
    # on the union would hand two templates from one category to the fit and
    # leave the other three categories with no contrast pairs at all -- which
    # fails loudly below, but only after the GPU time is spent.
    n_fit = spec.n_fit or 2
    fit_funcs: set[tuple[str, str]] = set()
    held_out: set[tuple[str, str]] = set()
    for category, group in sorted(by_category.items()):
        funcs = sorted({b.func for b in group})
        if not 0 < n_fit < len(funcs):
            raise ValueError(
                f"n_fit={n_fit} leaves no held-out template of {len(funcs)} in "
                f"{category!r}; a baseline fitted on every template is a "
                "ceiling, not a baseline"
            )
        fit_funcs |= {(category, f) for f in funcs[:n_fit]}
        held_out |= {(category, f) for f in funcs[n_fit:]}
    logger.info("fit templates %s", sorted(fit_funcs))
    logger.info("held out %s", sorted(held_out))

    # Dataset order, not spec.arms: the causal run cross-checks the target
    # list across every digest it loads, so this must match the others.
    words = all_args()
    index = {w: i for i, w in enumerate(words)}

    # residuals[(key, layer)] -> {"last": [d], "arg": [d]}
    acts: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
    missing_span = 0
    for base in bases:
        if (base.category, base.func) not in fit_funcs:
            continue
        prompt = EvalPrompt(group="eval", key=base.key, text=base.prompt, base=base)
        span = arg_span(model, prompt, max_seq_len=max_seq_len)
        if span is None:
            missing_span += 1
            continue
        input_ids = model.encode(base.prompt, max_length=max_seq_len)
        with torch.no_grad(), ActivationRecorder(model.layers, at=layers) as rec:
            model.forward(input_ids)
            for ll in layers:
                h = rec.activations[ll][0].detach().float().cpu()
                acts[(base.key, ll)] = {
                    "last": h[-1].clone(),
                    "arg": h[span[0] : span[1]].mean(dim=0).clone(),
                }
    if missing_span:
        logger.warning("%d prompts had no locatable argument span", missing_span)

    if not acts:
        raise RuntimeError("no activations recorded; every argument span failed")
    d_model = next(iter(acts.values()))["last"].shape[0]
    banks: dict[str, dict[int, torch.Tensor]] = {
        "diffmean_last": {},
        "diffmean_arg": {},
    }
    counts: dict[str, int] = collections.Counter()
    for ll in layers:
        for where, arm in (("last", "diffmean_last"), ("arg", "diffmean_arg")):
            V = torch.zeros(len(words), d_model)
            for category, group in by_category.items():
                args = sorted({b.arg for b in group})
                for arg in args:
                    if arg not in index:
                        continue
                    pos, neg = [], []
                    for b in group:
                        if (b.category, b.func) not in fit_funcs:
                            continue
                        key = (b.key, ll)
                        if key not in acts:
                            continue
                        (pos if b.arg == arg else neg).append(acts[key][where])
                    if not pos or not neg:
                        raise RuntimeError(f"{arg}@{ll}: {len(pos)} pos, {len(neg)} neg")
                    V[index[arg]] = torch.stack(pos).mean(0) - torch.stack(neg).mean(0)
                    counts[f"{arg}"] = len(pos)
            banks[arm][ll] = V

    donors = _derange(len(words), spec.seed or 1)
    banks["diffmean_shuf"] = {ll: banks["diffmean_arg"][ll][donors].clone() for ll in layers}

    digests: dict[str, dict] = {}
    report: dict[str, dict[str, float]] = {}
    for arm, bank in banks.items():
        norms = []
        for ll in layers:
            V = bank[ll]
            if V.shape[0] != len(words):
                raise RuntimeError(f"{arm}@{ll} has {V.shape} for {len(words)} words")
            if not torch.isfinite(V).all():
                raise RuntimeError(f"{arm}@{ll} has non-finite entries")
            if float(V.norm(dim=-1).min()) < 1e-8:
                raise RuntimeError(f"{arm}@{ll} has a zero row; a word had no contrast")
            norms.append(float(V.norm(dim=-1).mean()))
            digests[f"{arm}|{ll}"] = {
                "group": arm,
                "layer": ll,
                "n": 1,
                "pullbacks": V,
            }
        report[arm] = {"mean_norm": round(statistics.mean(norms), 4)}

    out = spec.resolve_out("rq1")
    tag = spec.label or "diffmean"
    path = out / f"group_mean_digests_{spec.config_name}_{tag}.pt"
    torch.save({"digests": digests, "targets": words, "pullback_only": True}, path)

    expected = len(banks) * len(layers)
    if len(digests) != expected:
        raise RuntimeError(f"wrote {len(digests)} entries, expected {expected}")
    logger.info("%d arms x %d layers -> %s", len(banks), len(layers), path)
    for arm, stats in sorted(report.items()):
        logger.info("  %-20s mean norm %.3f", arm, stats["mean_norm"])

    return {
        "arms": sorted(report),
        "layers": layers,
        "n_words": len(words),
        "fit_templates": sorted(f"{c}/{f}" for c, f in fit_funcs),
        "heldout_templates": sorted(f"{c}/{f}" for c, f in held_out),
        "prompts_per_arg": min(counts.values()) if counts else 0,
        "n_entries": len(digests),
        "norms": report,
        "namespace": f"{spec.config_name}_{tag}",
        "path": str(path),
    }

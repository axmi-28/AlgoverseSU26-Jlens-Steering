"""C32-C35: what a filtered J-lens write changes, how long it lasts, how precisely.

One panel (``data/heldout/concept-use.json``) serves every stage. A *unit* is
(source entity, clue, target entity): a prefix that makes the model think of
the source without naming it, and a target concept to write in its place. Each
unit carries several *queries* appended after the prefix:

* ``express`` -- "The name of this country is": saying the concept.
* ``use`` -- "The capital city of this country is": computing with it.
* ``neutral`` -- an unrelated question on the same edited prefix: disruption.

**The design point everything rests on.** Edits are confined to the prefix
positions, and the model is causal, so the prefix residuals -- hence a swap's
per-position delta -- are bitwise the same whichever query follows. Expression
and use are therefore read off *one* intervention, not two similar ones. Only
the ``prompt`` position set (the conventional "every prompt position" recipe,
kept as a localisation arm) breaks this, and it is labelled as such.

**Prompt-only.** Generation decodes from a KV cache filled by the edited
prefill; nothing is re-injected at generated tokens. That is the same
intervention :func:`jsteer.steering.greedy_continuation` applies by recomputing
the whole sequence each step, at a sixth of the cost
(``test_cached_decoding_matches_recompute``).

Stages
------
* :func:`run_concept_clean` -- clean-model eligibility, frozen before any edit.
* :func:`run_concept_swap` -- C32 (express vs use) and C33 (distance), rank-2
  swaps with per-position norm matching.
* :func:`run_concept_fit` -- supervised baselines (DiffMean, template lens).
* :func:`run_concept_additive` -- C34/C35 target-only additions over a grid of
  directions x layer sets x position sets x doses.
* :func:`run_concept_freegen` -- C33b free generation after a prompt-only edit.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from jsteer.config import REPO_ROOT, load_config
from jsteer.steering import answer_matches
from jsteer.write_controls import (
    coordinates,
    file_hash,
    random_projection_axis,
    remove_projection,
    unit,
)

logger = logging.getLogger(__name__)

PANEL_PATH = REPO_ROOT / "data" / "heldout" / "concept-use.json"
BAND = (13, 16, 19, 22, 25, 28, 31)

#: Unrelated sentences for the distance manipulation (C33). Chosen to mention
#: no entity type in the panel, so the intervening text cannot itself carry or
#: contradict the concept.
FILLER = (
    " The weather that morning was mild, and a light breeze moved through the trees.",
    " Several people waited at the station, reading newspapers and drinking coffee.",
    " A delivery truck stopped outside, and the driver unloaded a few boxes.",
    " Later, the meeting was postponed because two of the speakers were delayed.",
    " The library stayed open until nine, as it usually did on weekdays.",
    " Meanwhile, the children finished their homework and went outside to play.",
    " The printer in the office jammed twice before anyone noticed.",
    " By noon the clouds had cleared and the streets were busy again.",
)

#: Second neutral query. Two, so disruption is not one fact's idiosyncrasy.
NEUTRAL_2 = {"text": " Separately, the color of the sky on a clear day is", "aliases": ["blue"]}


# --------------------------------------------------------------------------
# Panel


@dataclass(frozen=True)
class Query:
    kind: str  # express | use | neutral
    relation: str  # express | <relation> | neutral | neutral2
    text: str
    target: tuple[str, ...]
    source: tuple[str, ...]


@dataclass(frozen=True)
class Unit:
    type: str
    source: str
    target: str
    clue: str  # "0", "1" (latent) or "name" (explicit)
    prefix: str
    split: str  # dev | test | cross
    queries: tuple[Query, ...]

    @property
    def key(self) -> str:
        return f"{self.type}/{self.source}/{self.clue}->{self.target}"


def load_panel(path: Path = PANEL_PATH) -> dict:
    return json.loads(Path(path).read_text())


def entity_split(panel: dict) -> dict[str, str]:
    """Half of each type's entities to dev, half to test, by hash.

    Split at the *entity* level and use only within-half pairs for tuning and
    testing, so no word whose direction was looked at during development
    appears at test, as either source or target.
    """
    out = {}
    for ty, spec in panel["types"].items():
        names = sorted(
            spec["entities"],
            key=lambda e: hashlib.sha256(f"split|{ty}|{e}".encode()).hexdigest(),
        )
        for i, name in enumerate(names):
            out[name] = "dev" if i < len(names) // 2 else "test"
    return out


def _disjoint(a, b) -> bool:
    fa = {x.casefold() for x in a}
    fb = {x.casefold() for x in b}
    return not (fa & fb) and not any(
        x.startswith(y + " ") or y.startswith(x + " ") for x in fa for y in fb
    )


def build_units(panel: dict | None = None, *, clues=("0", "1", "name")) -> list[Unit]:
    """Every ordered within-type pair, crossed with clue variants.

    A relation is attached to a pair only if both entities have it and their
    answer sets are disjoint -- otherwise a "success" could be the source's
    own answer. Neutral queries are attached to every unit.
    """
    panel = panel or load_panel()
    split = entity_split(panel)
    neutral = panel["neutral_query"]
    units = []
    for ty, spec in panel["types"].items():
        ents = spec["entities"]
        for s in sorted(ents):
            for t in sorted(ents):
                if s == t:
                    continue
                queries = [
                    Query(
                        "express", "express", spec["express"],
                        tuple(ents[t].get("names", [t])), tuple(ents[s].get("names", [s])),
                    )
                ]
                for rel, text in spec["relations"].items():
                    if rel in ents[s] and rel in ents[t] and _disjoint(ents[s][rel], ents[t][rel]):
                        queries.append(
                            Query("use", rel, text, tuple(ents[t][rel]), tuple(ents[s][rel]))
                        )
                for name, q in (("neutral", neutral), ("neutral2", NEUTRAL_2)):
                    queries.append(
                        Query("neutral", name, q["text"], tuple(q["aliases"]), tuple(q["aliases"]))
                    )
                sp = split[s] if split[s] == split[t] else "cross"
                for clue in clues:
                    body = s if clue == "name" else ents[s]["clues"][int(clue)]
                    units.append(
                        Unit(ty, s, t, clue, panel["prefix"].format(clue=body), sp, tuple(queries))
                    )
    return units


def panel_hash(path: Path = PANEL_PATH) -> str:
    return file_hash(path)[:12]


def hits(text: str, aliases) -> bool:
    return any(answer_matches(text, a) for a in aliases)


# --------------------------------------------------------------------------
# Batched, prefix-only, KV-cached engine


class _PrefixEdit:
    """Adds ``deltas[layer][b, :P]`` to block outputs while ``active``.

    ``deltas[layer]`` is ``[B, P, d]``; rows of zeros leave a condition
    unedited at that layer. Decode steps run with ``active`` False, which is
    what makes the intervention prompt-only.
    """

    def __init__(self, model, deltas: dict[int, torch.Tensor]):
        self.model, self.deltas, self.active, self.handles = model, deltas, True, []

    def __enter__(self):
        for layer, delta in self.deltas.items():
            self.handles.append(
                self.model.layers[layer].register_forward_hook(self._hook(delta))
            )
        return self

    def _hook(self, delta):
        def hook(module, inputs, output):
            if not self.active:
                return None
            tensor = output if torch.is_tensor(output) else output[0]
            P = delta.shape[1]
            if tensor.shape[1] < P:
                raise RuntimeError("prefix edit longer than the sequence it edits")
            edited = tensor.clone()
            edited[:, :P] += delta.to(device=tensor.device, dtype=tensor.dtype)
            return edited if torch.is_tensor(output) else (edited, *output[1:])

        return hook

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


@contextmanager
def _record_last(model):
    store = {}

    def hook(module, inputs, output):
        store["h"] = output if torch.is_tensor(output) else output[0]

    handle = model.layers[model.n_layers - 1].register_forward_hook(hook)
    try:
        yield store
    finally:
        handle.remove()


def _forward(model, ids, *, cache=None, use_cache=False):
    text = getattr(model, "_text_module", None)
    if text is None:
        model.forward(ids)
        return None
    out = text(input_ids=ids, past_key_values=cache, use_cache=use_cache)
    return out.past_key_values if use_cache else None


def encode(model, text: str) -> torch.Tensor:
    return model.encode(text, max_length=1024)


@torch.no_grad()
def evaluate(
    model,
    ids: torch.Tensor,
    deltas: dict[int, torch.Tensor],
    *,
    answers: dict[str, list[int]] | None = None,
    probe: list[int] | None = None,
    gen_tokens: int = 0,
    chunk: int = 64,
    ref_logprobs: torch.Tensor | None = None,
) -> dict:
    """Run B conditions on one prompt; each differs only in its prefix edit.

    Returns, per condition: ``lp_last`` (``[B, V]`` final-position log-probs,
    only if ``ref_logprobs`` is None), ``kl`` from ``ref_logprobs``,
    ``answers[name]`` teacher-forced sequence log-probs, and ``texts``.
    """
    B = next(iter(deltas.values())).shape[0] if deltas else 1
    T = ids.shape[1]
    answers = answers or {}
    out: dict = {"kl": [], "top1": [], "texts": [], "ans": {k: [] for k in answers}, "probe": []}
    lp_keep = []
    for lo in range(0, B, chunk):
        hi = min(B, lo + chunk)
        part = {l: d[lo:hi] for l, d in deltas.items()}
        n = hi - lo
        batch = ids.expand(n, T)
        # 1. prompt prefill (+ greedy decoding from its cache)
        with _PrefixEdit(model, part) as edit, _record_last(model) as rec:
            cache = _forward(model, batch, use_cache=gen_tokens > 0)
            lp = torch.log_softmax(model.unembed(rec["h"][:, -1]).float(), -1)
            if ref_logprobs is not None:
                ref = ref_logprobs.to(lp.device)
                out["kl"] += (ref.exp() * (ref - lp)).sum(-1).clamp_min(0).tolist()
            else:
                lp_keep.append(lp.cpu())
            out["top1"] += lp.argmax(-1).tolist()
            if probe:
                out["probe"] += lp[:, torch.tensor(probe, device=lp.device)].tolist()
            if gen_tokens:
                produced = [lp.argmax(-1)]
                seq = torch.cat([batch, produced[-1][:, None]], 1)
                for _ in range(gen_tokens - 1):
                    if cache is None:  # models without a cache: recompute
                        _forward(model, seq)
                    else:
                        edit.active = False
                        cache = _forward(model, seq[:, -1:], cache=cache, use_cache=True)
                    nxt = model.unembed(rec["h"][:, -1]).float().argmax(-1)
                    produced.append(nxt)
                    seq = torch.cat([seq, nxt[:, None]], 1)
                toks = torch.stack(produced, 1).tolist()
                out["texts"] += [model.tokenizer.decode(t) for t in toks]
            del cache
        # 2. teacher-forced answer log-probs
        for name, a_ids in answers.items():
            a = torch.tensor([a_ids], device=ids.device)
            full = torch.cat([ids, a], 1).expand(n, T + len(a_ids))
            with _PrefixEdit(model, part), _record_last(model) as rec:
                _forward(model, full)
                h = rec["h"][:, T - 1 : T - 1 + len(a_ids)]
                lps = torch.log_softmax(model.unembed(h).float(), -1)
                tok = a.to(lps.device).expand(n, -1)
                out["ans"][name] += lps.gather(-1, tok[..., None])[..., 0].sum(-1).tolist()
    if ref_logprobs is None:
        out["lp_last"] = torch.cat(lp_keep)
    return out


@torch.no_grad()
def band_residuals(model, ids, layers) -> dict[int, torch.Tensor]:
    store = {}
    handles = []
    for l in layers:
        def hook(module, inputs, output, l=l):
            t = output if torch.is_tensor(output) else output[0]
            store[l] = t[0].detach().float().cpu()
        handles.append(model.layers[l].register_forward_hook(hook))
    try:
        _forward(model, ids)
    finally:
        for h in handles:
            h.remove()
    return store


# --------------------------------------------------------------------------
# Direction banks


def _seeded_gaussian(word: str, layer: int, d: int, salt: str = "rand") -> torch.Tensor:
    seed = int(hashlib.sha256(f"{salt}|{word}|{layer}".encode()).hexdigest()[:8], 16)
    return torch.randn(d, generator=torch.Generator().manual_seed(seed))


class Bank:
    """``vec(method, word, layer)`` for every direction family.

    Lens families come from a component digest (``{corpus}_{full,diag,off}``);
    ``proj`` is ``full`` with the target's own unembedding row removed (the
    beta=1 correction); ``logit`` is that row; ``rand`` a seeded Gaussian;
    anything else (``diffmean``, ``template``, ``tuned``) from extra digests in
    the same format.
    """

    def __init__(self, model, digest_paths: list[str], words: list[str]):
        from jsteer.loading import first_token_id, unembedding_rows

        self.rows: dict[tuple[str, int], torch.Tensor] = {}
        self.index: dict[str, dict[str, int]] = {}
        self.hashes = {}
        for p in digest_paths:
            blob = torch.load(p, map_location="cpu", weights_only=False)
            self.hashes[Path(p).name] = file_hash(p)
            targets = list(blob["targets"])
            for key, entry in blob["digests"].items():
                group, layer = key.split("|")
                if (group, int(layer)) in self.rows:
                    raise ValueError(f"{key} defined twice across digests")
                self.rows[group, int(layer)] = entry["pullbacks"].float()
                self.index[group] = {w: i for i, w in enumerate(targets)}
        ids = [first_token_id(model, w) for w in words]
        self.u = dict(zip(words, unembedding_rows(model, ids).float()))
        self.d = next(iter(self.u.values())).shape[0]

    def vec(self, method: str, word: str, layer: int) -> torch.Tensor:
        if method == "logit":
            return self.u[word]
        if method == "rand":
            return _seeded_gaussian(word, layer, self.d)
        if "_lam" in method:
            # J_lambda = J_off + lambda * J_diag, pulled back: the pullback is
            # linear, so this is exact from the stored components (full and
            # diag+off agree to 4e-8 relative in the digest) and needs no refit.
            base, _, lam = method.partition("_lam")
            return self.vec(f"{base}_off", word, layer) + float(lam) * self.vec(
                f"{base}_diag", word, layer
            )
        if method.endswith("_proj"):
            return remove_projection(self.vec(method[: -len("_proj")] + "_full", word, layer), self.u[word])
        if method.endswith("_rproj"):
            full = self.vec(method[: -len("_rproj")] + "_full", word, layer)
            g = torch.Generator().manual_seed(
                int(hashlib.sha256(f"rproj|{word}|{layer}".encode()).hexdigest()[:8], 16)
            )
            return remove_projection(full, random_projection_axis(full, self.u[word], g))
        try:
            return self.rows[method, layer][self.index[method][word]]
        except KeyError as e:
            raise KeyError(f"no direction for {method}/{word}/L{layer}") from e


# --------------------------------------------------------------------------
# Specs and bookkeeping


@dataclass
class ConceptSpec:
    stage: str  # clean | swap | fit | additive | freegen
    config_name: str = "qwen3-8b"
    digest_paths: list[str] = field(default_factory=list)
    panel_path: str = ""  # eligibility JSON from the clean stage
    corpora: list[str] = field(default_factory=lambda: ["gsm8k", "wikitext_a"])
    clues: list[str] = field(default_factory=lambda: ["0", "1", "name"])
    splits: list[str] = field(default_factory=lambda: ["dev", "test", "cross"])
    fillers: list[int] = field(default_factory=lambda: [0])
    layers: list[int] = field(default_factory=lambda: list(BAND))
    # swap
    swap_arms: list[str] = field(default_factory=list)
    #: ``prefix`` edits only the shared prefix, so the edit is identical under
    #: every query (the paired design). ``prompt`` edits every prompt position
    #: including the question, which is the convention the J-lens paper and the
    #: earlier two-hop panel used; the edit then differs per query and the
    #: paired reading is lost. Kept so the two can be compared directly.
    #: ``query`` edits only the question tokens (P..T-1), leaving the prefix
    #: clean: the test of whether the express/use dissociation is caused by the
    #: unembedding component sitting at the position that emits the answer.
    swap_positions: str = "prefix"
    doses: list[float] = field(default_factory=lambda: [0.25, 0.5, 1.0])
    # additive grid
    methods: list[str] = field(default_factory=list)
    layer_sets: dict[str, list[int]] = field(default_factory=dict)
    position_sets: list[str] = field(default_factory=lambda: ["prefix"])
    alphas: list[float] = field(default_factory=list)
    #: Explicit ``[method, layer_set, position_set, alpha]`` rows; overrides
    #: the product above (used once per-method configurations are frozen).
    condition_list: list[list] = field(default_factory=list)
    kinds: list[str] = field(default_factory=lambda: ["express", "use", "neutral"])
    one_use_per_unit: bool = False
    max_units: int = 0
    gen_tokens: int = 8
    chunk: int = 64
    label: str = ""
    shard: int = 0
    n_shards: int = 1
    out_dir: str = str(REPO_ROOT / "results")

    def manifest(self) -> dict:
        d = asdict(self)
        for k in ("shard", "n_shards", "out_dir", "chunk"):
            d.pop(k)
        # Basenames, not paths: the local and the volume copy of a run must
        # share a tag. Content is pinned by the hashes below.
        d["digest_paths"] = [Path(p).name for p in self.digest_paths]
        d["panel_path"] = Path(self.panel_path).name if self.panel_path else ""
        d["panel_sha"] = file_hash(PANEL_PATH)
        d["digest_sha"] = {Path(p).name: file_hash(p) for p in self.digest_paths}
        if self.panel_path:
            d["eligibility_sha"] = file_hash(self.panel_path)
        d["source_sha"] = file_hash(Path(__file__))
        return d

    def key(self) -> dict:
        """The manifest minus the source hash.

        The source hash is *recorded* (and a resumed run must match it), but it
        must not name the file: otherwise an unrelated edit to this module
        renames every finished output, which is this repo's cache-key bug
        pointed the other way. Staleness is caught by comparing the recorded
        hash, not by silently orphaning results.
        """
        return {k: v for k, v in self.manifest().items() if k != "source_sha"}

    @property
    def tag(self) -> str:
        h = hashlib.sha256(json.dumps(self.key(), sort_keys=True).encode()).hexdigest()[:10]
        lab = f"_{self.label}" if self.label else ""
        return f"{self.stage}{lab}_{self.config_name}_{h}"

    def out_path(self) -> Path:
        p = Path(self.out_dir) / "concept" / f"{self.tag}_shard{self.shard}of{self.n_shards}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def _shard(items, spec):
    return items[spec.shard :: spec.n_shards]


def _prompt(unit: Unit, query: Query, n_filler: int) -> tuple[str, str]:
    filler = "".join(FILLER[:n_filler])
    return unit.prefix, unit.prefix + filler + query.text


def _answer_ids(model, aliases) -> list[int]:
    return model.tokenizer.encode(" " + aliases[0], add_special_tokens=False)


def _eligible_units(spec: ConceptSpec) -> list[tuple[Unit, list[Query], dict]]:
    """Units and queries whose clean baseline is correct, from the frozen file."""
    elig = json.loads(Path(spec.panel_path).read_text())
    ok = elig["eligible"]  # key -> {filler: [relation, ...]}
    by_key = {u.key: u for u in build_units(clues=tuple(spec.clues))}
    out = []
    for key in sorted(ok):
        u = by_key.get(key)
        if u is None or u.split not in spec.splits:
            continue
        rels = ok[key]
        # A unit is usable only where its express query is eligible: every
        # contrast is expression vs use on the same edit. Filtering here (not
        # in the loop) keeps ``max_units`` a subsample of *usable* units.
        if not any("express" in rels.get(str(f), []) for f in spec.fillers):
            continue
        qs = [q for q in u.queries if q.kind in spec.kinds]
        out.append((u, qs, rels))
    if spec.max_units:
        out.sort(key=lambda x: hashlib.sha256(f"sub|{x[0].key}".encode()).hexdigest())
        out = out[: spec.max_units]
        out.sort(key=lambda x: x[0].key)
    return out


def _save(spec, payload):
    path = spec.out_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, allow_nan=False))
    tmp.replace(path)
    return path


def _setup(spec, dtype: str | None = None):
    from jsteer.loading import load_model

    config = load_config(spec.config_name)
    model = load_model(config, dtype=dtype)
    return config, model


# --------------------------------------------------------------------------
# Stage: clean eligibility


@torch.no_grad()
def run_concept_clean(spec: ConceptSpec) -> dict:
    """Clean greedy answers for every (unit, query, filler); nothing edited.

    A (unit, query, filler) is eligible when the clean model gives the
    source's answer and not the target's (neutral: gives the right answer),
    and the unit's express query is eligible at that filler. Written once and
    hashed; every later stage reads only this file.
    """
    config, model = _setup(spec)
    units = build_units(clues=tuple(spec.clues))
    if spec.max_units:
        units = units[: spec.max_units]
    units = _shard(units, spec)
    rows = []
    start = time.time()
    seen = {}
    for i, u in enumerate(units):
        for q in u.queries:
            for nf in spec.fillers:
                pre, text = _prompt(u, q, nf)
                cache_key = text
                if cache_key not in seen:
                    ids = encode(model, text)
                    pids = encode(model, pre)
                    if not torch.equal(ids[0, : pids.shape[1]], pids[0]):
                        raise RuntimeError(f"prefix does not tokenize as a prefix: {text!r}")
                    res = evaluate(model, ids, {}, gen_tokens=spec.gen_tokens)
                    seen[cache_key] = res["texts"][0]
                gen = seen[cache_key]
                rows.append(
                    {
                        "key": u.key, "relation": q.relation, "kind": q.kind, "filler": nf,
                        "text": gen, "hit_source": hits(gen, q.source), "hit_target": hits(gen, q.target),
                    }
                )
        if (i + 1) % 50 == 0:
            logger.info("clean %d/%d units %.0fs", i + 1, len(units), time.time() - start)
    payload = {"manifest": spec.manifest(), "rows": rows}
    path = _save(spec, payload)
    return {"path": str(path), "units": len(units), "rows": len(rows)}


def freeze_eligibility(rows: list[dict]) -> dict:
    """``key -> {filler: [eligible relations]}``; express must be eligible."""
    by = {}
    for r in rows:
        ok = r["hit_source"] and (r["kind"] == "neutral" or not r["hit_target"])
        by.setdefault(r["key"], {}).setdefault(str(r["filler"]), {})[r["relation"]] = ok
    out = {}
    for key, per in by.items():
        keep = {}
        for nf, rel in per.items():
            if rel.get("express"):
                keep[nf] = sorted(k for k, v in rel.items() if v)
        if keep:
            out[key] = keep
    return out


# --------------------------------------------------------------------------
# Stage: swaps (C32, C33)


SWAP_WRITERS = {
    # arm -> (source/target direction family suffix or special, matched?)
    "full": ("full", False),
    "off_m": ("off", True),
    "diag_m": ("diag", True),
    "proj_m": ("proj", True),
    "rproj_m": ("rproj", True),
    "logit_m": ("logit", True),
    "rand_m": ("rand", True),
}


def arm_spec(arm: str) -> dict:
    """``{fam, matched, cn, kappa, kappa_from}`` for a swap arm name.

    The clamp ``h += V(sigma(c) - c)``, ``c = V+h`` is invariant to *uniform*
    rescaling of ``V = [v_s, v_t]`` but not to scaling one column against the
    other, so ``kappa = ||v_t|| / ||v_s||`` is a free parameter of every arm.
    Matching the resulting delta's row norm (what ``_m`` does) equalises the
    edit's size but cannot undo a mis-weighted kappa, because the geometry is
    already baked in. These arms control it directly:

    * ``<fam>_cn`` -- both columns unit-normalised, so kappa = 1 by
      construction and arms differ only in direction.
    * ``<fam>_k<value>`` -- unit columns, then the target column scaled by an
      explicit kappa: the sweep.
    * ``full_koff`` / ``off_kfull`` -- unit columns, then the target column
      given the *other* family's per-pair kappa. The decisive arms: if `full`
      at `off`'s kappa reproduces `off`, the filtering effect was a rescaling
      effect.

    Legacy names keep their exact previous meaning, so earlier runs reproduce.
    """
    if arm in SWAP_WRITERS:
        fam, matched = SWAP_WRITERS[arm]
        return {"fam": fam, "matched": matched, "cn": False, "kappa": None, "kappa_from": None}
    head, _, tail = arm.partition("_")
    spec = {"fam": head, "matched": True, "cn": True, "kappa": None, "kappa_from": None}
    if tail == "m":
        spec["cn"] = False
        return spec
    if tail == "cn":
        return spec
    if tail.startswith("k"):
        rest = tail[1:]
        if rest in ("off", "full", "diag", "proj"):
            spec["kappa_from"] = rest
        else:
            spec["kappa"] = float(rest)
        return spec
    raise ValueError(f"unknown swap arm {arm!r}")


def arm_columns(bank: Bank, unit_: Unit, corpus: str, arm: str, layer: int) -> torch.Tensor:
    """``V = [v_source, v_target]`` for one arm, with its column scaling applied."""
    spec = arm_spec(arm)
    name = _family(corpus, spec["fam"])
    vs = bank.vec(name, unit_.source, layer)
    vt = bank.vec(name, unit_.target, layer)
    if not spec["cn"]:
        return torch.stack([vs, vt], 1)
    vs, vt = unit(vs), unit(vt)
    kappa = spec["kappa"]
    if spec["kappa_from"] is not None:
        donor = _family(corpus, spec["kappa_from"])
        kappa = float(
            bank.vec(donor, unit_.target, layer).norm() / bank.vec(donor, unit_.source, layer).norm()
        )
    if kappa is not None:
        vt = vt * kappa
    return torch.stack([vs, vt], 1)


def _family(corpus: str, fam: str) -> str:
    return fam if fam in ("logit", "rand") else f"{corpus}_{fam}"


def _match_or_keep(delta, ref, report, key):
    """Rescale each row to ``ref``'s norm, leaving rows that have no direction.

    A row of a swap delta is zero when the arm's two directions give that
    position identical coordinates: the arm has nothing to write there, and no
    rescaling can invent a direction. Those rows stay zero and are counted, so
    the arm is reported as writing at fewer positions rather than crashing
    (which is what it did: ``c32query`` lost two shards to it) or being
    silently rescaled from noise.
    """
    norm = delta.norm(dim=-1, keepdim=True)
    budget = ref.norm(dim=-1, keepdim=True)
    dead = (norm < 1e-10) & (budget > 1e-8)
    if dead.any():
        report[key] = report.get(key, 0) + int(dead.sum())
        if bool(dead.all()):
            raise ValueError(f"{key}: every row is a zero edit; the arm writes nothing")
    return delta * torch.where(dead, torch.zeros_like(budget), budget / norm.clamp_min(1e-10))


def swap_deltas(bank: Bank, h: dict[int, torch.Tensor], unit_: Unit, corpus: str, arms, P: int,
                report: dict | None = None):
    """``{arm: {layer: [P, d]}}`` at dose 1; matched arms take swap_full's row norms."""
    report = {} if report is None else report
    out = {arm: {} for arm in arms}
    for layer, hl in h.items():
        hp = hl[:P]
        V = {arm: arm_columns(bank, unit_, corpus, arm, layer) for arm in arms}
        Vf = torch.stack(
            [bank.vec(f"{corpus}_full", unit_.source, layer), bank.vec(f"{corpus}_full", unit_.target, layer)], 1
        )
        cf = coordinates(hp, Vf)
        ref = (cf.flip(-1) - cf) @ Vf.T
        for arm in arms:
            c = coordinates(hp, V[arm])
            delta = (c.flip(-1) - c) @ V[arm].T
            if arm_spec(arm)["matched"]:
                delta = _match_or_keep(delta, ref, report, f"{corpus}/{arm}/L{layer}")
            if not torch.isfinite(delta).all():
                raise ValueError(f"nonfinite swap delta for {arm}")
            out[arm][layer] = delta
    return out


def _arm_geometry(bank: Bank, spec: ConceptSpec, words: list[str]) -> dict:
    """Per arm and layer: cosine to the off-diagonal direction, to the raw
    unembedding row, and the direction's norm, averaged over words.

    Free -- no forward passes. This is what turns a lambda sweep into a
    statement about *where* the direction is pointing, not only what it does.
    """
    out: dict[str, dict] = {}
    for corpus in spec.corpora:
        for arm in spec.swap_arms:
            fam = arm_spec(arm)["fam"]
            name = _family(corpus, fam)
            for layer in spec.layers:
                cos_off, cos_u, norms = [], [], []
                for w in words:
                    try:
                        v = bank.vec(name, w, layer)
                    except KeyError:
                        continue
                    o = bank.vec(_family(corpus, "off"), w, layer)
                    u = bank.u[w]
                    cos_off.append(float(unit(v) @ unit(o)))
                    cos_u.append(float(unit(v) @ unit(u)))
                    norms.append(float(v.norm()))
                if norms:
                    out[f"{corpus}/{arm}/L{layer}"] = {
                        "cos_off": round(statistics.fmean(cos_off), 4),
                        "cos_uy": round(statistics.fmean(cos_u), 4),
                        "norm": round(statistics.fmean(norms), 4),
                    }
    return out


@torch.no_grad()
def run_concept_swap(spec: ConceptSpec) -> dict:
    config, model = _setup(spec)
    units = _shard(_eligible_units(spec), spec)
    words = sorted({w for u, _, _ in units for w in (u.source, u.target)})
    bank = Bank(model, spec.digest_paths, words)
    conds = [(c, a, s) for c in spec.corpora for a in spec.swap_arms for s in spec.doses]
    geometry = _arm_geometry(bank, spec, sorted({w for u, _, _ in units for w in (u.source, u.target)}))
    path = spec.out_path()
    payload = {
        "manifest": spec.manifest(), "conditions": conds, "records": [], "geometry": geometry,
    }
    done = set()
    if path.exists():
        old = json.loads(path.read_text())
        if old["manifest"] != payload["manifest"]:
            raise ValueError("existing output has a different manifest")
        payload = old
        payload["geometry"] = geometry
        done = {r["key"] for r in payload["records"]}
    start = time.time()
    # One clean generation per (prefix, query): every target sharing a prefix
    # has the same clean continuation, and there are ~17 of them.
    clean_gen: dict[str, tuple[str, torch.Tensor]] = {}
    dead_rows: dict[str, int] = {}
    for i, (u, qs, rels) in enumerate(units):
        if u.key in done:
            continue
        pre_ids = encode(model, u.prefix)
        P = pre_ids.shape[1]
        dev = next(model.layers[0].parameters()).device

        def build(ids_, n_pos):
            h = band_residuals(model, ids_, spec.layers)
            per = {c: swap_deltas(bank, h, u, c, spec.swap_arms, n_pos, dead_rows) for c in spec.corpora}
            d = {
                l: torch.stack([per[c][a][l] * s for c, a, s in conds]).to(dev)
                for l in spec.layers
            }
            return per, d, {l: torch.zeros_like(d[l][:1]) for l in spec.layers}

        if spec.swap_positions == "prefix":
            per_corpus, deltas, zero = build(pre_ids, P)
        elif spec.swap_positions not in ("prompt", "query"):
            raise ValueError(
                f"swap_positions must be prefix, prompt or query, got {spec.swap_positions!r}"
            )
        rec_rows = []
        for nf in spec.fillers:
            ok = set(rels.get(str(nf), []))
            if "express" not in ok:
                continue
            for q in qs:
                if q.relation not in ok:
                    continue
                _, text = _prompt(u, q, nf)
                ids = encode(model, text)
                if not torch.equal(ids[0, :P], pre_ids[0]):
                    raise RuntimeError("prefix mismatch")
                if spec.swap_positions in ("prompt", "query"):
                    per_corpus, deltas, zero = build(ids, ids.shape[1])
                    if spec.swap_positions == "query":
                        # Same deltas, prefix rows zeroed: the prefix keeps its
                        # clean residual and only the question carries the edit.
                        for l in spec.layers:
                            deltas[l][:, :P] = 0
                            for c in spec.corpora:
                                for a in spec.swap_arms:
                                    per_corpus[c][a][l][:P] = 0
                answers = {} if q.kind == "neutral" else {
                    "t": _answer_ids(model, q.target), "s": _answer_ids(model, q.source)
                }
                if text not in clean_gen:
                    g = evaluate(model, ids, zero, gen_tokens=spec.gen_tokens)
                    clean_gen[text] = (g["texts"][0], g["lp_last"][0])
                clean_text, ref = clean_gen[text]
                clean = (
                    evaluate(model, ids, zero, answers=answers)
                    if answers
                    else {"ans": {}}
                )
                res = evaluate(
                    model, ids, deltas, answers=answers, gen_tokens=spec.gen_tokens,
                    chunk=spec.chunk, ref_logprobs=ref,
                )
                tfirst = answers["t"][0] if answers else None
                row = {
                    "key": u.key, "type": u.type, "source": u.source, "target": u.target,
                    "clue": u.clue, "split": u.split, "relation": q.relation, "kind": q.kind,
                    "filler": nf, "n_tokens": ids.shape[1], "prefix_len": P,
                    "clean": {
                        "text": clean_text,
                        "t": clean["ans"].get("t", [None])[0], "s": clean["ans"].get("s", [None])[0],
                    },
                    "text": res["texts"], "kl": [round(x, 5) for x in res["kl"]],
                    "t": [round(x, 4) for x in res["ans"].get("t", [])],
                    "s": [round(x, 4) for x in res["ans"].get("s", [])],
                    "hit_t": [hits(x, q.target) for x in res["texts"]],
                    "hit_s": [hits(x, q.source) for x in res["texts"]],
                    "name_t": [u.target.casefold() in x.casefold() for x in res["texts"]],
                    "top1_t": None if tfirst is None else [x == tfirst for x in res["top1"]],
                }
                rec_rows.append(row)
        norms = {
            c: {a: {l: round(float(per_corpus[c][a][l].norm(dim=-1).mean()), 4) for l in spec.layers}
                for a in spec.swap_arms}
            for c in spec.corpora
        }
        payload["records"].extend(rec_rows)
        payload.setdefault("norms", {})[u.key] = norms
        if (i + 1) % 10 == 0 or i + 1 == len(units):
            _save(spec, payload)
            logger.info("swap %d/%d units %.0fs", i + 1, len(units), time.time() - start)
    # Every unit this shard was handed must have produced at least one record:
    # it is in the list only because its express query is eligible. A unit that
    # contributes nothing means the loop skipped it, and a shard that returns
    # normally while short is the failure mode this repo keeps hitting.
    produced = {r["key"] for r in payload["records"]}
    silent = [u.key for u, _, _ in units if u.key not in produced]
    if silent:
        raise RuntimeError(
            f"{len(silent)} units produced no records (e.g. {silent[:5]}); "
            "the shard is incomplete and must not be reported"
        )
    payload["dead_rows"] = dead_rows
    _save(spec, payload)
    return {
        "path": str(path), "units": len(units), "records": len(payload["records"]),
        "dead_rows": sum(dead_rows.values()), "dead_keys": len(dead_rows),
    }


# --------------------------------------------------------------------------
# Stage: supervised baselines


#: Sentences that mention an entity explicitly, for DiffMean and the template
#: lens. None shares wording with a clue or a query.
FIT_TEMPLATES = {
    "country": [
        "Last year my cousin spent two weeks traveling around {X}.",
        "The documentary we watched last night was mostly about {X}.",
        "Our company recently opened a new office in {X}.",
        "She wrote her thesis on the economic history of {X}.",
        "The delegation from {X} arrived at the summit on Tuesday.",
        "He has always wanted to learn more about the culture of {X}.",
        "The article compared the healthcare system of {X} with others.",
        "A friend of mine just moved to {X} for work.",
        "The museum exhibit featured old maps of {X}.",
        "Tourism in {X} grew steadily over the past decade.",
        "The trade agreement with {X} was signed after long negotiations.",
        "My neighbor keeps a small flag of {X} on his desk.",
    ],
    "person": [
        "The lecture this morning focused on the life of {X}.",
        "My grandfather owned an old biography of {X}.",
        "The museum has a portrait of {X} in its main hall.",
        "Many students write essays about {X} every year.",
        "A new film about {X} will be released next spring.",
        "The quote on the poster was attributed to {X}.",
        "Historians still debate the legacy of {X}.",
        "The exhibition included several letters written by {X}.",
        "Our teacher often told stories about {X}.",
        "The podcast episode was entirely devoted to {X}.",
        "A statue of {X} stands in the town square.",
        "The library displayed a first edition connected to {X}.",
    ],
    "element": [
        "The chemistry lab ordered a fresh sample of {X}.",
        "The textbook chapter this week covers {X}.",
        "The students measured the density of {X} in class.",
        "Industrial demand for {X} rose sharply this year.",
        "The report discussed how {X} is extracted and refined.",
        "Our teacher showed a short video about {X}.",
        "The museum case contained a labeled specimen of {X}.",
        "The research team studied the properties of {X}.",
        "The price of {X} changed a lot over the decade.",
        "A worksheet asked us to describe {X} in detail.",
        "The safety sheet explained how to handle {X}.",
        "The engineer explained why {X} was chosen for the design.",
    ],
}
#: The template lens needs contexts whose *natural continuation* is the word
#: (J-lens App. A.9.1). Truncating each sentence right before the entity gives
#: exactly that, and needs no cue. An earlier cue-based version ("The person
#: mentioned in that sentence is") was wrong for people: the natural
#: continuation there is the first name, not the surname the lens rows are
#: indexed by, and all seven persons scored 0 on its own top-1 diagnostic.
#: A cue that makes the entity itself the natural next token. Truncating the
#: sentence before the entity does not: after "traveling around" the argmax is
#: a function word, and the entity scored top-1 in 0 of 396 contexts. The
#: person cue asks for the surname, because the lens rows are indexed by
#: surname while the natural continuation of "the person mentioned" is a first
#: name -- which is what made all seven people score 0 on the first attempt.
TEMPLATE_CUE = {
    "country": " The country mentioned in that sentence is",
    "person": " The surname of the person mentioned in that sentence is",
    "element": " The element mentioned in that sentence is",
}


@torch.no_grad()
def run_concept_fit(spec: ConceptSpec) -> dict:
    """DiffMean and template-lens directions for every panel entity.

    * ``diffmean`` (AxBench-style mean difference): residual at the entity
      token, averaged over the templates, minus the same average over the
      type's other entities.
    * ``template`` (J-lens paper App. A.9.1): residual at the final position of
      a context whose natural continuation is the entity (the sentence plus a
      cue naming the type), averaged over templates, centred on the type's
      mean. The share of contexts where X is the clean top-1 is
      recorded, since the recipe assumes it.

    Supervision cost: 12 labelled sentences per entity, plus the type's
    entity list as negatives. No eval clue or query wording is used.
    """
    config, model = _setup(spec)
    panel = load_panel()
    digests, targets, top1 = {}, [], {}
    means = {("diffmean", l): [] for l in spec.layers} | {("template", l): [] for l in spec.layers}
    for ty, tspec in panel["types"].items():
        ents = sorted(tspec["entities"])
        acts = {m: {l: [] for l in spec.layers} for m in ("diffmean", "template")}
        for e in ents:
            from jsteer.loading import first_token_id

            eid = first_token_id(model, e)
            per = {m: {l: [] for l in spec.layers} for m in acts}
            hit = 0
            for tpl in FIT_TEMPLATES[ty]:
                sent = tpl.format(X=e)
                ids = encode(model, sent)
                pos = [i for i, t in enumerate(ids[0].tolist()) if t == eid]
                if len(pos) != 1:
                    raise RuntimeError(f"{e!r} not found exactly once in {sent!r}")
                h = band_residuals(model, ids, spec.layers)
                for l in spec.layers:
                    per["diffmean"][l].append(h[l][pos[0]])
                ids2 = encode(model, sent + TEMPLATE_CUE[ty])
                with _record_last(model) as rec:
                    h2 = band_residuals(model, ids2, spec.layers)
                    hit += int(model.unembed(rec["h"][0, -1]).argmax()) == eid
                for l in spec.layers:
                    per["template"][l].append(h2[l][-1])
            top1[e] = hit / len(FIT_TEMPLATES[ty])
            for m in acts:
                for l in spec.layers:
                    acts[m][l].append(torch.stack(per[m][l]).mean(0))
        for m in acts:
            for l in spec.layers:
                A = torch.stack(acts[m][l])  # [n_ents, d]
                n = len(ents)
                # mean over the OTHER entities of this type
                others = (A.sum(0, keepdim=True) - A) / (n - 1)
                means[m, l].append(A - others if m == "diffmean" else A - A.mean(0, keepdim=True))
        targets += ents
    for (m, l), blocks in means.items():
        digests[f"{m}|{l}"] = {"group": m, "layer": l, "n": 12, "pullbacks": torch.cat(blocks).float()}
    path = Path(spec.out_dir) / "concept" / f"baselines_{spec.tag}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"digests": digests, "targets": targets, "template_top1": top1}, path)
    return {"path": str(path), "targets": len(targets), "template_top1_mean": sum(top1.values()) / len(top1)}


# --------------------------------------------------------------------------
# Stage: additive grid (C34, C35)


def position_index(name: str, P: int, T: int) -> list[int]:
    """Positions edited. Position 0 is excluded throughout: it is the
    attention sink, its norm is an order of magnitude above the rest, and a
    fixed-norm vector there is a different intervention from one elsewhere."""
    if name == "prefix":
        return list(range(1, P))
    if name == "last":
        return [P - 1]
    if name == "tail3":
        return list(range(max(1, P - 3), P))
    if name == "prompt":
        return list(range(1, T))
    raise ValueError(name)


def _conditions(spec: ConceptSpec):
    if spec.condition_list:
        return [tuple(c) for c in spec.condition_list]
    return [
        (m, ls, ps, a)
        for m in spec.methods
        for ls in spec.layer_sets
        for ps in spec.position_sets
        for a in spec.alphas
    ]


@torch.no_grad()
def run_concept_additive(spec: ConceptSpec) -> dict:
    """Positive target-only additions: ``alpha * s_l * unit(v_target)``.

    ``s_l`` is the clean prefix's mean residual norm at layer ``l`` over
    positions 1..P-1 -- shared by every method, so at a given alpha every
    method spends the same per-position norm (equal-norm comparison). The
    source is never subtracted.

    With ``gen_tokens=0`` only teacher-forced scores are computed (the dev
    grid); the test run adds greedy text.
    """
    config, model = _setup(spec)
    units = _shard(_eligible_units(spec), spec)
    words = sorted({w for u, _, _ in units for w in (u.source, u.target)})
    bank = Bank(model, spec.digest_paths, words)
    conds = _conditions(spec)
    dev = next(model.layers[0].parameters()).device
    path = spec.out_path()
    payload = {"manifest": spec.manifest(), "conditions": conds, "records": []}
    done = set()
    if path.exists():
        old = json.loads(path.read_text())
        if old["manifest"] != payload["manifest"]:
            raise ValueError("existing output has a different manifest")
        payload, done = old, {r["key"] for r in old["records"]}
    methods = sorted({m for m, *_ in conds})
    start = time.time()
    for i, (u, qs, rels) in enumerate(units):
        if u.key in done:
            continue
        pre_ids = encode(model, u.prefix)
        P = pre_ids.shape[1]
        h = band_residuals(model, pre_ids, spec.layers)
        scale = {l: float(h[l][1:P].norm(dim=-1).mean()) for l in spec.layers}
        U = {l: torch.stack([unit(bank.vec(m, u.target, l)) for m in methods]).to(dev) for l in spec.layers}
        m_idx = torch.tensor([methods.index(m) for m, *_ in conds], device=dev)
        ok = set(rels.get("0", []))
        qsel = [q for q in qs if q.relation in ok]
        if spec.one_use_per_unit:
            uses = [q for q in qsel if q.kind == "use"]
            keep = uses[int(hashlib.sha256(u.key.encode()).hexdigest(), 16) % len(uses)] if uses else None
            qsel = [q for q in qsel if q.kind != "use" or q is keep]
        for q in qsel:
            _, text = _prompt(u, q, 0)
            ids = encode(model, text)
            T = ids.shape[1]
            pos_mask = torch.zeros(len(conds), T)
            for k, (m, ls, ps, a) in enumerate(conds):
                pos_mask[k, position_index(ps, P, T)] = a
            pos_mask = pos_mask.to(dev)
            deltas = {}
            for l in spec.layers:
                in_set = torch.tensor(
                    [l in spec.layer_sets[ls] for _, ls, _, _ in conds], device=dev
                )
                coef = pos_mask * (in_set[:, None] * scale[l])
                deltas[l] = coef[:, :, None] * U[l][m_idx][:, None, :]
            t_ids, s_ids = _answer_ids(model, q.target), _answer_ids(model, q.source)
            # The dev grid is scored from ONE forward per condition: top-1,
            # the two answers' first-token log-probs, and the neutral KL. The
            # teacher-forced sequence scores cost two more passes each and are
            # only needed at test, where the equal-success crossing uses them.
            answers = {} if spec.gen_tokens == 0 or q.kind == "neutral" else {"t": t_ids, "s": s_ids}
            probe = [t_ids[0], s_ids[0]]
            zero = {l: torch.zeros_like(d[:1]) for l, d in deltas.items()}
            clean = evaluate(model, ids, zero, answers=answers, probe=probe)
            ref = clean["lp_last"][0]
            res = evaluate(
                model, ids, deltas, answers=answers, probe=probe,
                gen_tokens=spec.gen_tokens, chunk=spec.chunk, ref_logprobs=ref,
            )
            first = t_ids[0]
            row = {
                "key": u.key, "type": u.type, "source": u.source, "target": u.target, "clue": u.clue,
                "split": u.split, "relation": q.relation, "kind": q.kind, "prefix_len": P, "n_tokens": T,
                "clean": {k: v[0] for k, v in clean["ans"].items()}
                | {"top1": clean["top1"][0] == first, "pt": round(clean["probe"][0][0], 4),
                   "ps": round(clean["probe"][0][1], 4)},
                "kl": [round(x, 5) for x in res["kl"]],
                "top1": [x == first for x in res["top1"]],
                "pt": [round(x[0], 4) for x in res["probe"]],
                "ps": [round(x[1], 4) for x in res["probe"]],
                **{k: [round(x, 4) for x in v] for k, v in res["ans"].items()},
            }
            if spec.gen_tokens:
                row["text"] = res["texts"]
                row["hit_t"] = [hits(x, q.target) for x in res["texts"]]
                row["hit_s"] = [hits(x, q.source) for x in res["texts"]]
                row["name_t"] = [u.target.casefold() in x.casefold() for x in res["texts"]]
            payload["records"].append(row)
        if (i + 1) % 10 == 0 or i + 1 == len(units):
            _save(spec, payload)
            logger.info("additive %d/%d units %.0fs", i + 1, len(units), time.time() - start)
    _save(spec, payload)
    return {"path": str(path), "units": len(units), "records": len(payload["records"]), "conditions": len(conds)}


# --------------------------------------------------------------------------
# Stage: free generation (C33b)

FREEGEN_CUE = " Here is a short description of this country:"
FREEGEN_PROBE = "\n\nIn one word, the capital city of this country is"


@torch.no_grad()
def run_concept_freegen(spec: ConceptSpec) -> dict:
    """Prompt-only swap, then free text; then ask the capital after that text.

    Scores mentions of the target and source (name, capital, language) in
    three windows of the generated text, and the capital probe appended after
    the model's own text -- the concept surviving a self-generated distance.
    Disruption: the clean model's mean NLL of the steered text.
    """
    config, model = _setup(spec)
    panel = load_panel()
    ents = panel["types"]["country"]["entities"]
    units = [x for x in _eligible_units(spec) if x[0].type == "country"]
    units = _shard(units, spec)
    words = sorted({w for u, _, _ in units for w in (u.source, u.target)})
    bank = Bank(model, spec.digest_paths, words)
    conds = [(c, a, s) for c in spec.corpora for a in spec.swap_arms for s in spec.doses]
    dev = next(model.layers[0].parameters()).device
    records = []
    start = time.time()
    for i, (u, _, _) in enumerate(units):
        pre_ids = encode(model, u.prefix)
        P = pre_ids.shape[1]
        h = band_residuals(model, pre_ids, spec.layers)
        per = {c: swap_deltas(bank, h, u, c, spec.swap_arms, P) for c in spec.corpora}
        deltas = {
            l: torch.cat([
                torch.zeros(1, P, bank.d),
                torch.stack([per[c][a][l] * s for c, a, s in conds]),
            ]).to(dev)
            for l in spec.layers
        }
        ids = encode(model, u.prefix + FREEGEN_CUE)
        res = evaluate(model, ids, deltas, gen_tokens=spec.gen_tokens, chunk=spec.chunk)
        texts = res["texts"]
        # clean-model NLL of each continuation (disruption)
        nll, probe = [], []
        for text in texts:
            cont = model.tokenizer.encode(text, add_special_tokens=False)
            full = torch.cat([ids, torch.tensor([cont], device=ids.device)], 1)
            with _record_last(model) as rec:
                _forward(model, full)
                lp = torch.log_softmax(model.unembed(rec["h"][0, ids.shape[1] - 1 : -1]).float(), -1)
            nll.append(float(-lp.gather(-1, torch.tensor(cont, device=lp.device)[:, None]).mean()))
        # capital probe after each arm's own text, same prefix edit
        for k, text in enumerate(texts):
            pids = encode(model, u.prefix + FREEGEN_CUE + text + FREEGEN_PROBE)
            if not torch.equal(pids[0, :P], pre_ids[0]):
                raise RuntimeError("prefix mismatch in probe")
            one = {l: d[k : k + 1] for l, d in deltas.items()}
            r = evaluate(model, pids, one, gen_tokens=4)
            probe.append(r["texts"][0])
        tok = [model.tokenizer.encode(t, add_special_tokens=False) for t in texts]

        def windows(k, e):
            names = [e] + ents[e].get("capital", []) + ents[e].get("language", [])
            out = []
            for lo, hi in ((0, 16), (16, 32), (32, 64)):
                seg = model.tokenizer.decode(tok[k][lo:hi]).casefold()
                out.append(any(n.casefold() in seg for n in names))
            return out

        for k, cond in enumerate([("clean", "clean", 0.0)] + conds):
            records.append(
                {
                    "key": u.key, "source": u.source, "target": u.target, "clue": u.clue,
                    "split": u.split, "cond": list(cond), "text": texts[k], "nll": round(nll[k], 4),
                    "win_t": windows(k, u.target), "win_s": windows(k, u.source),
                    "probe": probe[k],
                    "probe_t": "capital" in ents[u.target] and hits(probe[k], ents[u.target]["capital"]),
                    "probe_s": "capital" in ents[u.source] and hits(probe[k], ents[u.source]["capital"]),
                }
            )
        if (i + 1) % 10 == 0:
            logger.info("freegen %d/%d %.0fs", i + 1, len(units), time.time() - start)
    payload = {"manifest": spec.manifest(), "conditions": conds, "records": records}
    path = _save(spec, payload)
    return {"path": str(path), "units": len(units), "records": len(records)}


# --------------------------------------------------------------------------
# Stage: realized edit magnitude (C38)


class _MeasuredEdit(_PrefixEdit):
    """``_PrefixEdit`` that first measures the activation arriving at each hook.

    For a unit-column swap the natural edit is the Householder reflection
    ``-2 w (w^T h)``, of norm ``2|w^T h|``. The swap stages compute that edit
    once, on the *clean* residual, and add it as a fixed tensor; downstream
    band layers therefore receive a state ``h~`` that earlier edits have moved.
    ``stats[layer] = (2|w^T h~|, ||h~||)``, each ``[B, P]``, is what a runtime
    Householder would have applied there, recorded before this hook edits.
    """

    def __init__(self, model, deltas, axes: dict[int, torch.Tensor]):
        super().__init__(model, deltas)
        self.axes, self.stats = axes, {}

    def __enter__(self):
        for layer, delta in self.deltas.items():
            self.handles.append(
                self.model.layers[layer].register_forward_hook(self._measure(layer, delta))
            )
        return self

    def _measure(self, layer, delta):
        edit = self._hook(delta)

        def hook(module, inputs, output):
            if self.active and layer not in self.stats:
                t = (output if torch.is_tensor(output) else output[0])[:, : delta.shape[1]].float()
                w = self.axes[layer].to(device=t.device, dtype=torch.float32)
                proj = torch.einsum("bpd,bd->bp", t, w)
                self.stats[layer] = ((2 * proj.abs()).cpu(), t.norm(dim=-1).cpu())
            return edit(module, inputs, output)

        return hook


@torch.no_grad()
def _teacher_forced(model, ids, a_ids, deltas, axes=None):
    """Sequence log-prob of ``a_ids`` after ``ids`` under each condition's edit."""
    B = next(iter(deltas.values())).shape[0] if deltas else 1
    T = ids.shape[1]
    a = torch.tensor([a_ids], device=ids.device)
    full = torch.cat([ids, a], 1).expand(B, T + len(a_ids))
    edit = _MeasuredEdit(model, deltas, axes) if axes is not None else _PrefixEdit(model, deltas)
    with edit, _record_last(model) as rec:
        _forward(model, full)
        lps = torch.log_softmax(model.unembed(rec["h"][:, T - 1 : T - 1 + len(a_ids)]).float(), -1)
        lp = lps.gather(-1, a.to(lps.device).expand(B, -1)[..., None])[..., 0].sum(-1)
    return lp.cpu(), getattr(edit, "stats", None)


def householder_sites(hp: torch.Tensor, V: torch.Tensor) -> dict:
    """Clean-state Householder quantities for one unit-column swap ``V = [s, t]``.

    ``native`` is the norm of the implemented pseudoinverse swap before any
    matching; for unit columns it equals ``m_clean = 2|w^T h|`` exactly (up to
    the solver's 1e-6 ridge), which is checked rather than assumed.
    """
    s, t = V[:, 0], V[:, 1]
    rho = float(s @ t)
    w = unit(t - s)
    c = coordinates(hp, V)
    native = ((c.flip(-1) - c) @ V.T).norm(dim=-1)
    return {
        "w": w, "rho": rho,
        "kappa2": math.sqrt((1 + abs(rho)) / (1 - abs(rho))),
        "m_clean": 2 * (hp @ w).abs(), "native": native,
    }


@torch.no_grad()
def run_concept_magnitude(spec: ConceptSpec) -> dict:
    """C38: the lambda sweep's edits, re-run with every site's magnitude logged.

    Per (use trial, condition, band layer, edited position): ``m_clean =
    2|w^T h0|``, ``m_actual = 2|w^T h~|`` on the intervened run, the norm of
    the edit actually applied (the full lens's clean-state delta norm, since
    every ``lam*_cn`` arm is row-matched to it), the unmatched native norm,
    ``||h0||`` and ``||h~||``. Per (condition, layer): rho, kappa_2. Per trial
    and condition: teacher-forced log-probs of the target answer, the source
    answer and the target's name, so the relational margin R = t - s and the
    leakage margin L = name - t are continuous. Deltas are built exactly as in
    ``run_concept_swap``, so the t/s columns must reproduce C37's.
    """
    if spec.doses != [1.0]:
        raise ValueError("the magnitude stage measures dose 1.0 only")
    if any(not arm_spec(a)["cn"] for a in spec.swap_arms):
        raise ValueError("Householder quantities are defined for unit-column arms only")
    config, model = _setup(spec)
    units = _shard(_eligible_units(spec), spec)
    bank = Bank(model, spec.digest_paths, sorted({w for u, _, _ in units for w in (u.source, u.target)}))
    conds = [(c, a) for c in spec.corpora for a in spec.swap_arms]
    dev = next(model.layers[0].parameters()).device
    records, sites = [], {}
    checks = {"native_vs_mclean": 0.0, "first_layer_actual_vs_clean": 0.0, "applied_vs_full": 0.0}
    dead_rows: dict[str, int] = {}
    start = time.time()
    first = min(spec.layers)
    for i, (u, qs, rels) in enumerate(units):
        pre_ids = encode(model, u.prefix)
        P = pre_ids.shape[1]
        n_before = len(records)
        for nf in spec.fillers:
            ok = set(rels.get(str(nf), []))
            if "express" not in ok:
                continue
            for q in qs:
                if q.kind != "use" or q.relation not in ok:
                    continue
                _, text = _prompt(u, q, nf)
                ids = encode(model, text)
                if not torch.equal(ids[0, :P], pre_ids[0]):
                    raise RuntimeError("prefix mismatch")
                if spec.swap_positions == "prefix":
                    h, n_pos = band_residuals(model, pre_ids, spec.layers), P
                elif spec.swap_positions == "prompt":
                    h, n_pos = band_residuals(model, ids, spec.layers), ids.shape[1]
                else:
                    raise ValueError("magnitude stage supports prefix and prompt positions")
                per = {c: swap_deltas(bank, h, u, c, spec.swap_arms, n_pos, dead_rows) for c in spec.corpora}
                geo = {
                    (c, a): {l: householder_sites(h[l][:n_pos], arm_columns(bank, u, c, a, l)) for l in spec.layers}
                    for c, a in conds
                }
                deltas = {l: torch.stack([per[c][a][l] for c, a in conds]).to(dev) for l in spec.layers}
                axes = {l: torch.stack([geo[k][l]["w"] for k in conds]) for l in spec.layers}
                ans = {
                    "t": _answer_ids(model, q.target), "s": _answer_ids(model, q.source),
                    "n": _answer_ids(model, (u.target,)),
                }
                lp, stats = {}, None
                for name, a_ids in ans.items():
                    lp[name], st = _teacher_forced(model, ids, a_ids, deltas, axes if stats is None else None)
                    stats = stats if stats is not None else st
                clean = {name: float(_teacher_forced(model, ids, a_ids, {})[0][0]) for name, a_ids in ans.items()}
                # Full-lens reference norm per site, for the applied-norm check.
                Vf = {c: {l: torch.stack([bank.vec(f"{c}_full", u.source, l), bank.vec(f"{c}_full", u.target, l)], 1)
                          for l in spec.layers} for c in spec.corpora}
                arr = {k: torch.zeros(len(conds), len(spec.layers), n_pos) for k in
                       ("m_clean", "m_actual", "native", "applied", "h_actual")}
                h_clean = torch.stack([h[l][:n_pos].norm(dim=-1) for l in spec.layers])
                for li, l in enumerate(spec.layers):
                    m_act, h_act = stats[l]
                    arr["m_actual"][:, li], arr["h_actual"][:, li] = m_act, h_act
                    for ci, (c, a) in enumerate(conds):
                        g = geo[c, a][l]
                        arr["m_clean"][ci, li], arr["native"][ci, li] = g["m_clean"], g["native"]
                        arr["applied"][ci, li] = per[c][a][l].norm(dim=-1)
                    for c in spec.corpora:
                        cf = coordinates(h[l][:n_pos], Vf[c][l])
                        ref = ((cf.flip(-1) - cf) @ Vf[c][l].T).norm(dim=-1)
                        for ci, (cc, a) in enumerate(conds):
                            if cc == c:
                                live = arr["applied"][ci, li] > 0
                                err = ((arr["applied"][ci, li] - ref).abs() / ref.clamp_min(1e-6))[live]
                                if err.numel():
                                    checks["applied_vs_full"] = max(checks["applied_vs_full"], float(err.max()))
                big = arr["m_clean"] > 1e-3 * arr["m_clean"].amax()
                checks["native_vs_mclean"] = max(
                    checks["native_vs_mclean"],
                    float(((arr["native"] - arr["m_clean"]).abs() / arr["m_clean"].clamp_min(1e-6))[big].max()),
                )
                li0 = spec.layers.index(first)
                # The first band layer sees no earlier edit: h~ must equal h0 there
                # (to bf16 precision), which validates the recording hook.
                b0 = big[:, li0]
                checks["first_layer_actual_vs_clean"] = max(
                    checks["first_layer_actual_vs_clean"],
                    float(((arr["m_actual"][:, li0] - arr["m_clean"][:, li0]).abs()
                           / arr["m_clean"][:, li0].clamp_min(1e-6))[b0].median()),
                )
                tid = f"{u.key}|{q.relation}|f{nf}"
                sites[tid] = {**arr, "h_clean": h_clean}
                records.append({
                    "trial": tid, "key": u.key, "type": u.type, "source": u.source, "target": u.target,
                    "clue": u.clue, "split": u.split, "relation": q.relation, "filler": nf,
                    "n_tokens": ids.shape[1], "prefix_len": P, "n_pos": n_pos,
                    "ans_len": {k: len(v) for k, v in ans.items()},
                    "clean": {k: round(v, 4) for k, v in clean.items()},
                    **{k: [round(float(x), 4) for x in v] for k, v in lp.items()},
                    "rho": [[round(geo[k][l]["rho"], 5) for l in spec.layers] for k in conds],
                    "kappa2": [[round(geo[k][l]["kappa2"], 4) for l in spec.layers] for k in conds],
                    # Projective distance to the lambda=0 and lambda=1 axes.
                    **{key: [[round(math.sqrt(max(0.0, 1 - float(geo[k][l]["w"] @ geo[k[0], ref][l]["w"]) ** 2)), 5)
                              for l in spec.layers] for k in conds]
                       for ref, key in (("lam0_cn", "dP0"), ("lam1_cn", "dP1")) if ref in spec.swap_arms},
                })
        if any(q.kind == "use" and q.relation in set(rels.get("0", [])) for q in qs) and len(records) == n_before:
            raise RuntimeError(f"{u.key} has an eligible use query but produced no record")
        if (i + 1) % 20 == 0 or i + 1 == len(units):
            logger.info("magnitude %d/%d units %.0fs %s", i + 1, len(units), time.time() - start, checks)
    payload = {"manifest": spec.manifest(), "conditions": conds, "layers": spec.layers,
               "records": records, "checks": checks, "dead_rows": dead_rows}
    path = _save(spec, payload)
    torch.save(sites, path.with_suffix(".sites.pt"))
    return {"path": str(path), "units": len(units), "records": len(records), "checks": checks,
            "dead_rows": sum(dead_rows.values())}


#: Sites where the arriving state barely loads on the axis are skipped, not
#: stabilised: the sign of w^T h~ is noise there and c = m*/m_current explodes.
#: Relative to ||h~|| so the threshold means the same at every layer.
TAU_M_REL = 1e-3


class _DynamicHouseholder:
    """Axis-preserving edit of fixed local norm, computed on the arriving state.

    At each hook: ``m_current = 2|w^T h~|``; where ``m_current > TAU_M_REL *
    ||h~||`` apply ``-2 c w (w^T h~)`` with ``c = m*/m_current`` (norm exactly
    ``m*``), else leave the site unedited and count it. Records the achieved
    norm of the edit as actually added (after the cast to the model dtype).
    """

    def __init__(self, model, axes, mstar):
        self.model, self.axes, self.mstar = model, axes, mstar
        self.handles, self.achieved, self.skipped = [], {}, {}

    def __enter__(self):
        for layer in self.axes:
            self.handles.append(self.model.layers[layer].register_forward_hook(self._hook(layer)))
        return self

    def _hook(self, layer):
        def hook(module, inputs, output):
            tensor = output if torch.is_tensor(output) else output[0]
            ms = self.mstar[layer].to(tensor.device, torch.float32)  # [B, P]
            P = ms.shape[1]
            h = tensor[:, :P].float()
            w = self.axes[layer].to(tensor.device, torch.float32)  # [B, d]
            proj = torch.einsum("bpd,bd->bp", h, w)
            m_cur = 2 * proj.abs()
            keep = (m_cur > TAU_M_REL * h.norm(dim=-1)) & (ms > 0)
            scale = torch.where(keep, -2 * ms / m_cur.clamp_min(1e-12) * proj, torch.zeros_like(proj))
            delta = (scale[..., None] * w[:, None, :]).to(tensor.dtype)
            edited = tensor.clone()
            edited[:, :P] += delta
            self.achieved[layer] = delta.float().norm(dim=-1).cpu()
            self.skipped[layer] = (~keep & (ms > 0)).cpu()
            return edited if torch.is_tensor(output) else (edited, *output[1:])

        return hook

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def _tf_dynamic(model, ids, a_ids, axes, mstar):
    B = next(iter(axes.values())).shape[0]
    T = ids.shape[1]
    a = torch.tensor([a_ids], device=ids.device)
    full = torch.cat([ids, a], 1).expand(B, T + len(a_ids))
    with _DynamicHouseholder(model, axes, mstar) as edit, _record_last(model) as rec:
        _forward(model, full)
        lps = torch.log_softmax(model.unembed(rec["h"][:, T - 1 : T - 1 + len(a_ids)]).float(), -1)
        lp = lps.gather(-1, a.to(lps.device).expand(B, -1)[..., None])[..., 0].sum(-1)
    return lp.cpu(), edit


@torch.no_grad()
def run_concept_magmatched(spec: ConceptSpec) -> dict:
    """C38b: the mentor's dynamic magnitude-matched control (spec section 3.5).

    Per trial, first reproduces the C38 run (fixed clean-state deltas, the
    measuring hook) to get ``m_actual`` for every lambda; the per-site target
    is ``m* = min over lambda of m_actual`` within each corpus, so the control
    mostly scales edits down. Then every lambda is re-run with the dynamic
    Householder of norm exactly ``m*`` at every site, along that lambda's own
    axis, on the state that actually arrives. Same teacher-forced t/s/n.
    """
    if spec.doses != [1.0]:
        raise ValueError("the matched stage measures dose 1.0 only")
    config, model = _setup(spec)
    units = _shard(_eligible_units(spec), spec)
    bank = Bank(model, spec.digest_paths, sorted({w for u, _, _ in units for w in (u.source, u.target)}))
    conds = [(c, a) for c in spec.corpora for a in spec.swap_arms]
    dev = next(model.layers[0].parameters()).device
    records = []
    checks = {"achieved_vs_mstar": 0.0, "skipped_sites": 0, "sites": 0}
    dead_rows: dict[str, int] = {}
    start = time.time()
    for i, (u, qs, rels) in enumerate(units):
        pre_ids = encode(model, u.prefix)
        P = pre_ids.shape[1]
        n_before = len(records)
        for nf in spec.fillers:
            ok = set(rels.get(str(nf), []))
            if "express" not in ok:
                continue
            for q in qs:
                if q.kind != "use" or q.relation not in ok:
                    continue
                _, text = _prompt(u, q, nf)
                ids = encode(model, text)
                if not torch.equal(ids[0, :P], pre_ids[0]):
                    raise RuntimeError("prefix mismatch")
                if spec.swap_positions == "prefix":
                    h, n_pos = band_residuals(model, pre_ids, spec.layers), P
                elif spec.swap_positions == "prompt":
                    h, n_pos = band_residuals(model, ids, spec.layers), ids.shape[1]
                else:
                    raise ValueError("matched stage supports prefix and prompt positions")
                per = {c: swap_deltas(bank, h, u, c, spec.swap_arms, n_pos, dead_rows) for c in spec.corpora}
                deltas = {l: torch.stack([per[c][a][l] for c, a in conds]).to(dev) for l in spec.layers}
                axes = {
                    l: torch.stack([householder_sites(h[l][:n_pos], arm_columns(bank, u, c, a, l))["w"]
                                    for c, a in conds])
                    for l in spec.layers
                }
                ans = {
                    "t": _answer_ids(model, q.target), "s": _answer_ids(model, q.source),
                    "n": _answer_ids(model, (u.target,)),
                }
                _, stats = _teacher_forced(model, ids, ans["t"], deltas, axes)
                mstar = {}
                for l in spec.layers:
                    m_act = stats[l][0]  # [B, n_pos]
                    ms = torch.empty_like(m_act)
                    for c in spec.corpora:
                        idx = [k for k, (cc, _) in enumerate(conds) if cc == c]
                        ms[idx] = m_act[idx].amin(0, keepdim=True).expand(len(idx), -1)
                    mstar[l] = ms
                lp, edit = {}, None
                for name, a_ids in ans.items():
                    lp[name], e = _tf_dynamic(model, ids, a_ids, axes, mstar)
                    edit = edit or e
                clean = {name: float(_teacher_forced(model, ids, a_ids, {})[0][0]) for name, a_ids in ans.items()}
                skipped = torch.stack([edit.skipped[l] for l in spec.layers], 1)  # [B, L, P]
                for l in spec.layers:
                    live = ~edit.skipped[l] & (mstar[l] > 0)
                    if live.any():
                        err = ((edit.achieved[l] - mstar[l]).abs() / mstar[l].clamp_min(1e-6))[live]
                        checks["achieved_vs_mstar"] = max(checks["achieved_vs_mstar"], float(err.max()))
                checks["skipped_sites"] += int(skipped.sum())
                checks["sites"] += skipped.numel()
                mst = torch.stack([mstar[l] for l in spec.layers], 1)  # [B, L, P]
                records.append({
                    "trial": f"{u.key}|{q.relation}|f{nf}", "key": u.key, "type": u.type,
                    "source": u.source, "target": u.target, "clue": u.clue, "split": u.split,
                    "relation": q.relation, "filler": nf, "n_pos": n_pos,
                    "ans_len": {k: len(v) for k, v in ans.items()},
                    "clean": {k: round(v, 4) for k, v in clean.items()},
                    **{k: [round(float(x), 4) for x in v] for k, v in lp.items()},
                    "mstar_L2": [round(float(x), 4) for x in mst.double().pow(2).sum((1, 2)).sqrt()],
                    "skipped": [int(x) for x in skipped.sum((1, 2))],
                })
        if any(q.kind == "use" and q.relation in set(rels.get("0", [])) for q in qs) and len(records) == n_before:
            raise RuntimeError(f"{u.key} has an eligible use query but produced no record")
        if (i + 1) % 20 == 0 or i + 1 == len(units):
            logger.info("magmatched %d/%d units %.0fs %s", i + 1, len(units), time.time() - start, checks)
    payload = {"manifest": spec.manifest(), "conditions": conds, "layers": spec.layers,
               "records": records, "checks": checks, "tau_m_rel": TAU_M_REL, "dead_rows": dead_rows}
    path = _save(spec, payload)
    return {"path": str(path), "units": len(units), "records": len(records), "checks": checks}


# --------------------------------------------------------------------------
# Stage: relational vs leakage gradient geometry (C39, mentor Experiment 2)


def clean_site_grads(model, ids, a_ids, layers, n_pos) -> dict[int, torch.Tensor]:
    """``d log p(a_ids | ids) / d h_l`` at every band layer's hook tensor, clean run.

    Total derivatives: the first band layer's output becomes a leaf, later band
    outputs keep their graph (``retain_grad``), so the gradient at layer 13
    includes every path through layers 16..31. Positions ``:n_pos`` only.
    """
    T = ids.shape[1]
    a = torch.tensor([a_ids], device=ids.device)
    full = torch.cat([ids, a], 1)
    first = min(layers)
    store, handles = {}, []

    def mk(l):
        def hook(module, inputs, output):
            t = output if torch.is_tensor(output) else output[0]
            if l == first:
                t = t.detach().requires_grad_(True)
                store[l] = t
                return t if torch.is_tensor(output) else (t, *output[1:])
            t.retain_grad()
            store[l] = t
            return None

        return hook

    for l in sorted(layers):
        handles.append(model.layers[l].register_forward_hook(mk(l)))
    try:
        with torch.enable_grad(), _record_last(model) as rec:
            _forward(model, full)
            lps = torch.log_softmax(model.unembed(rec["h"][:, T - 1 : T - 1 + len(a_ids)]).float(), -1)
            lp = lps.gather(-1, a.to(lps.device)[..., None])[..., 0].sum()
            lp.backward()
    finally:
        for h in handles:
            h.remove()
    return {l: store[l].grad[0, :n_pos].detach().float().cpu() for l in layers}


@torch.no_grad()
def run_concept_gradgeom(spec: ConceptSpec) -> dict:
    """C39: do clean-state gradients of R and L explain the lambda trend?

    Objectives (teacher-forced on the use query, sequence log-probs, the same
    margins C38 measured): R = lp(target answer) - lp(source answer),
    L = lp(target name) - lp(target answer). Gradients g_R, g_L at every band
    hook tensor on the clean graph (lambda-independent: one set per trial).

    Per (trial, cond, layer, position): signed g_F^T w and the sign-invariant
    |g_F^T w| / ||g_F||; the Householder first-order term
    C_F = -2 (w^T h0)(g_F^T w) (spec 4.5) and, because the run applies the
    row-matched delta, the applied term g_F^T delta. Per (trial, cond): their
    sums F_hat, the finite teacher-forced responses dF at full strength, and
    at eps = EPS strength (a local check of the gradients themselves).
    """
    if spec.doses != [1.0]:
        raise ValueError("the gradient stage measures dose 1.0 only")
    config, model = _setup(spec)
    units = _shard(_eligible_units(spec), spec)
    bank = Bank(model, spec.digest_paths, sorted({w for u, _, _ in units for w in (u.source, u.target)}))
    conds = [(c, a) for c in spec.corpora for a in spec.swap_arms]
    dev = next(model.layers[0].parameters()).device
    EPS = 0.01
    records, sites = [], {}
    dead_rows: dict[str, int] = {}
    start = time.time()
    for i, (u, qs, rels) in enumerate(units):
        pre_ids = encode(model, u.prefix)
        P = pre_ids.shape[1]
        n_before = len(records)
        for nf in spec.fillers:
            ok = set(rels.get(str(nf), []))
            if "express" not in ok:
                continue
            for q in qs:
                if q.kind != "use" or q.relation not in ok:
                    continue
                _, text = _prompt(u, q, nf)
                ids = encode(model, text)
                if not torch.equal(ids[0, :P], pre_ids[0]):
                    raise RuntimeError("prefix mismatch")
                if spec.swap_positions == "prefix":
                    h, n_pos = band_residuals(model, pre_ids, spec.layers), P
                elif spec.swap_positions == "prompt":
                    h, n_pos = band_residuals(model, ids, spec.layers), ids.shape[1]
                else:
                    raise ValueError("gradient stage supports prefix and prompt positions")
                per = {c: swap_deltas(bank, h, u, c, spec.swap_arms, n_pos, dead_rows) for c in spec.corpora}
                ans = {
                    "t": _answer_ids(model, q.target), "s": _answer_ids(model, q.source),
                    "n": _answer_ids(model, (u.target,)),
                }
                g = {k: clean_site_grads(model, ids, v, spec.layers, n_pos) for k, v in ans.items()}
                gF = {"R": {l: g["t"][l] - g["s"][l] for l in spec.layers},
                      "L": {l: g["n"][l] - g["t"][l] for l in spec.layers}}
                deltas = {l: torch.stack([per[c][a][l] for c, a in conds]).to(dev) for l in spec.layers}
                clean = {k: float(_teacher_forced(model, ids, v, {})[0][0]) for k, v in ans.items()}
                full = {k: _teacher_forced(model, ids, v, deltas)[0] for k, v in ans.items()}
                small = {k: _teacher_forced(model, ids, v, {l: d * EPS for l, d in deltas.items()})[0]
                         for k, v in ans.items()}
                shape = (len(conds), len(spec.layers), n_pos)
                arr = {f"{q_}_{F}": torch.zeros(shape) for q_ in ("gw", "C_house", "C_applied") for F in "RL"}
                gnorm = {F: torch.stack([gF[F][l].norm(dim=-1) for l in spec.layers]) for F in "RL"}
                proj = torch.zeros(shape)
                for ci, (c, a) in enumerate(conds):
                    for li, l in enumerate(spec.layers):
                        V = arm_columns(bank, u, c, a, l)
                        w = unit(V[:, 1] - V[:, 0])
                        pw = h[l][:n_pos] @ w
                        proj[ci, li] = pw
                        for F in "RL":
                            gw = gF[F][l] @ w
                            arr[f"gw_{F}"][ci, li] = gw
                            arr[f"C_house_{F}"][ci, li] = -2 * pw * gw
                            arr[f"C_applied_{F}"][ci, li] = (gF[F][l] * per[c][a][l]).sum(-1)
                R = lambda d: d["t"] - d["s"]  # noqa: E731
                Lm = lambda d: d["n"] - d["t"]  # noqa: E731
                tid = f"{u.key}|{q.relation}|f{nf}"
                sites[tid] = {**arr, "gnorm_R": gnorm["R"], "gnorm_L": gnorm["L"], "proj": proj}
                align = {F: (arr[f"gw_{F}"].abs() / gnorm[F].clamp_min(1e-12)[None]) for F in "RL"}
                records.append({
                    "trial": tid, "key": u.key, "type": u.type, "source": u.source, "target": u.target,
                    "clue": u.clue, "split": u.split, "relation": q.relation, "filler": nf, "n_pos": n_pos,
                    "ans_len": {k: len(v) for k, v in ans.items()},
                    "clean": {"R": round(clean["t"] - clean["s"], 5), "L": round(clean["n"] - clean["t"], 5)},
                    "dR": [round(float(x) - (clean["t"] - clean["s"]), 5) for x in R(full)],
                    "dL": [round(float(x) - (clean["n"] - clean["t"]), 5) for x in Lm(full)],
                    "dR_eps": [round((float(x) - (clean["t"] - clean["s"])) / EPS, 5) for x in R(small)],
                    "dL_eps": [round((float(x) - (clean["n"] - clean["t"])) / EPS, 5) for x in Lm(small)],
                    **{f"hat_{k}_{F}": [round(float(x), 5) for x in arr[f"C_{k}_{F}"].sum((1, 2))]
                       for k in ("house", "applied") for F in "RL"},
                    **{f"A_{F}": [round(float(x), 6) for x in align[F].mean((1, 2))] for F in "RL"},
                    **{f"A_{F}_layer": [[round(float(x), 6) for x in row] for row in align[F].mean(2)] for F in "RL"},
                })
        if any(q.kind == "use" and q.relation in set(rels.get("0", [])) for q in qs) and len(records) == n_before:
            raise RuntimeError(f"{u.key} has an eligible use query but produced no record")
        if (i + 1) % 20 == 0 or i + 1 == len(units):
            logger.info("gradgeom %d/%d units %.0fs", i + 1, len(units), time.time() - start)
    payload = {"manifest": spec.manifest(), "conditions": conds, "layers": spec.layers,
               "records": records, "eps": EPS, "dead_rows": dead_rows}
    path = _save(spec, payload)
    torch.save(sites, path.with_suffix(".sites.pt"))
    return {"path": str(path), "units": len(units), "records": len(records), "dead_rows": sum(dead_rows.values())}


@torch.no_grad()
def run_concept_gradcheck(spec: ConceptSpec) -> dict:
    """C39 gate: are the clean-state gradients locally right? float32, central differences.

    bf16 log-probs move in steps of 1/16-1/8 nat, so a 1%-strength response
    divided by 0.01 is quantisation (C39's own epsilon column is exactly
    12.5 / 6.25 at the median). Here the model runs in float32 and, per
    (trial, cond), F_hat = sum g_F . delta is compared with
    (F(+eps delta) - F(-eps delta)) / (2 eps) at eps in EPS_GRID, both sides
    in the same batch shape so batch-composition offsets cancel. The float32
    F_hat is also what the bf16 C39 F_hat is checked against, by trial id.
    """
    EPS_GRID = (0.01, 0.03, 0.1)
    config, model = _setup(spec, dtype="float32")
    units = _shard(_eligible_units(spec), spec)
    bank = Bank(model, spec.digest_paths, sorted({w for u, _, _ in units for w in (u.source, u.target)}))
    conds = [(c, a) for c in spec.corpora for a in spec.swap_arms]
    dev = next(model.layers[0].parameters()).device
    records, dead_rows = [], {}
    for u, qs, rels in units:
        pre_ids = encode(model, u.prefix)
        P = pre_ids.shape[1]
        for nf in spec.fillers:
            ok = set(rels.get(str(nf), []))
            if "express" not in ok:
                continue
            for q in qs:
                if q.kind != "use" or q.relation not in ok:
                    continue
                _, text = _prompt(u, q, nf)
                ids = encode(model, text)
                if spec.swap_positions == "prefix":
                    h, n_pos = band_residuals(model, pre_ids, spec.layers), P
                else:
                    h, n_pos = band_residuals(model, ids, spec.layers), ids.shape[1]
                per = {c: swap_deltas(bank, h, u, c, spec.swap_arms, n_pos, dead_rows) for c in spec.corpora}
                ans = {"t": _answer_ids(model, q.target), "s": _answer_ids(model, q.source),
                       "n": _answer_ids(model, (u.target,))}
                g = {k: clean_site_grads(model, ids, v, spec.layers, n_pos) for k, v in ans.items()}
                deltas = {l: torch.stack([per[c][a][l] for c, a in conds]).to(dev) for l in spec.layers}
                hat = {k: [float(sum((g[k][l] * per[c][a][l]).sum() for l in spec.layers)) for c, a in conds]
                       for k in ans}
                fd = {}
                for eps in EPS_GRID:
                    both = {l: torch.cat([d * eps, -d * eps]) for l, d in deltas.items()}
                    for k, v in ans.items():
                        lp = _teacher_forced(model, ids, v, both)[0].double()
                        B = len(conds)
                        fd[f"{k}@{eps}"] = [round(float(x), 6) for x in (lp[:B] - lp[B:]) / (2 * eps)]
                records.append({
                    "trial": f"{u.key}|{q.relation}|f{nf}", "source": u.source,
                    "hat": {k: [round(x, 6) for x in v] for k, v in hat.items()}, "fd": fd,
                })
    payload = {"manifest": spec.manifest(), "conditions": conds, "records": records,
               "eps": list(EPS_GRID), "dead_rows": dead_rows}
    path = _save(spec, payload)
    return {"path": str(path), "units": len(units), "records": len(records)}


STAGES = {
    "gradcheck": run_concept_gradcheck,
    "gradgeom": run_concept_gradgeom,
    "magmatched": run_concept_magmatched,
    "clean": run_concept_clean,
    "swap": run_concept_swap,
    "fit": run_concept_fit,
    "additive": run_concept_additive,
    "freegen": run_concept_freegen,
    "magnitude": run_concept_magnitude,
}


def run_concept(spec: ConceptSpec) -> dict:
    return STAGES[spec.stage](spec)

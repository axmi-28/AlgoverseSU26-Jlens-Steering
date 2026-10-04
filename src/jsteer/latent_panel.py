"""C40: an expanded, held-out latent-intermediate panel.

The original panel is the 55 probe-swap items (Gurnee et al.) whose clean
answer is correct and whose intermediate and replacement are both covered by
the component artifact. That coverage -- 70 target words -- is the binding
constraint, not prompt writing: a new item is only runnable if both of its
entity names have fitted full/off rows. So new items are generated from a
hand-written fact table (``data/heldout/latent-kb.json``) restricted to those
70 words, and no lens is refitted.

Three stages, each keyed by the hash of everything it reads:

1. ``generate_candidates`` (CPU, deterministic): templates x clues x swap
   partners, with leakage and disjointness checks.
2. ``run_latent_screen`` (GPU): clean greedy answers, plus one-hop checks that
   the model knows each entity's answer when named. Nothing is edited.
3. ``freeze_panel`` (CPU) keeps clean-correct, counterfactual-incorrect items
   under a rule fixed before screening; ``run_latent_controls`` (GPU) then runs
   the unchanged C22 arms on the frozen panel.

``write_controls.py`` is deliberately untouched: its source hash is part of
the frozen C22 manifest, and editing it would invalidate that verification.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from jsteer.config import REPO_ROOT, load_config
from jsteer.write_controls import ARMS, file_hash, layer_controls

KB_PATH = REPO_ROOT / "data" / "heldout" / "latent-kb.json"
CANDIDATES_PATH = REPO_ROOT / "data" / "heldout" / "latent-candidates.json"
PANEL_PATH = REPO_ROOT / "data" / "heldout" / "latent-panel.json"
#: Upper bound on items sharing an intermediate, so the 19 countries (four
#: relations, three clues each) cannot swamp the other categories.
PER_ENTITY_CAP = 6
GEN_SEED = 4001


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _word_start(word: str, text: str) -> bool:
    """``word`` begins a word of ``text`` (case-insensitive).

    Start-of-word, not substring: "ant" must catch "anthill" but not
    "elephant" or "important".
    """
    return re.search(r"(?<![\w])" + re.escape(word.casefold()), text.casefold()) is not None


def _whole_word(word: str, text: str) -> bool:
    return (
        re.search(r"(?<![\w])" + re.escape(word.casefold()) + r"(?![\w])", text.casefold())
        is not None
    )


def _clean_forms(forms, names):
    """Drop answer forms that contain an entity name.

    "Australian" contains "Australia": if it were a swap answer, a hit would
    also count as leakage, and the two outcomes are reported separately.
    """
    return [f for f in forms if not any(n.casefold() in f.casefold() for n in names)]


def _overlap(a, b) -> bool:
    return any(
        x.casefold().startswith(y.casefold()) or y.casefold().startswith(x.casefold())
        for x in a
        for y in b
    )


def item_problems(item: dict, kb: dict, covered: set[str]) -> list[str]:
    """Every reason ``item`` is not a valid latent-intermediate swap."""
    ents = kb["types"][item["type"]]["entities"]
    src, dst = item["intermediate"], item["swap_to"]
    p = item["prompt"]
    out = []
    if src == dst:
        out.append("self-swap")
    for name in (src, dst):
        if name not in covered:
            out.append(f"{name} not covered by the lens artifact")
        for w in [name] + ents[name]["forbid"]:
            if _word_start(w, p):
                out.append(f"prompt names {name} via {w!r}")
    for forms, label in ((item["answer_forms"], "answer"), (item["swap_answer_forms"], "swap answer")):
        if not forms:
            out.append(f"no usable {label} form")
        for f in forms:
            # Short forms ("O", "no", "1") only as whole words; longer ones also
            # as word prefixes, so "American" counts as naming "America".
            if (_word_start if len(f) >= 3 else _whole_word)(f, p):
                out.append(f"prompt contains {label} {f!r}")
            if any(n.casefold() in f.casefold() for n in (src, dst)):
                out.append(f"{label} form {f!r} contains an entity name")
    if _overlap(item["answer_forms"], item["swap_answer_forms"]):
        out.append("answer and swap answer are not disjoint")
    if p != p.rstrip():
        out.append("trailing whitespace")
    return out


def generate_candidates(kb: dict, covered: set[str], *, seed: int = GEN_SEED, cap: int = PER_ENTITY_CAP) -> list[dict]:
    """All valid items, one swap partner per (entity, clue, relation), capped.

    Partner and cap selection use hashes of names only, so the panel is fixed
    by the fact table and seed, never by model behaviour.
    """
    by_entity: dict[str, list[dict]] = {}
    for tname, t in kb["types"].items():
        ents = t["entities"]
        for rel, spec in t["relations"].items():
            for src, e in ents.items():
                if rel not in e:
                    continue
                for k, clue in enumerate(e["clues"]):
                    if rel in clue.get("skip", ()):
                        continue
                    prompt = spec["template"].format(clue=clue["text"])
                    options = []
                    for dst, d in ents.items():
                        if dst == src or rel not in d:
                            continue
                        item = {
                            "name": f"c40-{tname}-{rel}-{src}-{k}",
                            "category": f"{tname}-{rel}",
                            "type": tname,
                            "relation": rel,
                            "clue": clue["text"],
                            "prompt": prompt,
                            "intermediate": src,
                            "answer_forms": _clean_forms(e[rel], (src, dst)),
                            "swap_to": dst,
                            "swap_answer_forms": _clean_forms(d[rel], (src, dst)),
                            "one_hop_source": spec["one_hop"].format(name=src),
                            "one_hop_target": spec["one_hop"].format(name=dst),
                        }
                        if not item_problems(item, kb, covered):
                            options.append(item)
                    if options:
                        options.sort(key=lambda i: hashlib.sha256(f"{seed}|{i['name']}|{i['swap_to']}".encode()).hexdigest())
                        by_entity.setdefault(src, []).append(options[0])
    items = []
    for src, rows in sorted(by_entity.items()):
        # Round-robin over relations so a cap keeps relation diversity.
        rows.sort(key=lambda i: hashlib.sha256(f"{seed}|{i['name']}".encode()).hexdigest())
        queues: dict[str, list[dict]] = {}
        for r in rows:
            queues.setdefault(r["relation"], []).append(r)
        chosen = []
        while len(chosen) < cap and any(queues.values()):
            for rel in sorted(queues):
                if queues[rel] and len(chosen) < cap:
                    chosen.append(queues[rel].pop(0))
        items.extend(chosen)
    for i in items:
        i["answer"], i["swap_answer"] = i["answer_forms"][0], i["swap_answer_forms"][0]
    items.sort(key=lambda i: i["name"])
    return items


def _hits(text: str, forms) -> bool:
    from jsteer.steering import answer_matches

    return any(answer_matches(text, f) for f in forms)


@dataclass
class LatentScreenSpec:
    candidates_path: str = str(CANDIDATES_PATH)
    config_name: str = "qwen3-8b"
    gen_tokens: int = 6
    out_dir: str = "results/latent"


def run_latent_screen(spec: LatentScreenSpec) -> dict:
    """Clean and one-hop greedy answers for every candidate. No edits."""
    from jsteer.loading import load_model
    from jsteer.steering import greedy_continuation

    config = load_config(spec.config_name)
    blob = json.loads(Path(spec.candidates_path).read_text())
    items = blob["items"]
    manifest = {"spec": asdict(spec), "candidates_sha256": file_hash(spec.candidates_path), "n": len(items)}
    manifest["spec"]["candidates_path"] = Path(spec.candidates_path).name
    out = Path(spec.out_dir) / f"screen_{_sha(manifest)[:16]}.json"
    if out.exists():
        done = json.loads(out.read_text())
        if done["manifest"] == manifest and len(done["records"]) == len(items):
            return {"path": str(out), "records": len(done["records"]), "cached": True}
    model = load_model(config)
    start = time.monotonic()

    def gen(prompt):
        return greedy_continuation(model, prompt, n_tokens=spec.gen_tokens, max_seq_len=config.fit.max_seq_len)

    records = []
    for item in items:
        clean, src, dst = gen(item["prompt"]), gen(item["one_hop_source"]), gen(item["one_hop_target"])
        records.append(
            {
                "name": item["name"],
                "generated": clean,
                "hit_answer": _hits(clean, item["answer_forms"]),
                "hit_swap_answer": _hits(clean, item["swap_answer_forms"]),
                "emitted_substring": item["swap_to"].casefold() in clean.casefold(),
                "one_hop_source_text": src,
                "one_hop_source_ok": _hits(src, item["answer_forms"]),
                "one_hop_target_text": dst,
                "one_hop_target_ok": _hits(dst, item["swap_answer_forms"]),
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"manifest": manifest, "records": records}, indent=1))
    return {"path": str(out), "records": len(records), "seconds": round(time.monotonic() - start, 1)}


#: Frozen before screening. The one-hop checks are recorded, not filtered on,
#: so the inclusion rule matches the original 55 exactly (plus clean
#: non-emission of the replacement name, which the leakage metric needs).
INCLUSION_RULE = "clean greedy continuation hits an answer form, hits no swap-answer form, and does not contain the replacement name"


def freeze_panel(candidates: dict, screen: dict, screen_sha256: str) -> dict:
    by_name = {r["name"]: r for r in screen["records"]}
    if set(by_name) != {i["name"] for i in candidates["items"]}:
        raise ValueError("Screen does not cover exactly the candidate set")
    keep = []
    for item in candidates["items"]:
        r = by_name[item["name"]]
        if r["hit_answer"] and not r["hit_swap_answer"] and not r["emitted_substring"]:
            keep.append({**item, "one_hop_source_ok": r["one_hop_source_ok"], "one_hop_target_ok": r["one_hop_target_ok"]})
    return {
        "description": "C40 expanded latent-intermediate panel. Frozen from the screen below before any intervention run; not used in development.",
        "inclusion_rule": INCLUSION_RULE,
        "candidates_sha256": candidates["sha256"],
        "screen_sha256": screen_sha256,
        "n_candidates": len(candidates["items"]),
        "items": keep,
    }


@dataclass
class LatentControlSpec:
    directions_path: str
    panel_path: str = str(PANEL_PATH)
    config_name: str = "qwen3-8b"
    corpus: str = "gsm8k"
    strengths: tuple[float, ...] = (0.5, 1.0)
    layers: tuple[int, ...] = (13, 16, 19, 22, 25, 28, 31)
    seed: int = 2201
    gen_tokens: int = 6
    max_seconds: int = 3000
    out_dir: str = "results/latent"


def run_latent_controls(spec: LatentControlSpec) -> dict:
    """The C22 arms on the frozen C40 panel; resumable, checkpointed per item.

    Same edits, layers, strengths, decoding and scorer as C22; the only
    difference is that an answer may have several accepted forms.
    """
    from jlens.hooks import ActivationRecorder

    from jsteer.loading import first_token_id, load_model, unembedding_rows
    from jsteer.steering import ResidualEdit, greedy_continuation

    if not spec.strengths or any(not math.isfinite(s) or s <= 0 for s in spec.strengths):
        raise ValueError("Strengths must be finite and positive")
    config = load_config(spec.config_name)
    blob = torch.load(spec.directions_path, map_location="cpu", weights_only=False)
    index = {w: i for i, w in enumerate(blob["targets"])}
    matrices, counts = {}, {}
    for part in ("full", "off"):
        for layer in spec.layers:
            entry = blob["digests"][f"{spec.corpus}_{part}|{layer}"]
            if entry["pullbacks"].shape != (len(index), config.d_model):
                raise ValueError("Component dimensions do not match model and target list")
            matrices[part, layer] = entry["pullbacks"].float()
            counts[f"{part}|{layer}"] = entry["n"]
    if len(set(counts.values())) != 1:
        raise ValueError("Full/off artifact fit counts differ")
    panel = json.loads(Path(spec.panel_path).read_text())
    items = panel["items"]
    missing = [i["name"] for i in items if i["intermediate"] not in index or i["swap_to"] not in index]
    if missing:
        raise ValueError(f"Panel items without lens coverage: {missing[:3]}")
    manifest = json.loads(
        json.dumps(
            {
                "spec": {**asdict(spec), "directions_path": Path(spec.directions_path).name, "panel_path": Path(spec.panel_path).name},
                "arms": ARMS,
                "items": [i["name"] for i in items],
                "panel_sha256": file_hash(spec.panel_path),
                "directions_sha256": file_hash(spec.directions_path),
                "fit_counts": counts,
            }
        )
    )
    out = Path(spec.out_dir) / f"latent_controls_{spec.corpus}_{_sha(manifest)[:16]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"manifest": manifest, "records": [], "status": "running"}
    if out.exists():
        payload = json.loads(out.read_text())
        if payload["manifest"] != manifest:
            raise ValueError("Existing output does not match the run manifest")

    def checkpoint():
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=1, allow_nan=False))
        tmp.replace(out)

    settings = [("baseline", 0.0)] + [(a, float(s)) for a in ARMS for s in spec.strengths]
    done = set()
    for item in items:
        keys = {(r["arm"], r["strength"]) for r in payload["records"] if r["name"] == item["name"]}
        if keys and keys != set(settings):
            raise ValueError(f"Incomplete saved item: {item['name']}")
        if keys:
            done.add(item["name"])
    start = time.monotonic()
    if len(done) < len(items):
        model = load_model(config)
        for item in items:
            if item["name"] in done:
                continue
            if time.monotonic() - start > spec.max_seconds:
                break
            src, dst = item["intermediate"], item["swap_to"]
            ids = [first_token_id(model, w) for w in (src, dst)]
            u = unembedding_rows(model, ids).T.cpu().float()
            input_ids = model.encode(item["prompt"], max_length=config.fit.max_seq_len)
            with torch.no_grad(), ActivationRecorder(model.layers, at=list(spec.layers)) as rec:
                model.forward(input_ids)
                h = {l: rec.activations[l][0].detach().cpu().float() for l in spec.layers}
            si, ti = index[src], index[dst]
            banks = {
                l: layer_controls(
                    h[l], matrices["full", l][[si, ti]].T, matrices["off", l][[si, ti]].T, u, seed=spec.seed + l
                )
                for l in spec.layers
            }
            rows = []
            for arm, strength in settings:
                edit = None if arm == "baseline" else ResidualEdit(
                    {l: banks[l][arm] * strength for l in spec.layers},
                    positions=list(range(input_ids.shape[1])),
                )
                text = greedy_continuation(
                    model, item["prompt"], edit=edit, n_tokens=spec.gen_tokens, max_seq_len=config.fit.max_seq_len
                )
                rows.append(
                    {
                        "name": item["name"],
                        "category": item["category"],
                        "intermediate": src,
                        "swap_to": dst,
                        "arm": arm,
                        "strength": strength,
                        "generated": text,
                        "hit_answer": _hits(text, item["answer_forms"]),
                        "hit_swap_answer": _hits(text, item["swap_answer_forms"]),
                        "emitted_substring": dst.casefold() in text.casefold(),
                        "edit_row_norm_mean": None if edit is None else float(
                            torch.cat([v.norm(dim=-1) for v in edit.vectors.values()]).mean()
                        ),
                    }
                )
            payload["records"].extend(rows)
            done.add(item["name"])
            checkpoint()
            print(f"C40 {spec.corpus} {len(done)}/{len(items)} items, {time.monotonic() - start:.0f}s", flush=True)
    payload["status"] = "complete" if len(done) == len(items) else "time_limit"
    checkpoint()
    return {"path": str(out), "status": payload["status"], "items_done": len(done), "items": len(items), "records": len(payload["records"])}


@dataclass
class LatentLoadingSpec:
    """C41: workspace loading of the intermediate on every latent item.

    The J-Lens paper's definition averages over "the argument and readout
    positions". A latent item has no argument position -- the intermediate is
    never written -- so the primary quantity is the **readout** (final)
    position alone; the mean over all prompt positions, where the swap is
    applied, is recorded as a secondary. Same lens (the released one), band and
    cosine as the flexible-generalization loading in ``sweeps.workspace_loading``.
    """

    directions_path: str
    dev_path: str
    panel_path: str = str(PANEL_PATH)
    config_name: str = "qwen3-8b"
    layers: tuple[int, ...] = (13, 16, 19, 22, 25, 28, 31)
    out_dir: str = "results/latent"


def run_latent_loading(spec: LatentLoadingSpec) -> dict:
    from jlens.hooks import ActivationRecorder

    from jsteer.jacobian import averaged_pullback
    from jsteer.loading import first_token_id, load_lens, load_model, unembedding_rows

    config = load_config(spec.config_name)
    dev = json.loads(Path(spec.dev_path).read_text())["records"]
    dev_items = list(
        {r["name"]: {"name": r["name"], "prompt": r["prompt"], "intermediate": r["intermediate"], "swap_to": r["swap_to"], "set": "development"} for r in dev}.values()
    )
    new_items = [
        {"name": i["name"], "prompt": i["prompt"], "intermediate": i["intermediate"], "swap_to": i["swap_to"], "set": "new"}
        for i in json.loads(Path(spec.panel_path).read_text())["items"]
    ]
    items = dev_items + new_items
    manifest = {
        "spec": {**asdict(spec), "directions_path": Path(spec.directions_path).name, "dev_path": Path(spec.dev_path).name, "panel_path": Path(spec.panel_path).name},
        "dev_sha256": file_hash(spec.dev_path),
        "panel_sha256": file_hash(spec.panel_path),
        "directions_sha256": file_hash(spec.directions_path),
        "items": [i["name"] for i in items],
    }
    out = Path(spec.out_dir) / f"latent_loading_{_sha(manifest)[:16]}.json"
    if out.exists():
        done = json.loads(out.read_text())
        if done["manifest"] == manifest and len(done["records"]) == len(items):
            return {"path": str(out), "records": len(done["records"]), "cached": True}

    blob = torch.load(spec.directions_path, map_location="cpu", weights_only=False)
    index = {w: i for i, w in enumerate(blob["targets"])}
    arms = {
        f"{c}_{p}": {l: blob["digests"][f"{c}_{p}|{l}"]["pullbacks"].float() for l in spec.layers}
        for c in ("gsm8k", "wikitext_a")
        for p in ("full", "off")
    }
    del blob
    model = load_model(config)
    lens = load_lens(config)
    start = time.monotonic()

    def cos(h, v):
        return (h @ v) / (h.norm(dim=-1) * v.norm() + 1e-12)

    records = []
    for item in items:
        src = item["intermediate"]
        src_id = first_token_id(model, src)
        u_src = unembedding_rows(model, [src_id])
        input_ids = model.encode(item["prompt"], max_length=config.fit.max_seq_len)
        with torch.no_grad(), ActivationRecorder(model.layers, at=list(spec.layers)) as rec:
            model.forward(input_ids)
            acts = {l: rec.activations[l][0].detach().cpu().float() for l in spec.layers}
        readout, allpos = {}, {}
        for l in spec.layers:
            v = averaged_pullback(lens, u_src, l)[0].float().cpu()
            c = cos(acts[l], v)
            readout[l], allpos[l] = float(c[-1]), float(c.mean())
        by_arm = {}
        for arm, bank in arms.items():
            vals = [cos(acts[l], bank[l][index[src]]) for l in spec.layers]
            by_arm[arm] = {"readout": float(sum(float(c[-1]) for c in vals) / len(vals)), "all": float(sum(float(c.mean()) for c in vals) / len(vals))}
        records.append(
            {
                **{k: item[k] for k in ("name", "set", "intermediate", "swap_to")},
                "src_token_id": src_id,
                "src_n_tokens": len(model.tokenizer.encode(f" {src}", add_special_tokens=False)),
                "seq_len": int(input_ids.shape[1]),
                "loading_readout": sum(readout.values()) / len(readout),
                "loading_all": sum(allpos.values()) / len(allpos),
                "by_layer_readout": readout,
                "by_arm": by_arm,
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"manifest": manifest, "records": records}, indent=1, allow_nan=False))
    return {"path": str(out), "records": len(records), "dev": len(dev_items), "new": len(new_items), "seconds": round(time.monotonic() - start, 1)}


@dataclass
class LoadingProfileSpec:
    """C41c: per-layer, per-position cosine profiles on both evaluation sets.

    ``run_latent_loading`` stores one averaged number per item; to ask *where*
    loading matters (argument vs readout on flexible generalization, clue vs
    readout on latent items) the whole [layer, position] profile is kept, for
    the source concept and the replacement, under the released lens. Positions
    are summarized offline from the decoded tokens and the stored arg span.
    """

    dev_path: str
    panel_path: str = str(PANEL_PATH)
    config_name: str = "qwen3-8b"
    layers: tuple[int, ...] = (13, 16, 19, 22, 25, 28, 31)
    out_dir: str = "results/latent"


def run_loading_profiles(spec: LoadingProfileSpec) -> dict:
    from jlens.hooks import ActivationRecorder

    from jsteer.jacobian import averaged_pullback
    from jsteer.loading import first_token_id, load_lens, load_model, single_token_id, unembedding_rows
    from jsteer.run import arg_span, build_prompt_set

    config = load_config(spec.config_name)
    manifest = {
        "spec": {**asdict(spec), "dev_path": Path(spec.dev_path).name, "panel_path": Path(spec.panel_path).name},
        "dev_sha256": file_hash(spec.dev_path),
        "panel_sha256": file_hash(spec.panel_path),
    }
    out = Path(spec.out_dir) / f"loading_profiles_{_sha(manifest)[:16]}.json"
    if out.exists():
        done = json.loads(out.read_text())
        if done["manifest"] == manifest:
            return {"path": str(out), "records": len(done["records"]), "cached": True}
    model = load_model(config)
    lens = load_lens(config)
    start = time.monotonic()

    def profile(text, src_id, dst_id):
        input_ids = model.encode(text, max_length=config.fit.max_seq_len)
        with torch.no_grad(), ActivationRecorder(model.layers, at=list(spec.layers)) as rec:
            model.forward(input_ids)
            acts = {l: rec.activations[l][0].detach().cpu().float() for l in spec.layers}
        u = unembedding_rows(model, [src_id, dst_id])
        prof = {"source": {}, "target": {}}
        for l in spec.layers:
            v = averaged_pullback(lens, u, l).float().cpu()
            h = acts[l]
            cos = (h @ v.T) / (h.norm(dim=-1, keepdim=True) * v.norm(dim=-1) + 1e-12)
            prof["source"][l] = [round(float(x), 5) for x in cos[:, 0]]
            prof["target"][l] = [round(float(x), 5) for x in cos[:, 1]]
        tokens = [model.tokenizer.decode([int(i)]) for i in input_ids[0]]
        return input_ids, tokens, prof

    records = []
    # Flexible generalization: one profile per base prompt; target = the first
    # replacement is not meaningful here, so the source is profiled twice and
    # replacements are scored offline from the source profile only.
    for prompt in build_prompt_set(config, n_fit=0, tokenizer=model.tokenizer):
        if prompt.base is None:
            continue
        try:
            src_id = single_token_id(model, prompt.base.arg)
        except ValueError:
            continue
        input_ids, tokens, prof = profile(prompt.text, src_id, src_id)
        records.append({"set": "flexible", "key": prompt.key, "arg": prompt.base.arg, "tokens": tokens,
                        "arg_span": arg_span(model, prompt, max_seq_len=config.fit.max_seq_len),
                        "source": prof["source"]})
    dev = json.loads(Path(spec.dev_path).read_text())["records"]
    latent = list({r["name"]: {"name": r["name"], "prompt": r["prompt"], "intermediate": r["intermediate"], "swap_to": r["swap_to"], "set": "development", "clue": None} for r in dev}.values())
    latent += [{"name": i["name"], "prompt": i["prompt"], "intermediate": i["intermediate"], "swap_to": i["swap_to"], "set": "new", "clue": i["clue"]}
               for i in json.loads(Path(spec.panel_path).read_text())["items"]]
    for item in latent:
        _, tokens, prof = profile(item["prompt"], first_token_id(model, item["intermediate"]), first_token_id(model, item["swap_to"]))
        records.append({**item, "tokens": tokens, "source": prof["source"], "target": prof["target"]})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"manifest": manifest, "records": records}, allow_nan=False))
    sets = collections.Counter(r["set"] for r in records)
    return {"path": str(out), "records": len(records), "by_set": dict(sets), "seconds": round(time.monotonic() - start, 1)}

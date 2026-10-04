"""Budgeted cross-family filtering replication; existing scientific primitives.

One model load, fixed inputs, clean-only eligibility, checkpointed stages.
No published full lens or Qwen-specific eligibility is used.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from jsteer.config import REPO_ROOT, load_config
from jsteer.concept_use import (
    Bank, band_residuals, build_units, encode, evaluate, hits, swap_deltas,
)
from jsteer.jacobian import component_pullbacks_for_prompt, pullback_for_prompt
from jsteer.loading import load_model, single_token_id, unembedding_rows
from jsteer.write_controls import file_hash

INPUT = REPO_ROOT / "data/protocols/cross_model_filtering.json"


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, allow_nan=False))
    tmp.replace(path)


def select_units(eligible, units, per_type=40):
    """Balanced round-robin over source entities; no edited outcomes used."""
    selected = []
    for ty in sorted({u.type for u in units}):
        buckets = defaultdict(list)
        for u in units:
            if u.type == ty and u.key in eligible:
                buckets[u.source].append(u.key)
        for source in buckets:
            buckets[source].sort(key=lambda k: sha([42102, k]))
        sources = sorted(buckets, key=lambda s: sha([42102, ty, s]))
        chosen = []
        while len(chosen) < per_type and any(buckets.values()):
            for source in sources:
                if buckets[source] and len(chosen) < per_type:
                    chosen.append(buckets[source].pop(0))
        selected.extend(chosen)
    return selected


@dataclass
class CrossModelSpec:
    config_name: str
    out_dir: str = "/results/cross_model"
    max_seconds: int = 21000
    expected_source_sha: str = ""


def run_cross_model(spec: CrossModelSpec):
    started = time.monotonic()
    source_sha = file_hash(Path(__file__))
    if spec.expected_source_sha != source_sha:
        raise ValueError("Deployed cross_model.py differs from launch source")
    config = load_config(spec.config_name)
    protocol = json.loads(INPUT.read_text())
    # Preserve the original Qwen seven-point relative-depth grid exactly.
    layers = [round(l / 35 * (config.n_layers - 1)) for l in (13, 16, 19, 22, 25, 28, 31)]
    manifest = {
        "config": asdict(config), "protocol_sha": file_hash(INPUT),
        "source_sha": source_sha, "layers": layers,
        "dependency_sha": {name: file_hash(Path(__file__).with_name(name)) for name in
                           ("concept_use.py", "jacobian.py", "loading.py", "steering.py", "write_controls.py")},
    }
    root = Path(spec.out_dir) / config.name / sha(manifest)[:12]
    root.mkdir(parents=True, exist_ok=True)
    save(root / "manifest.json", manifest)

    def checkpoint(stage, detail=None):
        save(root / "status.json", {"stage": stage, "seconds": time.monotonic() - started,
                                   "detail": detail, "manifest_sha": sha(manifest)})
        # Persist during the invocation, including when a timeout stops it.
        import modal
        modal.Volume.from_name("jsteer-results").commit()
        print(json.dumps({"stage": stage, "seconds": round(time.monotonic()-started), "detail": detail}), flush=True)

    def budget():
        if time.monotonic() - started > spec.max_seconds:
            checkpoint("budget_stop")
            raise TimeoutError("Per-model compute limit reached; resume requires remaining budget")

    checkpoint("loading")
    model = load_model(config)
    # Compare the wrapped decoder's logits to the checkpoint's own forward,
    # then independently check a tiny exact-position Jacobian pullback.
    from jlens.hooks import ActivationRecorder
    ids = model.encode("The capital of France is", max_length=32)
    with torch.no_grad(), ActivationRecorder(model.layers, at=[model.n_layers - 1]) as rec:
        model.forward(ids)
        wrapped = model.unembed(rec.activations[model.n_layers - 1][:, -1]).float()
        native = model._hf_model(input_ids=ids, use_cache=False).logits[:, -1].float()
        logit_error = float((wrapped-native).abs().max())
        if not torch.allclose(wrapped, native, atol=0.05, rtol=0.01):
            raise RuntimeError(f"Wrapped logits differ from native checkpoint: {logit_error}")
    cot = unembedding_rows(model, [single_token_id(model, "France")])
    tiny = component_pullbacks_for_prompt(model, protocol["fit_texts"][0], layers[:1], cot,
                                          dim_batch=1, max_seq_len=32, skip_first=16)
    direct = pullback_for_prompt(model, protocol["fit_texts"][0], layers[:1], cot,
                                dim_batch=1, max_seq_len=32, skip_first=16)
    relative_error = float((tiny[layers[0]]["total"]-direct[layers[0]]).norm()/direct[layers[0]].norm())
    if relative_error > 0.05:
        raise RuntimeError(f"Independent pullback check failed: {relative_error}")
    save(root / "smoke.json", {"logit_max_error": logit_error, "pullback_relative_error": relative_error})
    checkpoint("smoke_passed", {"logit_error": logit_error, "pullback_error": relative_error})
    candidates = protocol["latent_candidates"]
    units = build_units(clues=("0", "1"))
    requested = sorted({w for i in candidates for w in (i["intermediate"], i["swap_to"])} |
                       {w for u in units for w in (u.source, u.target)})
    token_ids, excluded = {}, {}
    for word in requested:
        try:
            token_ids[word] = single_token_id(model, word)
        except ValueError as e:
            excluded[word] = str(e)
    save(root / "token_coverage.json", {"supported": token_ids, "excluded": excluded})
    units = [u for u in units if u.source in token_ids and u.target in token_ids]
    candidates = [i for i in candidates if i["intermediate"] in token_ids and i["swap_to"] in token_ids]

    # Clean generation cache is shared across source-target pairs and stages.
    clean_path = root / "clean_generations.json"
    clean = json.loads(clean_path.read_text()) if clean_path.exists() else {}

    def generate(text, n):
        key = sha([text, n])
        if key not in clean:
            budget()
            with torch.no_grad():
                clean[key] = evaluate(model, encode(model, text), {}, gen_tokens=n)["texts"][0]
            save(clean_path, clean)
        return clean[key]

    screen = []
    for i, item in enumerate(candidates):
        text = generate(item["prompt"], 6)
        ok = hits(text, item["answer_forms"]) and not hits(text, item["swap_answer_forms"]) and item["swap_to"].casefold() not in text.casefold()
        screen.append({"name": item["name"], "text": text, "eligible": ok})
        if (i + 1) % 50 == 0:
            checkpoint("screen_latent", i + 1)
    save(root / "latent_screen.json", screen)
    keep = {r["name"] for r in screen if r["eligible"]}
    latent = [i for i in candidates if i["name"] in keep]
    eligible = {}
    for u in units:
        rels = []
        for q in u.queries:
            if q.kind not in ("express", "use"):
                continue
            text = generate(u.prefix + q.text, 8)
            if hits(text, q.source) and not hits(text, q.target):
                rels.append(q.relation)
        if "express" in rels and len(rels) > 1:
            eligible[u.key] = rels
    chosen = select_units(eligible, units)
    selected = [u for u in units if u.key in chosen]
    save(root / "eligibility.json", {"latent": latent, "concept_eligible": eligible, "concept_selected": chosen})
    checkpoint("eligibility_frozen", {"latent": len(latent), "concept": len(selected), "excluded_tokens": len(excluded)})
    if not latent or not selected:
        raise RuntimeError("Insufficient clean eligibility: inspect screening before any intervention")

    words = sorted({w for i in latent for w in (i["intermediate"], i["swap_to"])} |
                   {w for u in selected for w in (u.source, u.target)})
    cotangents = unembedding_rows(model, [token_ids[w] for w in words])
    fit_path = root / "components.pt"
    fit_started = time.monotonic()
    sums, count = {}, 0
    partial_path = root / "fit_partial.pt"
    if partial_path.exists():
        partial = torch.load(partial_path, weights_only=False, map_location="cpu")
        if partial["words"] != words or partial["manifest_sha"] != sha(manifest):
            raise ValueError("Partial fit manifest mismatch")
        sums, count = partial["sums"], partial["count"]
    for index, prompt in enumerate(protocol["fit_texts"]):
        if fit_path.exists() or index < count:
            continue
        budget()
        tick = time.monotonic()
        parts = component_pullbacks_for_prompt(model, prompt, layers, cotangents, dim_batch=8,
                                               max_seq_len=128, skip_first=16, t_stride=4)
        for l in layers:
            for part in ("total", "diag", "off"):
                value = parts[l][part]
                if not torch.isfinite(value).all():
                    raise ValueError("Nonfinite component")
                key = (l, part)
                sums[key] = sums.get(key, torch.zeros_like(value)) + value
        count = index + 1
        tmp = partial_path.with_suffix(".tmp")
        torch.save({"words": words, "manifest_sha": sha(manifest), "sums": sums, "count": count}, tmp)
        tmp.replace(partial_path)
        checkpoint("fitting", {"prompts": count, "total": 32, "last_seconds": round(time.monotonic()-tick, 1)})
        if count == 1:
            projected = (time.monotonic() - tick) * 31
            if time.monotonic() - started + projected + 1800 > spec.max_seconds:
                checkpoint("projection_exceeds_budget", {"fit_remaining_seconds": projected})
                raise TimeoutError("Measured fit speed would exceed approved per-model allowance")
    if not fit_path.exists():
        digests = {}
        for (l, part), value in sums.items():
            family = "full" if part == "total" else part
            digests[f"gsm8k_{family}|{l}"] = {"pullbacks": value / count, "n": count}
        torch.save({"targets": words, "digests": digests, "manifest_sha": sha(manifest)}, fit_path)
    bank = Bank(model, [str(fit_path)], words)
    checkpoint("fit_complete", {"words": len(words), "fit_seconds": round(time.monotonic()-fit_started)})

    # Validate normalization at every actual edit site, not only average norms.
    def intervene(prompt, pair, arms, strengths, tokens):
        budget()
        ids = encode(model, prompt)
        h = band_residuals(model, ids, layers)
        report = {}
        per = swap_deltas(bank, h, pair, "gsm8k", list(dict.fromkeys(["full"] + arms)), ids.shape[1], report)
        max_error = 0.
        for arm in arms:
            if arm == "full":
                continue
            for l in layers:
                ref = per["full"][l].norm(dim=-1)
                err = (per[arm][l].norm(dim=-1) - ref).abs() / ref.clamp_min(1e-6)
                max_error = max(max_error, float(err.max()))
        if max_error > 1e-4 or sum(report.values()):
            raise RuntimeError(f"Edit matching failed: {max_error}, {report}")
        settings = [(a, s) for a in arms for s in strengths]
        deltas = {l: torch.stack([per[a][l] * s for a, s in settings]).to(ids.device) for l in layers}
        result = evaluate(model, ids, deltas, gen_tokens=tokens, chunk=5)
        return settings, result["texts"], max_error

    path = root / "latent_results.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {r["name"] for r in rows}
    with torch.no_grad():
        for item in latent:
            if item["name"] in done:
                continue
            pair = SimpleNamespace(source=item["intermediate"], target=item["swap_to"])
            settings, texts, error = intervene(item["prompt"], pair, ["full", "off_m"], [0.5, 1.0], 6)
            for (arm, strength), text in zip(settings, texts, strict=True):
                rows.append({"name": item["name"], "set": item["set"], "source": pair.source, "target": pair.target,
                             "arm": arm, "strength": strength, "text": text,
                             "success": hits(text, item["swap_answer_forms"]),
                             "leakage": pair.target.casefold() in text.casefold(), "norm_error": error})
            save(path, rows)
            if len(rows) % 80 == 0:
                checkpoint("latent_interventions", len(rows) // 4)
    checkpoint("latent_complete", len(rows) // 4)
    path = root / "concept_results.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {(r["key"], r["relation"]) for r in rows}
    arms = ["lam0_cn", "lam0.5_cn", "lam1_cn", "lam2_cn", "rand_m"]
    with torch.no_grad():
        for u in selected:
            for q in u.queries:
                if q.relation not in eligible[u.key] or (u.key, q.relation) in done:
                    continue
                settings, texts, error = intervene(u.prefix + q.text, u, arms, [1.0], 8)
                for (arm, strength), text in zip(settings, texts, strict=True):
                    rows.append({"key": u.key, "type": u.type, "source": u.source, "target": u.target,
                                 "relation": q.relation, "kind": q.kind, "arm": arm, "strength": strength,
                                 "text": text, "success": hits(text, q.target),
                                 "leakage": u.target.casefold() in text.casefold(), "norm_error": error})
                save(path, rows)
            checkpoint("concept_interventions", len({r["key"] for r in rows}))
    expected = sum(len(eligible[u.key]) for u in selected) * len(arms)
    if len(rows) != expected:
        raise RuntimeError(f"Concept rows {len(rows)} != {expected}")
    checkpoint("complete", {"latent_items": len(latent), "concept_units": len(selected), "concept_rows": len(rows)})
    return {"path": str(root), "status": "complete", "seconds": time.monotonic()-started}

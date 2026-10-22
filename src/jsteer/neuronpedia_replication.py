"""Published Neuronpedia full lenses minus an estimated WikiText diagonal.

The full lens is never fitted or replaced. Filtering is an estimate because
the diagonal sample is not the released lens's original fitting run.
"""
from __future__ import annotations

import importlib.metadata
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from jsteer.concept_use import (
    Bank,
    band_residuals,
    build_units,
    encode,
    evaluate,
    hits,
    swap_deltas,
)
from jsteer.config import REPO_ROOT, load_config
from jsteer.cross_model import save, select_units, sha
from jsteer.jacobian import component_pullbacks_for_prompt, pullback_for_prompt
from jsteer.loading import load_model, single_token_id, unembedding_rows
from jsteer.write_controls import file_hash

PROTOCOL = REPO_ROOT / "data/protocols/neuronpedia_replication.json"
PANEL = REPO_ROOT / "data/protocols/cross_model_filtering.json"


def input_hashes(config_name):
    paths = [PROTOCOL, PANEL, REPO_ROOT / "data/heldout/concept-use.json",
             REPO_ROOT / "data/protocols/lens_source_policy.json",
             REPO_ROOT / "data/protocols/neuronpedia_gemma_llama_inventory.json",
             REPO_ROOT / "configs" / f"{config_name}.yaml"]
    paths += [Path(__file__).with_name(name) for name in
              ("neuronpedia_replication.py", "cross_model.py", "concept_use.py",
               "jacobian.py", "loading.py", "steering.py", "write_controls.py", "config.py")]
    return {str(p.relative_to(REPO_ROOT)): file_hash(p) for p in paths}


def derived_digests(full, diagonal64, diagonal32, count=64):
    """Keep the released full directions intact; subtraction creates new tensors."""
    digests = {}
    for layer, published in full.items():
        if published.shape != diagonal64[layer].shape or published.shape != diagonal32[layer].shape:
            raise ValueError("Published/estimated direction shapes differ")
        for family, value, n in (
            ("full", published, None), ("diag", diagonal64[layer], count),
            ("off", published - diagonal64[layer], count),
            ("off32", published - diagonal32[layer], count // 2),
        ):
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite published or derived direction")
            digests[f"published_{family}|{layer}"] = {"pullbacks": value, "n": n}
    return digests


@dataclass
class NeuronpediaSpec:
    config_name: str
    out_dir: str = "/results/neuronpedia_replication"
    max_seconds: int = 3500
    expected_source_sha: str = ""
    expected_inputs_sha: str = ""


def run_neuronpedia(spec: NeuronpediaSpec):
    import modal
    from huggingface_hub import hf_hub_download
    from jlens import JacobianLens
    from jlens.hooks import ActivationRecorder

    started = time.monotonic()
    source_sha = file_hash(Path(__file__))
    if source_sha != spec.expected_source_sha or sha(input_hashes(spec.config_name)) != spec.expected_inputs_sha:
        raise ValueError("Deployed published-lens science inputs differ from launch")
    config = load_config(spec.config_name)
    p = json.loads(PROTOCOL.read_text())
    panel = json.loads(PANEL.read_text())
    if config.lens_repo != p["lens_repository"] or config.lens_revision != p["lens_revision"]:
        raise ValueError("Published lens repository/revision differs from frozen protocol")
    inventory = json.loads((REPO_ROOT / "data/protocols/neuronpedia_gemma_llama_inventory.json").read_text())
    entry = next(i for i in inventory["models"] if i["config_name"] == config.name)
    if (config.lens_filename, config.model_revision) != (entry["lens_filename"], entry["model_revision"]):
        raise ValueError("Published lens/model pair differs from frozen inventory")
    layers = [round(l / 35 * (config.n_layers - 1)) for l in p["reference_layers"]]
    manifest = {"config": asdict(config), "protocol_sha": file_hash(PROTOCOL),
                "source_sha": source_sha, "inputs": input_hashes(config.name), "layers": layers,
                "method": "published Neuronpedia full minus estimated WikiText diagonal; not exact released-fit decomposition"}
    root = Path(spec.out_dir) / config.name / sha(manifest)[:12]
    root.mkdir(parents=True, exist_ok=True)
    save(root / "manifest.json", manifest)
    volume = modal.Volume.from_name("jsteer-results")

    def checkpoint(stage, detail=None):
        body = {"stage": stage, "seconds": time.monotonic()-started, "detail": detail,
                "manifest_sha": sha(manifest)}
        save(root / "status.json", body)
        volume.commit()
        print(json.dumps({"model": config.name, **body}), flush=True)

    def budget():
        if time.monotonic()-started > spec.max_seconds:
            checkpoint("budget_stop")
            raise TimeoutError("Published-lens model budget reached")

    checkpoint("loading")
    model = load_model(config)
    save(root / "environment.json", {
        "packages": {n: importlib.metadata.version(n) for n in
                     ("torch", "transformers", "jlens", "numpy", "huggingface_hub")},
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
    })
    ids = model.encode("The capital of France is", max_length=32)
    with torch.no_grad(), ActivationRecorder(model.layers, at=[model.n_layers-1]) as rec:
        model.forward(ids)
        wrapped = model.unembed(rec.activations[model.n_layers-1][:, -1]).float()
        native = model._hf_model(input_ids=ids, use_cache=False).logits[:, -1].float()
        error = float((wrapped-native).abs().max())
        if not torch.allclose(wrapped, native, atol=.05, rtol=.01):
            raise RuntimeError(f"Native/wrapped logits disagree: {error}")
    cot = unembedding_rows(model, [single_token_id(model, "France")])
    text = p["diagonal_fit"]["fit_texts"][0]
    tiny = component_pullbacks_for_prompt(model, text, layers[:1], cot, dim_batch=1, max_seq_len=32, skip_first=16)
    direct = pullback_for_prompt(model, text, layers[:1], cot, dim_batch=1, max_seq_len=32, skip_first=16)
    relative = float((tiny[layers[0]]["total"]-direct[layers[0]]).norm()/direct[layers[0]].norm())
    if not relative < .05:
        raise RuntimeError(f"Independent pullback disagreement: {relative}")
    save(root / "smoke.json", {"logit_max_error": error, "pullback_relative_error": relative})
    checkpoint("smoke_passed", {"logit_error": error, "pullback_error": relative})

    units = build_units(clues=("0", "1"))
    candidates = panel["latent_candidates"]
    requested = sorted({w for i in candidates for w in (i["intermediate"], i["swap_to"])} |
                       {w for u in units for w in (u.source, u.target)})
    token_ids, excluded = {}, {}
    for word in requested:
        try:
            token_ids[word] = single_token_id(model, word)
        except ValueError as exc:
            excluded[word] = str(exc)
    save(root / "token_coverage.json", {"supported": token_ids, "excluded": excluded})
    units = [u for u in units if u.source in token_ids and u.target in token_ids]
    candidates = [i for i in candidates if i["intermediate"] in token_ids and i["swap_to"] in token_ids]
    clean_path = root / "clean_generations.json"
    clean = json.loads(clean_path.read_text()) if clean_path.exists() else {}

    def generate(text, n):
        key = sha([text, n])
        if key not in clean:
            budget()
            clean[key] = evaluate(model, encode(model, text), {}, gen_tokens=n)["texts"][0]
            save(clean_path, clean)
        return clean[key]

    screen = []
    for index, item in enumerate(candidates):
        text = generate(item["prompt"], 6)
        ok = hits(text, item["answer_forms"]) and not hits(text, item["swap_answer_forms"]) and item["swap_to"].casefold() not in text.casefold()
        screen.append({"name": item["name"], "text": text, "eligible": ok})
        if (index+1) % 50 == 0:
            checkpoint("screen_latent", index+1)
    save(root / "latent_screen.json", screen)
    keep = {r["name"] for r in screen if r["eligible"]}
    latent = [i for i in candidates if i["name"] in keep]
    eligible = {}
    for u in units:
        rels = []
        for q in u.queries:
            if q.kind in ("express", "use"):
                text = generate(u.prefix+q.text, 8)
                if hits(text, q.source) and not hits(text, q.target):
                    rels.append(q.relation)
        if "express" in rels and len(rels) > 1:
            eligible[u.key] = rels
    chosen = select_units(eligible, units)
    selected = [u for u in units if u.key in chosen]
    save(root / "eligibility.json", {"latent": latent, "concept_eligible": eligible, "concept_selected": chosen})
    checkpoint("eligibility_frozen", {"latent": len(latent), "concept": len(selected), "excluded_tokens": len(excluded)})
    if not latent or not selected:
        raise RuntimeError("Insufficient clean eligibility; no interventions executed")

    words = sorted({w for i in latent for w in (i["intermediate"], i["swap_to"])} |
                   {w for u in selected for w in (u.source, u.target)})
    cotangents = unembedding_rows(model, [token_ids[w] for w in words])
    lens_path = hf_hub_download(config.lens_repo, config.lens_filename, revision=config.lens_revision, local_files_only=True)
    lens = JacobianLens.load(lens_path)
    if lens.d_model != config.d_model or lens.n_prompts != entry["published_fit_prompts"] or not set(layers) <= set(lens.source_layers):
        raise ValueError("Released lens prompt count or layer coverage differs")
    full = {l: cotangents @ lens.jacobians[l] for l in layers}
    full_copy = {l: value.clone() for l, value in full.items()}
    published_n = lens.n_prompts
    save(root / "published_metadata.json", {"repository": config.lens_repo, "revision": config.lens_revision,
         "filename": config.lens_filename, "file_sha256": file_hash(lens_path),
         "n_prompts": published_n, "d_model": lens.d_model, "source_layers": lens.source_layers,
         "full_lens_refitted": False})
    del lens
    checkpoint("published_lens_loaded", {"published_prompts": published_n, "words": len(words)})
    partial_path = root / "diagonal_partial.pt"
    count, sums, first32 = 0, {}, None
    if partial_path.exists():
        part = torch.load(partial_path, map_location="cpu", weights_only=False)
        if part["manifest_sha"] != sha(manifest) or part["words"] != words:
            raise ValueError("Diagonal partial input mismatch")
        count, sums, first32 = part["count"], part["sums"], part["first32"]
    fit = p["diagonal_fit"]
    texts = fit["fit_texts"]
    if len(texts) != 64:
        raise ValueError("Frozen protocol requires 64 diagonal fitting texts")
    for index, text in enumerate(texts):
        if index < count:
            continue
        budget()
        if model.encode(text, max_length=128).shape[1] != 128:
            raise ValueError("Diagonal fitting text did not fill the 128-token window")
        tick = time.monotonic()
        parts = component_pullbacks_for_prompt(model, text, layers, cotangents, dim_batch=8,
                                               max_seq_len=128, skip_first=16, t_stride=4)
        # Only the diagonal is accumulated. The released full directions above
        # remain the baseline; generic helper's total/off outputs are discarded.
        for l in layers:
            value = parts[l]["diag"]
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite diagonal estimate")
            sums[l] = sums.get(l, torch.zeros_like(value)) + value
        count = index+1
        if count == 32:
            first32 = {l: value.clone() for l, value in sums.items()}
        temp = partial_path.with_suffix(".tmp")
        torch.save({"manifest_sha": sha(manifest), "words": words, "count": count,
                    "sums": sums, "first32": first32}, temp)
        temp.replace(partial_path)
        checkpoint("estimating_diagonal", {"prompts": count, "total": 64, "last_seconds": round(time.monotonic()-tick, 2)})
        if count == 1 and time.monotonic()-started + (time.monotonic()-tick)*63 + 900 > spec.max_seconds:
            raise TimeoutError("Measured diagonal-fit speed exceeds the per-model allowance")
    if count != 64 or first32 is None:
        raise RuntimeError("Incomplete diagonal estimate")
    diagonal64 = {l: value/64 for l, value in sums.items()}
    diagonal32 = {l: value/32 for l, value in first32.items()}
    diagnostics = {}
    for l in layers:
        second = (sums[l]-first32[l])/32
        diagnostics[l] = {
            "split_half_mean_cosine": float(F.cosine_similarity(diagonal32[l], second, dim=1).mean()),
            "diag64_to_published_mean_norm_ratio": float((diagonal64[l].norm(dim=1)/full[l].norm(dim=1).clamp_min(1e-12)).mean()),
            "filtered32_vs_filtered64_mean_cosine": float(F.cosine_similarity(full[l]-diagonal32[l], full[l]-diagonal64[l], dim=1).mean()),
        }
    digests = derived_digests(full, diagonal64, diagonal32)
    for l in layers:
        digests[f"published_full|{l}"]["n"] = published_n
        if not torch.equal(full[l], full_copy[l]):
            raise RuntimeError("Published full directions were modified")
    fit_path = root / "components.pt"
    torch.save({"targets": words, "digests": digests, "manifest_sha": sha(manifest),
                "published_full_refitted": False, "filter_is_estimated": True}, fit_path)
    save(root / "fit_diagnostics.json", {"diagonal_prompts": 64, "published_prompts": published_n,
         "published_full_unchanged": True, "filter_is_estimated": True, "layers": diagnostics})
    bank = Bank(model, [str(fit_path)], words)
    checkpoint("diagonal_estimate_complete", {"prompts": count, "published_full_unchanged": True})

    def intervene(prompt, pair, arms, strengths, tokens):
        budget()
        ids = encode(model, prompt)
        h = band_residuals(model, ids, layers)
        report = {}
        per = swap_deltas(bank, h, pair, "published", list(dict.fromkeys(["full"]+arms)), ids.shape[1], report)
        max_error = 0.
        for arm in arms:
            if arm != "full":
                for l in layers:
                    norm = per["full"][l].norm(dim=-1)
                    max_error = max(max_error, float(((per[arm][l].norm(dim=-1)-norm).abs()/norm.clamp_min(1e-6)).max()))
        if max_error > 1e-4 or sum(report.values()):
            raise RuntimeError(f"Per-site matching failed: {max_error}, {report}")
        settings = [(a, strength) for a in arms for strength in strengths]
        deltas = {l: torch.stack([per[a][l]*strength for a, strength in settings]).to(ids.device) for l in layers}
        texts = evaluate(model, ids, deltas, gen_tokens=tokens, chunk=6)["texts"]
        return settings, texts, max_error

    path = root / "latent_results.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {r["name"] for r in rows}
    for item in latent:
        if item["name"] in done:
            continue
        pair = SimpleNamespace(source=item["intermediate"], target=item["swap_to"])
        settings, texts, error = intervene(item["prompt"], pair, p["latent_arms"], p["latent_strengths"], 6)
        for (arm, strength), text in zip(settings, texts, strict=True):
            rows.append({"name": item["name"], "set": item["set"], "source": pair.source, "target": pair.target,
                         "arm": arm, "strength": strength, "text": text, "success": hits(text, item["swap_answer_forms"]),
                         "leakage": pair.target.casefold() in text.casefold(), "norm_error": error})
        save(path, rows)
        if len(rows) % 120 == 0:
            checkpoint("latent_interventions", len(rows)//6)
    expected = {(i["name"], a, s) for i in latent for a in p["latent_arms"] for s in p["latent_strengths"]}
    if len(rows) != len(expected) or {(r["name"], r["arm"], r["strength"]) for r in rows} != expected:
        raise RuntimeError("Incomplete or duplicated latent results")
    checkpoint("latent_complete", len(latent))
    path = root / "concept_results.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {(r["key"], r["relation"]) for r in rows}
    for u in selected:
        for q in u.queries:
            if q.relation not in eligible[u.key] or (u.key, q.relation) in done:
                continue
            settings, texts, error = intervene(u.prefix+q.text, u, p["concept_arms"], [1.], 8)
            for (arm, strength), text in zip(settings, texts, strict=True):
                rows.append({"key": u.key, "type": u.type, "source": u.source, "target": u.target,
                             "relation": q.relation, "kind": q.kind, "arm": arm, "strength": strength,
                             "text": text, "success": hits(text, q.target),
                             "leakage": u.target.casefold() in text.casefold(), "norm_error": error})
            save(path, rows)
        checkpoint("concept_interventions", len({r["key"] for r in rows}))
    expected = {(u.key, relation, arm) for u in selected for relation in eligible[u.key] for arm in p["concept_arms"]}
    if len(rows) != len(expected) or {(r["key"], r["relation"], r["arm"]) for r in rows} != expected:
        raise RuntimeError("Incomplete or duplicated concept results")
    checkpoint("complete", {"latent_items": len(latent), "concept_units": len(selected), "concept_rows": len(rows)})
    return {"path": str(root), "status": "complete", "seconds": time.monotonic()-started}

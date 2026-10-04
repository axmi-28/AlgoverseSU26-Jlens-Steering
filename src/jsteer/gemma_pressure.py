"""Frozen exploratory Gemma diagnostics. Selection precedes validation.

All scoring uses existing answer aliases; no revised grading based on edited
outputs. Released full-lens projection is a proxy control, not an exact
position-pair-filtered lens. Exact filtering uses jointly fitted components.
"""
from __future__ import annotations

import gc
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from jsteer.config import REPO_ROOT, load_config
from jsteer.concept_use import Bank, band_residuals, build_units, encode, evaluate, hits, swap_deltas
from jsteer.cross_model import save, sha
from jsteer.jacobian import component_pullbacks_for_prompt
from jsteer.loading import load_model, single_token_id, unembedding_rows
from jsteer.steering import ResidualEdit, answer_matches, greedy_continuation
from jsteer.write_controls import file_hash

PROTOCOL = REPO_ROOT / "data/protocols/gemma_pressure.json"
QWEN_DIGEST = "rq1/group_mean_digests_qwen3-8b_dom2x32_968c0ba0_pb_t89d64f82_comp_s4.pt"
OLD_ROOT = "cross_model/gemma-3-12b-it/22b06e8c0089"


def task_mean(rows, metric):
    """Equal task weight, equal unit weight within task, query mean within unit."""
    tasks = defaultdict(lambda: defaultdict(list))
    for r in rows:
        tasks[r["task"]][r["unit"]].append(float(r[metric]))
    means = [sum(sum(v)/len(v) for v in units.values())/len(units) for units in tasks.values()]
    return sum(means)/len(means) if means else float("nan")


def choose_settings(rows, settings):
    """Use development outcomes only; preserve all failed settings in output."""
    if any(r["split"] != "dev" for r in rows):
        raise ValueError("Validation data cannot enter selection")
    baseline = task_mean([r for r in rows if r["setting"] == "gsm8k/original/1/native" and r["arm"] == "full"], "success")
    stats = []
    for s in settings:
        if s["corpus"] == "released":
            continue
        full = task_mean([r for r in rows if r["setting"] == s["id"] and r["arm"] == "full"], "success")
        off = task_mean([r for r in rows if r["setting"] == s["id"] and r["arm"] == "filtered"], "success")
        stats.append({"id": s["id"], "corpus": s["corpus"], "full": full, "filtered": off,
                      "gain": off-full, "qualifies": off > baseline and off > full})
    selected = []
    for corpus in ("gsm8k", "wikitext_a"):
        pool = [r for r in stats if r["corpus"] == corpus]
        good = [r for r in pool if r["qualifies"]]
        ranked = sorted(good, key=lambda r: (-r["filtered"], -r["gain"], r["id"])) if good else sorted(pool, key=lambda r: (-r["gain"], -r["filtered"], r["id"]))
        selected.append(ranked[0])
    return {"baseline_dev_accuracy": baseline, "selected": selected, "all_settings": stats}


@dataclass
class PressureSpec:
    expected_source_sha: str
    out_dir: str = "/results/gemma_pressure"
    max_seconds: int = 10200


def run_pressure(spec: PressureSpec):
    import modal
    from huggingface_hub import hf_hub_download
    from jlens import JacobianLens
    start = time.monotonic()
    if spec.expected_source_sha != file_hash(Path(__file__)):
        raise ValueError("Pressure source did not ship")
    p = json.loads(PROTOCOL.read_text())
    config = load_config("gemma-3-12b-it")
    volume = modal.Volume.from_name("jsteer-results")
    manifest = {"protocol_sha": file_hash(PROTOCOL), "source_sha": spec.expected_source_sha,
                "model": asdict(config), "dependencies": {n: file_hash(Path(__file__).with_name(n)) for n in
                ("concept_use.py", "jacobian.py", "loading.py", "steering.py", "write_controls.py")}}
    root = Path(spec.out_dir) / sha(manifest)[:12]
    root.mkdir(parents=True, exist_ok=True)
    save(root / "manifest.json", manifest)

    def checkpoint(stage, detail=None):
        body = {"stage": stage, "seconds": round(time.monotonic()-start, 2), "detail": detail}
        save(root / "status.json", body)
        volume.commit()
        print(json.dumps(body), flush=True)

    def budget():
        if time.monotonic()-start > spec.max_seconds:
            checkpoint("budget_stop")
            raise TimeoutError("Frozen pressure-test budget reached")

    units = {u.key: u for u in build_units(clues=("0", "1"))}
    # End-to-end positive control with an existing fit and known Qwen outputs.
    checkpoint("qwen_positive_control")
    model = load_model(load_config("qwen3-8b"))
    words = sorted({w for c in p["qwen_check"] for w in (units[c["key"]].source, units[c["key"]].target)})
    bank = Bank(model, [str(Path("/results")/QWEN_DIGEST)], words)
    checks = []
    with torch.no_grad():
        for c in p["qwen_check"]:
            u = units[c["key"]]
            q = next(q for q in u.queries if q.relation == c["relation"])
            ids = encode(model, u.prefix+q.text)
            h = band_residuals(model, ids, [13,16,19,22,25,28,31])
            per = swap_deltas(bank, h, u, "gsm8k", ["lam1_cn", "lam0_cn"], ids.shape[1])
            delta = {l: torch.stack([per[a][l] for a in ("lam1_cn", "lam0_cn")]).to(ids.device) for l in h}
            with torch.no_grad():
                texts = evaluate(model, ids, delta, gen_tokens=8, chunk=2)["texts"]
            observed = [hits(t, q.target) for t in texts]
            checks.append({**c, "observed": observed, "texts": texts})
    save(root/"qwen_positive_control.json", checks)
    mismatch = sum(a != b for c in checks for a,b in zip(c["expected"], c["observed"]))
    del model, bank, delta, per, h, ids
    gc.collect(); torch.cuda.empty_cache()
    if mismatch:
        checkpoint("qwen_control_mismatch", mismatch)
        raise RuntimeError("Historical Qwen replay differs; inspect before interpreting new interventions")

    checkpoint("loading_gemma")
    model = load_model(config)
    layers = sorted({l for ls in p["layers"].values() for l in ls})
    old_path = Path("/results")/OLD_ROOT/"components.pt"
    old = torch.load(old_path, map_location="cpu", weights_only=False)
    words = old["targets"]
    ids = [single_token_id(model, w) for w in words]
    cot = unembedding_rows(model, ids)
    oldbank = Bank(model, [str(old_path)], words)
    cases = []
    for i in p["latent"]:
        cases.append({"unit": i["name"], "query": i["name"], "task": "latent", "split": i["split"],
                      "source": i["intermediate"], "target": i["swap_to"], "prompt": i["prompt"],
                      "source_answers": i["answer_forms"], "target_answers": i["swap_answer_forms"],
                      "one_hop": i.get("one_hop_target"), "type": i.get("type", i["category"])})
    panel = json.loads((REPO_ROOT/"data/heldout/concept-use.json").read_text())
    for entry in p["concept"]:
        u = units[entry["key"]]
        for q in u.queries:
            if q.relation in entry["relations"]:
                cases.append({"unit": u.key, "query": u.key+"/"+q.relation, "task": "concept", "split": entry["split"],
                              "source": u.source, "target": u.target, "prompt": u.prefix+q.text,
                              "source_answers": list(q.source), "target_answers": list(q.target),
                              "one_hop": panel["prefix"].format(clue=u.target)+q.text, "type": u.type})
    if any(c["source"] not in words or c["target"] not in words for c in cases):
        raise RuntimeError("New cases lack fitted-token coverage")
    # Fresh clean correctness and target-knowledge checks, before new edits.
    clean_cache = {}
    def clean(text, n=8):
        if text not in clean_cache:
            budget()
            with torch.no_grad():
                clean_cache[text] = evaluate(model, encode(model,text), {}, gen_tokens=n)["texts"][0]
        return clean_cache[text]
    for c in cases:
        c["clean_text"] = clean(c["prompt"])
        c["clean_ok"] = hits(c["clean_text"],c["source_answers"]) and not hits(c["clean_text"],c["target_answers"])
        c["target_known"] = hits(clean(c["one_hop"]), c["target_answers"]) if c["one_hop"] else None
        c["strict_question"] = not any(c["target"].casefold() in a.casefold() for a in c["target_answers"])
    save(root/"question_audit.json", cases)
    # Preserve the frozen population. A changed clean outcome is a diagnostic,
    # never a reason to silently choose a more favorable subset.
    if any(not c["clean_ok"] for c in cases):
        checkpoint("clean_replay_mismatch", sum(not c["clean_ok"] for c in cases))
        raise RuntimeError("Frozen clean eligibility did not replay")
    engine=[]
    with torch.no_grad():
        for c in [c for c in cases if c["split"] == "dev"][:4]:
            ids = encode(model,c["prompt"])
            h = band_residuals(model,ids,p["layers"]["original"])
            per = swap_deltas(oldbank,h,SimpleNamespace(source=c["source"],target=c["target"]),"gsm8k",["full","off_m"],ids.shape[1])
            for a in ["full","off_m"]:
                cached = evaluate(model,ids,{l:d[None].to(ids.device) for l,d in per[a].items()},gen_tokens=8)["texts"][0]
                uncached = greedy_continuation(model,c["prompt"],edit=ResidualEdit(per[a],positions=list(range(ids.shape[1]))),n_tokens=8,max_seq_len=1024)
                engine.append({"query":c["query"],"arm":a,"cached":cached,"uncached":uncached,
                               "hit_equal":hits(cached,c["target_answers"])==hits(uncached,c["target_answers"])})
    save(root/"generation_parity.json",engine)
    if not all(r["hit_equal"] for r in engine):
        checkpoint("generation_parity_failed")
        raise RuntimeError("Cached and uncached scoring differ")
    checkpoint("evaluation_audit_passed", {"queries":len(cases),"qwen_predictions":len(checks)*2})

    # Jointly fit both components; retain every prompt estimate for stability
    # checks and independent split-half comparisons without future GPU work.
    fit_path = root/"components.pt"
    stability=[]
    if not fit_path.exists():
        digests={}
        for corpus,texts in p["fit_texts"].items():
            sums={}; half=None
            for j,text in enumerate(texts):
                budget()
                part_path=root/"fit_prompts"/f"{corpus}_{j:02d}.pt"
                if part_path.exists():
                    parts=torch.load(part_path,map_location="cpu",weights_only=False)
                else:
                    parts=component_pullbacks_for_prompt(model,text,layers,cot,dim_batch=8,max_seq_len=128,skip_first=16,t_stride=4)
                    part_path.parent.mkdir(parents=True,exist_ok=True)
                    torch.save(parts,part_path)
                for l in layers:
                    for part in ("total","diag","off"):
                        v=parts[l][part]
                        if not torch.isfinite(v).all():raise ValueError("Nonfinite fit")
                        sums[l,part]=sums.get((l,part),torch.zeros_like(v))+v
                if j==15:half={k:v.clone() for k,v in sums.items()}
                if (j+1)%4==0:checkpoint("fitting",{"corpus":corpus,"prompts":j+1})
            for (l,part),v in sums.items():
                fam="full" if part=="total" else part
                digests[f"{corpus}_{fam}|{l}"]={"pullbacks":v/32,"n":32}
                a=half[l,part]/16;b=(v-half[l,part])/16
                cs=torch.nn.functional.cosine_similarity(a,b,dim=-1)
                stability.append({"corpus":corpus,"layer":l,"part":part,"split_half_cos_mean":float(cs.mean()),"split_half_cos_min":float(cs.min())})
        torch.save({"targets":words,"digests":digests,"manifest":manifest},fit_path)
        save(root/"fit_stability.json",stability)

    released_path=hf_hub_download(p["released_repo"],p["released_file"],revision=p["released_revision"],local_files_only=True)
    released=JacobianLens.load(released_path)
    if released.d_model!=config.d_model:raise ValueError("Released lens dimension mismatch")
    released_digest=root/"released_directions.pt"
    torch.save({"targets":words,"digests":{f"released_full|{l}":{"pullbacks":cot@J,"n":released.n_prompts} for l,J in released.jacobians.items()}},released_digest)
    save(root/"released_metadata.json",{"n_prompts":released.n_prompts,"layers":released.source_layers,"d_model":released.d_model,"file_sha":file_hash(released_path),"revision":p["released_revision"]})
    bank=Bank(model,[str(fit_path),str(released_digest)],words)
    del released
    settings=[]
    for corpus in ("gsm8k","wikitext_a"):
        for grid in p["layers"]:
            for strength in p["strengths"]:
                for variant in p["column_variants"]:
                    settings.append({"id":f"{corpus}/{grid}/{strength:g}/{variant}","corpus":corpus,"grid":grid,"strength":strength,"variant":variant})
    # Released lens is a benchmark plus an explicitly labeled unembedding-
    # projection proxy. Never call this the released off-diagonal lens.
    for grid in ("original","global"):
        if not set(p["layers"][grid])<=set(bank_layer for group,bank_layer in bank.rows if group=="released_full"):
            continue
        for strength in p["strengths"]:
            settings.append({"id":f"released/{grid}/{strength:g}/native","corpus":"released","grid":grid,"strength":strength,"variant":"native"})
    save(root/"settings.json",settings)
    checkpoint("fits_ready",{"settings":len(settings)})

    @torch.no_grad()
    def run_cases(split,chosen):
        path=root/f"{split}_results.json"
        rows=json.loads(path.read_text()) if path.exists() else []
        done={r["query"] for r in rows}
        subset=[c for c in cases if c["split"]==split]
        for number,c in enumerate(subset):
            if c["query"] in done:continue
            budget()
            ids=encode(model,c["prompt"])
            h=band_residuals(model,ids,layers)
            pair=SimpleNamespace(source=c["source"],target=c["target"])
            builds={}; conditions=[]
            for s in chosen:
                k=(s["corpus"],s["grid"],s["variant"])
                arms=["full","proj_m"] if s["corpus"]=="released" else (["full","off_m"] if s["variant"]=="native" else ["lam1_cn","lam0_cn"])
                if k not in builds:
                    report={}
                    hs={l:h[l] for l in p["layers"][s["grid"]]}
                    builds[k]=swap_deltas(bank,hs,pair,s["corpus"],list(dict.fromkeys(["full"]+arms)),ids.shape[1],report)
                    if sum(report.values()):raise RuntimeError("Unmatchable zero edit")
                    for arm in arms:
                        for l in hs:
                            ref=builds[k]["full"][l].norm(dim=-1)
                            err=(builds[k][arm][l].norm(dim=-1)-ref).abs()/ref.clamp_min(1e-6)
                            if float(err.max())>1e-4:raise RuntimeError("Norm matching failed")
                for arm,label in zip(arms,["full","projection_proxy" if s["corpus"]=="released" else "filtered"]):
                    conditions.append((s,arm,label,k))
            for lo in range(0,len(conditions),6):
                batch=conditions[lo:lo+6]
                d={l:torch.stack([builds[k][a][l]*s["strength"] if l in p["layers"][s["grid"]] else torch.zeros_like(h[l]) for s,a,label,k in batch]).to(ids.device) for l in layers}
                texts=evaluate(model,ids,d,gen_tokens=8,chunk=6)["texts"]
                for (s,a,label,k),text in zip(batch,texts):
                    success=hits(text,c["target_answers"])
                    name=c["target"].casefold() in text.casefold()
                    rows.append({"query":c["query"],"unit":c["unit"],"source":c["source"],"target":c["target"],"type":c["type"],
                                 "task":c["task"],"split":split,"setting":s["id"],"arm":label,"text":text,"success":success,
                                 "source_answer":hits(text,c["source_answers"]),"name_emission":name,"incorrect_name_emission":name and not success,
                                 "answer_initial_name_error":not success and answer_matches(text,c["target"]),
                                 "target_known":c["target_known"],"strict_question":c["strict_question"]})
            save(path,rows)
            if (number+1)%4==0 or number+1==len(subset):checkpoint(split,{"queries":number+1,"total":len(subset)})
        if len(rows)!=len(subset)*len(chosen)*2:raise RuntimeError("Incomplete paired results")
        return rows

    dev=run_cases("dev",settings)
    selection_path=root/"selection.json"
    if not selection_path.exists():
        selection=choose_settings(dev,settings)
        selection["dev_results_sha"]=file_hash(root/"dev_results.json")
        save(selection_path,selection)
    else:selection=json.loads(selection_path.read_text())
    checkpoint("selection_frozen",selection["selected"])
    selected_ids={s["id"] for s in selection["selected"]}|{"gsm8k/original/1/native","gsm8k/original/1/unit","wikitext_a/original/1/native","released/original/1/native"}
    chosen=[s for s in settings if s["id"] in selected_ids]
    run_cases("validation",chosen)
    checkpoint("complete",{"selected":selection["selected"],"validation_settings":len(chosen)})
    return {"path":str(root),"status":"complete","seconds":time.monotonic()-start}

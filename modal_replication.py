"""One persistent app for the frozen multi-model filtering campaign.

Deploy once with `modal deploy modal_replication.py`. Launches use deployed
function handles, never `modal run`; at most two GPU containers can run.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import modal

if not modal.is_local():
    sys.path.insert(0, "/root/repo")

from modal_app import CACHE, RESULTS, cache_volume, image, results_volume  # noqa: E402

replication_image = image.env({"HF_XET_CACHE": "/tmp/hf-xet", "HF_XET_LOG_DEST": "/tmp/hf-xet-logs/"})

APP_NAME = "jlens-filtering-replication"
app = modal.App(APP_NAME)
VOLUMES = {CACHE: cache_volume, RESULTS: results_volume}
ROOT = Path("/results/filtering_replication")


def _save_campaign(state):
    from jsteer.cross_model import save
    results_volume.reload()
    save(Path(state.get("out_dir", str(ROOT))) / "campaign.json", state)
    results_volume.commit()
    print(json.dumps({"campaign": state.get("out_dir", str(ROOT)), "stage": state["stage"],
                      "models": {name: row["stage"] for name, row in state["models"].items()}}), flush=True)


def _wait_for_gpu_slot(calls, state, limit=2):
    """Bound outstanding jobs even when recovery spans deployment versions."""
    while True:
        pending = [name for name in calls if state["models"][name]["stage"] not in ("complete", "failed")]
        if len(pending) < limit:
            return
        name = pending[0]
        try:
            state["models"][name].update(stage="complete", result=calls[name].get())
        except Exception as exc:
            state["models"][name].update(stage="failed", error=f"{type(exc).__name__}: {exc}")
        _save_campaign(state)


@app.function(image=replication_image, volumes=VOLUMES, cpu=2, memory=8192,
              timeout=1200, max_containers=1, retries=0, scaledown_window=2)
def prepare(config_name: str, published_lens: bool = False):
    from huggingface_hub import hf_hub_download, snapshot_download

    from jsteer.config import load_config
    config = load_config(config_name)
    if not config.model_revision:
        raise ValueError("Every replication checkpoint must have a pinned revision")
    cache_volume.reload()
    snapshot_download(config.hf_model_id, revision=config.model_revision,
                      allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"],
                      ignore_patterns=["consolidated.safetensors"])
    result = {"config": config_name, "revision": config.model_revision}
    if published_lens:
        from jsteer.write_controls import file_hash
        if config.lens_repo != "neuronpedia/jacobian-lens" or not config.lens_revision:
            raise ValueError("Published Neuronpedia lens with a pinned revision required")
        path = hf_hub_download(config.lens_repo, config.lens_filename, revision=config.lens_revision)
        result.update(lens_filename=config.lens_filename, lens_revision=config.lens_revision,
                      lens_sha256=file_hash(path))
    cache_volume.commit()
    return result


@app.function(image=replication_image, volumes=VOLUMES, gpu="H100", cpu=4, memory=32768,
              timeout=3600, max_containers=2, buffer_containers=0,
              retries=0, scaledown_window=2)
def run_model(spec_kwargs: dict, kind: str = "self_fitted"):
    from jsteer.cross_model import CrossModelSpec, run_cross_model, save
    results_volume.reload()
    cache_volume.reload()
    try:
        if kind == "neuronpedia":
            from jsteer.neuronpedia_replication import NeuronpediaSpec, run_neuronpedia
            return run_neuronpedia(NeuronpediaSpec(**spec_kwargs))
        if kind != "self_fitted":
            raise ValueError(f"Unknown model-run kind: {kind}")
        return run_cross_model(CrossModelSpec(**spec_kwargs))
    except Exception as exc:
        save(Path(spec_kwargs.get("out_dir", str(ROOT))) / "failures" / f"{spec_kwargs['config_name']}.json",
             {"error": f"{type(exc).__name__}: {exc}", "time": time.time(), "spec": spec_kwargs})
        results_volume.commit()
        raise


@app.function(image=replication_image, volumes=VOLUMES, cpu=1, memory=2048,
              timeout=21600, max_containers=1, retries=0, scaledown_window=2)
def campaign(expected: dict, campaign_name: str = "filtering_replication"):
    """Durable CPU supervisor; GPU calls outlive the local launcher."""
    import subprocess

    from jsteer.config import REPO_ROOT
    from jsteer.cross_model import CrossModelSpec, input_hashes, sha
    from jsteer.write_controls import file_hash
    if campaign_name not in ("filtering_replication", "neuronpedia_replication"):
        raise ValueError("Unknown frozen campaign")
    published = campaign_name == "neuronpedia_replication"
    root = ROOT.with_name(campaign_name) if published else ROOT
    spec_class = CrossModelSpec
    if published:
        from jsteer.neuronpedia_replication import NeuronpediaSpec
        from jsteer.neuronpedia_replication import input_hashes as published_hashes
        input_hashes = published_hashes
        spec_class = NeuronpediaSpec
    protocol_path = REPO_ROOT / "data/protocols" / f"{campaign_name}.json"
    protocol = json.loads(protocol_path.read_text())
    if file_hash(protocol_path) != expected["protocol_sha"]:
        raise ValueError("Deployed campaign protocol differs from launch")
    for name in protocol["models"]:
        if sha(input_hashes(name)) != expected["inputs_sha"][name]:
            raise ValueError(f"Deployed inputs differ: {name}")
    results_volume.reload()
    if (root / "campaign.json").exists():
        state = json.loads((root / "campaign.json").read_text())
        if state["expected"] != expected:
            raise RuntimeError("Existing campaign has different inputs")
        if state["stage"] in ("complete", "complete_with_failures"):
            return state
        # Infrastructure recovery reattaches to recorded calls. It never
        # repeats a model invocation because a CPU supervisor was restarted.
        state.setdefault("recoveries", []).append(time.time())
    else:
        state = {"app": APP_NAME, "stage": "running", "started": time.time(),
                 "protocol": protocol, "expected": expected, "models": {}}
    state["out_dir"] = str(root)
    _save_campaign(state)
    warm_fn = modal.Function.from_name(APP_NAME, "prepare")
    gpu_fn = modal.Function.from_name(APP_NAME, "run_model")
    if published and any(not state["models"].get(name, {}).get("gpu_call_id")
                         and state["models"].get(name, {}).get("stage") != "failed"
                         for name in protocol["models"]):
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("Gated-model preparation requires the ephemeral HF credential")
        secrets = [modal.Secret.from_dict({"HF_TOKEN": token})]
        warm_fn = warm_fn.with_options(secrets=secrets)
        gpu_fn = gpu_fn.with_options(secrets=secrets)
    calls = {name: modal.FunctionCall.from_id(row["gpu_call_id"])
             for name, row in state["models"].items() if row.get("gpu_call_id")}
    for name in protocol["models"]:
        record = state["models"].setdefault(name, {"stage": "preparing"})
        if record.get("gpu_call_id"):
            continue
        if record["stage"] == "failed":
            continue
        if record["stage"] == "submitting_gpu":
            raise RuntimeError(f"Uncertain prior submission for {name}; reconcile call graph before launching")
        _save_campaign(state)
        try:
            warm = (modal.FunctionCall.from_id(record["prepare_call_id"])
                    if record.get("prepare_call_id") else
                    (warm_fn.spawn(name, published_lens=True) if published else warm_fn.spawn(name)))
            record["prepare_call_id"] = warm.object_id
            _save_campaign(state)
            record["prepared"] = warm.get()
            _wait_for_gpu_slot(calls, state)
            spec = spec_class(config_name=name, out_dir=str(root),
                                  max_seconds=protocol["scientific_budget_seconds_per_model"],
                                  expected_source_sha=expected["source_sha"],
                                  expected_inputs_sha=expected["inputs_sha"][name])
            record["stage"] = "submitting_gpu"
            _save_campaign(state)
            call = (gpu_fn.spawn(asdict(spec), kind="neuronpedia") if published
                    else gpu_fn.spawn(asdict(spec)))
            calls[name] = call
            record.update(stage="submitted", gpu_call_id=call.object_id)
        except Exception as exc:
            record.update(stage="failed", error=f"{type(exc).__name__}: {exc}")
        _save_campaign(state)
    for name, call in calls.items():
        try:
            state["models"][name].update(stage="complete", result=call.get())
        except Exception as exc:
            state["models"][name].update(stage="failed", error=f"{type(exc).__name__}: {exc}")
        _save_campaign(state)
    results_volume.reload()
    if any(r["stage"] == "complete" for r in state["models"].values()):
        scripts = ("73_neuronpedia_report.py",) if published else ("60_cross_model_report.py", "67_filtering_replication_report.py")
        for script in scripts:
            subprocess.run(["python", str(REPO_ROOT / "scripts" / script), str(root)], check=True)
        results_volume.commit()
    state.update(stage="complete" if all(r["stage"] == "complete" for r in state["models"].values())
                 else "complete_with_failures", finished=time.time())
    _save_campaign(state)
    return state

"""Launch one sequential cross-model replication under the deployed jsteer app.

No modal run. A six-hour remote timeout includes load, screening, fit and eval.
The local credential is passed as an ephemeral Modal secret, never written.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import modal
from huggingface_hub import get_token

from jsteer.cross_model import CrossModelSpec
from jsteer.write_controls import file_hash

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/cross_model"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", choices=["gemma-3-12b-it", "llama-3.1-8b-instruct"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    # One launcher for this protocol, including CPU warm-up. Other existing
    # workspace jobs are not touched.
    import fcntl
    lock = (OUT / "launcher.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = OUT / f"launch_{args.config}.json"
    previous = json.loads(path.read_text()) if path.exists() else None
    if previous and previous.get("gpu_call_id"):
        raise SystemExit(f"GPU launch already recorded: {path}; inspect before relaunch")
    token = get_token()
    if not token:
        raise SystemExit("No local Hugging Face credential")
    secrets = [modal.Secret.from_dict({"HF_TOKEN": token})]
    state = previous or {"config": args.config, "started": time.time(), "max_gpu_seconds": 21600,
                         "source_sha": file_hash(ROOT / "src/jsteer/cross_model.py")}
    def save():
        path.write_text(json.dumps(state, indent=2))
    warm = (modal.FunctionCall.from_id(state["warm_call_id"]) if previous else
            modal.Function.from_name("jsteer", "prepare_cross_model").with_options(
                secrets=secrets, max_containers=1, retries=0, scaledown_window=2,
            ).spawn(args.config))
    state.update(stage="warming", warm_call_id=state["warm_call_id"] if previous else warm.object_id)
    save()
    print(json.dumps(state), flush=True)
    state["warm_result"] = warm.get()
    from dataclasses import asdict
    spec = CrossModelSpec(config_name=args.config, expected_source_sha=state["source_sha"])
    gpu = modal.Function.from_name("jsteer", "gpu_task").with_options(
        gpu="A100-80GB", secrets=secrets, max_containers=1,
        buffer_containers=0, scaledown_window=2, retries=0, timeout=21600,
    ).spawn("cross_model", asdict(spec))
    state.update(stage="gpu_running", gpu_call_id=gpu.object_id, gpu_started=time.time())
    save()
    print(json.dumps(state), flush=True)
    try:
        state["result"] = gpu.get()
        state["stage"] = "complete"
    except BaseException as exc:
        state.update(stage="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        state["finished"] = time.time()
        save()
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()

"""Launch the frozen Gemma pressure test under the existing deployed app."""
import fcntl
import json
import time
from dataclasses import asdict
from pathlib import Path

import modal
from huggingface_hub import get_token

from jsteer.gemma_pressure import PressureSpec
from jsteer.write_controls import file_hash

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/gemma_pressure"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT/"results/cross_model/launcher.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = OUT/"launch.json"
    if path.exists():
        raise SystemExit("Pressure launch ledger exists; inspect before relaunch")
    state = {"started":time.time(),"max_gpu_seconds":10800,"source_sha":file_hash(ROOT/"src/jsteer/gemma_pressure.py")}
    def save():path.write_text(json.dumps(state,indent=2))
    token = get_token()
    if not token:raise RuntimeError("Missing local HF credential")
    secrets=[modal.Secret.from_dict({"HF_TOKEN":token})]
    cpu=modal.Function.from_name("jsteer","prepare_gemma_pressure").with_options(secrets=secrets,max_containers=1,retries=0,scaledown_window=2).spawn()
    state.update(stage="preparing",cpu_call_id=cpu.object_id);save();print(json.dumps(state),flush=True)
    state["released"]=cpu.get()
    call=modal.Function.from_name("jsteer","gpu_task").with_options(gpu="A100-80GB",secrets=secrets,max_containers=1,buffer_containers=0,retries=0,scaledown_window=2,timeout=10800).spawn("gemma_pressure",asdict(PressureSpec(expected_source_sha=state["source_sha"])))
    state.update(stage="running",gpu_call_id=call.object_id,gpu_started=time.time());save();print(json.dumps(state),flush=True)
    try:
        state["result"]=call.get();state["stage"]="complete"
    except BaseException as exc:
        state.update(stage="failed",error=f"{type(exc).__name__}: {exc}");raise
    finally:
        state["finished"]=time.time();save();print(json.dumps(state),flush=True)


if __name__ == "__main__":main()

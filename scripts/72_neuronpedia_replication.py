"""Launch/fetch the published-lens Gemma/Llama campaign on the existing app."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

import modal
from huggingface_hub import get_token

from jsteer.cross_model import sha
from jsteer.neuronpedia_replication import PROTOCOL, input_hashes
from jsteer.write_controls import file_hash

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/neuronpedia_replication"
APP = "jlens-filtering-replication"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["plan", "start", "recover", "status", "fetch"])
    ap.add_argument("--json-only", action="store_true")
    args = ap.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    expected = {"protocol_sha": file_hash(PROTOCOL),
                "source_sha": file_hash(ROOT / "src/jsteer/neuronpedia_replication.py"),
                "inputs_sha": {name: sha(input_hashes(name)) for name in protocol["models"]}}
    if args.action == "plan":
        print(json.dumps({"expected": expected, "models": protocol["models"],
                          "diagonal_fit_prompts": len(protocol["diagonal_fit"]["fit_texts"]),
                          "max_concurrent_gpus": 2}, indent=2))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    volume = modal.Volume.from_name("jsteer-results")
    remote_ledger = "/neuronpedia_replication/campaign.json"
    if args.action in ("start", "recover"):
        with (OUT / "launcher.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = OUT / ("launch.json" if args.action == "start" else "recovery.json")
            try:
                ledger = json.loads(b"".join(volume.read_file(remote_ledger)))
            except (FileNotFoundError, modal.exception.NotFoundError):
                ledger = None
            if args.action == "start" and (path.exists() or ledger is not None):
                raise SystemExit("Existing launch/campaign ledger; inspect before duplicate compute")
            if args.action == "recover":
                if not ledger or ledger["expected"] != expected:
                    raise SystemExit("Recovery requires an existing campaign with identical scientific inputs")
                if any(not row.get("gpu_call_id") and row["stage"] != "failed" for row in ledger["models"].values()):
                    raise SystemExit("Incomplete preparation/submission; inspect before recovery")
            token = get_token()
            if not token:
                raise SystemExit("No local HF credential for the already-accessible gated checkpoints")
            path.write_text(json.dumps({"stage": "submitting", "expected": expected}, indent=2))
            call = modal.Function.from_name(APP, "campaign").with_options(
                secrets=[modal.Secret.from_dict({"HF_TOKEN": token})]).spawn(
                    expected, campaign_name="neuronpedia_replication")
            state = {"stage": "submitted", "app": APP, "campaign_call_id": call.object_id,
                     "expected": expected, "campaign": "neuronpedia_replication"}
            path.write_text(json.dumps(state, indent=2))
            print(json.dumps(state, indent=2))
        return
    if args.action == "status":
        data = b"".join(volume.read_file(remote_ledger))
        (OUT / "campaign.json").write_bytes(data)
        ledger = json.loads(data)
        print(json.dumps({"stage": ledger["stage"], "models": ledger["models"]}, indent=2))
        return
    for entry in volume.iterdir("/neuronpedia_replication", recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE or (args.json_only and not entry.path.endswith(".json")):
            continue
        remote = entry.path.lstrip("/")
        dest = OUT / Path(remote).relative_to("neuronpedia_replication")
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix(dest.suffix+".download")
        with temp.open("wb") as stream:
            for block in volume.read_file(remote):
                stream.write(block)
        temp.replace(dest)
    print(OUT)


if __name__ == "__main__":
    main()

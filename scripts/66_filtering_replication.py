"""Start, inspect or fetch one durable, bounded Modal replication campaign."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

import modal

from jsteer.cross_model import input_hashes, sha
from jsteer.write_controls import file_hash

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/filtering_replication"
APP = "jlens-filtering-replication"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["plan", "start", "status", "fetch"])
    ap.add_argument("--json-only", action="store_true", help="Fetch progress and raw JSON without fitted tensor banks")
    args = ap.parse_args()
    protocol_path = ROOT / "data/protocols/filtering_replication.json"
    protocol = json.loads(protocol_path.read_text())
    expected = {"protocol_sha": file_hash(protocol_path),
                "source_sha": file_hash(ROOT / "src/jsteer/cross_model.py"),
                "inputs_sha": {name: sha(input_hashes(name)) for name in protocol["models"]}}
    if args.action == "plan":
        print(json.dumps({"protocol": protocol, "expected": expected}, indent=2))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    if args.action == "start":
        with (OUT / "launcher.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = OUT / "launch.json"
            if path.exists():
                raise SystemExit("Launch ledger exists; inspect it before any relaunch")
            volume = modal.Volume.from_name("jsteer-results")
            try:
                next(volume.read_file("/filtering_replication/campaign.json"))
            except (FileNotFoundError, modal.exception.NotFoundError):
                pass
            else:
                raise SystemExit("Remote campaign ledger exists; inspect before relaunch")
            # Reserve the local ledger before making a billable submission.
            path.write_text(json.dumps({"stage": "submitting", "expected": expected}, indent=2))
            call = modal.Function.from_name(APP, "campaign").spawn(expected)
            state = {"stage": "submitted", "app": APP, "campaign_call_id": call.object_id,
                     "expected": expected}
            path.write_text(json.dumps(state, indent=2))
            print(json.dumps(state, indent=2))
        return
    volume = modal.Volume.from_name("jsteer-results")
    if args.action == "status":
        for block in volume.read_file("/filtering_replication/campaign.json"):
            print(block.decode(), end="")
        return
    for entry in volume.iterdir("/filtering_replication", recursive=True):
        # Directory entries have no file content to retrieve.
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        if args.json_only and not entry.path.endswith(".json"):
            continue
        remote = entry.path.lstrip("/")
        relative = Path(remote).relative_to("filtering_replication")
        dest = OUT / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix(dest.suffix + ".download")
        with temp.open("wb") as stream:
            for block in volume.read_file(remote):
                stream.write(block)
        temp.replace(dest)
    print(OUT)


if __name__ == "__main__":
    main()

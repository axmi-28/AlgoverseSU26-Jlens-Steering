"""Wait for the submitted diagnostic run, fetch small artifacts, build report.

This creates no Modal app or GPU invocation. It leaves per-prompt tensors on
the results volume to avoid downloading gigabytes of fitting checkpoints.
"""
import json
from pathlib import Path
import subprocess
import sys

import modal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/gemma_pressure"


def main():
    ledger = json.loads((OUT / "launch.json").read_text())
    result = modal.FunctionCall.from_id(ledger["gpu_call_id"]).get()
    remote = result["path"].removeprefix("/results")
    local = OUT / Path(remote).name
    local.mkdir(parents=True, exist_ok=True)
    names = ["manifest.json", "status.json", "qwen_positive_control.json",
             "question_audit.json", "generation_parity.json", "fit_stability.json",
             "released_metadata.json", "settings.json", "dev_results.json",
             "selection.json", "validation_results.json"]
    for name in names:
        subprocess.run([str(ROOT/".venv/bin/modal"), "volume", "get", "--force",
                        "jsteer-results", f"{remote}/{name}", str(local/name)], check=True)
    subprocess.run([sys.executable, str(ROOT/"scripts/62_gemma_pressure_report.py")], check=True)
    print(json.dumps({"status": "report_ready", "report": str(OUT/"report.md"), "result": result}), flush=True)


if __name__ == "__main__":
    main()

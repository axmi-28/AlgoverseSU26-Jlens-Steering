"""C22 preflight, bounded launch, and report. Default action spends no GPU time."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from jsteer.data import probe_swap_items
from jsteer.write_controls import (
    ARMS,
    WriteControlSpec,
    file_hash,
    run_write_controls,
    source_hashes,
)

ROOT = Path(__file__).resolve().parents[1]
DIGEST = "group_mean_digests_qwen3-8b_dom2x32_968c0ba0_pb_t89d64f82_comp_s4.pt"
REFERENCE = "two_hop_qwen3-8b_all_g595bfbbf_twohop89d9c5_shard0of1.json"


def report(path):
    payload = json.loads(Path(path).read_text())
    records = payload["records"]
    base = {r["name"]: r for r in records if r["arm"] == "baseline"}
    usable = {
        name for name, r in base.items() if r["hit_answer"] and not r["hit_swap_answer"]
    }
    strengths = payload["manifest"]["spec"]["strengths"]
    lines = [
        "# C22 pilot: write and swap controls",
        "",
        f"Status: {payload['status']}. Completed items: {len(base)}; current clean-correct, counterfactual-incorrect: {len(usable)}.",
        "This is a development pilot, not a confirmatory test. All arm means below use that same current-clean subset.",
        "",
        "| Arm | Strength | Answer hits / n | Emission / n | Mean edit row norm |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for strength in strengths:
            rs = [
                r
                for r in records
                if r["name"] in usable and r["arm"] == arm and r["strength"] == strength
            ]
            if len(rs) != len(usable):
                raise ValueError("Incomplete paired panel")
            norms = [n for r in rs for ns in r["edit_row_norms"].values() for n in ns]
            mean_norm = sum(norms) / len(norms) if norms else 0
            lines.append(
                f"| {arm} | {strength} | {sum(r['hit_swap_answer'] for r in rs)}/{len(rs)} | {sum(r['emitted_substring'] for r in rs)}/{len(rs)} | {mean_norm:.4f} |"
            )
    lines.extend(
        [
            "",
            "Full manifest, individual generations, tokenization counts, and per-layer/per-position edit norms are in the source JSON.",
        ]
    )
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=("preflight", "local", "remote", "report"),
        default="preflight",
    )
    ap.add_argument("--directions", default=str(ROOT / "results/rq1" / DIGEST))
    ap.add_argument("--reference", default=str(ROOT / "results/causal" / REFERENCE))
    ap.add_argument("--corpus", choices=("gsm8k", "wikitext_a"), default="gsm8k")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--max-seconds", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=2201)
    ap.add_argument("--gpu", default="A100-80GB")
    ap.add_argument("--memory-mib", type=int, default=65536)
    ap.add_argument("--result")
    args = ap.parse_args()
    if args.memory_mib < 1:
        ap.error("--memory-mib must be positive")
    if args.mode == "report":
        if not args.result:
            ap.error("--report requires --result")
        report(args.result)
        return
    spec = WriteControlSpec(
        directions_path=args.directions,
        reference_path=args.reference,
        corpus=args.corpus,
        limit=args.limit,
        strengths=tuple(args.strengths),
        max_seconds=args.max_seconds,
        seed=args.seed,
        out_dir=str(ROOT / "results/c22"),
        expected_sources=source_hashes(),
    )
    if args.mode == "preflight":
        blob = torch.load(spec.directions_path, map_location="cpu", weights_only=False)
        words = set(blob["targets"])
        baseline = json.loads(Path(spec.reference_path).read_text())
        names = {
            r["name"] for r in baseline if r["arm"] == "baseline" and r["baseline_ok"]
        }
        eligible = [
            i
            for i in probe_swap_items()
            if i.name in names and i.intermediate in words and i.swap_to in words
        ]
        counts = {}
        for part in ("full", "off"):
            for layer in spec.layers:
                entry = blob["digests"][f"{spec.corpus}_{part}|{layer}"]
                if entry["pullbacks"].shape != (len(words), 4096):
                    raise ValueError("Wrong Qwen3-8B artifact dimensions")
                counts[f"{part}|{layer}"] = entry["n"]
        if len(set(counts.values())) != 1:
            raise ValueError("Unequal fit counts")
        print(
            json.dumps(
                {
                    "spec": asdict(spec),
                    "eligible_items": len(eligible),
                    "pilot_items": min(spec.limit, len(eligible)),
                    "arms": ARMS,
                    "generations": min(spec.limit, len(eligible))
                    * (1 + len(ARMS) * len(spec.strengths)),
                    "fit_counts": counts,
                    "artifact_sha256": file_hash(spec.directions_path),
                },
                indent=2,
            )
        )
    elif args.mode == "local":
        print(json.dumps(run_write_controls(spec), indent=2))
    else:
        import modal

        spec.directions_path = f"/results/rq1/{Path(spec.directions_path).name}"
        spec.reference_path = f"/results/causal/{Path(spec.reference_path).name}"
        spec.out_dir = "/results/c22"
        # One invocation, no automatic paid retries, no warm buffer. A hard
        # timeout bounds model-loading or individual-forward stalls as well.
        fn = modal.Function.from_name("jsteer", "gpu_task").with_options(
            gpu=args.gpu,
            max_containers=1,
            buffer_containers=0,
            scaledown_window=2,
            retries=0,
            timeout=args.max_seconds + 120,
            memory=args.memory_mib,
        )
        call = fn.spawn("write_controls", asdict(spec))
        print(f"call_id={call.object_id}", flush=True)
        print(json.dumps(call.get(), indent=2))


if __name__ == "__main__":
    main()

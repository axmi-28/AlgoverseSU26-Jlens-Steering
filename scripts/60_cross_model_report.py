"""CPU reports for completed cross-model runs; paired source-cluster intervals."""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def emission_metrics(rows):
    """Keep raw outputs intact; distinguish name presence from failed answers.

    Historical `leakage` is a substring flag over the entire continuation.
    Correct answers can contain the name (Russian/Russia, Australia as a
    continent), or mention it after a correct initial answer. Neither is an
    incorrect-answer/name-emission event. Express queries ask for the name
    and are reported only as naming success.
    """
    return [dict(r, name_emission=bool(r["leakage"]),
                 incorrect_name_emission=bool(r["leakage"] and not r["success"]),
                 correct_name_emission=bool(r["leakage"] and r["success"])) for r in rows]


def paired(rows, arm_a, arm_b, metric, *, concept=False):
    by = defaultdict(dict)
    for r in rows:
        if r["arm"] in (arm_a, arm_b):
            key = (r["key"], r["relation"]) if concept else r["name"]
            if r["arm"] in by[key]:
                raise ValueError(f"Duplicate paired observation: {key}")
            by[key][r["arm"]] = r
    if not by or any(set(v) != {arm_a, arm_b} for v in by.values()):
        raise ValueError("Empty or incomplete paired result")
    units = defaultdict(list)
    for key, v in by.items():
        uid = key[0] if concept else key
        units[(v[arm_a]["source"], uid)].append((int(v[arm_a][metric]), int(v[arm_b][metric])))
    clusters = defaultdict(list)
    for (source, _), values in units.items():
        clusters[source].append(tuple(statistics.fmean(v[j] for v in values) for j in (0, 1)))
    pairs = [p for group in clusters.values() for p in group]
    base, filtered = [statistics.fmean(p[j] for p in pairs) for j in (0, 1)]
    names = sorted(clusters)
    rng = random.Random(42102)
    draws = []
    for _ in range(2000):
        sampled = [p for source in rng.choices(names, k=len(names)) for p in clusters[source]]
        draws.append(statistics.fmean(b-a for a, b in sampled)*100)
    draws.sort()
    return {"n": len(pairs), "clusters": len(clusters), "baseline": base*100,
            "filtered": filtered*100, "delta": (filtered-base)*100,
            "ci": [draws[50], draws[1949]],
            "wins": sum(b>a for a,b in pairs), "losses": sum(b<a for a,b in pairs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, nargs="?", default=Path("results/cross_model"))
    args = ap.parse_args()
    lines = ["# Cross-model filtering replication", "", "Paired 95% source-entity cluster-bootstrap intervals; conditional on one lens fit per model.", "",
             "`success`: accepted target answer at the beginning of the continuation. "
             "`name_emission`: target-name substring anywhere in the continuation (the raw field historically called `leakage`); it can overlap correct answers. "
             "`incorrect_name_emission`: name emission AND an incorrect target answer, matching the paper's success-first outcome breakdown. "
             "This still measures co-occurrence, not necessarily a name replacing the answer: a name may appear later, or as part of another word. "
             "Express queries ask for the name, so only naming success is reported for them. "
             "Original raw outputs and accuracy scores are unchanged.", "",
             "| Model | Evaluation | Metric | N | Full (%) | Filtered (%) | Change (pp; 95% CI) |", "|---|---|---|---:|---:|---:|---|"]
    found = 0
    for status in sorted(args.root.glob("*/*/status.json")):
        if json.loads(status.read_text())["stage"] != "complete":
            continue
        root = status.parent
        model = root.parent.name
        found += 1
        latent = emission_metrics(json.loads((root / "latent_results.json").read_text()))
        concept = emission_metrics(json.loads((root / "concept_results.json").read_text()))
        for strength in (0.5, 1.0):
            for subset in ("all", "upstream", "expanded"):
                rows = [r for r in latent if r["strength"] == strength and (subset == "all" or r["set"] == subset)]
                if not rows:
                    continue
                for metric in ("success", "name_emission", "incorrect_name_emission"):
                    d = paired(rows, "full", "off_m", metric)
                    lines.append(f"| {model} | Latent {subset}, strength {strength} | {metric} | {d['n']} | {d['baseline']:.1f} | {d['filtered']:.1f} | {d['delta']:+.1f} [{d['ci'][0]:+.1f}, {d['ci'][1]:+.1f}] |")
        for kind in ("use", "express"):
            for metric in (("success", "name_emission", "incorrect_name_emission") if kind == "use" else ("success",)):
                d = paired([r for r in concept if r["kind"] == kind], "lam1_cn", "lam0_cn", metric, concept=True)
                lines.append(f"| {model} | Concept {kind}, unit columns | {metric} | {d['n']} | {d['baseline']:.1f} | {d['filtered']:.1f} | {d['delta']:+.1f} [{d['ci'][0]:+.1f}, {d['ci'][1]:+.1f}] |")
    if not found:
        raise SystemExit("No completed runs: no results reported")
    path = args.root / "report.md"
    path.write_text("\n".join(lines) + "\n")
    print(path)


if __name__ == "__main__":
    main()

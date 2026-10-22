"""Exploratory entity-type and common-query analyses of saved replications."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, nargs="?", default=Path("results/filtering_replication"))
    ap.add_argument("--historical", type=Path, default=Path("results/cross_model"))
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("cross_report", Path(__file__).with_name("60_cross_model_report.py"))
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    models = {}
    for directory, historical in ((args.root, False), (args.historical, True)):
        for status in directory.glob("*/*/status.json"):
            if json.loads(status.read_text())["stage"] != "complete":
                continue
            model = status.parent.parent.name
            if model in models:
                raise ValueError(f"Multiple completed fits for {model}; explicit fit selection required")
            rows = report.emission_metrics(json.loads((status.parent / "concept_results.json").read_text()))
            models[model] = {"historical": historical, "rows": [r for r in rows if r["kind"] == "use"]}
    output = {"analysis": "Exploratory, post-hoc, saved outputs only; no model or setting selection", "models": {}}
    lines = ["# Concept replication population audit", "",
             "Exploratory analyses of saved outputs, prompted by differences in clean eligibility. "
             "All comparisons use strength 1, unit columns and the fixed whole-prompt seven-layer intervention. "
             "Intervals are paired 95% source-entity cluster bootstrap intervals, conditional on one fit, "
             "pointwise and unadjusted. Small per-type samples have limited precision. "
             "Historical Gemma/Llama runs are labeled with an asterisk.", "",
             "## By entity type", "",
             "| Model | Type | Units | Source clusters | Full (%) | Filtered (%) | Change, pp (95% CI) |",
             "|---|---|---:|---:|---:|---:|---|"]
    for name, info in sorted(models.items()):
        stats = {}
        for ty in sorted({r["type"] for r in info["rows"]}):
            d = report.paired([r for r in info["rows"] if r["type"] == ty], "lam1_cn", "lam0_cn", "success", concept=True)
            stats[ty] = d
            label = name + ("*" if info["historical"] else "")
            lines.append(f"| {label} | {ty} | {d['n']} | {d['clusters']} | {d['baseline']:.1f} | {d['filtered']:.1f} | "
                         f"{d['delta']:+.1f} [{d['ci'][0]:+.1f}, {d['ci'][1]:+.1f}] |")
        output["models"][name] = {"historical": info["historical"], "by_type": stats}
    lines += ["", "## Exactly shared concept-use queries with Qwen3-8B", "",
              "Each row recomputes both models on the same unit/relation keys. Queries are averaged "
              "within units, then units equally. This controls the observed query population; it does "
              "not isolate architecture, training, tokenization or layer placement.", "",
              "| Other model | Shared units | Shared queries | Qwen full → filtered | Other full → filtered | Other change, pp (95% CI) |",
              "|---|---:|---:|---|---|---|"]
    if "qwen3-8b" in models:
        qwen_rows = models["qwen3-8b"]["rows"]
        qwen_keys = {(r["key"], r["relation"]) for r in qwen_rows}
        for name, info in sorted(models.items()):
            if name == "qwen3-8b":
                continue
            keys = {(r["key"], r["relation"]) for r in info["rows"]} & qwen_keys
            if not keys:
                output["models"][name]["matched_qwen"] = {"shared_queries": 0}
                continue
            selected = [r for r in info["rows"] if (r["key"], r["relation"]) in keys]
            reference = [r for r in qwen_rows if (r["key"], r["relation"]) in keys]
            d = report.paired(selected, "lam1_cn", "lam0_cn", "success", concept=True)
            q = report.paired(reference, "lam1_cn", "lam0_cn", "success", concept=True)
            output["models"][name]["matched_qwen"] = {"shared_queries": len(keys), "qwen": q, "other": d}
            label = name + ("*" if info["historical"] else "")
            lines.append(f"| {label} | {d['n']} | {len(keys)} | {q['baseline']:.1f}% → {q['filtered']:.1f}% | "
                         f"{d['baseline']:.1f}% → {d['filtered']:.1f}% | {d['delta']:+.1f} [{d['ci'][0]:+.1f}, {d['ci'][1]:+.1f}] |")
    (args.root / "population_audit.json").write_text(json.dumps(output, indent=2)+"\n")
    (args.root / "population_audit.md").write_text("\n".join(lines)+"\n")
    print(args.root / "population_audit.md")


if __name__ == "__main__":
    main()

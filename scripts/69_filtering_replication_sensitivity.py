"""Post-hoc alias and non-currency sensitivities; original labels stay intact."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from jsteer.concept_use import hits
from jsteer.write_controls import file_hash

ROOT = Path(__file__).resolve().parents[1]


def rescore(rows, candidates, aliases):
    rescored, audit = [], []
    for row in rows:
        item = candidates[row["name"]]
        currency = "currency" in item["prompt"].casefold()
        success = bool(row["success"])
        if currency:
            success = success or hits(row["text"], aliases.get(item["swap_to"], []))
        result = dict(row, success=success)
        rescored.append(result)
        if success != row["success"]:
            audit.append(dict(row, expanded_success=success, prompt=item["prompt"],
                              added_aliases=aliases.get(item["swap_to"], [])))
    return rescored, audit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, nargs="?", default=Path("results/filtering_replication"))
    args = ap.parse_args()
    plan_path = ROOT / "data/protocols/filtering_currency_sensitivity.json"
    plan = json.loads(plan_path.read_text())
    candidates = {i["name"]: i for i in json.loads((ROOT / "data/protocols/cross_model_filtering.json").read_text())["latent_candidates"]}
    spec = importlib.util.spec_from_file_location("cross_report", Path(__file__).with_name("60_cross_model_report.py"))
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    output = {"plan_sha": file_hash(plan_path), "plan": plan, "models": {}}
    lines = ["# Currency scoring sensitivities", "",
             "Exploratory CPU analysis prompted by a grading edge case seen in the first completed models. "
             "Original accuracy labels, eligibility and primary reports remain unchanged. "
             "Expanded aliases add full currency names (e.g. Indian rupee); excluding every currency question "
             "checks the effect without relying on this alias expansion. Neither is exhaustive semantic grading. "
             "All intervals are paired 95% source-entity cluster intervals, pointwise, conditional on one fit.", "",
             "| Model | Strength | Scoring / population | N | Full (%) | Filtered (%) | Change, pp (95% CI) |",
             "|---|---:|---|---:|---:|---:|---|"]
    for status in sorted(args.root.glob("*/*/status.json")):
        if json.loads(status.read_text())["stage"] != "complete":
            continue
        root = status.parent
        model = root.parent.name
        rows = json.loads((root / "latent_results.json").read_text())
        expanded, audit = rescore(rows, candidates, plan["new_aliases"])
        stats = {}
        for strength in (0.5, 1.0):
            for label, selected in (
                ("frozen / all", rows),
                ("expanded currency aliases / all", expanded),
                ("frozen / exclude all currency", [r for r in rows if "currency" not in candidates[r["name"]]["prompt"].casefold()]),
            ):
                selected = [r for r in selected if r["strength"] == strength]
                d = report.paired(selected, "full", "off_m", "success")
                e = report.paired(report.emission_metrics(selected), "full", "off_m", "incorrect_name_emission")
                stats[f"{strength:g} / {label}"] = {"success": d, "incorrect_name_emission": e}
                lines.append(f"| {model} | {strength:g} | {label} | {d['n']} | {d['baseline']:.1f} | {d['filtered']:.1f} | "
                             f"{d['delta']:+.1f} [{d['ci'][0]:+.1f}, {d['ci'][1]:+.1f}] |")
        output["models"][model] = {"stats": stats, "changed_rows": audit}
    (args.root / "currency_sensitivity.json").write_text(json.dumps(output, indent=2)+"\n")
    (args.root / "currency_sensitivity.md").write_text("\n".join(lines)+"\n")
    print(args.root / "currency_sensitivity.md")


if __name__ == "__main__":
    main()

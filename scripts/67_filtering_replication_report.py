"""Compact campaign findings and fixed lambda controls; no outcome selection."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, nargs="?", default=Path("results/filtering_replication"))
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("cross_report", Path(__file__).with_name("60_cross_model_report.py"))
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    lines = ["# Broader filtering replication", "",
             "Frozen extension: one 32-text GSM8K fit per model, raw continuation prompts, "
             "seven layers mapped by relative depth, no model-specific tuning. "
             "Model-specific clean eligibility is frozen before interventions; populations differ across models. "
             "Intervals are paired 95% source-entity cluster bootstrap intervals, conditional on one fit, "
             "pointwise and unadjusted for multiple comparisons.", "",
             "A positive accuracy change supports filtering for that setting. Reduced incorrect name emission "
             "alone does not establish improved relational use. `incorrect name` means a name substring "
             "anywhere in an incorrect continuation, not necessarily a name replacing the answer.", "",
             "## Primary contrasts", "",
             "| Model | Task / strength | N | Full accuracy | Filtered accuracy | Accuracy change, pp (95% CI) | Incorrect name, full → filtered |",
             "|---|---|---:|---:|---:|---|---|"]
    output = {"models": {}}
    for status in sorted(args.root.glob("*/*/status.json")):
        if json.loads(status.read_text())["stage"] != "complete":
            continue
        root = status.parent
        name = root.parent.name
        latent = report.emission_metrics(json.loads((root / "latent_results.json").read_text()))
        concept = report.emission_metrics(json.loads((root / "concept_results.json").read_text()))
        contrasts = []
        for strength in (0.5, 1.0):
            rows = [r for r in latent if r["strength"] == strength]
            contrasts.append((f"latent / {strength:g}", rows, "full", "off_m", False))
        contrasts.append(("concept use / 1", [r for r in concept if r["kind"] == "use"], "lam1_cn", "lam0_cn", True))
        stats = {}
        for label, rows, a, b, is_concept in contrasts:
            d = report.paired(rows, a, b, "success", concept=is_concept)
            e = report.paired(rows, a, b, "incorrect_name_emission", concept=is_concept)
            stats[label] = {"success": d, "incorrect_name_emission": e}
            lines.append(f"| {name} | {label} | {d['n']} | {d['baseline']:.1f}% | {d['filtered']:.1f}% | "
                         f"{d['delta']:+.1f} [{d['ci'][0]:+.1f}, {d['ci'][1]:+.1f}] | {e['baseline']:.1f}% → {e['filtered']:.1f}% |")
        output["models"][name] = {"path": str(root), "contrasts": stats}
    lines += ["", "## Fixed continuum controls", "",
              "Unit-normalized direction columns; all edit norms matched at every layer and position. "
              "Use accuracy averages queries within units, then units equally. The random arm uses the same edit norm.", "",
              "| Model | Arm | Concept-use accuracy | Naming accuracy | Incorrect name on use |", "|---|---|---:|---:|---:|"]
    for name, info in output["models"].items():
        rows = report.emission_metrics(json.loads((Path(info["path"]) / "concept_results.json").read_text()))
        for arm in ("lam0_cn", "lam0.5_cn", "lam1_cn", "lam2_cn", "rand_m"):
            # paired() supplies exactly the same within-unit weighting as the primary table.
            other = "lam0_cn" if arm == "lam1_cn" else "lam1_cn"
            use = report.paired([r for r in rows if r["kind"] == "use"], arm, other, "success", concept=True)
            express = report.paired([r for r in rows if r["kind"] == "express"], arm, other, "success", concept=True)
            emission = report.paired([r for r in rows if r["kind"] == "use"], arm, other, "incorrect_name_emission", concept=True)
            lines.append(f"| {name} | {arm} | {use['baseline']:.1f}% | {express['baseline']:.1f}% | {emission['baseline']:.1f}% |")
    ledger_path = args.root / "campaign.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        lines += ["", "## Run accounting", ""]
        for name in ledger["protocol"]["models"]:
            row = ledger["models"].get(name, {})
            if row.get("stage") == "complete":
                lines.append(f"- {name}: complete; GPU function time {row['result']['seconds']/60:.1f} min.")
            else:
                lines.append(f"- {name}: {row.get('stage', 'not launched')}; {row.get('error', '')}")
    lines += ["", "Scope: these runs test latent-intermediate and concept express/use results. "
              "They do not replicate the full paper, a second fitting corpus, fitting variance, "
              "or model-specific layer/strength optimization. The Qwen3-8B control uses the "
              "same clean-selection pipeline as the new models; absolute rates need not match the paper's broader panel."]
    (args.root / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    (args.root / "summary.md").write_text("\n".join(lines) + "\n")
    print(args.root / "summary.md")


if __name__ == "__main__":
    main()

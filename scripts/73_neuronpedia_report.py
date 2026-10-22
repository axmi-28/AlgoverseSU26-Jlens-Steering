"""Published-lens + estimated diagonal results, with fixed 32/64 sensitivity."""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from collections import defaultdict
from pathlib import Path


def unit_mean(rows, metric):
    units = defaultdict(list)
    for row in rows:
        units[row["key"]].append(float(row[metric]))
    return statistics.fmean(statistics.fmean(values) for values in units.values())*100


def make_figure(summary, out):
    import matplotlib.pyplot as plt

    order = [n for n in ("gemma-2-2b-it-np", "gemma-2-9b-it-np", "gemma-3-4b-it-np",
                        "gemma-3-12b-pt-np", "llama-3.1-8b-np", "llama-3.1-8b-instruct-np") if n in summary]
    labels = [n.replace("-np", "").replace("-pt", " (base)").replace("-instruct", " instruct") for n in order]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.6), sharey=True)
    settings = [("latent / 0.5", "Latent swaps · strength 0.5"), ("latent / 1", "Latent swaps · strength 1"),
                ("concept use / 1", "Concept use · strength 1")]
    all_ci = [summary[n]["contrasts"][key]["success"]["ci"] for n in order for key, _ in settings]
    low, high = min(-5, min(v[0] for v in all_ci))-4, max(5, max(v[1] for v in all_ci))+4
    for ax, (key, title) in zip(axes, settings, strict=True):
        ax.axvline(0, lw=1, color="#74818c")
        ax.set_axisbelow(True)
        ax.grid(axis="x", lw=.7, color="#e7eaee")
        for y, name in enumerate(order):
            d = summary[name]["contrasts"][key]["success"]
            color = "#b66b2a" if name.startswith("gemma") else "#216987"
            ax.errorbar(d["delta"], y, xerr=[[max(0, d["delta"]-d["ci"][0])], [max(0, d["ci"][1]-d["delta"])]],
                        fmt="o", capsize=3, color=color, ms=6, lw=1.5)
            ax.annotate(f"{d['delta']:+.1f}", (d["delta"], y), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8, color=color)
        ax.set(xlim=(low, high), ylim=(len(order)-.5, -.65),
               xlabel="Estimated filtered − published full\naccuracy change (percentage points)")
        ax.set_title(title, fontsize=10.5, loc="left", pad=18, weight="bold")
        ax.xaxis.label.set_size(8.5)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=9)
    axes[0].set_yticks(range(len(order)), labels)
    fig.suptitle("Filtering published Neuronpedia J-lenses", x=.025, y=.98, ha="left", fontsize=18, weight="bold")
    fig.text(.025, .9, "Published full lens fixed; subtract a diagonal estimated on 64 WikiText chunks. Positive values favor the estimated filter.", fontsize=9.5)
    fig.text(.025, .045, "Bars: paired 95% source-entity cluster intervals; pointwise, conditional on one diagonal estimate.\n"
             "This is an estimated subtraction, not an exact decomposition of Neuronpedia’s original fit. Clean eligibility differs by model.", fontsize=8.5)
    fig.subplots_adjust(left=.235, right=.985, top=.80, bottom=.19, wspace=.20)
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(out/f"accuracy_contrasts.{suffix}", dpi=200, facecolor="white")
    (out/"figure_data.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(out/"accuracy_contrasts.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, nargs="?", default=Path("results/neuronpedia_replication"))
    ap.add_argument("--figure", action="store_true")
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("cross_report", Path(__file__).with_name("60_cross_model_report.py"))
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    lines = ["# Published Neuronpedia lenses plus estimated diagonal filtering", "",
             "The full lens is downloaded from a pinned Neuronpedia release and kept fixed. "
             "The filtered direction subtracts a same-position contribution estimated on 64 frozen WikiText chunks. "
             "It is **not an exact position-pair decomposition of the released lens**. The first 32 chunks supply "
             "a fixed sensitivity; no edited outcomes select models, layers, strengths or fit size. "
             "Raw continuation prompts, clean-only eligibility and up to 40 source-balanced concept units per type "
             "match the previous evaluation pipeline. Intervals are paired 95% source-entity cluster bootstrap "
             "intervals (2000 draws), conditional on the released lens and one diagonal estimate, pointwise and unadjusted.", "",
             "`Incorrect name` means an incorrect answer plus target-name substring presence in the continuation. "
             "Correct answers containing the target name are retained as successes.", "",
             "## Primary 64-chunk estimated-filter contrasts", "",
             "| Model | Task / strength | N | Published full (%) | Estimated filtered (%) | Accuracy change, pp (95% CI) | Incorrect name, full → estimated |",
             "|---|---|---:|---:|---:|---|---|"]
    models, sensitivity_lines, continuum_lines, source_lines = {}, [], [], []
    for status in sorted(args.root.glob("*/*/status.json")):
        if json.loads(status.read_text())["stage"] != "complete":
            continue
        root = status.parent
        name = root.parent.name
        if name in models:
            raise ValueError(f"Multiple completed estimates for {name}; explicit selection required")
        latent = report.emission_metrics(json.loads((root/"latent_results.json").read_text()))
        concept = report.emission_metrics(json.loads((root/"concept_results.json").read_text()))
        source = json.loads((root/"published_metadata.json").read_text())
        diagnostics = json.loads((root/"fit_diagnostics.json").read_text())
        contrasts, sensitivity, by_type = {}, {}, {}
        settings = [(f"latent / {s:g}", [r for r in latent if r["strength"] == s], "full", "off_m", "off32_m", False) for s in (.5, 1.)]
        settings.append(("concept use / 1", [r for r in concept if r["kind"] == "use"], "full_cn", "lam0_cn", "off32_cn", True))
        for label, rows, a, b, sensitivity_arm, is_concept in settings:
            d = report.paired(rows, a, b, "success", concept=is_concept)
            e = report.paired(rows, a, b, "incorrect_name_emission", concept=is_concept)
            contrasts[label] = {"success": d, "incorrect_name_emission": e}
            lines.append(f"| {name} | {label} | {d['n']} | {d['baseline']:.1f} | {d['filtered']:.1f} | "
                         f"{d['delta']:+.1f} [{d['ci'][0]:+.1f}, {d['ci'][1]:+.1f}] | {e['baseline']:.1f}% → {e['filtered']:.1f}% |")
            s = report.paired(rows, a, sensitivity_arm, "success", concept=is_concept)
            sensitivity[label] = s
            sensitivity_lines.append(f"| {name} | {label} | {s['n']} | {s['filtered']:.1f} | {d['filtered']:.1f} | "
                                     f"{s['delta']:+.1f} [{s['ci'][0]:+.1f}, {s['ci'][1]:+.1f}] |")
        for ty in sorted({r["type"] for r in concept}):
            by_type[ty] = report.paired([r for r in concept if r["kind"] == "use" and r["type"] == ty], "full_cn", "lam0_cn", "success", concept=True)
        for arm in ("lam0_cn", "lam0.5_cn", "full_cn", "lam2_cn", "off32_cn", "rand_m"):
            use = [r for r in concept if r["arm"] == arm and r["kind"] == "use"]
            express = [r for r in concept if r["arm"] == arm and r["kind"] == "express"]
            continuum_lines.append(f"| {name} | {arm} | {unit_mean(use,'success'):.1f} | {unit_mean(express,'success'):.1f} | {unit_mean(use,'incorrect_name_emission'):.1f} |")
        mean_cos = statistics.fmean(v["split_half_mean_cosine"] for v in diagnostics["layers"].values())
        off_cos = statistics.fmean(v["filtered32_vs_filtered64_mean_cosine"] for v in diagnostics["layers"].values())
        source_lines.append(f"| {name} | {source['n_prompts']} | 64 | {mean_cos:.3f} | {off_cos:.3f} |")
        models[name] = {"path": str(root), "contrasts": contrasts, "sensitivity32": sensitivity,
                        "published_metadata": source, "fit_diagnostics": diagnostics, "by_type": by_type,
                        "max_norm_error": max(r["norm_error"] for r in latent+concept)}
    lines += ["", "## Fixed first-32-chunk sensitivity", "",
              "The published baseline stays identical. The last column is the 32-chunk filter's change against that baseline.", "",
              "| Model | Task / strength | N | Filtered 32 (%) | Filtered 64 (%) | 32-chunk change, pp (95% CI) |", "|---|---|---:|---:|---:|---|"]+sensitivity_lines
    lines += ["", "## Concept controls", "",
              "`full_cn` is the released full lens directly (diagonal weight 1). `lam0_cn`, `lam0.5_cn`, "
              "and `lam2_cn` use published full + (weight − 1) × estimated diagonal. Direction columns are "
              "unit-normalized and every edit-row norm is matched to the published full native swap.", "",
              "| Model | Arm | Use accuracy (%) | Naming accuracy (%) | Incorrect name on use (%) |", "|---|---|---:|---:|---:|"]+continuum_lines
    lines += ["", "## Source and estimation stability", "",
              "Split-half cosine compares diagonal estimates on the first and last 32 chunks, averaged over words and layers. "
              "Stability within this sample does not establish equality to the released fit's true diagonal.", "",
              "| Model | Released fit prompts | Diagonal fit prompts | Diagonal split-half cosine | Filtered 32 vs 64 cosine |", "|---|---:|---:|---:|---:|"]+source_lines
    ledger_path = args.root/"campaign.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        lines += ["", "## Run accounting", ""]
        for name in ledger["protocol"]["models"]:
            row = ledger["models"].get(name, {})
            if row.get("stage") == "complete":
                lines.append(f"- {name}: complete; GPU function time {row['result']['seconds']/60:.1f} min.")
            else:
                lines.append(f"- {name}: {row.get('stage','not launched')}; {row.get('error','')}")
    lines += ["", "Scope: one diagonal estimate, one corpus, fixed relative layers and doses, no model-specific calibration. "
              "Clean eligibility differs by model. This condition differs from the earlier jointly fitted 32-prompt "
              "GSM8K full/off comparisons, so changes across the two campaigns do not isolate lens provenance alone."]
    (args.root/"summary.json").write_text(json.dumps({"method": "published full minus estimated diagonal", "models": models}, indent=2)+"\n")
    (args.root/"summary.md").write_text("\n".join(lines)+"\n")
    print(args.root/"summary.md")
    if args.figure and models:
        make_figure(models, Path("output/neuronpedia_replication"))


if __name__ == "__main__":
    main()

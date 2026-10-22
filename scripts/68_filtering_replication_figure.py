"""Forest plot of fixed filtering contrasts, including labeled historical runs."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def main():
    import matplotlib.pyplot as plt
    import numpy as np

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, nargs="?", default=Path("results/filtering_replication"))
    ap.add_argument("--historical", type=Path, default=Path("results/cross_model"))
    ap.add_argument("--out", type=Path, default=Path("output/filtering_replication"))
    args = ap.parse_args()
    summary = json.loads((args.root / "summary.json").read_text())["models"]
    spec = importlib.util.spec_from_file_location("cross_report", Path(__file__).with_name("60_cross_model_report.py"))
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    for status in args.historical.glob("*/*/status.json"):
        if json.loads(status.read_text())["stage"] != "complete":
            continue
        root = status.parent
        name = root.parent.name
        latent = report.emission_metrics(json.loads((root / "latent_results.json").read_text()))
        concept = report.emission_metrics(json.loads((root / "concept_results.json").read_text()))
        stats = {}
        for strength in (0.5, 1.0):
            rows = [r for r in latent if r["strength"] == strength]
            stats[f"latent / {strength:g}"] = {"success": report.paired(rows, "full", "off_m", "success")}
        stats["concept use / 1"] = {"success": report.paired(
            [r for r in concept if r["kind"] == "use"], "lam1_cn", "lam0_cn", "success", concept=True)}
        summary[name] = {"historical": True, "contrasts": stats}
    order = [n for n in ("qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen2.5-7b-instruct",
                        "llama-3.1-8b-instruct", "mistral-7b-instruct-v0.3",
                        "olmo-2-1124-7b-instruct", "gemma-3-12b-it") if n in summary]
    labels = {"qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B (control)", "qwen3-14b": "Qwen3-14B",
              "qwen2.5-7b-instruct": "Qwen2.5-7B-Instruct", "llama-3.1-8b-instruct": "Llama-3.1-8B-Instruct*",
              "mistral-7b-instruct-v0.3": "Mistral-7B-Instruct-v0.3", "olmo-2-1124-7b-instruct": "OLMo-2-7B-Instruct",
              "gemma-3-12b-it": "Gemma-3-12B-IT*"}
    settings = [("latent / 0.5", "Latent swaps · strength 0.5"), ("latent / 1", "Latent swaps · strength 1"),
                ("concept use / 1", "Concept use · strength 1")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.8), sharey=True)
    ys = np.arange(len(order))
    for ax, (key, title) in zip(axes, settings, strict=True):
        ax.axvline(0, color="#687380", lw=1, zorder=1)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e7eaee", lw=.7)
        for y, name in zip(ys, order, strict=True):
            d = summary[name]["contrasts"][key]["success"]
            color = "#1b6488" if not summary[name].get("historical") else "#91877d"
            ax.errorbar(d["delta"], y, xerr=[[max(0, d["delta"]-d["ci"][0])], [max(0, d["ci"][1]-d["delta"])]],
                        fmt="o", color=color, capsize=3, ms=6, lw=1.5, zorder=3)
            ax.annotate(f"{d['delta']:+.1f}", (d["delta"], y), xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=8, color=color)
        ax.set_title(title, fontsize=11, loc="left", pad=18, weight="bold")
        ax.set_xlabel("Filtered − full accuracy (percentage points)", fontsize=9)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=9)
        ax.set_ylim(len(order)-.5, -.65)
    axes[0].set_yticks(ys, [labels[n] for n in order])
    limits = [(summary[n]["contrasts"][key]["success"]["ci"]) for n in order for key, _ in settings]
    low = min(-5, min(v[0] for v in limits))-3
    high = max(5, max(v[1] for v in limits))+3
    for ax in axes:
        ax.set_xlim(low, high)
    fig.suptitle("Where does position-pair filtering improve relational use?", x=.025, y=.98,
                 ha="left", fontsize=17, weight="bold")
    fig.text(.025, .905, "Same fixed protocol across new models; positive values favor filtering.", fontsize=10, color="#43515d")
    fig.text(.025, .045, "Bars: paired 95% source-entity cluster bootstrap intervals, conditional on one fit; pointwise, unadjusted.\n"
             "* Historical Gemma/Llama runs. Clean eligibility differs by model. One GSM8K fit; seven relative-depth layers; raw prompts.",
             fontsize=8.5, color="#43515d", linespacing=1.5)
    fig.subplots_adjust(left=.235, right=.985, top=.805, bottom=.19, wspace=.20)
    args.out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(args.out / f"accuracy_contrasts.{suffix}", dpi=200, facecolor="white")
    (args.out / "figure_data.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(args.out / "accuracy_contrasts.png")


if __name__ == "__main__":
    main()

"""Two readable cross-model figures from the audited master table; CPU only."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/core-results/full_table.json"
OUT = ROOT / "output/core-results/figures"
COLORS = ("#24789B", "#B86C25")
INK, MUTED = "#172C3C", "#526777"
TASK = "latent / 1"
METRICS = (("success", "Latent swap success"), ("incorrect_name_emission", "Leakage"))
LABELS = {
    "qwen3-4b": "Qwen3-4B",
    "qwen3-8b": "Qwen3-8B",
    "qwen3-14b": "Qwen3-14B",
    "qwen2.5-7b-instruct": "Qwen2.5-7B Instruct",
    "mistral-7b-instruct-v0.3": "Mistral-7B Instruct v0.3",
    "olmo-2-1124-7b-instruct": "OLMo-2-7B Instruct",
    "gemma-3-12b-it": "Gemma 3 12B IT",
    "gemma-3-12b-pt": "Gemma 3 12B base",
    "gemma-3-4b-it": "Gemma 3 4B IT",
    "gemma-2-2b-it": "Gemma 2 2B IT",
    "gemma-2-9b-it": "Gemma 2 9B IT",
    "llama-3.1-8b": "Llama 3.1 8B base",
    "llama-3.1-8b-instruct": "Llama 3.1 8B Instruct",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows():
    payload = json.loads(SOURCE.read_text())
    provenance = json.loads(SOURCE.with_name("provenance.json").read_text())
    for filename, expected in provenance["sources_sha256"].items():
        if sha(ROOT / filename) != expected:
            raise ValueError(
                f"Master-table input changed: {filename}; rebuild script 75 first"
            )
    rows = payload["core"]
    assert len(rows) == 16 and len({r["id"] for r in rows}) == 16
    for row in rows:
        for metric, _ in METRICS:
            stat = row["contrasts"][TASK][metric]
            assert stat["n"] > 0 and all(
                math.isfinite(x) for x in [stat["delta"], *stat["ci"]]
            )
            assert math.isclose(
                stat["delta"], stat["filtered"] - stat["baseline"], abs_tol=1e-9
            )
            assert stat["ci"][0] <= stat["delta"] <= stat["ci"][1]
        assert (
            row["contrasts"][TASK]["success"]["n"]
            == row["contrasts"][TASK]["incorrect_name_emission"]["n"]
        )
    return {r["id"]: r for r in rows}


def make_figure(rows, filename, published, limits, group_counts=None):
    """Publication-size forest plot, with a shared outcome definition and axes."""
    combined = group_counts is not None
    height = max(2.80, 0.90 + 0.34 * len(rows) + (0.55 if combined else 0))
    fig, axes = plt.subplots(1, 2, figsize=(4.8, height), sharey=True)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.365, right=0.985, bottom=0.80/height, top=1-0.37/height, wspace=0.24)
    positions = list(range(len(rows)))
    if combined:
        split = group_counts[0]
        positions = [y + (1.45 if y >= split else 0) for y in positions]
    labels = [LABELS[r["model"]] for r in rows]
    tick_labels = [
        f"{label}\n" + r"$n=" + str(row["contrasts"][TASK]["success"]["n"]) + "$"
        for label, row in zip(labels, rows, strict=True)
    ]
    plotted = []
    for panel, (ax, (metric, title), color) in enumerate(
        zip(axes, METRICS, COLORS, strict=True)
    ):
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E5EBEF", linewidth=0.5)
        ax.axvline(0, color="#667885", linewidth=0.8, zorder=1)
        for y, row, label in zip(positions, rows, labels, strict=True):
            stat = row["contrasts"][TASK][metric]
            value, (low, high) = stat["delta"], stat["ci"]
            ax.errorbar(
                value, y, xerr=[[value-low], [high-value]], fmt="o",
                color=color, ecolor=color, markersize=3.8, linewidth=1.0,
                capsize=2, capthick=0.8, markeredgecolor="white", markeredgewidth=0.4,
                zorder=3,
            )
            ax.annotate(
                f"{value:+.1f}", (value, y), xytext=(0, 5),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=6.5, color=color,
            )
            plotted.append({
                "id": row["id"], "model": row["model"], "label": label,
                "lens": row["lens"], "task": TASK, "metric": metric,
                "metric_label": title, "outputs_in_paired_comparison": 2*stat["n"], **stat,
            })
        ax.set(xlim=limits[metric], ylim=(positions[-1]+0.65, -1.45 if combined else -0.85))
        ax.xaxis.set_major_locator(MultipleLocator(20 if metric == "incorrect_name_emission" else 10))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:+g}" if value else "0"))
        ax.tick_params(axis="x", labelsize=6.5, colors=MUTED, length=0, pad=5)
        ax.tick_params(axis="y", length=0, labelsize=6.8, colors=INK, pad=6)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#B8C6CF")
        ax.set_title(title, loc="left", pad=12, fontsize=8, color=color, fontweight="bold")
        if panel == 0:
            ax.set_yticks(positions, tick_labels)
    # Put the figure description in the manuscript caption, keeping only the
    # outcome headings and lens-source row labels inside the artwork.
    group_labels = []
    if combined:
        for y, text in [(-1.08, "Self-fitted lenses\n(32 GSM8K texts)"),
                        (split+0.45, "Neuronpedia lenses\n(estimated filtering)")]:
            loc = fig.transFigure.inverted().transform(axes[0].transData.transform((0,y)))[1]
            group_labels.append(fig.text(
                0.345, loc, text, ha="right", va="center", fontsize=7,
                color=INK, fontweight="bold", linespacing=1.25,
            ))
    fig.text(0.67, 0.36/height, "Filtered minus full (percentage points)", ha="center", fontsize=7, color=INK)
    fig.text(0.025, 0.16/height, "Success: positive is better. Leakage: negative is better.", fontsize=6.4, color=MUTED)
    fig.text(0.025, 0.025/height, "Paired 95% source-entity intervals; conditional on lens estimates.", fontsize=6.2, color=MUTED)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert all(
        label.get_window_extent(renderer).x1 < axes[0].get_window_extent(renderer).x0
        for label in group_labels
    ), "Lens-source labels must stay outside the plotting area"
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{filename}.{suffix}", dpi=220, facecolor="white")
    plt.close(fig)
    return plotted


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": True,
        }
    )
    rows = load_rows()
    # Presentation policy is independent of outcome values: prefer IT over base,
    # published over self-fitted for the same checkpoint, and the newer Qwen replay.
    np_ids = ["NP-01", "NP-02", "NP-04", "NP-05"]
    other_ids = ["F-05", "F-06", "F-04", "F-03", "F-01", "F-02", "H-01"]
    omitted = {
        "P-GSM": "Original Qwen3-8B GSM8K reference; newer F-06 shown",
        "P-WIKI": "Original Qwen3-8B WikiText reference; newer F-06 shown",
        "H-02": "Self-fitted Llama Instruct; published NP-05 shown",
        "NP-03": "Gemma 3 12B base; instruction-tuned H-01 shown",
        "NP-06": "Llama 3.1 8B base; instruction-tuned NP-05 shown",
    }
    selected = np_ids + other_ids
    assert len({rows[k]["model"] for k in selected}) == len(selected) == 11
    assert set(selected).isdisjoint(omitted) and set(selected) | set(omitted) == set(rows)
    assert all(rows[k]["published_metadata"] is not None for k in np_ids)
    assert all(rows[k]["method"] == "jointly fitted full / off-diagonal" for k in other_ids)
    limits = {}
    for metric, _ in METRICS:
        values = [row["contrasts"][TASK][metric]["ci"] for row in rows.values()]
        limits[metric] = (
            10 * math.floor((min(0, min(c[0] for c in values)) - 2) / 10),
            10 * math.ceil((max(0, max(c[1] for c in values)) + 2) / 10),
        )
    OUT.mkdir(parents=True, exist_ok=True)
    data = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha(SOURCE),
        "script_sha256": sha(Path(__file__)),
        "strength": 1,
        "display_selection": {"rule": "Prefer IT over base; published over self-fitted for the same checkpoint; newer Qwen replay over original references. No outcome-based selection or averaging.", "shown_ids": other_ids + np_ids, "omitted_from_display": omitted, "all_16_conditions_retained_in_source_table": True},
        "shared_x_limits": limits,
        "sample_definition": "N eligible prompt-target swap trials, paired across full and filtered lenses; one generated continuation per lens per trial, 2N outputs; both metrics use the same N trials",
        "leakage_definition": "Target-name substring in an incorrect continuation; correct answers containing the name are excluded",
        "confidence_interval": "Paired 95% source-entity bootstrap; 2000 draws; pointwise, conditional on one fit/diagonal estimate",
        "neuronpedia": make_figure(
            [rows[k] for k in np_ids], "neuronpedia_across_models", True, limits
        ),
        "self_fitted": make_figure(
            [rows[k] for k in other_ids], "self_fitted_across_models", False, limits
        ),
        "combined": make_figure(
            [rows[k] for k in other_ids + np_ids], "cross_model_deduplicated", False,
            limits, group_counts=(len(other_ids), len(np_ids)),
        ),
    }
    (OUT / "figure_data.json").write_text(json.dumps(data, indent=2) + "\n")
    captions = """# Cross-model filtering figures

The manuscript figure `cross_model_deduplicated.pdf` / `.svg` / `.png` combines
seven self-fitted and four published-lens conditions. The separate
`self_fitted_across_models` and `neuronpedia_across_models` files show the same
disjoint selections. Vector PDF exports can be included directly in Overleaf.

## Display selection

One row per model/size: prefer instruction-tuned over base variants, published
lenses over self-fitted lenses for the same checkpoint, and the newer Qwen3-8B
replication over its original GSM8K/WikiText references. The rule does not use
accuracy or leakage outcomes, and no conditions are averaged together.
The full source table retains all 16 conditions on 13 checkpoints; the plot
shows 11 distinct checkpoints. Its selection must not be described as all 16
conditions or as a controlled comparison of lens sources.

Omitted from the plot only: P-GSM and P-WIKI (original Qwen reference rows),
H-02 (self-fitted Llama Instruct), NP-03 (Gemma 3 12B base), and NP-06 (Llama base).
Their complete results remain in `../full_table.json` and `../full_table.md`.

## Outcomes and methods

Both panels show the same eligible latent-intermediate prompt-target swap
trials at strength 1. Left: downstream target-answer success; right: target-name
substring in an incorrect continuation. Correct answers containing the name
count as successes, not leakage. Effects are filtered minus full in percentage
points. Each n is a paired-trial count; one output per lens yields 2n outputs.
The separate concept-use/naming task is not plotted here.

Self-fitted directions are jointly estimated on 32 GSM8K texts. Published full
lenses are held fixed while subtracting a same-position contribution estimated
on 64 WikiText texts; this filtering is approximate. Error bars are paired,
pointwise 95% source-entity bootstrap intervals (2,000 draws), conditional on
one lens/estimate, without multiple-comparison correction. Model eligibility
populations differ; rows are not head-to-head comparisons of models.

## Main-figure caption

Cross-model latent-intermediate swaps at strength 1. Effects are filtered minus
full in percentage points; both panels use the same n paired trials per row.
Self-fitted lenses use jointly estimated full and filtered directions; published
Neuronpedia lenses use approximate filtering with a separately estimated
same-position contribution. Error bars are paired, pointwise 95% source-entity
bootstrap intervals conditional on lens estimates. One row per model/size is
shown using the source/variant selection rule above; all conditions, including
base variants and original Qwen reference panels, are retained in the full table.

Build: `.venv/bin/python scripts/76_core_results_figures.py` (CPU only).
"""
    captions += "\n## Exact evaluation counts\n\n"
    captions += "| Condition | Model | Paired swap trials (n) | Outputs (2n) | Source-entity clusters |\n"
    captions += "|---|---|---:|---:|---:|\n"
    for row_id in np_ids + other_ids:
        row = rows[row_id]
        stat = row["contrasts"][TASK]["success"]
        captions += (
            f"| {row_id} | {LABELS[row['model']]} | {stat['n']} | "
            f"{2 * stat['n']} | {stat['clusters']} |\n"
        )
    (OUT / "captions.md").write_text(captions)
    print(
        json.dumps(
            {
                "output": str(OUT),
                "figures": 3,
                "published_rows": len(np_ids),
                "self_fitted_rows": len(other_ids),
                "shared_x_limits": limits,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

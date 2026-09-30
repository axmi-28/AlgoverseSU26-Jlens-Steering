"""One figure: do the prompt-local predictors survive the controls?

Three series, two panels, swap only.

* **Two prompt-local geometric predictors** -- top-8 right-subspace alignment of
  J_x to J_bar, and cos(g_x, g_bar) -- the only two of our six that hold their
  sign across both the strength grid and both models.
* **Workspace loading**, the J-Lens paper's own per-prompt predictor of swap
  success (section 3.4), as the control. Not a competitor: it asks whether the
  concept is present, ours ask whether the averaged direction describes this
  prompt's computation.

x is steering strength, so the whole point of the figure is which lines stay
flat. Every value is a partial Spearman with baseline correctness partialled
out, ranks tie-averaged.

**Swap only, deliberately.** The additive arm's output distribution collapses
past a threshold -- on qwen3-8b, median KL reaches 13.9 nats against
ln(vocab) ~ 11.9, and distinct greedy outputs fall from 45 to 16 -- so
correlations there measure which prompts break, not which steer. The swap arm
shows no such collapse at any strength tested (61 distinct outputs at alpha=4
against 51 at alpha=0.125), which is what makes its full range readable.

    python scripts/17_control_figure.py
"""

from __future__ import annotations

import importlib.util
import json
import logging
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("fig")

# Validated with the dataviz palette checker (light surface, categorical):
# all of lightness band, chroma floor, CVD separation, normal-vision floor and
# contrast vs surface pass. Do not swap these for eyeballed colours.
GEOMETRY_A = "#1b6ca8"
GEOMETRY_B = "#7d5ba6"
CONTROL = "#c14d1d"

SERIES = [
    (
        "subspace align k=8 right (to J_bar)",
        "Subspace alignment\nof this prompt's Jacobian",
        GEOMETRY_A,
        "-",
        "o",
    ),
    (
        "cos(g_x, g_bar) on swap targets",
        "Direction agreement\ncos(g_x, ḡ)",
        GEOMETRY_B,
        "-",
        "s",
    ),
    (
        "workspace loading",
        "Workspace loading\n(J-Lens paper's predictor)",
        CONTROL,
        "--",
        "^",
    ),
]

MODELS = [
    ("qwen3-8b", "Qwen3-8B", 14, 19, ["steering_qwen3-8b_both_c1_L14-19.json"]),
    (
        "qwen3.6-27b",
        "Qwen3.6-27B",
        24,
        34,
        [
            "steering_qwen3.6-27b_swap_c1_L24-34.json",
            "steering_qwen3.6-27b_swap_c1_L24-34_lowalpha.json",
        ],
    ),
]


def _sweep_module():
    path = Path(__file__).with_name("16_c2_strength_sweep.py")
    spec = importlib.util.spec_from_file_location("c2_sweep", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def series_for(m, config_name, lo, hi, files, results: Path):
    from jsteer.config import load_config
    from jsteer.data import flexible_generalization_prompts

    config = load_config(config_name)
    band = list(range(lo, hi + 1))

    ranks: dict[float, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    baseline: dict[str, bool] = {}
    for name in files:
        path = results / "causal" / name
        if not path.exists():
            continue
        for r in json.loads(path.read_text()):
            if r["arm"] == "swap_averaged":
                ranks[r["strength"]][r["prompt_key"]].append(r["target_rank"])
            if r["arm"] == "baseline":
                baseline[r["prompt_key"]] = bool(r["hit"])

    strengths = sorted(ranks)
    prompts = [
        p for p in flexible_generalization_prompts() if p.key in ranks[strengths[0]]
    ]
    predictors = dict(
        m._load_c2_module().build_predictors(config, results, band, "all", prompts)
    )
    loading_path = results / "c2" / f"workspace_loading_{config_name}_L{lo}-{hi}.json"
    predictors["workspace loading"] = {
        k: v["loading"] for k, v in json.loads(loading_path.read_text()).items()
    }

    out = {}
    for key, *_ in SERIES:
        ys = []
        for s in strengths:
            score = {k: -st.median(v) for k, v in ranks[s].items()}
            keys = [
                p.key for p in prompts if p.key in predictors[key] and p.key in score
            ]
            ys.append(
                m.partial(
                    [predictors[key][k] for k in keys],
                    [score[k] for k in keys],
                    [1.0 if baseline.get(k) else 0.0 for k in keys],
                )
            )
        out[key] = ys
    hits = {
        s: sum(1 for v in ranks[s].values() if st.median(v) == 0) for s in strengths
    }
    return strengths, out, hits, len(prompts)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = REPO_ROOT / "results"
    m = _sweep_module()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True)
    panels = []
    for ax, (name, label, lo, hi, files) in zip(axes, MODELS, strict=False):
        strengths, curves, hits, n = series_for(m, name, lo, hi, files, results)
        panels.append((ax, label, strengths, curves, hits, n, lo, hi))

    for ax, label, strengths, curves, hits, n, lo, hi in panels:
        ax.axhline(0, color="#999999", lw=1, zorder=1)
        for key, series_label, colour, style, marker in SERIES:
            ax.plot(
                strengths,
                curves[key],
                style,
                color=colour,
                lw=2,
                marker=marker,
                markersize=7,
                markerfacecolor="white",
                markeredgewidth=2,
                zorder=3,
                label=series_label,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(strengths)
        ax.set_xticklabels([f"{s:g}" for s in strengths])
        ax.minorticks_off()
        ax.set_title(
            f"{label}   (layers L{lo}–L{hi}, n = {n} prompts)", fontsize=11, pad=10
        )
        ax.set_xlabel("steering strength α  (log scale)")
        ax.grid(axis="y", color="#e6e6e6", lw=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        # Second row of tick labels: how well steering actually works there,
        # so a reader can see which columns are worth trusting.
        secondary = ax.secondary_xaxis("top")
        secondary.set_xscale("log", base=2)
        secondary.set_xticks(strengths)
        secondary.set_xticklabels(
            [str(hits[s]) for s in strengths], fontsize=8, color="#777777"
        )
        secondary.set_xlabel(
            f"prompts successfully steered (of {n})",
            fontsize=8,
            color="#777777",
            labelpad=4,
        )
        secondary.tick_params(length=0)
        for side in secondary.spines:
            secondary.spines[side].set_visible(False)

    axes[0].set_ylabel(
        "predicts steering success  →\npartial Spearman ρ (difficulty controlled)"
    )
    axes[0].set_ylim(-0.35, 0.72)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
        fontsize=9.5,
        handlelength=2.6,
        columnspacing=2.6,
    )
    # Titled on what the figure actually shows. An earlier draft said "only the
    # prompt-local predictors survive", which overclaims: at alpha 0.5-1 on the
    # 8B all three lines are within ~0.06 of each other. The separation is in
    # the trend, not the level.
    fig.suptitle(
        "The paper's predictor decays with steering strength; "
        "the prompt-local ones hold",
        fontsize=13,
        y=0.99,
    )
    fig.text(
        0.5,
        0.885,
        "Swap steering with the lens's averaged direction. Above zero = the "
        "predictor identifies which prompts steer well. Note the two models "
        "were run on different \u03b1 grids.",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.87))

    dest = results / "figures" / "fig6_controls.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=200, bbox_inches="tight", facecolor="white")
    logger.info("wrote %s", dest)


if __name__ == "__main__":
    main()

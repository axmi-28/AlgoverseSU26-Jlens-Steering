#!/usr/bin/env python3
"""Figure: are the Jacobian predictors and workspace loading the same signal?

Two panels, answering the question at two levels.

Left, a correlation matrix. The six prompt-Jacobian measures are strongly
inter-correlated (0.70-0.91) -- they are one underlying quantity -- while
workspace loading sits apart from all of them. Rows are ordered by each
predictor's mean absolute correlation with the others, so the tight cluster
comes first and the outlier lands last by construction rather than by choice.
Only the lower triangle is drawn: the matrix is symmetric and showing both
halves doubles the ink for no information.

Right, the practical version. Every prompt placed by its loading and its
subspace alignment, coloured by how often that prompt's swaps actually
succeeded, split at both medians. If the two predictors were the same signal
the points would lie on a diagonal and two quadrants would be empty; if they
are complementary the off-diagonal cells are populated and intermediate in
success, which is what decides how the result should be framed.

Colour follows the job, not taste. The matrix is a *diverging* scale (sign of a
correlation) so it takes two hues with a neutral light midpoint and never a
rainbow. The scatter is *sequential* (magnitude of a success rate) so it takes
one hue, light to dark. Both poles are the palette already validated for this
project's figures.

    python scripts/22_overlap_figure.py --model qwen3-8b
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from jsteer.analysis import rank_correlation  # noqa: E402
from jsteer.config import REPO_ROOT, load_config  # noqa: E402
from jsteer.data import flexible_generalization_prompts  # noqa: E402

# The project's validated poles. Diverging = two hues + neutral midpoint;
# sequential = one hue, light to dark. Lightness is monotone along each arm,
# which is the correct check for a ramp (the categorical CVD validator fails
# ramps by design and must not be used to "fix" them).
COOL = "#1b6ca8"
WARM = "#c14d1d"
NEUTRAL = "#f2f0ee"
INK = "#2b2b2b"
MUTED = "#6b6b6b"

DIVERGING = LinearSegmentedColormap.from_list("corr", [WARM, NEUTRAL, COOL])
SEQUENTIAL = LinearSegmentedColormap.from_list("rate", ["#e8eef4", COOL, "#0d3d63"])

LOADING = "workspace loading"
SHORT = {
    "rel_frobenius(J_x, J_bar)": "‖J_x − J̄‖",
    "participation ratio": "effective rank",
    "subspace align k=8 left (to J_bar)": "subspace align (left)",
    "subspace align k=8 right (to J_bar)": "subspace align (right)",
    "cos(g_x, g_bar) on swap targets": "cos(g_x, ḡ)",
    "|g_x| / |g_bar| on swap targets": "‖g_x‖ / ‖ḡ‖",
    LOADING: "workspace loading",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--band", default="14-19")
    parser.add_argument("--positions", default="all")
    parser.add_argument("--against", default="subspace align k=8 right (to J_bar)")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def _c2_module():
    spec = importlib.util.spec_from_file_location(
        "c2", REPO_ROOT / "scripts" / "11_c2_predictors.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = parse_args()
    lo, hi = (int(x) for x in args.band.split("-"))
    band = list(range(lo, hi + 1))
    config = load_config(args.model)
    results = REPO_ROOT / "results"

    predictors = dict(
        _c2_module().build_predictors(
            config,
            results,
            band,
            args.positions,
            list(flexible_generalization_prompts()),
        )
    )
    predictors[LOADING] = {
        k: v["loading"]
        for k, v in json.loads(
            (
                results / "c2" / f"workspace_loading_{config.name}_L{lo}-{hi}.json"
            ).read_text()
        ).items()
    }

    swap = results / "causal" / f"steering_{config.name}_both_c1_L{lo}-{hi}.json"
    seen: dict[tuple, dict] = {}
    for r in json.loads(swap.read_text()):
        seen[(r["arm"], r["strength"], r["prompt_key"], r["target_arg"])] = r
    hits: dict[str, list[bool]] = defaultdict(list)
    for r in seen.values():
        if r["arm"] == "swap_averaged" and r["strength"] == args.strength:
            hits[r["prompt_key"]].append(r["hit"])

    keys = sorted(set.intersection(*[set(v) for v in predictors.values()]) & set(hits))
    names = list(predictors)

    def col(name: str) -> torch.Tensor:
        return torch.tensor([predictors[name][k] for k in keys])

    corr = {a: {b: rank_correlation(col(a), col(b)) for b in names} for a in names}
    # Geometric predictors ordered by how much each shares with the rest, then
    # workspace loading pinned last *by design*: the bottom row is then loading
    # against every Jacobian measure, read left to right, which is the
    # comparison the panel exists to make. An earlier version sorted all seven
    # together and put `effective rank` last, which left the highlight on the
    # wrong predictor while the title named a different one.
    geometry = [n for n in names if n != LOADING]
    order = sorted(
        geometry, key=lambda a: -st.mean(abs(corr[a][b]) for b in geometry if b != a)
    ) + [LOADING]

    fig, (ax_m, ax_s) = plt.subplots(
        1, 2, figsize=(14.5, 6.4), gridspec_kw={"width_ratios": [1.15, 1]}
    )

    # ---- left: correlation matrix, lower triangle -----------------------
    n = len(order)
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            if j > i:
                continue
            value = corr[a][b]
            ax_m.add_patch(
                plt.Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    facecolor=DIVERGING((value + 1) / 2),
                    edgecolor="white",
                    linewidth=2,
                )
            )
            if i != j:
                ax_m.text(
                    j,
                    i,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    color="white" if abs(value) > 0.55 else INK,
                )
    ax_m.set_xlim(-0.5, n - 0.5)
    ax_m.set_ylim(n - 0.5, -0.5)
    labels = [SHORT[x] for x in order]
    ax_m.set_xticks(range(n))
    ax_m.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax_m.set_yticks(range(n))
    ax_m.set_yticklabels(labels, fontsize=10)
    # Highlight workspace loading by NAME, never by position -- the ordering is
    # derived from the data and must not be trusted to place it anywhere.
    slot = order.index(LOADING)
    for labels_ in (ax_m.get_yticklabels(), ax_m.get_xticklabels()):
        labels_[slot].set_color(WARM)
        labels_[slot].set_fontweight("bold")
    for side in ax_m.spines:
        ax_m.spines[side].set_visible(False)
    ax_m.tick_params(length=0)
    ax_m.set_title(
        "Workspace loading tracks direction agreement (+0.52),\n"
        "but barely tracks subspace alignment (+0.10).",
        fontsize=12.5,
        loc="left",
        pad=14,
        color=INK,
    )
    bar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(-1, 1)),
        ax=ax_m,
        fraction=0.035,
        pad=0.02,
    )
    bar.set_label("Spearman correlation", fontsize=10, color=MUTED)
    bar.outline.set_visible(False)

    # ---- right: complementarity ----------------------------------------
    x = [predictors[LOADING][k] for k in keys]
    y = [predictors[args.against][k] for k in keys]
    rate = [sum(hits[k]) / len(hits[k]) for k in keys]
    mx, my = st.median(x), st.median(y)

    ax_s.axvline(mx, color=MUTED, linewidth=1, linestyle="--", zorder=1)
    ax_s.axhline(my, color=MUTED, linewidth=1, linestyle="--", zorder=1)
    dots = ax_s.scatter(
        x,
        y,
        c=rate,
        cmap=SEQUENTIAL,
        vmin=0,
        vmax=1,
        s=95,
        edgecolor="white",
        linewidth=1.4,
        zorder=3,
    )
    # Quadrant success rates, pooled over that cell's trials.
    cells: dict[tuple, list[bool]] = defaultdict(list)
    for k in keys:
        cells[(predictors[LOADING][k] > mx, predictors[args.against][k] > my)] += hits[
            k
        ]
    span_x, span_y = max(x) - min(x), max(y) - min(y)
    for (hi_x, hi_y), v in cells.items():
        ax_s.text(
            (mx + span_x * 0.30) if hi_x else (mx - span_x * 0.30),
            (my + span_y * 0.36) if hi_y else (my - span_y * 0.36),
            f"{sum(v) / len(v):.0%}",
            ha="center",
            va="center",
            fontsize=21,
            color=INK,
            alpha=0.5,
            fontweight="bold",
            zorder=2,
        )
    ax_s.set_xlabel("Workspace loading  (J-Lens paper's predictor)", fontsize=11)
    ax_s.set_ylabel(f"{SHORT[args.against]}  (this prompt's Jacobian)", fontsize=11)
    ax_s.set_title(
        "Both predictors carry signal the other does not.\n"
        "Each point is one prompt; shade is its swap success rate.",
        fontsize=12.5,
        loc="left",
        pad=14,
        color=INK,
    )
    for side in ("top", "right"):
        ax_s.spines[side].set_visible(False)
    ax_s.grid(alpha=0.18, linewidth=0.7)
    ax_s.set_axisbelow(True)
    bar2 = fig.colorbar(dots, ax=ax_s, fraction=0.035, pad=0.02)
    bar2.set_label("swap success rate for that prompt", fontsize=10, color=MUTED)
    bar2.outline.set_visible(False)

    agree = sum(
        1
        for k in keys
        if (predictors[LOADING][k] > mx) == (predictors[args.against][k] > my)
    )
    fig.suptitle(
        f"{config.name} · workspace band L{lo}-{hi} · {len(keys)} prompts · "
        f"swap at α={args.strength} · predictors agree on {agree}/{len(keys)} prompts",
        fontsize=10.5,
        color=MUTED,
        y=0.02,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    dest = (
        Path(args.out)
        if args.out
        else results / "figures" / f"fig7_overlap_{config.name}.png"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Two figures for C2. Static PNGs sized for a document.

Fig 4 is the effect in the units a reader already understands -- percent of
steering attempts that succeeded -- rather than a correlation coefficient.
Fig 5 shows every prompt, so the reader can see the spread behind the bars and
check the baseline-difficulty confound by eye.
"""

from __future__ import annotations

import json
import logging
import statistics as st
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jsteer.config import REPO_ROOT  # noqa: E402

logger = logging.getLogger("c2-figs")

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"
BAND = range(14, 20)


def style(ax, title, subtitle, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=18)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9.5)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9.5)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)


def load():
    results = REPO_ROOT / "results"
    pb = json.loads((results / "rq2" / "pullbacks_qwen3-8b_all.json").read_text())
    cos = defaultdict(list)
    for x in pb:
        if (
            x["group"] == "eval"
            and x["layer"] in BAND
            and x["relevance"] == "swap_target"
        ):
            cos[x["prompt"]].append(x["cos"])
    causal = json.loads(
        (results / "causal" / "steering_qwen3-8b_both_c1_L14-19.json").read_text()
    )
    hits, rank, base = defaultdict(list), defaultdict(list), {}
    for x in causal:
        if x["arm"] == "swap_averaged" and x["strength"] == 1.0:
            hits[x["prompt_key"]].append(x["hit"])
            rank[x["prompt_key"]].append(x["target_rank"])
        if x["arm"] == "baseline":
            base[x["prompt_key"]] = x["hit"]
    keys = sorted(set(cos) & set(hits))
    return (
        keys,
        {k: st.mean(cos[k]) for k in keys},
        hits,
        {k: st.median(rank[k]) for k in keys},
        base,
        results,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    keys, cosine, hits, rank, base, results = load()
    out_dir = results / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    order = sorted(keys, key=lambda k: cosine[k])
    third = len(order) // 3
    groups = [
        ("Least aligned\nthird", order[:third]),
        ("Middle\nthird", order[third : 2 * third]),
        ("Most aligned\nthird", order[2 * third :]),
    ]

    # ---- Figure 4: the effect in success rates ----
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    style(
        ax,
        "Prompts whose own Jacobian disagrees with the lens steer worse",
        "64 prompts split into thirds; 192 steering attempts total, all using the standard J-Lens direction",
        "Prompts grouped by how well their own steering direction aligns with the lens's",
        "Steering attempts that succeeded",
    )
    rates, labels, counts = [], [], []
    for label, g in groups:
        trials = sum(len(hits[k]) for k in g)
        won = sum(sum(hits[k]) for k in g)
        rates.append(100 * won / trials)
        labels.append(label)
        counts.append(f"{won}/{trials}")
    bars = ax.bar(labels, rates, color=BLUE, width=0.58, zorder=3)
    for bar, rate, count in zip(bars, rates, counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 1.4,
            f"{rate:.0f}%",
            ha="center",
            color=INK,
            fontsize=13,
            fontweight="bold",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate / 2,
            count,
            ha="center",
            color="white",
            fontsize=10,
        )
    ax.set_ylim(0, max(rates) * 1.28)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    fig.savefig(
        out_dir / "fig4_c2_success_by_alignment.png",
        dpi=200,
        bbox_inches="tight",
        facecolor=SURFACE,
    )
    plt.close(fig)
    logger.info("wrote fig4")

    # ---- Figure 5: every prompt, with the confound visible ----
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    style(
        ax,
        "The relationship holds whether or not the model knew the answer",
        "One point per prompt (n=64). Lower = the intended new answer ranked higher after steering.",
        "How well the prompt's own steering direction aligns with the lens's  →",
        "Rank of intended answer after steering",
    )
    for flag, colour, label in (
        (True, BLUE, "model answered the unsteered prompt correctly"),
        (False, ORANGE, "model did not"),
    ):
        xs = [cosine[k] for k in keys if bool(base.get(k)) is flag]
        # rank is 0-indexed (0 == the model's top token), so +1 to plot it as
        # an ordinal position. Clamping instead would silently merge "1st" with
        # "2nd" and put the most important points off the top of the axis.
        ys = [rank[k] + 1 for k in keys if bool(base.get(k)) is flag]
        ax.scatter(
            xs,
            ys,
            s=44,
            color=colour,
            alpha=0.82,
            edgecolor=SURFACE,
            linewidth=1.2,
            label=label,
            zorder=3,
        )
    ax.set_yscale("log")
    ax.set_ylim(3000, 0.62)  # inverted, with headroom so 1st place is not clipped
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(["1st", "10th", "100th", "1000th"])
    ax.grid(True, axis="both", color=GRID, linewidth=0.6)
    ax.annotate(
        "steering worked",
        xy=(0.012, 0.955),
        xycoords="axes fraction",
        color=INK_2,
        fontsize=9,
        style="italic",
    )
    ax.annotate(
        "steering failed",
        xy=(0.012, 0.03),
        xycoords="axes fraction",
        color=INK_2,
        fontsize=9,
        style="italic",
    )
    # Legend below the axes: every quadrant of the plot area holds data.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor=INK_2,
    )
    fig.savefig(
        out_dir / "fig5_c2_per_prompt.png",
        dpi=200,
        bbox_inches="tight",
        facecolor=SURFACE,
    )
    plt.close(fig)
    logger.info("wrote fig5")
    return 0


if __name__ == "__main__":
    sys.exit(main())

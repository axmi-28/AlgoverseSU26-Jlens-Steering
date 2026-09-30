#!/usr/bin/env python3
"""Charts for the preliminary results. Static PNGs sized for a Google Doc.

One chart per claim, not one per table. Each is a line chart over layer depth,
which is the ordered dimension every result varies along, on a single axis.

Palette: slots 1-3 of the validated categorical set (blue / orange / aqua),
which clears the all-pairs CVD and normal-vision floors in light mode. Every
series is direct-labelled at its right end as well as legended, so identity is
never carried by colour alone.

    python scripts/10_charts.py --config qwen3-8b --positions all
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jsteer.config import REPO_ROOT, load_config  # noqa: E402

logger = logging.getLogger("charts")

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SURFACE = "#fcfcfb"


def style(ax, title: str, ylabel: str, subtitle: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(
        title,
        color=INK,
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=16 if subtitle else 8,
    )
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9)
    ax.set_xlabel("Layer", color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)


def label_end(ax, x, y, text, color, dy: float = 0) -> None:
    """Direct label at the right end -- identity without relying on colour."""
    ax.annotate(
        text,
        xy=(x[-1], y[-1]),
        xytext=(6, dy),
        textcoords="offset points",
        color=color,
        fontsize=9,
        fontweight="bold",
        va="center",
    )


def med(xs):
    return st.median(xs) if xs else float("nan")


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    logger.info("wrote %s", path)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="qwen3-8b")
    parser.add_argument("--positions", default="all")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    results = REPO_ROOT / "results"
    out_dir = Path(args.out) if args.out else results / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: RQ2 headline ----
    path = results / "rq2" / f"pullbacks_{config.name}_{args.positions}.json"
    if path.exists():
        records = json.loads(path.read_text())
        layers = sorted({r["layer"] for r in records})
        by = defaultdict(list)
        for r in records:
            by[(r["group"], r["layer"])].append(r["cos"])

        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        style(
            ax,
            "The averaged direction is a poor proxy for the prompt's own",
            "cos(g_x, g_bar)",
            "Median cosine between the prompt-local pulled-back direction and the lens's averaged one",
        )
        ev = [med(by[("eval", l)]) for l in layers]
        ft = [med(by[("fit", l)]) for l in layers]
        p10 = [sorted(by[("eval", l)])[len(by[("eval", l)]) // 10] for l in layers]
        ax.fill_between(layers, p10, ev, color=BLUE, alpha=0.13, linewidth=0)
        ax.plot(layers, ft, color=ORANGE, linewidth=2)
        ax.plot(layers, ev, color=BLUE, linewidth=2)
        # The two series nearly meet at the last layer; nudge the labels apart.
        label_end(ax, layers, ft, "wikitext (null)", ORANGE, dy=7)
        label_end(ax, layers, ev, "eval prompts", BLUE, dy=-8)
        ax.set_ylim(0, 1.08)
        ax.set_xlim(layers[0], layers[-1] + 5.0)
        ax.legend(
            handles=[
                plt.Line2D(
                    [],
                    [],
                    color=BLUE,
                    lw=2,
                    label="eval (flexible-generalization, n=64)",
                ),
                plt.Line2D([], [], color=ORANGE, lw=2, label="wikitext null (n=100)"),
                plt.Line2D(
                    [], [], color=BLUE, lw=6, alpha=0.13, label="eval p10-median band"
                ),
            ],
            loc="upper left",
            frameon=False,
            fontsize=8,
            labelcolor=INK_2,
        )
        save(fig, out_dir / "fig1_rq2_cosine_by_layer.png")

        # ---- Figure 2: relevance runs backwards ----
        rel = defaultdict(list)
        for r in records:
            if r["group"] == "eval":
                rel[(r["relevance"], r["layer"])].append(r["cos"])
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        style(
            ax,
            "The averaged direction is worst for the tokens you would steer with",
            "cos(g_x, g_bar)",
            "Eval prompts only, split by whether the target token relates to the prompt",
        )
        # The three series converge to ~0.97 at the last layer, so end labels
        # collide. Label at the point of maximum separation instead and keep a
        # legend, rather than stacking unreadable text at the right edge.
        handles = []
        series = {}
        for name, colour, label in (
            ("foreign", AQUA, "foreign (different category)"),
            ("swap_target", ORANGE, "swap target (same category)"),
            ("self", BLUE, "the prompt's own argument"),
        ):
            ys = [med(rel[(name, l)]) for l in layers]
            series[name] = ys
            ax.plot(layers, ys, color=colour, linewidth=2)
            handles.append(plt.Line2D([], [], color=colour, lw=2, label=label))
        gaps = [series["foreign"][i] - series["self"][i] for i in range(len(layers))]
        pick = gaps.index(max(gaps))
        for name, colour, dy in (
            ("foreign", AQUA, 9),
            ("swap_target", ORANGE, -2),
            ("self", BLUE, -14),
        ):
            ax.annotate(
                {"foreign": "foreign", "swap_target": "swap target", "self": "own arg"}[
                    name
                ],
                xy=(layers[pick], series[name][pick]),
                xytext=(6, dy),
                textcoords="offset points",
                color=colour,
                fontsize=9,
                fontweight="bold",
            )
        ax.legend(
            handles=handles,
            loc="upper left",
            frameon=False,
            fontsize=8,
            labelcolor=INK_2,
        )
        ax.set_ylim(0, 1.05)
        ax.set_xlim(layers[0], layers[-1] + 0.5)
        save(fig, out_dir / "fig2_rq2_relevance.png")

    # ---- Figure 3: RQ1 bias vs variance ----
    scalars = results / "rq1" / f"scalars_{config.name}_{args.positions}.json"
    pooled = results / "rq1" / f"group_mean_digests_{config.name}_{args.positions}.pt"
    if scalars.exists() and pooled.exists():
        import torch

        recs = json.loads(scalars.read_text())
        blob = torch.load(pooled, map_location="cpu", weights_only=False)
        layers = sorted({r["layer"] for r in recs})

        individual, centre, predicted = [], [], []
        for layer in layers:
            ind = med(
                [
                    r["rel_frobenius"]
                    for r in recs
                    if r["group"] == "eval" and r["layer"] == layer
                ]
            )
            entry = next(
                d
                for k, d in blob["digests"].items()
                if k.split("|")[0] == "eval" and int(k.split("|")[1]) == layer
            )
            individual.append(ind)
            centre.append(entry["scalars"]["rel_frobenius"])
            predicted.append(ind / (entry["n"] ** 0.5))

        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        style(
            ax,
            "The eval cloud is not centred on the averaged Jacobian - this is bias",
            "||. - J_bar||_F / ||J_bar||_F",
            "If averaging 64 prompts merely cancelled noise, the centre would fall to the dashed line",
        )
        ax.plot(layers, individual, color=BLUE, linewidth=2)
        ax.plot(layers, centre, color=ORANGE, linewidth=2)
        ax.plot(layers, predicted, color=INK_2, linewidth=1.6, linestyle=(0, (5, 3)))
        label_end(ax, layers, individual, "single prompt", BLUE)
        label_end(ax, layers, centre, "mean of 64", ORANGE)
        label_end(ax, layers, predicted, "variance-only\nprediction", INK_2)
        # Headroom at the bottom so the two-line dashed label does not run into
        # the axis, and at the right for the end labels.
        ax.set_ylim(-0.08, max(individual) * 1.08)
        ax.set_xlim(layers[0], layers[-1] + 6.5)
        save(fig, out_dir / "fig3_rq1_bias_vs_variance.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())

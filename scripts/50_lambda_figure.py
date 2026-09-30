"""C37 figures: the diagonal as a continuum. Reads the runs, writes PNGs + CSV.

    python scripts/50_lambda_figure.py [--dose 1.0] [--variant _cn]

Three separate figures rather than one crowded grid:

    fig_c37a_tradeoff_question.png   outcomes vs lambda, edit reaches the question
    fig_c37b_tradeoff_prefix.png     the same, edit on the prefix only (control)
    fig_c37c_geometry.png            where the direction points, shared by both

Percentages share one axis within a panel and cosines share one axis in the
geometry figure, so no panel carries two scales. Figures a and b use identical
axes so they can be read against each other.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("rep", ROOT / "scripts/49_concept_report.py")
rep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rep)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECOND = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
USE = "#2a78d6"      # downstream use, and the off-diagonal alignment that tracks it
EMIT = "#eb6834"     # leakage, and the unembedding alignment that tracks it
EXPRESS = "#1baf7a"
LAYERS = (13, 16, 19, 22, 25, 28, 31)
TITLES = {"gsm8k": "GSM8K-fitted lens", "wikitext_a": "WikiText-fitted lens"}
OUT = ROOT / "results/figures"


def series(preset: str, dose: float, variant: str):
    _, records, conds = rep.load(preset)
    units = rep.unit_table(records, conds, clues={"0", "1"})
    idx = {c: i for i, c in enumerate(conds)}
    _, _, shards = rep.find_shards(preset)
    geom = json.loads(shards[0].read_text()).get("geometry", {})
    out: dict[str, dict[str, list]] = {}
    for corpus in sorted({c for c, _, _ in conds}):
        rows = defaultdict(list)
        for arm in dict.fromkeys(a for _, a, _ in conds):
            if not (arm.startswith("lam") and arm.endswith(variant)):
                continue
            lam = float(arm[len("lam") : -len(variant)])
            k = idx[(corpus, arm, dose)]
            g = [geom[f"{corpus}/{arm}/L{l}"] for l in LAYERS if f"{corpus}/{arm}/L{l}" in geom]
            rows["lam"].append(lam)
            rows["use"].append(100 * statistics.fmean(rep.per_unit(u, k, "hit")[1] for u in units.values()))
            rows["express"].append(100 * statistics.fmean(rep.per_unit(u, k, "hit")[0] for u in units.values()))
            rows["leak"].append(100 * statistics.fmean(rep.per_unit(u, k, "leak")[1] for u in units.values()))
            rows["cos_uy"].append(statistics.fmean(x["cos_uy"] for x in g))
            rows["cos_off"].append(statistics.fmean(x["cos_off"] for x in g))
        order = sorted(range(len(rows["lam"])), key=lambda i: rows["lam"][i])
        out[corpus] = {k: [v[i] for i in order] for k, v in rows.items()}
    return out, len(units)


def style(ax, *, marks_at_top=True):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=12)
    ax.set_xlim(-0.1, 2.95)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xlabel("λ   (weight on the diagonal term)", color=SECOND, fontsize=13, labelpad=8)
    for lam, text in ((0.0, "off-diagonal"), (1.0, "full J-lens")):
        ax.axvline(lam, color=MUTED, linewidth=1.2, linestyle=(0, (2, 3)), zorder=1)
        y, dy, va = (1.0, -15, "top") if marks_at_top else (0.0, 8, "bottom")
        ax.annotate(text, (lam, y), xycoords=("data", "axes fraction"), va=va,
                    xytext=(4, dy), textcoords="offset points", color=MUTED, fontsize=11)


def label_end(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x[-1], y[-1]), xytext=(9, dy), textcoords="offset points",
                color=color, fontsize=13, va="center", fontweight="medium")


def tradeoff_figure(data, where, note, fname):
    corpora = [c for c in ("gsm8k", "wikitext_a") if c in data]
    fig, axes = plt.subplots(1, len(corpora), figsize=(14.5, 7.0), facecolor=SURFACE)
    fig.subplots_adjust(left=0.065, right=0.975, top=0.70, bottom=0.125, wspace=0.32)
    for ax, corpus in zip(axes, corpora):
        d = data[corpus]
        style(ax)
        for key, color, name in (("express", EXPRESS, "says it"),
                                 ("use", USE, "uses it"),
                                 ("leak", EMIT, "leaks the name")):
            ax.plot(d["lam"], d[key], color=color, linewidth=2.6, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3)
            label_end(ax, d["lam"], d[key], name, color)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_ylabel("% of trials", color=SECOND, fontsize=13, labelpad=8)
        ax.set_title(TITLES[corpus], color=INK, fontsize=15, fontweight="bold", loc="left", pad=12)
    fig.suptitle(where, color=INK, fontsize=18, fontweight="bold", x=0.065, ha="left", y=0.965)
    fig.text(0.065, 0.912, note, color=SECOND, fontsize=12.5, ha="left", va="top", linespacing=1.7)
    path = OUT / fname
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def geometry_figure(data, fname):
    corpora = [c for c in ("gsm8k", "wikitext_a") if c in data]
    fig, ax = plt.subplots(figsize=(10.4, 6.8), facecolor=SURFACE)
    fig.subplots_adjust(left=0.095, right=0.745, top=0.68, bottom=0.125)
    style(ax, marks_at_top=False)
    for corpus, dash in zip(corpora, ["-", (0, (5, 2))]):
        d = data[corpus]
        for key, color, name in (("cos_off", USE, "with the\noff-diagonal"),
                                 ("cos_uy", EMIT, "with the raw\nunembedding $u_y$")):
            ax.plot(d["lam"], d[key], color=color, linewidth=2.6, linestyle=dash, marker="o",
                    markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3)
            if corpus == corpora[0]:
                label_end(ax, d["lam"], d[key], name, color)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("cosine of the write direction", color=SECOND, fontsize=13, labelpad=8)
    ax.plot([], [], color=MUTED, linestyle="-", linewidth=2.6, label="GSM8K fit")
    ax.plot([], [], color=MUTED, linestyle=(0, (5, 2)), linewidth=2.6, label="WikiText fit")
    ax.legend(frameon=False, fontsize=12, loc="lower right", labelcolor=SECOND,
              bbox_to_anchor=(1.0, 0.06))
    fig.suptitle("Where the write direction points", color=INK, fontsize=18, fontweight="bold",
                 x=0.095, ha="left", y=0.955)
    fig.text(0.095, 0.885,
             "As λ grows the direction rotates away from the off-diagonal and toward the\n"
             "token's own unembedding row. Identical under both position conventions —\n"
             "only the behaviour they produce differs.",
             color=SECOND, fontsize=12.5, ha="left", va="top", linespacing=1.7)
    path = OUT / fname
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dose", type=float, default=1.0)
    ap.add_argument("--variant", default="_cn")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    prompt, n_units = series("c37lambdaprompt", a.dose, a.variant)
    prefix, _ = series("c37lambdaprefix", a.dose, a.variant)

    common = (f"$J_λ = J_O + λJ_D$, pulled back to $v_λ = v_{{off}} + λv_{{diag}}$.   Qwen3-8B, {n_units} held-out concept units,\n"
              "rank-2 swap, every arm norm-matched per position and column-normalised.")
    tradeoff_figure(prompt, "The edit reaches the question", common,
                    "fig_c37a_tradeoff_question.png")
    tradeoff_figure(prefix, "Control: the edit stays on the prefix",
                    common + "\nSame directions and norms as above; only the edited positions differ.",
                    "fig_c37b_tradeoff_prefix.png")
    geometry_figure(prompt, "fig_c37c_geometry.png")

    csv_path = OUT / "fig_c37_lambda_interpolation.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["positions", "corpus", "lambda", "express_pct", "use_pct", "leak_pct", "cos_off", "cos_uy"])
        for where, dat in (("question", prompt), ("prefix", prefix)):
            for corpus, d in dat.items():
                for i, lam in enumerate(d["lam"]):
                    w.writerow([where, corpus, lam] + [round(d[k][i], 4) for k in
                                                       ("express", "use", "leak", "cos_off", "cos_uy")])
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()

"""Mechanism-section figure: why the same-position term favors leakage on Qwen3-8B. CPU only.

  fig_mechanism  (a) cosine between the write direction and the target's unembedding row
                 (b) relational and leakage margins, change from the unedited run
                 (c) share of queries where the target answer beats the source answer (R > 0)
                     and where the target name beats the target answer (L > 0)

Every prompt position edited, strength 1, edit norms matched to the full lens.
Style, lens encodings and the bootstrap come from 63_paper_figures_v3.py.

Run: .venv/bin/python scripts/64_mechanism_figure.py
"""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("v3", ROOT / "scripts/63_paper_figures_v3.py")
v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v3)
b, LAM, CORPORA, LENS, MM, W = v3.b, v3.LAM, v3.CORPORA, v3.LENS, v3.MM, v3.W


def geometry():
    """cos(v_lambda, u_y), averaged over the 7 band layers and 33 target words (C37)."""
    out = {c: {} for c in CORPORA}
    with open(ROOT / "results/figures/fig_c37_lambda_interpolation.csv") as f:
        for r in csv.DictReader(f):
            if r["positions"] == "question" and r["corpus"] in out:
                out[r["corpus"]][float(r["lambda"])] = float(r["cos_uy"])
    for c in CORPORA:
        assert sorted(out[c]) == LAM, (c, sorted(out[c]))
    return {c: np.array([out[c][x] for x in LAM]) for c in CORPORA}


def margins():
    blobs = [b.read(f"results/concept/gradgeom_c39gradprompt_qwen3-8b_ec4eca685b_shard{i}of2.json")
             for i in range(2)]
    rows = sorted([r for bb in blobs for r in bb["records"]], key=lambda r: r["trial"])
    assert len(rows) == 1339
    cond = [tuple(x) for x in blobs[0]["conditions"]]
    groups = [r["type"] + "/" + r["source"] for r in rows]
    out = {}
    for corp in CORPORA:
        ids = [cond.index((corp, f"lam{x:g}_cn")) for x in LAM]
        for F in "RL":
            d = np.array([[r[f"d{F}"][i] for i in ids] for r in rows])
            level = d + np.array([r["clean"][F] for r in rows])[:, None]
            out[corp, "d" + F] = b.stats(d, groups, f"mech_d{F}_{corp}")
            out[corp, F + ">0"] = tuple(100 * np.asarray(s) for s in
                                        b.stats((level > 0).astype(float), groups, f"mech_{F}pos_{corp}"))
    return out


def lam_axis(ax):
    ax.set_xlim(-.08, 2.08)
    ax.set_xticks([0, .5, 1, 1.5, 2], ["0", "0.5", "1", "1.5", "2"])
    ax.set_xlabel(r"Same-position weight $\lambda$")
    ax.axvline(1, color=v3.MUTED, lw=.6, ls=(0, (1, 2)), zorder=0)


def line(ax, y, corp, color, band=None):
    st = LENS[corp]
    if band is not None:
        ax.fill_between(LAM, band[0], band[1], color=color, alpha=.12, lw=0)
    ax.plot(LAM, y, color=color, ls=st["ls"], marker=st["marker"], ms=2.6, lw=1.2,
            mfc=color if corp == "gsm8k" else "white", mew=.8)


def tag(ax, x, y, text, color):
    ax.text(x, y, text, color=color, fontsize=6.6, ha="left", va="center",
            bbox=dict(fc="white", ec="none", pad=.6))


def fig_mechanism(G, M):
    fig, axs = plt.subplots(1, 3, figsize=(W, 50 * MM))
    fig.subplots_adjust(left=.085, right=.99, bottom=.21, top=.76, wspace=.50)

    ax = axs[0]
    v3.ygrid(ax)
    for corp in CORPORA:
        line(ax, G[corp], corp, v3.INK)
    lam_axis(ax)
    ax.set_ylim(0, .7)
    ax.set_yticks([0, .2, .4, .6])
    ax.set_ylabel("Cosine")
    ax.set_title("Unembedding alignment", fontsize=7.1, pad=4)
    v3.letter(ax, "a", x=-.34)

    ax = axs[1]
    v3.ygrid(ax)
    for corp in CORPORA:
        for F, color in (("dR", v3.USE), ("dL", v3.LEAK)):
            m, lo, hi = M[corp, F]
            line(ax, m, corp, color, (lo, hi))
    lam_axis(ax)
    ax.set_ylim(0, 40)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_ylabel("Change (nats)")
    ax.set_title("Margins after the question", fontsize=7.1, pad=4)
    tag(ax, .05, 38, r"relational $\Delta R$", v3.USE)
    tag(ax, .05, 12.5, r"leakage $\Delta L$", v3.LEAK)
    v3.letter(ax, "b", x=-.34)

    ax = axs[2]
    v3.ygrid(ax)
    for corp in CORPORA:
        for F, color in (("R>0", v3.USE), ("L>0", v3.LEAK)):
            m, lo, hi = M[corp, F]
            line(ax, m, corp, color, (lo, hi))
    lam_axis(ax)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Queries (%)")
    ax.set_title("Which candidate wins", fontsize=7.1, pad=4)
    tag(ax, .05, 84, "new answer > old answer", v3.USE)
    tag(ax, .05, 47, "name > new answer", v3.LEAK)
    v3.letter(ax, "c", x=-.34)

    handles = [plt.Line2D([], [], color=v3.INK, ls=LENS[c]["ls"], marker=LENS[c]["marker"], ms=2.6,
                          lw=1.2, mfc=v3.INK if c == "gsm8k" else "white", mew=.8, label=LENS[c]["label"])
               for c in CORPORA]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.01), handlelength=2.6)
    v3.save(fig, "fig_mechanism")


if __name__ == "__main__":
    v3.OUT.mkdir(parents=True, exist_ok=True)
    G, M = geometry(), margins()
    for corp in CORPORA:
        print(corp, "cos", np.round(G[corp][[0, 5, 7]], 3),
              "dR", np.round(M[corp, "dR"][0][[0, 7]], 1), "dL", np.round(M[corp, "dL"][0][[0, 7]], 1),
              "R>0", np.round(M[corp, "R>0"][0][[0, 7]], 1), "L>0", np.round(M[corp, "L>0"][0][[0, 7]], 1))
    fig_mechanism(G, M)

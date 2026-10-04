"""Main-text figures for the restructured paper (Oct 2026). CPU only, no GPU calls.

Builds four figures into paper/overleaf/figures/ and copies every figure that
main.tex references into paper/overleaf/upload_figures/ for Overleaf:

  fig_filtering    Sec. 4.1   full vs off-diagonal lens on all three evaluation sets
  fig_lambda       Sec. 4.2/5.3  relational use and leakage vs lambda, whole prompt vs prefix only
  fig_competition  Sec. 5.2/5.4  margin changes vs lambda and first-order leakage prediction
  fig_corpus       Sec. 6     five independent fits per corpus

Run: .venv/bin/python scripts/63_paper_figures_v3.py
"""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


b = module("style", "50_paper_figures.py")      # read, stats, load_concept, unit_arrays, col
c = module("corpus", "53_corpus_figures.py")    # corpus_rows
s58 = module("s58", "58_loading_split.py")      # load_trials, latent_outcomes

OVERLEAF = ROOT / "paper/overleaf"
OUT = OVERLEAF / "figures"
UPLOAD = OVERLEAF / "upload_figures"

# ---------------------------------------------------------------- style
MM = 1 / 25.4
W = 122 * MM                      # LNCS text width
INK = "#222222"
MUTED = "#6B6B6B"
GRID = "#E6E6E6"
USE = "#0072B2"                   # concept use / success (matches Fig. 1)
LEAK = "#D55E00"                  # name leakage (matches Fig. 1)
LENS = {"gsm8k": dict(color="#222222", ls="-", marker="o", label="GSM8K lens"),
        "wikitext_a": dict(color="#8C8C8C", ls="--", marker="s", label="WikiText lens")}
CORPORA = ["gsm8k", "wikitext_a"]
LAM = [0, .125, .25, .5, .75, 1, 1.5, 2]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "legend.fontsize": 7,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "mathtext.fontset": "dejavusans",
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#444444",
    "xtick.color": INK, "ytick.color": INK, "axes.linewidth": .6,
    "xtick.major.width": .6, "ytick.major.width": .6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.3, "lines.markersize": 3.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def letter(ax, s, x=-.16, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontweight="bold", fontsize=8.5, va="bottom", ha="left")


def ygrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=.5)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=400 if ext == "png" else None)
    plt.close(fig)
    print("built", name)


def pct(v):
    return 100 * np.asarray(v, float)


# ---------------------------------------------------------------- data
def flexible():
    """(corpus, arm) -> success, leakage arrays over 144 trials; groups = base prompt."""
    rows = s58.c9.load(s58.PATTERN, 2, 15)
    s58.c9.regrade(rows)
    ok = {t.prompt for t, r in rows if r["arm"] == "baseline" and r["hit_re"]}
    out = {}
    for corp in CORPORA:
        for arm in ("full", "off"):
            sel = sorted(((t, r) for t, r in rows if r["arm"] == f"swap_{corp}_{arm}"
                          and r["strength"] == 1.0 and t.prompt in ok),
                         key=lambda p: (p[0].prompt, p[0].target_arg))
            assert len(sel) == 144, len(sel)
            succ = np.array([bool(r["hit_re"]) for _, r in sel], float)
            leak = np.array([t.target_arg.casefold() in (r.get("generated") or "").casefold() for t, r in sel], float)
            groups = [f"{t.category}/{t.func}/{t.source_arg}" for t, _ in sel]
            out[corp, arm] = (succ, leak, groups)
    assert out["gsm8k", "full"][0].sum() == 46 and out["gsm8k", "off"][0].sum() == 78
    assert out["wikitext_a", "full"][0].sum() == 20 and out["wikitext_a", "off"][0].sum() == 50
    return out


def latent():
    """(corpus, arm) -> success, leakage arrays over 194 items; groups = intermediate entity."""
    recs = {}
    for corp in CORPORA:
        for path in (s58.DEV[corp], s58.NEW[corp]):
            for r in json.loads(path.read_text())["records"]:
                if r["strength"] == 1.0 and r["arm"] in ("swap_full", "swap_off_matched"):
                    recs[corp, r["arm"], r["name"]] = r
    out = {}
    for corp in CORPORA:
        names = sorted(n for (cc, a, n) in recs if cc == corp and a == "swap_full")
        assert len(names) == 194
        for arm, key in (("full", "swap_full"), ("off", "swap_off_matched")):
            rs = [recs[corp, key, n] for n in names]
            out[corp, arm] = (np.array([r["hit_swap_answer"] for r in rs], float),
                              np.array([r["emitted_substring"] for r in rs], float),
                              [r["intermediate"] for r in rs])
    assert out["gsm8k", "full"][0].sum() == 62 and out["gsm8k", "off"][0].sum() == 92
    assert out["gsm8k", "full"][1].sum() == 67 and out["gsm8k", "off"][1].sum() == 18
    assert out["wikitext_a", "full"][0].sum() == 68 and out["wikitext_a", "off"][0].sum() == 96
    return out


def interp():
    out = {}
    for pos, stem in (("prompt", "swap_c37lambdaprompt_qwen3-8b_034da26ce9"),
                      ("prefix", "swap_c37lambdaprefix_qwen3-8b_75f74629d0")):
        rows, conds = b.load_concept(stem)
        a = b.unit_arrays(rows, conds)
        assert len(a["keys"]) == 661 and a["n_queries"] == 1339
        out[pos] = a
    return out


def lam_series(a, corp, metric):
    ids = [b.col(a, corp, f"lam{x:g}_cn", 1) for x in LAM]
    return a[metric][:, ids]


# ---------------------------------------------------------------- Fig: filtering
def fig_filtering(F, L, P):
    fig, axs = plt.subplots(1, 3, figsize=(W, 62 * MM), sharey=True)
    fig.subplots_adjust(left=.085, right=.985, bottom=.25, top=.88, wspace=.16)
    sets = [("Flexible generalization", "144 trials", F),
            ("Latent intermediates", "194 items", L),
            ("Held-out concept panel", "661 units", None)]
    for j, (title, n, data) in enumerate(sets):
        ax = axs[j]
        ygrid(ax)
        for corp in CORPORA:
            st = LENS[corp]
            for metric, color in (("use", USE), ("leak", LEAK)):
                if data is None:
                    a = P["prompt"]
                    v = np.stack([a[metric][:, b.col(a, corp, f"lam{x:g}_cn", 1)] for x in (1, 0)], 1)
                    groups = a["groups"]
                else:
                    k = 0 if metric == "use" else 1
                    v = np.stack([data[corp, "full"][k], data[corp, "off"][k]], 1)
                    groups = data[corp, "full"][2]
                m, lo, hi = b.stats(v, groups, f"v3_filter_{j}_{corp}_{metric}")
                x = np.array([0, 1]) + (-.04 if corp == "gsm8k" else .04)
                ax.errorbar(x, pct(m), yerr=[pct(m - lo), pct(hi - m)], color=color, ls=st["ls"],
                            marker=st["marker"], ms=3.4, lw=1.2, elinewidth=.7, capsize=1.6,
                            mfc=color if corp == "gsm8k" else "white", mec=color, zorder=3)
        ax.set_xticks([0, 1], ["Full", "Off-diagonal"])
        ax.set_xlim(-.35, 1.35)
        ax.set_ylim(0, 85)
        ax.set_yticks([0, 20, 40, 60, 80])
        ax.set_title(f"{title}\n({n})", loc="left", pad=3, fontsize=7.1, linespacing=1.15)
        letter(ax, "abc"[j], x=-.10 if j else -.30, y=1.13)
    axs[0].set_ylabel("Share of swaps (%)")
    handles = [Line2D([], [], color=USE, lw=1.4, label="Success"),
               Line2D([], [], color=LEAK, lw=1.4, label="Leakage"),
               Line2D([], [], color=MUTED, ls="-", marker="o", ms=3.4, label="GSM8K lens"),
               Line2D([], [], color=MUTED, ls="--", marker="s", mfc="white", ms=3.4, label="WikiText lens")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, .0), ncol=4, handlelength=2.4,
               columnspacing=1.6)
    save(fig, "fig_filtering")


# ---------------------------------------------------------------- Fig: lambda
def fig_lambda(P):
    fig, axs = plt.subplots(2, 2, figsize=(W, 92 * MM), sharex=True, sharey=True)
    fig.subplots_adjust(left=.135, right=.985, bottom=.10, top=.86, wspace=.10, hspace=.30)
    rows = [("prompt", "Whole prompt edited"), ("prefix", "Prefix only edited")]
    for i, (pos, rlab) in enumerate(rows):
        for j, corp in enumerate(CORPORA):
            ax = axs[i, j]
            ygrid(ax)
            ax.axvline(1, color="#BBBBBB", lw=.6, ls=(0, (2, 2)), zorder=0)
            for metric, color in (("use", USE), ("leak", LEAK)):
                v = lam_series(P[pos], corp, metric)
                m, lo, hi = b.stats(v, P[pos]["groups"], f"v3_lambda_{pos}_{corp}_{metric}")
                ax.fill_between(LAM, pct(lo), pct(hi), color=color, alpha=.14, lw=0)
                ax.plot(LAM, pct(m), color=color, marker="o", ms=2.8, lw=1.2)
            ax.set_ylim(0, 90)
            ax.set_yticks([0, 20, 40, 60, 80])
            ax.set_xlim(-.08, 2.08)
            ax.set_xticks([0, .5, 1, 1.5, 2], ["0", "0.5", "1", "1.5", "2"])
            if i == 0:
                ax.set_title(LENS[corp]["label"], fontsize=7.5, pad=4)
            if i == 1:
                ax.set_xlabel(r"Diagonal weight $\lambda$")
            if j == 0:
                ax.set_ylabel("Share of queries (%)")
                ax.text(-.30, .5, rlab, transform=ax.transAxes, rotation=90, ha="center", va="center",
                        fontsize=7.5, fontweight="bold")
            letter(ax, "abcd"[2 * i + j], x=-.16 if j == 0 else -.06)
    for ax in axs[0]:
        ax.text(1, 88, r"full lens", fontsize=6.2, color=MUTED, ha="center", va="top",
                bbox=dict(facecolor="white", edgecolor="none", pad=.6))
    handles = [Line2D([], [], color=USE, marker="o", ms=2.8, lw=1.2, label="Relational use"),
               Line2D([], [], color=LEAK, marker="o", ms=2.8, lw=1.2, label="Leakage")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.56, 1.0), ncol=2, columnspacing=2)
    save(fig, "fig_lambda")


# ---------------------------------------------------------------- Fig: competition
def fig_competition():
    blobs = [b.read(f"results/concept/gradgeom_c39gradprompt_qwen3-8b_ec4eca685b_shard{i}of2.json") for i in range(2)]
    rows = sorted([r for bb in blobs for r in bb["records"]], key=lambda r: r["trial"])
    assert len(rows) == 1339
    cond = [tuple(x) for x in blobs[0]["conditions"]]
    groups = [r["type"] + "/" + r["source"] for r in rows]

    fig = plt.figure(figsize=(W, 100 * MM))
    gs = fig.add_gridspec(2, 2, left=.10, right=.86, bottom=.09, top=.94, hspace=.50, wspace=.34)
    arrays = {}
    for j, (F, lab) in enumerate((("R", r"Relational margin $\Delta R$ (nats)"),
                                  ("L", r"Leakage margin $\Delta L$ (nats)"))):
        ax = fig.add_subplot(gs[0, j])
        ygrid(ax)
        for corp in CORPORA:
            ids = [cond.index((corp, f"lam{x:g}_cn")) for x in LAM]
            obs = np.array([[r[f"d{F}"][i] for i in ids] for r in rows])
            pred = np.array([[r[f"hat_applied_{F}"][i] for i in ids] for r in rows])
            arrays[corp, F] = (obs, pred)
            m, lo, hi = b.stats(obs, groups, f"v3_margin_{corp}_{F}")
            st = LENS[corp]
            ax.fill_between(LAM, lo, hi, color=st["color"], alpha=.10, lw=0)
            ax.plot(LAM, m, color=st["color"], ls=st["ls"], marker=st["marker"], ms=2.8, lw=1.2,
                    label=st["label"])
        ax.set_xlim(-.08, 2.08)
        ax.set_xticks([0, .5, 1, 1.5, 2], ["0", "0.5", "1", "1.5", "2"])
        ax.set_xlabel(r"Diagonal weight $\lambda$")
        ax.set_ylabel(lab)
        if F == "R":
            ax.set_ylim(20, 40)
            ax.set_yticks([20, 25, 30, 35, 40])
        else:
            ax.set_ylim(0, 12)
            ax.set_yticks([0, 3, 6, 9, 12])
        letter(ax, "ab"[j], x=-.24)
        if j == 0:
            ax.legend(loc="lower right", handlelength=2.4)
    cmap = LinearSegmentedColormap.from_list("seq", ["#F7DCCB", "#E89B6C", LEAK, "#7A3500"])
    for j, corp in enumerate(CORPORA):
        ax = fig.add_subplot(gs[1, j])
        obs, pred = arrays[corp, "L"]
        x = (pred[:, 1:] - pred[:, [0]]).ravel()
        y = (obs[:, 1:] - obs[:, [0]]).ravel()
        r = float(np.corrcoef(x, y)[0, 1])
        ax.hexbin(x, y, gridsize=32, extent=(-15, 30, -15, 30), mincnt=1, cmap=cmap,
                  norm=LogNorm(1, 1000), linewidths=0)
        ax.plot([-15, 30], [-15, 30], color="#555555", lw=.7, ls=(0, (3, 2)))
        ax.axhline(0, color=GRID, lw=.5, zorder=0)
        ax.axvline(0, color=GRID, lw=.5, zorder=0)
        ax.set(xlim=(-15, 30), ylim=(-15, 30), xticks=[-10, 0, 10, 20, 30], yticks=[-10, 0, 10, 20, 30])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"Predicted $\Delta L$ (nats)")
        ax.set_ylabel(r"Observed $\Delta L$ (nats)")
        ax.set_title(LENS[corp]["label"], fontsize=7.3, pad=3)
        ax.text(.04, .95, f"$r$ = {r:.2f}", transform=ax.transAxes, va="top", fontsize=6.8)
        letter(ax, "cd"[j], x=-.30)
    cax = fig.add_axes([.90, .11, .014, .29])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=LogNorm(1, 1000), cmap=cmap), cax=cax, ticks=[1, 10, 100, 1000])
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=6.2, length=2)
    cb.outline.set_linewidth(.5)
    cb.set_label("Points per bin", fontsize=6.6)
    save(fig, "fig_competition")


# ---------------------------------------------------------------- Fig: corpus
def fig_corpus():
    rep, aq = c.corpus_rows("tier0"), c.corpus_rows("aqua")
    specs = [("WikiText", "wikirep", rep), ("GSM8K", "gsm8k", rep), ("AQuA-RAT", "aqua_rat", aq)]
    expected = [[[1, 0, 1, 0, 2], [39, 41, 41, 41, 42]], [[10, 8, 8, 8, 9], [44, 42, 41, 39, 43]],
                [[14, 12, 12, 15, 10], [41, 42, 38, 42, 41]]]
    fig, axs = plt.subplots(1, 2, figsize=(W, 50 * MM), sharey=True)
    fig.subplots_adjust(left=.10, right=.98, bottom=.17, top=.88, wspace=.10)
    for i, (name, prefix, fn) in enumerate(specs):
        counts = [[fn(f"swap_{prefix}_r{k}", numeric)[0] for k in range(5)] for numeric in (True, False)]
        assert counts == expected[i], (name, counts)
        for j, denom in enumerate((39, 105)):
            ax = axs[j]
            vals = 100 * np.array(counts[j]) / denom
            for v in sorted(set(vals)):
                n = int((vals == v).sum())
                xs = i + (np.linspace(-.12, .12, n) if n > 1 else np.zeros(1))
                ax.scatter(xs, [v] * n, s=13, color=INK, zorder=3, linewidths=0)
            ax.plot([i - .22, i + .22], [vals.mean()] * 2, color=INK, lw=.9, zorder=2)
    for j, (ax, title) in enumerate(zip(axs, ("Numerical arguments (39 trials)",
                                               "Other arguments (105 trials)"))):
        ygrid(ax)
        ax.set_xticks([0, 1, 2], [s[0] for s in specs])
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-.5, 2.5)
        ax.set_ylim(0, 50)
        ax.set_yticks([0, 10, 20, 30, 40, 50])
        ax.set_title(title, loc="left", pad=3, fontsize=7.3)
        ax.set_xlabel("Fitting corpus")
        letter(ax, "ab"[j], x=-.20 if j == 0 else -.06)
    axs[0].set_ylabel("Swap success (%)")
    save(fig, "fig_corpus")


# ---------------------------------------------------------------- Fig: outcome breakdown (appendix)
def fig_breakdown():
    """Per-unit shares of success / leakage-without-success / other over lambda."""
    from collections import defaultdict
    stems = (("prompt", "swap_c37lambdaprompt_qwen3-8b_034da26ce9"),
             ("prefix", "swap_c37lambdaprefix_qwen3-8b_75f74629d0"))
    fig, axs = plt.subplots(2, 2, figsize=(W, 92 * MM), sharex=True, sharey=True)
    fig.subplots_adjust(left=.135, right=.985, bottom=.10, top=.86, wspace=.10, hspace=.30)
    for i, (pos, stem) in enumerate(stems):
        rows, conds = b.load_concept(stem)
        units = defaultdict(list)
        for r in rows:
            if r["kind"] == "use" and r["clue"] in ("0", "1") and r.get("filler", 0) == 0:
                units[r["key"]].append(r)
        ref = b.unit_arrays(rows, conds)
        keys = ref["keys"]
        assert len(keys) == 661
        for j, corp in enumerate(CORPORA):
            ax = axs[i, j]
            ids = [b.col(ref, corp, f"lam{x:g}_cn", 1) for x in LAM]
            succ, leak = [], []
            for k in keys:
                hit = np.array([r["hit_t"] for r in units[k]], float)[:, ids]
                name = np.array([r["name_t"] for r in units[k]], float)[:, ids]
                succ.append(hit.mean(0))
                leak.append(((name > 0) & (hit == 0)).mean(0))
            succ, leak = 100 * np.mean(succ, 0), 100 * np.mean(leak, 0)
            assert np.allclose(succ, 100 * ref["use"][:, ids].mean(0))
            other = 100 - succ - leak
            x = np.arange(len(LAM))
            ax.bar(x, succ, width=.72, color=USE, edgecolor="white", linewidth=.6)
            ax.bar(x, leak, width=.72, bottom=succ, color=LEAK, edgecolor="white", linewidth=.6)
            ax.bar(x, other, width=.72, bottom=succ + leak, color="#D9D9D9", edgecolor="white", linewidth=.6)
            ax.set_xticks(x, ["0", ".125", ".25", ".5", ".75", "1", "1.5", "2"])
            ax.tick_params(axis="x", length=0, labelsize=6.3)
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 25, 50, 75, 100])
            if i == 0:
                ax.set_title(LENS[corp]["label"], fontsize=7.5, pad=4)
            if i == 1:
                ax.set_xlabel(r"Diagonal weight $\lambda$")
            if j == 0:
                ax.set_ylabel("Share of queries (%)")
                ax.text(-.30, .5, ["Whole prompt edited", "Prefix only edited"][i], transform=ax.transAxes,
                        rotation=90, ha="center", va="center", fontsize=7.5, fontweight="bold")
            letter(ax, "abcd"[2 * i + j], x=-.16 if j == 0 else -.06)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=USE, label="Success"), Patch(color=LEAK, label="Leakage, answer incorrect"),
                        Patch(color="#D9D9D9", label="Other")],
               loc="upper center", bbox_to_anchor=(.56, 1.0), ncol=3, columnspacing=1.6, handlelength=1.2)
    save(fig, "fig_breakdown")


# ---------------------------------------------------------------- upload folder
def collect():
    tex = (OVERLEAF / "main.tex").read_text()
    names = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{figures/([^}]+)\}", tex)
    if UPLOAD.exists():
        shutil.rmtree(UPLOAD)
    UPLOAD.mkdir()
    for n in names:
        stem = Path(n).stem
        src = next((OUT / f"{stem}{ext}" for ext in (".pdf", ".png") if (OUT / f"{stem}{ext}").exists()), None)
        assert src is not None, f"missing figure for {n}"
        shutil.copy(src, UPLOAD / src.name)
    print("upload folder:", sorted(p.name for p in UPLOAD.iterdir()))


if __name__ == "__main__":
    P = interp()
    fig_filtering(flexible(), latent(), P)
    fig_lambda(P)
    fig_competition()
    fig_corpus()
    fig_breakdown()
    if "--collect" in sys.argv:
        collect()

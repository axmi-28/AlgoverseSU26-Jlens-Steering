"""C39 report: do relational / leakage gradients explain the lambda trend?

    python scripts/55_gradgeom_report.py

Mentor spec, Experiment 2 (sections 4.4-4.8). Reads the ``gradgeom_c39grad*``
runs and writes

    results/c39_gradgeom_qwen3-8b.md
    results/figures/fig_c39*.png

Per (trial, lambda): F_hat = sum over sites of the first-order term (applied
delta, and the spec's Householder form), dF at full strength, dF at eps
strength / eps. Primary test: within swap, dpred = F_hat(l) - F_hat(0) against
dobs = dF(l) - dF(0), lambda != 0; cluster bootstrap over source entities.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONCEPT = ROOT / "results/concept"
FIG = ROOT / "results/figures"
LAMS = [0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
N_BOOT = 1000
SEED = 0
TITLES = {"gsm8k": "GSM8K-fitted lens", "wikitext_a": "WikiText-fitted lens"}
WHERE = {"prompt": "edit at every prompt position", "prefix": "edit on the prefix only"}
SURFACE, INK, SECOND, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
USE, EMIT = "#2a78d6", "#eb6834"


def load(label):
    shards = sorted(CONCEPT.glob(f"gradgeom_{label}_qwen3-8b_*_shard*of*.json"))
    n = int(shards[0].stem.split("of")[-1])
    if len(shards) != n:
        raise SystemExit(f"{label}: {len(shards)} of {n} shards")
    blobs = [json.loads(p.read_text()) for p in shards]
    recs = [r for b in blobs for r in b["records"]]
    if len({r["trial"] for r in recs}) != len(recs):
        raise SystemExit("duplicate trials")
    conds = [tuple(c) for c in blobs[0]["conditions"]]
    rows = []
    for r in recs:
        for ci, (c, a) in enumerate(conds):
            row = {"trial": r["trial"], "key": r["key"], "relation": r["relation"], "source": r["source"],
                   "target": r["target"], "corpus": c, "arm": a, "lam": float(a[3:-3]),
                   "clean_tok": all(v == 1 for v in r["ans_len"].values())}
            for F in "RL":
                row[f"d{F}"] = r[f"d{F}"][ci]
                row[f"d{F}_eps"] = r[f"d{F}_eps"][ci]
                row[f"hat_applied_{F}"] = r[f"hat_applied_{F}"][ci]
                row[f"hat_house_{F}"] = r[f"hat_house_{F}"][ci]
                row[f"A_{F}"] = r[f"A_{F}"][ci]
            rows.append(row)
    return rows, len(recs), sum(sum(b["dead_rows"].values()) for b in blobs)


def c37_labels(label):
    out = {}
    for p in sorted(CONCEPT.glob(f"swap_{label}_qwen3-8b_*_shard*of*.json")):
        b = json.loads(p.read_text())
        conds = [tuple(c) for c in b["conditions"]]
        for r in b["records"]:
            if r["kind"] == "use":
                for k, (c, a, d) in enumerate(conds):
                    if d == 1.0 and a.startswith("lam") and a.endswith("_cn"):
                        out[r["key"], r["relation"], c, a] = (float(r["hit_t"][k]), float(r["name_t"][k]))
    return out


def c38_outcomes(label):
    out = {}
    for p in sorted(CONCEPT.glob(f"magnitude_{label}_qwen3-8b_*_shard*of*.json")):
        b = json.loads(p.read_text())
        conds = [tuple(c) for c in b["conditions"]]
        for r in b["records"]:
            for ci, (c, a) in enumerate(conds):
                R = r["t"][ci] - r["s"][ci] - (r["clean"]["t"] - r["clean"]["s"])
                out[r["trial"], c, a] = R
    return out


def within(rows, pred, obs):
    """Pairs (dpred, dobs, cluster) relative to lambda=0 within each trial."""
    by = defaultdict(dict)
    for r in rows:
        by[r["trial"]][r["lam"]] = r
    out = []
    for t, d in by.items():
        if 0.0 not in d:
            continue
        for l, r in d.items():
            if l == 0.0:
                continue
            out.append((r[pred] - d[0.0][pred], r[obs] - d[0.0][obs], r["source"], t))
    return out


def stats(pairs):
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    corr = float(np.corrcoef(x, y)[0, 1])
    slope = float((x @ y) / (x @ x))
    big = np.abs(y) > 0.1
    sign = float(np.mean(np.sign(x[big]) == np.sign(y[big])))
    mae = float(np.median(np.abs(x - y)))
    # Spearman as a scale-free companion
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    rho = float(np.corrcoef(rx, ry)[0, 1])
    return corr, rho, slope, sign, mae


def boot(pairs):
    cl = defaultdict(list)
    for p in pairs:
        cl[p[2]].append(p)
    keys = sorted(cl)
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(keys), len(keys))
        draws.append(stats([p for j in pick for p in cl[keys[j]]]))
    d = np.array(draws)
    return stats(pairs), np.percentile(d, 2.5, 0), np.percentile(d, 97.5, 0)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=12)


def fig_alignment(data, fname):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.0), facecolor=SURFACE, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.70, bottom=0.12, wspace=0.1)
    for ax, corpus in zip(axes, ("gsm8k", "wikitext_a")):
        style(ax)
        rows = [r for r in data["prompt"][0] if r["corpus"] == corpus]
        for F, color, name in (("R", USE, "relational gradient  |g_Rᵀw| / ‖g_R‖"),
                               ("L", EMIT, "leakage gradient  |g_Lᵀw| / ‖g_L‖")):
            m = [np.mean([r[f"A_{F}"] for r in rows if r["lam"] == l]) for l in LAMS]
            ax.plot(LAMS, m, color=color, linewidth=2.6, marker="o", markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.4, label=name)
        ax.set_title(TITLES[corpus], color=INK, fontsize=14, fontweight="bold", loc="left")
        ax.set_xlabel("λ", color=SECOND, fontsize=13)
        ax.set_xticks([0, 0.5, 1, 1.5, 2])
    axes[0].set_ylabel("mean alignment over sites", color=SECOND, fontsize=12)
    axes[0].legend(frameon=False, fontsize=12, loc="best", labelcolor=SECOND)
    fig.suptitle("Where each λ axis points relative to the two objectives", color=INK, fontsize=18,
                 fontweight="bold", x=0.08, ha="left", y=0.955)
    fig.text(0.08, 0.875, "Sign-invariant cosine between the Householder axis w_λ and the clean-state gradient, "
             "averaged over band layers and edited positions.\nEdit at every prompt position; the prefix "
             "convention differs only in which positions are averaged.", color=SECOND, fontsize=12,
             ha="left", va="top", linespacing=1.6)
    fig.savefig(FIG / fname, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_scatter(data, where, pred, fname, eps=False):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11.5), facecolor=SURFACE)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.87, bottom=0.07, hspace=0.32, wspace=0.25)
    for col, corpus in enumerate(("gsm8k", "wikitext_a")):
        rows = [r for r in data[where][0] if r["corpus"] == corpus]
        for row, F in enumerate("RL"):
            ax = axes[row, col]
            style(ax)
            obs = f"d{F}_eps" if eps else f"d{F}"
            pr = within(rows, f"{pred}_{F}", obs)
            x = np.array([p[0] for p in pr])
            y = np.array([p[1] for p in pr])
            lim = np.percentile(np.abs(np.concatenate([x, y])), 99)
            ax.scatter(x, y, s=6, alpha=0.18, color=USE if F == "R" else EMIT, linewidths=0, rasterized=True)
            ax.plot([-lim, lim], [-lim, lim], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
            ax.axhline(0, color=GRID, linewidth=1)
            ax.axvline(0, color=GRID, linewidth=1)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            c, rho, *_ = stats(pr)
            name = "relational R" if F == "R" else "leakage L"
            ax.set_title(f"{TITLES[corpus].replace('-fitted', '')}, {name}", color=INK, fontsize=13,
                         fontweight="bold", loc="left")
            ax.text(0.03, 0.95, f"r = {c:.2f}   ρ = {rho:.2f}", transform=ax.transAxes, color=SECOND,
                    fontsize=12, va="top")
            ax.set_xlabel("predicted change from λ=0  (first-order, nats)", color=SECOND, fontsize=12)
            ax.set_ylabel(("local" if eps else "observed") + " change from λ=0  (nats)", color=SECOND, fontsize=12)
    title = ("Gradient check: first-order prediction vs response at 1% strength" if eps
             else "Does the gradient geometry predict the within-swap λ effect?")
    fig.suptitle(title, color=INK, fontsize=17, fontweight="bold", x=0.09, ha="left", y=0.965)
    fig.text(0.09, 0.915, f"{WHERE[where].capitalize()}. One point per (swap, λ≠0). Dashed line: perfect calibration. "
             "Axes clipped at the 99th percentile of |value|.", color=SECOND, fontsize=12, ha="left")
    fig.savefig(FIG / fname, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    data = {w: load(f"c39grad{w}") for w in ("prompt", "prefix")}
    L = ["# C39: relational vs leakage gradient geometry across λ (Qwen3-8B)", ""]
    L.append("Mentor spec, Experiment 2. Objectives, teacher-forced on the use query: "
             "R = lp(target answer) − lp(source answer), L = lp(target name) − lp(target answer). "
             "Gradients g_R, g_L on the clean graph at every band hook. F̂ = Σ_sites g_Fᵀδ, with δ the applied "
             "(row-matched) delta; the Householder form −2(wᵀh⁰)(g_Fᵀw) is reported alongside.")
    L.append("")
    L.append("## Checks")
    L.append("")
    for w in ("prompt", "prefix"):
        rows, n, dead = data[w]
        c38 = c38_outcomes(f"c38mag{w}")
        a = [(r["dR"], c38[r["trial"], r["corpus"], r["arm"]]) for r in rows if (r["trial"], r["corpus"], r["arm"]) in c38]
        cr = np.corrcoef(*zip(*a))[0, 1]
        L.append(f"- {w}: {n} trials, {len(rows)} trial×cond rows, dead rows {dead}; dR vs C38 corr {cr:.5f} "
                 f"over {len(a)} rows.")
    L.append("")
    L.append("### Gradient gate (float32, central differences; replaces the bf16 ε column)")
    L.append("")
    L.append("40-unit hash subsample per convention, all 16 conditions. F̂ = Σ g·δ in float32 vs "
             "(F(+εδ) − F(−εδ))/2ε in one batch; last column: C39's bf16 F̂ vs float32 F̂, same trials.")
    L.append("")
    L.append("| convention | F | trials | ε=0.01 corr / slope | ε=0.03 | ε=0.1 | bf16 F̂ vs fp32 F̂ corr / slope |")
    L.append("|---|---|---|---|---|---|---|")
    for w in ("prompt", "prefix"):
        b = json.loads(next(CONCEPT.glob(f"gradcheck_c39check{w}_qwen3-8b_*_shard0of1.json")).read_text())
        main = {}
        for pth in CONCEPT.glob(f"gradgeom_c39grad{w}_qwen3-8b_*_shard*of*.json"):
            for r in json.loads(pth.read_text())["records"]:
                main[r["trial"]] = r
        for F, (a, c) in (("R", ("t", "s")), ("L", ("n", "t"))):
            hat = np.concatenate([np.array(r["hat"][a]) - np.array(r["hat"][c]) for r in b["records"]])
            cells = []
            for e in b["eps"]:
                fd = np.concatenate([np.array(r["fd"][f"{a}@{e}"]) - np.array(r["fd"][f"{c}@{e}"]) for r in b["records"]])
                cells.append(f"{np.corrcoef(hat, fd)[0, 1]:.4f} / {(hat @ fd) / (hat @ hat):.3f}")
            bf = np.concatenate([np.array(main[r["trial"]][f"hat_applied_{F}"]) for r in b["records"]])
            cells.append(f"{np.corrcoef(hat, bf)[0, 1]:.4f} / {(hat @ bf) / (hat @ hat):.3f}")
            L.append(f"| {w} | {F} | {len(b['records'])} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("### (Superseded) bf16 ε = 0.01 column: quantised, kept for the record")
    L.append("")
    L.append("| convention | lens | F | corr(F̂_applied, dF_ε) | slope | median |F̂ − dF_ε| | median |dF_ε| |")
    L.append("|---|---|---|---|---|---|---|")
    for w in ("prompt", "prefix"):
        for corpus in ("gsm8k", "wikitext_a"):
            rows = [r for r in data[w][0] if r["corpus"] == corpus]
            for F in "RL":
                x = np.array([r[f"hat_applied_{F}"] for r in rows])
                y = np.array([r[f"d{F}_eps"] for r in rows])
                L.append(f"| {w} | {TITLES[corpus]} | {F} | {np.corrcoef(x, y)[0, 1]:.3f} | "
                         f"{(x @ y) / (x @ x):.2f} | {np.median(np.abs(x - y)):.3f} | {np.median(np.abs(y)):.3f} |")
    L.append("")
    L.append("### Level comparison at full strength: corr(F̂, dF)")
    L.append("")
    L.append("| convention | lens | F | F̂_applied | F̂_house |")
    L.append("|---|---|---|---|---|")
    for w in ("prompt", "prefix"):
        for corpus in ("gsm8k", "wikitext_a"):
            rows = [r for r in data[w][0] if r["corpus"] == corpus]
            for F in "RL":
                y = np.array([r[f"d{F}"] for r in rows])
                cs = [np.corrcoef([r[f"hat_{k}_{F}"] for r in rows], y)[0, 1] for k in ("applied", "house")]
                L.append(f"| {w} | {TITLES[corpus]} | {F} | {cs[0]:.3f} | {cs[1]:.3f} |")
    L.append("")
    for w in ("prompt", "prefix"):
        L.append(f"## {WHERE[w].capitalize()}")
        L.append("")
        L.append("### Means per λ (over trials)")
        L.append("")
        for corpus in ("gsm8k", "wikitext_a"):
            rows = [r for r in data[w][0] if r["corpus"] == corpus]
            L.append(f"**{TITLES[corpus]}**")
            L.append("")
            L.append("| λ | A_R | A_L | F̂_R | F̂_L | dR (ε, ÷ε) | dL (ε, ÷ε) | dR full | dL full |")
            L.append("|---|---|---|---|---|---|---|---|---|")
            for l in LAMS:
                rr = [r for r in rows if r["lam"] == l]
                m = lambda k: np.mean([r[k] for r in rr])  # noqa: E731
                L.append(f"| {l:g} | {m('A_R'):.4f} | {m('A_L'):.4f} | {m('hat_applied_R'):+.2f} | "
                         f"{m('hat_applied_L'):+.2f} | {m('dR_eps'):+.2f} | {m('dL_eps'):+.2f} | "
                         f"{m('dR'):+.2f} | {m('dL'):+.2f} |")
            L.append("")
        L.append("### Primary within-swap test (§4.6): Δpred = F̂_λ − F̂_0 vs Δobs = dF_λ − dF_0")
        L.append("")
        L.append("95% cluster-bootstrap CIs over source entities. Sign agreement on pairs with |Δobs| > 0.1 nats; "
                 "calibration = median |Δpred − Δobs|.")
        L.append("")
        L.append("| lens | F | predictor | observed | Pearson r | Spearman ρ | slope | sign agree | calib. err |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for corpus in ("gsm8k", "wikitext_a"):
            rows = [r for r in data[w][0] if r["corpus"] == corpus]
            for F in "RL":
                for pred, obs in ((f"hat_applied_{F}", f"d{F}"), (f"hat_house_{F}", f"d{F}"),
                                  (f"A_{F}", f"d{F}")):
                    (c, rho, sl, sg, mae), lo, hi = boot(within(rows, pred, obs))
                    L.append(f"| {TITLES[corpus]} | {F} | {pred} | {obs} | {c:+.2f} [{lo[0]:+.2f}, {hi[0]:+.2f}] | "
                             f"{rho:+.2f} | {sl:.3f} | {100*sg:.0f}% [{100*lo[3]:.0f}, {100*hi[3]:.0f}] | {mae:.2f} |")
        L.append("")
        L.append("Tokenization-clean subset (target answer, source answer and name all one token), primary predictor:")
        L.append("")
        for corpus in ("gsm8k", "wikitext_a"):
            rows = [r for r in data[w][0] if r["corpus"] == corpus and r["clean_tok"]]
            for F in "RL":
                c, rho, sl, sg, mae = stats(within(rows, f"hat_applied_{F}", f"d{F}"))
                L.append(f"- {TITLES[corpus]} {F}: r {c:+.2f}, ρ {rho:+.2f}, sign {100*sg:.0f}% "
                         f"({len(rows)//len(LAMS)} trials)")
        L.append("")
        L.append("### Free-generation external validity (§4.8): within swap, Δpred vs change in C37 binary labels")
        L.append("")
        lab = c37_labels(f"c37lambda{w}")
        L.append("| lens | Δpred_R vs Δuse-hit | Δpred_L vs Δleak | Δpred_R vs Δleak | Δpred_L vs Δuse-hit |")
        L.append("|---|---|---|---|---|")
        for corpus in ("gsm8k", "wikitext_a"):
            rows = [dict(r, hit=lab[r["key"], r["relation"], r["corpus"], r["arm"]][0],
                         leak=lab[r["key"], r["relation"], r["corpus"], r["arm"]][1])
                    for r in data[w][0] if r["corpus"] == corpus]
            cells = []
            for pred, obs in (("hat_applied_R", "hit"), ("hat_applied_L", "leak"),
                              ("hat_applied_R", "leak"), ("hat_applied_L", "hit")):
                pr = within(rows, pred, obs)
                cells.append(f"{np.corrcoef([p[0] for p in pr], [p[1] for p in pr])[0, 1]:+.2f}")
            L.append(f"| {TITLES[corpus]} | " + " | ".join(cells) + " |")
        L.append("")
    fig_alignment(data, "fig_c39a_alignment.png")
    fig_scatter(data, "prompt", "hat_applied", "fig_c39b_within_swap_prompt.png")
    fig_scatter(data, "prefix", "hat_applied", "fig_c39c_within_swap_prefix.png")
    (ROOT / "results/c39_gradgeom_qwen3-8b.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

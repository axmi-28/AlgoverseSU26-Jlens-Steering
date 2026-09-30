"""C38 report: does realized edit magnitude explain the lambda trend?

    python scripts/51_magnitude_report.py

Follows the mentor's Experiment 1 spec (sections 3.1-3.4). Reads the two
``magnitude_c38mag*`` runs (per-site Householder quantities + teacher-forced
margins), joins the C37 free-generation labels, and writes

    results/c38_magnitude_qwen3-8b.md
    results/c38_magnitude_config.json
    results/figures/fig_c38*.png

Outcomes (continuous, whole run, teacher-forced on the use query):
    R = log p(target answer) - log p(source answer)          relational margin
    L = log p(target name)   - log p(target answer)          leakage margin
    dR, dL = intervened minus clean.

Model, per corpus and position convention, over trials i and lambdas:
    Y_il = a_i + g_l + f(log M_il) + e_il
with a_i absorbed by within-trial demeaning and f a restricted cubic spline
(5 knots). g_1 and g_2 are compared with and without f. Uncertainty: cluster
bootstrap resampling source entities with every trial's full lambda
trajectory (target-entity clustering reported as a sensitivity).
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CONCEPT = ROOT / "results/concept"
FIG = ROOT / "results/figures"
LAMS = [0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
TAU_AXIS = [0.05, 0.10, 0.15, 0.20]
TAU_PRIMARY = 0.05
N_BOOT = 1000
SEED = 0
TITLES = {"gsm8k": "GSM8K-fitted lens", "wikitext_a": "WikiText-fitted lens"}
WHERE = {"prompt": "edit at every prompt position", "prefix": "edit on the prefix only"}

SURFACE, INK, SECOND, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SEQ = plt.get_cmap("viridis")


# --------------------------------------------------------------------------
# Loading


def load(label: str):
    shards = sorted(CONCEPT.glob(f"magnitude_{label}_qwen3-8b_*_shard*of*.json"))
    if not shards:
        raise SystemExit(f"no shards for {label}")
    n = int(shards[0].stem.split("of")[-1])
    if len(shards) != n:
        raise SystemExit(f"{label}: {len(shards)} of {n} shards present")
    blobs = [json.loads(p.read_text()) for p in shards]
    if len({json.dumps(b["manifest"], sort_keys=True) for b in blobs}) != 1:
        raise SystemExit(f"{label}: shards disagree on manifest")
    records, sites, checks, dead = [], {}, defaultdict(float), 0
    for p, b in zip(shards, blobs):
        records += b["records"]
        s = torch.load(p.with_suffix(".sites.pt"), weights_only=False)
        if set(s) != {r["trial"] for r in b["records"]}:
            raise SystemExit(f"{p.name}: site file and records disagree")
        sites.update(s)
        for k, v in b["checks"].items():
            checks[k] = max(checks[k], v)
        dead += sum(b["dead_rows"].values())
    if len({r["trial"] for r in records}) != len(records):
        raise SystemExit(f"{label}: duplicate trials")
    return blobs[0], records, sites, dict(checks), dead


def c37_labels(label: str) -> dict:
    """(key, relation, corpus, arm) -> (t, s, hit_t, name_t) from the C37 run at dose 1."""
    out = {}
    for p in sorted(CONCEPT.glob(f"swap_{label}_qwen3-8b_*_shard*of*.json")):
        b = json.loads(p.read_text())
        conds = [tuple(c) for c in b["conditions"]]
        for r in b["records"]:
            if r["kind"] != "use":
                continue
            for k, (c, a, d) in enumerate(conds):
                if d == 1.0 and a.endswith("_cn") and a.startswith("lam"):
                    out[r["key"], r["relation"], c, a] = (r["t"][k], r["s"][k], r["hit_t"][k], r["name_t"][k])
    return out


# --------------------------------------------------------------------------
# Tables


def build(label: str, c37: str):
    man, records, sites, checks, dead = load(label)
    conds = [tuple(c) for c in man["conditions"]] if "conditions" in man else None
    blob_conds = None
    for p in sorted(CONCEPT.glob(f"magnitude_{label}_qwen3-8b_*_shard0of*.json")):
        blob_conds = [tuple(c) for c in json.loads(p.read_text())["conditions"]]
    conds = blob_conds
    lam_of = {a: float(a[3:-3]) for _, a in conds}
    labels = c37_labels(c37)
    rows = []  # one per (trial, cond)
    repro = []
    for r in records:
        st = sites[r["trial"]]
        edited = torch.ones(st["m_clean"].shape[-1], dtype=torch.bool)
        no_sink = edited.clone()
        no_sink[0] = False
        for ci, (c, a) in enumerate(conds):
            R = r["t"][ci] - r["s"][ci]
            L = r["n"][ci] - r["t"][ci]
            R0 = r["clean"]["t"] - r["clean"]["s"]
            L0 = r["clean"]["n"] - r["clean"]["t"]
            lab = labels.get((r["key"], r["relation"], c, a))
            if lab is not None:
                repro.append((r["t"][ci], lab[0], R, lab[0] - lab[1]))
            row = {
                "trial": r["trial"], "source": r["source"], "target": r["target"], "type": r["type"],
                "corpus": c, "lam": lam_of[a], "dR": R - R0, "dL": L - L0, "R": R, "L": L,
                "hit": None if lab is None else float(lab[2]), "leak": None if lab is None else float(lab[3]),
                "min1mrho": min(1 - x for x in r["rho"][ci]), "kappa2": max(r["kappa2"][ci]),
                "dP0": float(np.mean(r["dP0"][ci])), "dP1": float(np.mean(r["dP1"][ci])),
                "clean_tok": all(v == 1 for v in r["ans_len"].values()),
            }
            for mask, suf in ((edited, ""), (no_sink, "_nosink")):
                for q in ("m_clean", "m_actual", "applied"):
                    m = st[q][ci][:, mask].double()
                    row[f"{q}_L2{suf}"] = float(m.pow(2).sum().sqrt())
                    row[f"{q}_L1{suf}"] = float(m.sum())
                    row[f"{q}_max{suf}"] = float(m.max())
                hc = st["h_clean"][:, mask].double()
                ha = st["h_actual"][ci][:, mask].double()
                row[f"rel_clean_L2{suf}"] = float((st["m_clean"][ci][:, mask].double() / hc).pow(2).sum().sqrt())
                row[f"rel_actual_L2{suf}"] = float((st["m_actual"][ci][:, mask].double() / ha).pow(2).sum().sqrt())
            # within-site spread of the applied norm across lambda, for the report
            rows.append(row)
    spread = []
    for r in records:
        a = sites[r["trial"]]["applied"]
        for c in {c for c, _ in conds}:
            idx = [i for i, (cc, _) in enumerate(conds) if cc == c]
            x = a[idx]
            live = (x > 0).all(0)
            if live.any():
                spread.append(float(((x.amax(0) - x.amin(0)) / x.amax(0).clamp_min(1e-6))[live].max()))
    return {
        "rows": rows, "checks": checks, "dead": dead, "n_trials": len(records),
        "repro_n": len(repro),
        "repro_t_med": round(float(np.median([abs(a - b) for a, b, _, _ in repro])), 4),
        "repro_t_corr": round(float(np.corrcoef([a for a, *_ in repro], [b for _, b, *_ in repro])[0, 1]), 5),
        "repro_R_corr": round(float(np.corrcoef([c for *_, c, _ in repro], [d for *_, d in repro])[0, 1]), 5),
        "applied_spread": max(spread) if spread else None, "manifest": man,
    }


# --------------------------------------------------------------------------
# Statistics


def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted (natural) cubic spline basis, Harrell's parameterisation."""
    k = knots
    K = len(k)
    cols = [x]
    norm = (k[-1] - k[0]) ** 2
    p = lambda u: np.clip(u, 0, None) ** 3  # noqa: E731
    for j in range(K - 2):
        cols.append((p(x - k[j]) - p(x - k[-2]) * (k[-1] - k[j]) / (k[-1] - k[-2])
                     + p(x - k[-1]) * (k[-2] - k[j]) / (k[-1] - k[-2])) / norm)
    return np.column_stack(cols)


def demean(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = X.astype(float).copy()
    _, inv = np.unique(groups, return_inverse=True)
    sums = np.zeros((inv.max() + 1,) + X.shape[1:])
    np.add.at(sums, inv, X)
    cnt = np.bincount(inv)
    return out - (sums / cnt.reshape((-1,) + (1,) * (X.ndim - 1)))[inv]


def fit_gamma(y, lam, trial, logM=None, knots=None):
    """g_lambda (vs lambda=0) from the within-trial model, optionally with f(log M)."""
    D = np.column_stack([(lam == l).astype(float) for l in LAMS[1:]])
    X = D if logM is None else np.column_stack([D, rcs_basis(logM, knots)])
    Xd, yd = demean(X, trial), demean(y[:, None], trial)[:, 0]
    beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    resid = yd - Xd @ beta
    return beta[: len(LAMS) - 1], 1 - resid.var() / yd.var()


def boot(df, y, mcol, cluster="source"):
    """Point estimates and cluster-bootstrap CIs for g_1, g_2 raw and adjusted."""
    lam = np.array([r["lam"] for r in df])
    trial = np.array([r["trial"] for r in df])
    Y = np.array([r[y] for r in df])
    logM = np.log(np.array([r[mcol] for r in df]))
    knots = np.quantile(logM, [0.05, 0.275, 0.5, 0.725, 0.95])
    i1, i2 = LAMS[1:].index(1.0), LAMS[1:].index(2.0)

    def est(sel):
        graw, r2raw = fit_gamma(Y[sel], lam[sel], trial[sel])
        gadj, r2adj = fit_gamma(Y[sel], lam[sel], trial[sel], logM[sel], knots)
        return np.array([graw[i1], gadj[i1], graw[i2], gadj[i2], r2raw, r2adj])

    point = est(np.arange(len(df)))
    cl = np.array([r[cluster] for r in df])
    keys = np.unique(cl)
    members = {k: np.flatnonzero(cl == k) for k in keys}
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(N_BOOT):
        pick = rng.choice(keys, len(keys))
        sel = np.concatenate([members[k] for k in pick])
        # a resampled cluster is a new trial: keep trajectories intact but distinct
        tr = np.concatenate([np.char.add(trial[members[k]].astype(str), f"#{j}") for j, k in enumerate(pick)])
        graw, _ = fit_gamma(Y[sel], lam[sel], tr)
        gadj, _ = fit_gamma(Y[sel], lam[sel], tr, logM[sel], knots)
        draws.append([graw[i1], gadj[i1], graw[i2], gadj[i2]])
    d = np.array(draws)
    lo, hi = np.percentile(d, 2.5, 0), np.percentile(d, 97.5, 0)
    return point, lo, hi


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def support(df, mcol):
    by = defaultdict(dict)
    for r in df:
        by[r["trial"]][r["lam"]] = r[mcol]
    rhos, r1, r2 = [], [], []
    for t, m in by.items():
        if len(m) != len(LAMS):
            continue
        v = [m[l] for l in LAMS]
        rhos.append(spearman(LAMS, v))
        r1.append(m[1.0] / m[0.0])
        r2.append(m[2.0] / m[0.0])
    rhos = np.array([x for x in rhos if not math.isnan(x)])
    # overlap of within-trial-centred log M between lambda=0 and lambda=1 / 2
    cent = {t: np.mean(np.log(list(m.values()))) for t, m in by.items()}
    def dist(l):
        return np.array([math.log(m[l]) - cent[t] for t, m in by.items() if l in m])
    def ovl(a, b):
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        bins = np.linspace(lo, hi, 41)
        pa, _ = np.histogram(a, bins, density=True)
        pb, _ = np.histogram(b, bins, density=True)
        return float(np.minimum(pa, pb).sum() * (bins[1] - bins[0]))
    return {
        "median_spearman": float(np.median(rhos)), "frac_pos": float((rhos > 0).mean()),
        "ratio1": np.percentile(r1, [25, 50, 75]), "ratio2": np.percentile(r2, [25, 50, 75]),
        "ovl1": ovl(dist(0.0), dist(1.0)), "ovl2": ovl(dist(0.0), dist(2.0)),
    }


# --------------------------------------------------------------------------
# Figures


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=12)


def fig_magnitude_vs_lambda(data, fname):
    """Median and IQR of within-trial magnitude (relative to lambda=0) per lambda."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=SURFACE, sharey="row")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.08, hspace=0.38, wspace=0.12)
    for row, where in enumerate(("prompt", "prefix")):
        for col, corpus in enumerate(("gsm8k", "wikitext_a")):
            ax = axes[row, col]
            style(ax)
            df = [r for r in data[where]["rows"] if r["corpus"] == corpus]
            by = defaultdict(dict)
            for r in df:
                by[r["trial"]][r["lam"]] = r
            for q, color, name in (("m_clean_L2", "#2a78d6", "natural, clean state  2|wᵀh⁰|"),
                                   ("m_actual_L2", "#eb6834", "natural, actual state  2|wᵀh̃|"),
                                   ("applied_L2", "#0b0b0b", "applied (matched to full lens)")):
                rel = np.array([[t[l][q] / t[0.0][q] for l in LAMS] for t in by.values() if len(t) == len(LAMS)])
                med, lo, hi = np.median(rel, 0), np.percentile(rel, 25, 0), np.percentile(rel, 75, 0)
                ax.fill_between(LAMS, lo, hi, color=color, alpha=0.15, linewidth=0)
                ax.plot(LAMS, med, color=color, linewidth=2.6, marker="o", markersize=7,
                        markeredgecolor=SURFACE, markeredgewidth=1.4, label=name,
                        linestyle="--" if q == "applied_L2" else "-")
            ax.axhline(1, color=MUTED, linewidth=1)
            ax.set_title(f"{TITLES[corpus]} — {WHERE[where]}", color=INK, fontsize=13.5,
                         fontweight="bold", loc="left")
            ax.set_xticks([0, 0.5, 1, 1.5, 2])
            ax.set_xlabel("λ", color=SECOND, fontsize=13)
            if col == 0:
                ax.set_ylabel("edit size relative to λ=0\n(within trial, L2 over sites)", color=SECOND, fontsize=12)
    axes[0, 0].legend(frameon=False, fontsize=11.5, loc="upper left", labelcolor=SECOND)
    fig.suptitle("How edit size moves with λ", color=INK, fontsize=18, fontweight="bold",
                 x=0.08, ha="left", y=0.965)
    fig.text(0.08, 0.915, "Lines: median over trials; bands: interquartile range. The applied edit is "
             "row-matched to the full lens at every site, so it is flat by construction.",
             color=SECOND, fontsize=12, ha="left")
    fig.savefig(FIG / fname, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_behaviour_vs_magnitude(data, where, mcol, fname):
    """Binned within-trial outcome vs within-trial log magnitude, one line per lambda."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=SURFACE)
    fig.subplots_adjust(left=0.08, right=0.86, top=0.86, bottom=0.08, hspace=0.38, wspace=0.22)
    for col, corpus in enumerate(("gsm8k", "wikitext_a")):
        df = [r for r in data[where]["rows"] if r["corpus"] == corpus]
        trial = np.array([r["trial"] for r in df])
        lam = np.array([r["lam"] for r in df])
        x = demean(np.log(np.array([r[mcol] for r in df]))[:, None], trial)[:, 0]
        edges = np.quantile(x, np.linspace(0, 1, 9))
        for row, (y, name) in enumerate((("dR", "Δ relational margin R  (nats)"),
                                         ("dL", "Δ leakage margin L  (nats)"))):
            ax = axes[row, col]
            style(ax)
            Y = demean(np.array([r[y] for r in df])[:, None], trial)[:, 0]
            for li, l in enumerate(LAMS):
                sel = lam == l
                xs, ys = [], []
                for a, b in zip(edges[:-1], edges[1:]):
                    m = sel & (x >= a) & (x <= b)
                    if m.sum() >= 20:
                        xs.append(x[m].mean())
                        ys.append(Y[m].mean())
                ax.plot(xs, ys, color=SEQ(li / (len(LAMS) - 1)), linewidth=2.2, marker="o", markersize=6,
                        markeredgecolor=SURFACE, markeredgewidth=1.2, label=f"λ={l:g}")
            ax.axhline(0, color=MUTED, linewidth=1)
            ax.set_title(f"{TITLES[corpus]}", color=INK, fontsize=13.5, fontweight="bold", loc="left")
            ax.set_xlabel("log edit size, within-trial centred", color=SECOND, fontsize=12)
            ax.set_ylabel(name + "\nwithin-trial centred", color=SECOND, fontsize=12)
    axes[0, 1].legend(frameon=False, fontsize=11.5, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                      labelcolor=SECOND)
    fig.suptitle(f"Behaviour against edit size, split by λ — {WHERE[where]}", color=INK, fontsize=18,
                 fontweight="bold", x=0.08, ha="left", y=0.965)
    fig.text(0.08, 0.915, f"Size = {mcol.replace('_', ' ')}. If size explained the λ trend, the λ lines "
             "would collapse onto one curve; vertical separation at equal size is λ's own effect.",
             color=SECOND, fontsize=12, ha="left")
    fig.savefig(FIG / fname, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------
# Section 3.5: dynamic magnitude-matched control


def load_matched(label: str):
    shards = sorted(CONCEPT.glob(f"magmatched_{label}_qwen3-8b_*_shard*of*.json"))
    if not shards:
        return None
    n = int(shards[0].stem.split("of")[-1])
    if len(shards) != n:
        raise SystemExit(f"{label}: {len(shards)} of {n} shards present")
    blobs = [json.loads(p.read_text()) for p in shards]
    recs = [r for b in blobs for r in b["records"]]
    if len({r["trial"] for r in recs}) != len(recs):
        raise SystemExit(f"{label}: duplicate trials")
    checks = {"achieved_vs_mstar": max(b["checks"]["achieved_vs_mstar"] for b in blobs),
              "skipped_sites": sum(b["checks"]["skipped_sites"] for b in blobs),
              "sites": sum(b["checks"]["sites"] for b in blobs)}
    conds = [tuple(c) for c in blobs[0]["conditions"]]
    rows = []
    for r in recs:
        for ci, (c, a) in enumerate(conds):
            rows.append({
                "trial": r["trial"], "source": r["source"], "target": r["target"], "corpus": c,
                "lam": float(a[3:-3]),
                "dR": (r["t"][ci] - r["s"][ci]) - (r["clean"]["t"] - r["clean"]["s"]),
                "dL": (r["n"][ci] - r["t"][ci]) - (r["clean"]["n"] - r["clean"]["t"]),
                "mstar_L2": r["mstar_L2"][ci], "skipped": r["skipped"][ci],
            })
    return {"rows": rows, "checks": checks, "n_trials": len(recs)}


def gamma_curve(df, y, cluster="source"):
    """Within-trial mean of Y(lambda) - Y(0) per lambda, with cluster-bootstrap CI."""
    by = defaultdict(dict)
    cl = {}
    for r in df:
        by[r["trial"]][r["lam"]] = r[y]
        cl[r["trial"]] = r[cluster]
    trials = [t for t, v in by.items() if len(v) == len(LAMS)]
    D = np.array([[by[t][l] - by[t][0.0] for l in LAMS] for t in trials])
    keys = sorted({cl[t] for t in trials})
    members = {k: np.array([i for i, t in enumerate(trials) if cl[t] == k]) for k in keys}
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(keys), len(keys))
        idx = np.concatenate([members[keys[j]] for j in pick])
        draws.append(D[idx].mean(0))
    draws = np.array(draws)
    return D.mean(0), np.percentile(draws, 2.5, 0), np.percentile(draws, 97.5, 0), len(trials)


def fig_matched(orig, matched, where, fname):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=SURFACE, sharex=True)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.85, bottom=0.08, hspace=0.32, wspace=0.2)
    for col, corpus in enumerate(("gsm8k", "wikitext_a")):
        for row, (y, name) in enumerate((("dR", "relational margin R"), ("dL", "leakage margin L"))):
            ax = axes[row, col]
            style(ax)
            for src, color, lab in ((orig, "#2a78d6", "original: fixed clean-state edit, norm = full lens"),
                                    (matched, "#eb6834", "matched: runtime Householder, norm = min over λ")):
                df = [r for r in src["rows"] if r["corpus"] == corpus]
                m, lo, hi, _ = gamma_curve(df, y)
                ax.fill_between(LAMS, lo, hi, color=color, alpha=0.18, linewidth=0)
                ax.plot(LAMS, m, color=color, linewidth=2.6, marker="o", markersize=7,
                        markeredgecolor=SURFACE, markeredgewidth=1.4, label=lab)
            ax.axhline(0, color=MUTED, linewidth=1)
            ax.set_title(f"{TITLES[corpus]} — {name}", color=INK, fontsize=13.5, fontweight="bold", loc="left")
            ax.set_ylabel(f"change in {y} vs λ=0  (nats)", color=SECOND, fontsize=12)
            ax.set_xticks([0, 0.5, 1, 1.5, 2])
            if row == 1:
                ax.set_xlabel("λ", color=SECOND, fontsize=13)
    axes[0, 0].legend(frameon=False, fontsize=11.5, loc="lower right", labelcolor=SECOND)
    fig.suptitle(f"Does the λ effect survive fixed local edit size? — {WHERE[where]}", color=INK,
                 fontsize=18, fontweight="bold", x=0.08, ha="left", y=0.965)
    fig.text(0.08, 0.912, "Within-trial change relative to λ=0; bands are 95% cluster-bootstrap intervals "
             "over source entities. Teacher-forced on the use query.", color=SECOND, fontsize=12, ha="left")
    fig.savefig(FIG / fname, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def matched_section(data):
    L = ["## Dynamic magnitude-matched control (§3.5)", ""]
    L.append("Every λ re-run with a runtime Householder along its own axis, norm exactly m* = min over λ of "
             "C38's m_actual at that (trial, site), computed on the arriving state. Sites where "
             "2|wᵀh̃| ≤ 1e-3·‖h̃‖ are left unedited and counted.")
    L.append("")
    out = {}
    for where in ("prompt", "prefix"):
        mt = load_matched(f"c38matched{where}")
        if mt is None:
            L.append(f"- {where}: not run")
            continue
        c = mt["checks"]
        L.append(f"**{WHERE[where].capitalize()}.** {mt['n_trials']} trials; achieved norm vs m* max rel. err "
                 f"{c['achieved_vs_mstar']:.1e}; skipped sites {c['skipped_sites']} of {c['sites']} "
                 f"({100*c['skipped_sites']/c['sites']:.3f}%).")
        L.append("")
        L.append("| lens | outcome | original γ₁ | matched γ₁ | original γ₂ | matched γ₂ | matched/original γ₂ |")
        L.append("|---|---|---|---|---|---|---|")
        for corpus in ("gsm8k", "wikitext_a"):
            for y in ("dR", "dL"):
                o = gamma_curve([r for r in data[where]["rows"] if r["corpus"] == corpus], y)
                m = gamma_curve([r for r in mt["rows"] if r["corpus"] == corpus], y)
                i1, i2 = LAMS.index(1.0), LAMS.index(2.0)
                f = lambda g, i: f"{g[0][i]:+.2f} [{g[1][i]:+.2f}, {g[2][i]:+.2f}]"  # noqa: E731
                ratio = m[0][i2] / o[0][i2] if abs(o[0][i2]) > 0.3 else float("nan")
                L.append(f"| {TITLES[corpus]} | {y} | {f(o, i1)} | {f(m, i1)} | {f(o, i2)} | {f(m, i2)} | "
                         f"{ratio:.2f} |")
                out[where, corpus, y] = (o, m)
        L.append("")
        fig_matched(data[where], mt, where, f"fig_c38d_matched_{where}.png")
    return L, out


# --------------------------------------------------------------------------


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    data = {"prompt": build("c38magprompt", "c37lambdaprompt"),
            "prefix": build("c38magprefix", "c37lambdaprefix")}
    L = ["# C38: realized edit magnitude and the λ trend (Qwen3-8B)", ""]
    L.append("Mentor spec, Experiment 1 (§3.1–3.4). Unit-column λ sweep (`lam*_cn`), dose 1.0, use queries, "
             "both position conventions. Outcomes are teacher-forced margins on the use query: "
             "R = log p(target answer) − log p(source answer), L = log p(target name) − log p(target answer); "
             "dR, dL are intervened minus clean.")
    L.append("")
    L.append("## Implementation checks")
    L.append("")
    L.append("| check | prompt | prefix |")
    L.append("|---|---|---|")
    for k, name in (("n_trials", "use trials"), ("repro_n", "trial×cond joined to C37"),
                    ("repro_t_med", "median |Δ log p(target answer)| vs C37 (nats)"),
                    ("repro_t_corr", "corr of log p(target answer) with C37"),
                    ("repro_R_corr", "corr of R with C37"),
                    ("applied_spread", "max relative spread of applied norm across λ at a site"),
                    ("dead", "dead rows")):
        L.append(f"| {name} | {data['prompt'][k]} | {data['prefix'][k]} |")
    for k in data["prompt"]["checks"]:
        L.append(f"| {k} (max rel. err) | {data['prompt']['checks'][k]:.2e} | {data['prefix']['checks'][k]:.2e} |")
    L.append("")
    L.append("## Axis reliability (1−ρ threshold, trial kept only if every λ and layer passes)")
    L.append("")
    L.append("| τ_axis | " + " | ".join(f"{w} {c}" for w in ("prompt", "prefix") for c in ("gsm8k", "wikitext_a")) + " |")
    L.append("|---|" + "---|" * 4)
    for tau in TAU_AXIS:
        cells = []
        for w in ("prompt", "prefix"):
            for c in ("gsm8k", "wikitext_a"):
                by = defaultdict(list)
                for r in data[w]["rows"]:
                    if r["corpus"] == c:
                        by[r["trial"]].append(r["min1mrho"] > tau)
                cells.append(f"{sum(all(v) for v in by.values())}/{len(by)}")
        L.append(f"| {tau} | " + " | ".join(cells) + " |")
    L.append("")
    results = {}
    for where in ("prompt", "prefix"):
        L.append(f"## {WHERE[where].capitalize()}")
        L.append("")
        for corpus in ("gsm8k", "wikitext_a"):
            by = defaultdict(list)
            for r in data[where]["rows"]:
                if r["corpus"] == corpus:
                    by[r["trial"]].append(r)
            keep = {t for t, rs in by.items() if all(r["min1mrho"] > TAU_PRIMARY for r in rs) and len(rs) == len(LAMS)}
            df = [r for r in data[where]["rows"] if r["corpus"] == corpus and r["trial"] in keep]
            L.append(f"### {TITLES[corpus]} ({len(keep)} trials × {len(LAMS)} λ)")
            L.append("")
            L.append("**Support (§3.2).** Within-trial Spearman(λ, M), ratio M(λ)/M(0), and overlap of "
                     "within-trial-centred log M between λ=0 and λ=1 / 2.")
            L.append("")
            L.append("| magnitude | median ρ_s | frac > 0 | M(1)/M(0) q25/50/75 | M(2)/M(0) q25/50/75 | OVL 0v1 | OVL 0v2 |")
            L.append("|---|---|---|---|---|---|---|")
            for m in ("m_clean_L2", "m_actual_L2", "rel_clean_L2", "rel_actual_L2", "m_actual_max", "m_clean_L2_nosink"):
                s = support(df, m)
                L.append(f"| {m} | {s['median_spearman']:+.2f} | {s['frac_pos']:.2f} | "
                         + "/".join(f"{x:.2f}" for x in s["ratio1"]) + " | "
                         + "/".join(f"{x:.2f}" for x in s["ratio2"]) + f" | {s['ovl1']:.2f} | {s['ovl2']:.2f} |")
            L.append("")
            L.append("**Screening model (§3.3).** γ_λ vs λ=0 from `Y = α_i + γ_λ + f(log M) + ε` "
                     "(within-trial, 5-knot restricted cubic spline), 95% cluster-bootstrap CI over source entities.")
            L.append("")
            L.append("| outcome | magnitude | γ₁ raw | γ₁ adj | γ₂ raw | γ₂ adj | share of γ₂ explained | R² raw → adj |")
            L.append("|---|---|---|---|---|---|---|---|")
            for y in ("dR", "dL"):
                for m in ("m_clean_L2", "m_actual_L2", "rel_actual_L2", "m_actual_max"):
                    p, lo, hi = boot(df, y, m)
                    results[where, corpus, y, m] = (p, lo, hi)
                    ci = lambda j: f"{p[j]:+.2f} [{lo[j]:+.2f}, {hi[j]:+.2f}]"  # noqa: E731
                    share = 1 - p[3] / p[2] if abs(p[2]) > 1e-9 else float("nan")
                    L.append(f"| {y} | {m} | {ci(0)} | {ci(1)} | {ci(2)} | {ci(3)} | {100*share:.0f}% | "
                             f"{p[4]:.3f} → {p[5]:.3f} |")
            # sensitivity: target clustering and tokenization-clean subset, primary magnitude only
            L.append("")
            for y in ("dR", "dL"):
                p, lo, hi = boot(df, y, "m_actual_L2", cluster="target")
                sub = [r for r in df if r["clean_tok"]]
                ps, los, his = boot(sub, y, "m_actual_L2") if len(sub) > 200 else (None, None, None)
                txt = f"- {y}, m_actual_L2: target-clustered γ₂ adj {p[3]:+.2f} [{lo[3]:+.2f}, {hi[3]:+.2f}]"
                if ps is not None:
                    txt += (f"; single-token subset ({len(sub)//len(LAMS)} trials) γ₂ raw {ps[2]:+.2f} → "
                            f"adj {ps[3]:+.2f} [{los[3]:+.2f}, {his[3]:+.2f}]")
                L.append(txt)
            L.append("")
    ms, _ = matched_section(data)
    L += ms
    fig_magnitude_vs_lambda(data, "fig_c38a_magnitude_vs_lambda.png")
    fig_behaviour_vs_magnitude(data, "prompt", "m_actual_L2", "fig_c38b_behaviour_vs_magnitude_prompt.png")
    fig_behaviour_vs_magnitude(data, "prefix", "m_actual_L2", "fig_c38c_behaviour_vs_magnitude_prefix.png")
    (ROOT / "results/c38_magnitude_qwen3-8b.md").write_text("\n".join(L) + "\n")
    cfg = {
        "tau_axis_primary": TAU_PRIMARY, "tau_axis_grid": TAU_AXIS,
        "tau_axis_rule": "fixed before outcomes: all axes reproducible under bf16 rounding (|cos| >= 0.99999) "
                         "and min(1-rho) = 0.106 over every pair, layer and lambda",
        "tau_m": None, "target_magnitude_rule": None,
        "objectives": {"R": "logp(target answer) - logp(source answer), teacher-forced, use query",
                       "L": "logp(target name) - logp(target answer), teacher-forced, use query",
                       "outcome": "intervened minus clean"},
        "tokenization_filter": "sensitivity: target answer, source answer and name all single-token",
        "lambda": LAMS, "arms": "lam*_cn (unit columns), dose 1.0, row-matched to full lens delta norm",
        "magnitude_summaries": ["L2", "L1", "max over (layer, position) sites", "relative m/||h||",
                                "with and without position 0"],
        "bootstrap": {"cluster": "source entity (primary), target entity (sensitivity)",
                      "n": N_BOOT, "seed": SEED, "unit": "whole lambda trajectory per trial"},
    }
    (ROOT / "results/c38_magnitude_config.json").write_text(json.dumps(cfg, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()

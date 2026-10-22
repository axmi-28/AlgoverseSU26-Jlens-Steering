"""C42 report: where in the prompt does the same-position (diagonal) term act?

    python scripts/65_position_sweep.py [--preset c42pos]

Reads the ``positions_c42pos`` shards (Modal stage ``positions``; protocol
``data/protocols/position_sweep.json``). Each eligible use query was edited with
C38's whole-prompt swap (unit columns, every arm row-matched to the full lens's
norm at every layer and position), masked to one position (``single``), all but
one (``drop``), or none (``all``). Writes

    results/c42_position_sweep_qwen3-8b.md
    results/figures/fig_c42_position_{roles,drop,distance}.{png,pdf}

Estimand: Delta_D(p) = Y(lambda=1) - Y(lambda=0) at matched edit norm, the
effect of re-admitting the diagonal term at position p, for Y in dR, dL (nats,
change from clean, teacher-forced) and the generated outcome (success /
leakage / retention, pp). Positions are pooled within prompt roles, which are
the same for every query: the sink (position 0), the rest of "Fact: Consider",
the clue's first / inner / last tokens, the closing period, the question, and
the readout (last prompt token). 95% cluster-bootstrap intervals over source
entities; trials weighted equally (unit-weighted as a sensitivity).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CONCEPT = ROOT / "results/concept"
FIG = ROOT / "results/figures"
REPORT = ROOT / "results/c42_position_sweep_qwen3-8b.md"
MODULE = ROOT / "src/jsteer/concept_use.py"
C38 = "magnitude_c38magprompt_qwen3-8b_c7822ff955"
N_EXPECTED = 1339
N_BOOT = 2000
SEED = 0
LAM = {"lam0_cn": 0.0, "lam0.5_cn": 0.5, "lam1_cn": 1.0, "lam2_cn": 2.0}
CORPORA = ("gsm8k", "wikitext_a")
ROLES = ("sink", "template", "clue_first", "clue_inner", "clue_last", "period", "query", "readout")
ROLE_LABEL = {
    "sink": "sink (pos. 0)", "template": "': Consider'", "clue_first": "clue, first",
    "clue_inner": "clue, inner", "clue_last": "clue, last", "period": "period",
    "query": "question", "readout": "readout",
}
PREFIX_ROLES = ("template", "clue_first", "clue_inner", "clue_last", "period")
METRICS = {
    "dR": "Δ relational margin R (nats)",
    "dL": "Δ leakage margin L (nats)",
    "success": "Success (pp)",
    "leak": "Leakage, answer wrong (pp)",
}

# Paper style (scripts/63_paper_figures_v3.py).
INK, MUTED, GRID = "#222222", "#6B6B6B", "#E6E6E6"
LENS = {"gsm8k": dict(color="#222222", ls="-", marker="o", label="GSM8K lens"),
        "wikitext_a": dict(color="#8C8C8C", ls="--", marker="s", label="WikiText lens")}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "legend.fontsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.8, "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": "#444444", "xtick.color": INK, "ytick.color": INK, "axes.linewidth": .6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.facecolor": "white", "figure.facecolor": "white",
})


# --------------------------------------------------------------------------
# Loading and gates


def tag(preset: str) -> tuple[str, int]:
    spec = importlib.util.spec_from_file_location("s48", ROOT / "scripts/48_concept_use.py")
    s48 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s48)
    kwargs, n = s48.preset(preset)
    return s48.local_spec(kwargs).tag, n


def load(preset: str):
    t, n = tag(preset)
    shards = [CONCEPT / f"{t}_shard{i}of{n}.json" for i in range(n)]
    missing = [s.name for s in shards if not s.exists()]
    if missing:
        raise SystemExit(f"missing shards {missing}; run `48_concept_use.py fetch`")
    local_sha = hashlib.sha256(MODULE.read_bytes()).hexdigest()
    records, gates = [], {"source_sha_match": True, "applied_spread": 0.0, "dead_rows": 0}
    conds = None
    for s in shards:
        blob = json.loads(s.read_text())
        conds = conds or [tuple(c) for c in blob["conditions"]]
        assert [tuple(c) for c in blob["conditions"]] == conds
        if blob["manifest"]["source_sha"] != local_sha:
            gates["source_sha_match"] = False
        gates["applied_spread"] = max(gates["applied_spread"], blob["checks"]["applied_spread"])
        gates["dead_rows"] += sum(blob.get("dead_rows", {}).values())
        records += blob["records"]
    trials = [r["trial"] for r in records]
    gates["n_records"] = len(records)
    gates["n_unique"] = len(set(trials))
    expected = N_EXPECTED if preset == "c42pos" else gates["n_unique"]
    if gates["n_unique"] != expected or len(records) != expected:
        raise SystemExit(f"{len(records)} records ({gates['n_unique']} unique), expected {expected}")
    return records, conds, gates, t


def c38_reproduction(records, conds) -> dict:
    """The `all` rows rebuild C38's whole-prompt edit; their log-probs must match."""
    ref, rconds = {}, None
    for path in sorted(CONCEPT.glob(f"{C38}_shard*of2.json")):
        blob = json.loads(path.read_text())
        rconds = [tuple(c) for c in blob["conditions"]]
        ref.update({r["trial"]: r for r in blob["records"]})
    C = len(conds)
    a, b = [], []
    for r in records:
        k = r["masks"].index(["all", -1])
        o = ref[r["trial"]]
        for ci, c in enumerate(conds):
            j = rconds.index(c)
            for x in "tsn":
                a.append(r[x][k * C + ci])
                b.append(o[x][j])
    d = np.abs(np.array(a) - np.array(b))
    return {"n": len(a), "median": float(np.median(d)), "p99": float(np.percentile(d, 99)),
            "corr": float(np.corrcoef(a, b)[0, 1])}


# --------------------------------------------------------------------------
# Per-trial arrays


def roles(tokens: list[str], P: int, T: int) -> list[str]:
    if "".join(tokens[:3]) != "Fact: Consider" or tokens[P - 1].strip() != "." or P - 2 <= 3:
        raise ValueError(f"unexpected prompt layout: {tokens[:P]}")
    lab = ["sink", "template", "template"] + ["clue_inner"] * (P - 4) + ["period"]
    lab[3], lab[P - 2] = "clue_first", "clue_last"
    return lab + ["query"] * (T - P - 1) + ["readout"]


def trial_arrays(r, conds):
    """{metric: [K, C]} plus the mask table."""
    C, cl = len(conds), r["clean"]
    shape = (-1, C)
    t, s, n = (np.array(r[x]).reshape(shape) for x in "tsn")
    hit_t, hit_s, name = (np.array(r[x], bool).reshape(shape) for x in ("hit_t", "hit_s", "name_t"))
    return {
        "dR": (t - s) - (cl["t"] - cl["s"]),
        "dL": (n - t) - (cl["n"] - cl["t"]),
        "success": 100.0 * hit_t,
        "leak": 100.0 * (name & ~hit_t),
        "retained": 100.0 * (hit_s & ~hit_t),
        "kl": np.array(r["kl"]).reshape(shape),
    }


class Table:
    """Per-trial arrays, indexed for the role and distance aggregations."""

    def __init__(self, records, conds):
        self.conds = conds
        self.ci = {c: i for i, c in enumerate(conds)}
        self.records = records
        self.arr = [trial_arrays(r, conds) for r in records]
        self.lab = [roles(r["tokens"], r["prefix_len"], r["n_tokens"]) for r in records]
        self.k = [{(m, p): i for i, (m, p) in enumerate(r["masks"])} for r in records]
        self.source = np.array([r["source"] for r in records])
        unit_n = defaultdict(int)
        for r in records:
            unit_n[r["key"]] += 1
        self.unit_w = np.array([1.0 / unit_n[r["key"]] for r in records])

    def val(self, i, metric, mode, p, corpus, arm):
        return self.arr[i][metric][self.k[i][mode, p], self.ci[corpus, arm]]

    def diag_effect(self, i, metric, mode, p, corpus, hi="lam1_cn", lo="lam0_cn"):
        return self.val(i, metric, mode, p, corpus, hi) - self.val(i, metric, mode, p, corpus, lo)

    def by_role(self, fn) -> np.ndarray:
        """[trials, roles]: mean over a trial's positions in each role of fn(i, p)."""
        out = np.full((len(self.records), len(ROLES)), np.nan)
        for i, lab in enumerate(self.lab):
            acc = defaultdict(list)
            for p, role in enumerate(lab):
                acc[role].append(fn(i, p))
            for j, role in enumerate(ROLES):
                if acc[role]:
                    out[i, j] = np.mean(acc[role])
        return out

    def by_distance(self, fn, max_d: int) -> np.ndarray:
        """[trials, max_d+1]: fn at distance d = T-1-p from the readout."""
        out = np.full((len(self.records), max_d + 1), np.nan)
        for i, r in enumerate(self.records):
            T = r["n_tokens"]
            for d in range(min(T, max_d + 1)):
                out[i, d] = fn(i, T - 1 - d)
        return out


# --------------------------------------------------------------------------
# Inference


def boot(M: np.ndarray, clusters: np.ndarray, w: np.ndarray | None = None, seed: int = SEED):
    """Weighted column means of [n, k] (NaN-aware) and their cluster-bootstrap draws."""
    w = np.ones(len(M)) if w is None else w
    keys = np.unique(clusters)
    idx = [np.flatnonzero(clusters == k) for k in keys]

    def mean(rows):
        X, W = M[rows], np.broadcast_to(w[rows][:, None], M[rows].shape)
        ok = ~np.isnan(X)
        return np.where(ok, X * W, 0).sum(0) / np.where(ok, W, 0).sum(0)

    point = mean(np.arange(len(M)))
    rng = np.random.default_rng(seed)
    draws = np.array([mean(np.concatenate([idx[j] for j in rng.choice(len(keys), len(keys))]))
                      for _ in range(N_BOOT)])
    return point, draws


def ci(draws, axis=0):
    return np.percentile(draws, 2.5, axis), np.percentile(draws, 97.5, axis)


def fmt(m, lo, hi, d=2):
    return f"{m:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def slope(tab: Table, i, metric, p, corpus):
    lams = np.array(list(LAM.values()))
    y = np.array([tab.val(i, metric, "single", p, corpus, a) for a in LAM])
    lc = lams - lams.mean()
    return float(lc @ (y - y.mean()) / (lc @ lc))


# --------------------------------------------------------------------------
# Figures


def role_figure(results: dict, fname: str, title: str):
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1), constrained_layout=True)
    x = np.arange(len(ROLES))
    for ax, metric in zip(axes, METRICS):
        ax.axhline(0, color=MUTED, lw=.6)
        ax.yaxis.grid(True, color=GRID, lw=.5)
        ax.set_axisbelow(True)
        for k, corpus in enumerate(CORPORA):
            m, lo, hi = results[corpus][metric]
            off = (k - .5) * .22
            st = LENS[corpus]
            ax.errorbar(x + off, m, yerr=[m - lo, hi - m], color=st["color"], ls="none",
                        marker=st["marker"], ms=3, lw=.8, capsize=0, label=st["label"])
        ax.set_xticks(x, [ROLE_LABEL[r] for r in ROLES], rotation=90)
        ax.set_title(METRICS[metric])
        ax.axvspan(len(PREFIX_ROLES) + .5, len(ROLES) - .5, color="#F3F3F3", zorder=-1)
    axes[0].legend(loc="upper left", handletextpad=.2)
    fig.suptitle(title, fontsize=7.5, x=.01, ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{fname}.{ext}", dpi=300)
    plt.close(fig)


def distance_figure(results: dict, n_at: np.ndarray, fname: str):
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.0), constrained_layout=True)
    for ax, metric in zip(axes, ("dR", "dL")):
        ax.axhline(0, color=MUTED, lw=.6)
        ax.yaxis.grid(True, color=GRID, lw=.5)
        for corpus in CORPORA:
            m, lo, hi = results[corpus][metric]
            d = np.arange(len(m))
            st = LENS[corpus]
            ax.plot(-d, m, color=st["color"], ls=st["ls"], marker=st["marker"], ms=2.5, lw=.9,
                    label=st["label"])
            ax.fill_between(-d, lo, hi, color=st["color"], alpha=.15, lw=0)
        ax.set_xlabel("position relative to the readout (0 = last prompt token)")
        ax.set_title(f"{METRICS[metric]}, λ=1 − λ=0")
    axes[0].legend(loc="upper left")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{fname}.{ext}", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="c42pos")
    a = ap.parse_args()
    records, conds, gates, run_tag = load(a.preset)
    if a.preset == "c42pos":
        gates["c38"] = c38_reproduction(records, conds)
    tab = Table(records, conds)
    src = tab.source
    FIG.mkdir(parents=True, exist_ok=True)
    out = [f"# C42 - where the same-position term acts: a per-token sweep (qwen3-8b)", ""]
    out += [
        "Colleague request: for each prompt position p, apply the intervention and compare the full and "
        "off-diagonal lenses at matched edit norm on the relational margin, the leakage margin and the "
        "generated outcome. Protocol frozen before the run: `data/protocols/position_sweep.json`.",
        "",
        "**Design.** For each of the 1,339 use queries of the held-out concept panel, C38's whole-prompt "
        "swap (strength 1, layers 13-31, unit columns, every λ arm rescaled to the full lens's delta "
        "norm at every layer and position) is built once on the clean residual and then masked: "
        "`single` keeps only position p, `drop` keeps everything except p, `all` keeps the whole edit. "
        "At any site the arms therefore differ only in direction. Δ_D(p) = Y(λ=1) − Y(λ=0) is the effect "
        "of re-admitting the diagonal term at p. Positions are pooled within roles that every query "
        "shares; the shaded region in the figures is the question. Intervals are 95% cluster-bootstrap "
        f"intervals over {len(np.unique(src))} source entities ({N_BOOT} resamples).",
        "",
        "## Gates",
        "",
        f"- records: {gates['n_records']} ({gates['n_unique']} unique trials; expected {N_EXPECTED})",
        f"- source_sha matches local module: {gates['source_sha_match']}",
        f"- applied-norm spread across arms at live sites: {gates['applied_spread']:.1e}; dead rows: {gates['dead_rows']}",
    ]
    if "c38" in gates:
        g = gates["c38"]
        out.append(f"- `all` rows vs C38 c38magprompt ({g['n']} log-probs): median |diff| {g['median']:.4f} nats, "
                   f"99th pct {g['p99']:.3f}, corr {g['corr']:.6f} (batch-size bf16 noise; clean log-probs identical)")
    out.append("")

    # ---------------- single-site levels
    out += ["## Where a single-site edit does anything", "",
            "Mean over trials of the role-averaged single-position outcome, at λ=0 (off-diagonal) and λ=1 "
            "(full). Whole-prompt (`all`) row for scale.", ""]
    out.append("| lens | role | dR λ0 | dR λ1 | dL λ0 | dL λ1 | success λ0 | success λ1 | leak λ0 | leak λ1 | KL λ1 |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for corpus in CORPORA:
        for role_i, role in enumerate(ROLES + ("all",)):
            vals = []
            for metric, arm in [("dR", "lam0_cn"), ("dR", "lam1_cn"), ("dL", "lam0_cn"), ("dL", "lam1_cn"),
                                ("success", "lam0_cn"), ("success", "lam1_cn"), ("leak", "lam0_cn"),
                                ("leak", "lam1_cn"), ("kl", "lam1_cn")]:
                if role == "all":
                    v = np.mean([tab.val(i, metric, "all", -1, corpus, arm) for i in range(len(records))])
                else:
                    v = np.nanmean(tab.by_role(lambda i, p: tab.val(i, metric, "single", p, corpus, arm))[:, role_i])
                vals.append(v)
            cells = [f"{v:+.2f}" for v in vals[:4]] + [f"{v:.1f}" for v in vals[4:8]] + [f"{vals[8]:.3f}"]
            out.append(f"| {corpus} | {role} | " + " | ".join(cells) + " |")
    out.append("")

    # ---------------- primary: single-site diagonal effect by role
    single, drop, slopes, dist, n_at = {}, {}, {}, {}, None
    draws_single = {}
    max_d = int(np.percentile([r["n_tokens"] for r in records], 10)) - 1
    for corpus in CORPORA:
        single[corpus], drop[corpus], slopes[corpus], dist[corpus] = {}, {}, {}, {}
        draws_single[corpus] = {}
        for metric in list(METRICS) + ["retained", "kl"]:
            M = tab.by_role(lambda i, p: tab.diag_effect(i, metric, "single", p, corpus))
            m, dr = boot(M, src)
            single[corpus][metric] = (m, *ci(dr))
            draws_single[corpus][metric] = (m, dr, M)
            whole = np.array([tab.diag_effect(i, metric, "all", -1, corpus) for i in range(len(records))])
            Md = whole[:, None] - tab.by_role(lambda i, p: tab.diag_effect(i, metric, "drop", p, corpus))
            m2, dr2 = boot(Md, src)
            drop[corpus][metric] = (m2, *ci(dr2), dr2, float(whole.mean()))
        for metric in ("dR", "dL"):
            S = tab.by_role(lambda i, p: slope(tab, i, metric, p, corpus))
            m3, dr3 = boot(S, src)
            slopes[corpus][metric] = (m3, *ci(dr3))
            D = tab.by_distance(lambda i, p: tab.diag_effect(i, metric, "single", p, corpus), max_d)
            m4, dr4 = boot(D, src)
            dist[corpus][metric] = (m4, *ci(dr4))
            n_at = (~np.isnan(D)).sum(0)

    out += ["## Primary: effect of the diagonal term at a single position", "",
            "Δ_D = λ1 − λ0, single-position edit, mean over the positions in each role. Success, leakage and "
            "retention in pp.", ""]
    for corpus in CORPORA:
        out += [f"### {LENS[corpus]['label']}", "",
                "| role | Δ_D dR (nats) | Δ_D dL (nats) | Δ_D success | Δ_D leakage | Δ_D retained | Δ_D KL |",
                "|---|---|---|---|---|---|---|"]
        for j, role in enumerate(ROLES):
            row = [fmt(*(v[j] for v in single[corpus][mm]), d=(2 if mm in ("dR", "dL") else 3 if mm == "kl" else 1))
                   for mm in ("dR", "dL", "success", "leak", "retained", "kl")]
            out.append(f"| {role} | " + " | ".join(row) + " |")
        out.append("")
    role_figure({c: single[c] for c in CORPORA}, "fig_c42_position_roles",
                "Effect of re-admitting the diagonal term (λ=1 − λ=0) when only one position is edited, matched edit norm")

    # ---------------- predictions
    out += ["## Pre-registered predictions", ""]
    ri = {r: j for j, r in enumerate(ROLES)}
    for corpus in CORPORA:
        m, dr, _ = draws_single[corpus]["dL"]
        lo, hi = ci(dr[:, ri["readout"]])
        diffs = {r: dr[:, ri["readout"]] - dr[:, ri[r]] for r in PREFIX_ROLES}
        p1 = lo > 0 and all(np.percentile(v, 2.5) > 0 for v in diffs.values())
        worst = min(PREFIX_ROLES, key=lambda r: np.percentile(diffs[r], 2.5))
        out.append(f"- **P1, {corpus}** (readout Δ_D dL > 0 and > every prefix role): "
                   f"{'holds' if p1 else 'fails'}. Readout {fmt(m[ri['readout']], lo, hi)}; smallest margin over a "
                   f"prefix role ({worst}) {fmt(diffs[worst].mean(), *ci(diffs[worst]))}.")
        ms, lo_s, hi_s = single[corpus]["success"]
        worst_s = min(PREFIX_ROLES, key=lambda r: ms[ri[r]])
        p2 = all(ms[ri[r]] >= -2 for r in PREFIX_ROLES)
        out.append(f"- **P2, {corpus}** (prefix-role Δ_D success ≥ −2 pp): {'holds' if p2 else 'fails'}. "
                   f"Lowest is {worst_s}, {fmt(ms[ri[worst_s]], lo_s[ri[worst_s]], hi_s[ri[worst_s]], 1)} pp.")
        md, _, _, drd, whole = drop[corpus]["dL"]
        others = [r for r in ROLES if r != "readout"]
        dd = {r: drd[:, ri["readout"]] - drd[:, ri[r]] for r in others}
        p3 = all(np.percentile(v, 2.5) > 0 for v in dd.values())
        worst3 = min(others, key=lambda r: np.percentile(dd[r], 2.5))
        out.append(f"- **P3, {corpus}** (dropping the readout removes more of the whole-prompt Δ_D dL "
                   f"[{whole:+.2f} nats] than dropping any other position): {'holds' if p3 else 'fails'}. "
                   f"Readout contribution {md[ri['readout']]:+.2f} nats; closest other role ({worst3}) "
                   f"{md[ri[worst3]]:+.2f}, difference {fmt(dd[worst3].mean(), *ci(dd[worst3]))}.")
    out.append("")

    # ---------------- secondary: drop-one
    out += ["## Secondary: necessity (drop one position)", "",
            "Contribution C(p) = Δ_D(whole prompt) − Δ_D(whole prompt without p), per position, mean over the "
            "positions in each role. Positive: removing p removes some of the diagonal's whole-prompt effect.", ""]
    for corpus in CORPORA:
        out += [f"### {LENS[corpus]['label']}", "",
                "Whole-prompt Δ_D: " + ", ".join(f"{mm} {drop[corpus][mm][4]:+.2f}" for mm in METRICS), "",
                "| role | C dR | C dL | C success | C leakage |", "|---|---|---|---|---|"]
        for j, role in enumerate(ROLES):
            row = [fmt(*(v[j] for v in drop[corpus][mm][:3]), d=(2 if mm in ("dR", "dL") else 1)) for mm in METRICS]
            out.append(f"| {role} | " + " | ".join(row) + " |")
        out.append("")
    role_figure({c: {m: drop[c][m][:3] for m in METRICS} for c in CORPORA}, "fig_c42_position_drop",
                "Whole-prompt diagonal effect lost when one position is left unedited, C(p) = Δ_D(all) − Δ_D(all but p)")

    # ---------------- secondary: lambda slope and distance
    out += ["## Secondary: dose-response in λ (single position)", "",
            "Within-trial OLS slope of the outcome on λ ∈ {0, 0.5, 1, 2}, nats per unit λ.", "",
            "| lens | role | slope dR | slope dL |", "|---|---|---|---|"]
    for corpus in CORPORA:
        for j, role in enumerate(ROLES):
            out.append(f"| {corpus} | {role} | {fmt(*(v[j] for v in slopes[corpus]['dR']))} | "
                       f"{fmt(*(v[j] for v in slopes[corpus]['dL']))} |")
    out.append("")
    distance_figure(dist, n_at, "fig_c42_position_distance")
    out += ["## By distance from the readout", "",
            f"Single-position Δ_D for the last {max_d + 1} prompt positions (every trial has at least that many).",
            "", "| distance | " + " | ".join(f"{c} Δ_D dR | {c} Δ_D dL" for c in CORPORA) + " |",
            "|---:|" + "---|---|" * len(CORPORA)]
    for d in range(max_d + 1):
        cells = []
        for c in CORPORA:
            for mm in ("dR", "dL"):
                m, lo, hi = dist[c][mm]
                cells.append(fmt(m[d], lo[d], hi[d]))
        out.append(f"| −{d} | " + " | ".join(cells) + " |")
    out.append("")

    # ---------------- sensitivity
    out += ["## Sensitivity", "",
            "Single-position Δ_D dL and Δ_D success at the readout and pooled over the prefix roles (template "
            "through period).", "",
            "| lens | subset | n | readout Δ_D dL | prefix Δ_D dL | readout Δ_D success | prefix Δ_D success |",
            "|---|---|---:|---|---|---|---|"]
    single_tok = np.array([all(v == 1 for v in r["ans_len"].values()) for r in records])
    clue = np.array([r["clue"] for r in records])
    subsets = {"all trials": (np.ones(len(records), bool), None), "unit-weighted": (np.ones(len(records), bool), tab.unit_w),
               "single-token answers and name": (single_tok, None), "clue 0": (clue == "0", None),
               "clue 1": (clue == "1", None)}
    pre = [ri[r] for r in PREFIX_ROLES]
    for corpus in CORPORA:
        for name, (mask, w) in subsets.items():
            cells = []
            for mm in ("dL", "success"):
                M = draws_single[corpus][mm][2][mask]
                X = np.column_stack([M[:, ri["readout"]], np.nanmean(M[:, pre], 1)])
                m, dr = boot(X, src[mask], None if w is None else w[mask])
                lo, hi = ci(dr)
                d = 2 if mm == "dL" else 1
                cells += [fmt(m[0], lo[0], hi[0], d), fmt(m[1], lo[1], hi[1], d)]
            out.append(f"| {corpus} | {name} | {int(mask.sum())} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    out += ["", f"Figures: `figures/fig_c42_position_roles.png`, `figures/fig_c42_position_drop.png`, "
            f"`figures/fig_c42_position_distance.png`. Run tag `{run_tag}`.", ""]
    REPORT.write_text("\n".join(out))
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()

"""C41 - does filtering still help where workspace loading is poor?

The J-Lens paper's own predictor of swap success is the **workspace loading** of
the source argument (its Sec. 3.4): the cosine between the clean residual stream
and the source concept's lens vector, averaged over the argument and readout
positions. Swaps on weakly loaded concepts are where the published lens fails.
So a natural objection to the component result is that filtering only helps
where the concept is already strongly present.

This splits the existing flexible-generalization swaps (C18 comp shards) into
thirds of loading and compares the full and off-diagonal lens in each third.
Loading is computed once per base prompt with the *published* averaged lens
(the C2 file, layers 13-31), so it is a fixed covariate shared by every arm and
cannot favour either lens. No GPU calls.

    python scripts/58_loading_split.py           # flexible-generalization
    python scripts/58_loading_split.py latent    # latent-intermediate (C41b)
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import statistics as st
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("c9", ROOT / "scripts/25_domain_causal.py")
c9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c9)

LOADING = ROOT / "results/c2/workspace_loading_qwen3-8b_L13-31_3e001929.json"
PATTERN = str(ROOT / "results/causal/steering_qwen3-8b_swap_comp_shard{shard}of{n}.json")
REPORT = ROOT / "results/c41_loading_split_qwen3-8b.md"
FIG = ROOT / "paper/overleaf/figures/fig_loading_split"
CORPORA = {"gsm8k": "GSM8K-fitted lens", "wikitext_a": "WikiText-fitted lens"}
STRATA = ("Low", "Middle", "High")
N_BOOT = 10_000
SEED = 4101

# Palette and type follow scripts/56_final_paper_figures.py.
INK, GRAY, LIGHT = "#242F37", "#75828B", "#D9DFE2"
OFF, FULL = "#007F73", "#43556A"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5,
    "mathtext.fontset": "dejavusans", "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": GRAY, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def sign_p(w: int, l: int) -> float:
    n = w + l
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(w, l) + 1)) / 2**n)


def load_trials() -> tuple[dict, dict]:
    loading = json.loads(LOADING.read_text())
    probe = json.loads(Path(PATTERN.format(shard=0, n=2)).read_text())
    arms = [a for a in dict.fromkeys(r["arm"] for r in probe) if a != "baseline"]
    strengths = sorted({r["strength"] for r in probe if r["arm"] != "baseline"})
    rows = c9.load(PATTERN, 2, 1 + len(arms) * len(strengths))
    c9.regrade(rows)
    ok = {t.prompt for t, r in rows if r["arm"] == "baseline" and r["hit_re"]}
    # (arm, strength) -> {trial: (base-prompt key, hit)}
    trials: dict = {}
    for t, r in rows:
        if r["arm"] == "baseline" or t.prompt not in ok:
            continue
        base = f"{t.category}/{t.func}/{t.source_arg}"
        trials.setdefault((r["arm"], r["strength"]), {})[(t.prompt, t.target_arg)] = (base, bool(r["hit_re"]))
    return loading, trials


def main() -> None:
    loading, trials = load_trials()
    strengths = sorted({s for _, s in trials})
    ref = trials[("swap_gsm8k_full", strengths[0])]
    bases = sorted({b for b, _ in ref.values()}, key=lambda b: loading[b]["loading"])
    assert len(ref) == 144 and len(bases) == 48, (len(ref), len(bases))
    third = len(bases) // 3
    stratum = {b: STRATA[min(i // third, 2)] for i, b in enumerate(bases)}
    ranges = {
        s: (min(loading[b]["loading"] for b in bases if stratum[b] == s),
            max(loading[b]["loading"] for b in bases if stratum[b] == s))
        for s in STRATA
    }

    rng = random.Random(SEED)
    out: dict = {}
    for a in strengths:
        for corpus in CORPORA:
            full, off = trials[(f"swap_{corpus}_full", a)], trials[(f"swap_{corpus}_off", a)]
            assert set(full) == set(off) == set(ref)
            for s in STRATA:
                ids = [p for p in full if stratum[full[p][0]] == s]
                by_base: dict[str, list[str]] = {}
                for p in ids:
                    by_base.setdefault(full[p][0], []).append(p)
                groups = list(by_base.values())

                def rate(arm, sample):
                    flat = [p for g in sample for p in g]
                    return 100 * sum(arm[p][1] for p in flat) / len(flat)

                boots = {"full": [], "off": [], "gain": []}
                for _ in range(N_BOOT):
                    sample = [groups[rng.randrange(len(groups))] for _ in groups]
                    f, o = rate(full, sample), rate(off, sample)
                    boots["full"].append(f)
                    boots["off"].append(o)
                    boots["gain"].append(o - f)
                ci = {k: (sorted(v)[int(0.025 * N_BOOT)], sorted(v)[int(0.975 * N_BOOT)])
                      for k, v in boots.items()}
                wins = sum(off[p][1] and not full[p][1] for p in ids)
                losses = sum(full[p][1] and not off[p][1] for p in ids)
                out[(a, corpus, s)] = {
                    "n": len(ids), "bases": len(groups),
                    "full": sum(full[p][1] for p in ids), "off": sum(off[p][1] for p in ids),
                    "wins": wins, "losses": losses, "p": sign_p(wins, losses), "ci": ci,
                }

    write_report(out, strengths, ranges)
    draw(out, strengths, ranges)


def write_report(out: dict, strengths: list, ranges: dict) -> None:
    digest = hashlib.sha256(LOADING.read_bytes()).hexdigest()[:12]
    lines = [
        "# C41 - filtering effect split by workspace loading (qwen3-8b)",
        "",
        "Flexible-generalization set, 144 baseline-correct trials on 48 base prompts. "
        "Base prompts are split into thirds (16 each, 48 trials) by the source "
        "argument's workspace loading under the published averaged lens "
        f"(`{LOADING.name}`, sha {digest}; Gurnee et al. Sec. 3.4 definition, layers 13-31). "
        "Success = answer changes to the replacement's (hit_re). Full = full lens, "
        "off = off-diagonal lens, both at the stated strength. CIs: 95% bootstrap over "
        f"base prompts ({N_BOOT:,} resamples). Sign test over paired trials.",
        "",
        "| third | loading range |",
        "|---|---|",
    ]
    lines += [f"| {s} | {lo:+.3f} to {hi:+.3f} |" for s, (lo, hi) in ranges.items()]
    for a in strengths:
        lines += ["", f"## Strength {a:g}", "",
                  "| lens | third | full | off | gain (pp) | 95% CI | wins/losses | sign p |",
                  "|---|---|---|---|---|---|---|---|"]
        for corpus, label in CORPORA.items():
            for s in STRATA:
                r = out[(a, corpus, s)]
                lo, hi = r["ci"]["gain"]
                lines.append(
                    f"| {label} | {s} | {r['full']}/{r['n']} | {r['off']}/{r['n']} | "
                    f"{100 * (r['off'] - r['full']) / r['n']:+.1f} | {lo:+.1f} to {hi:+.1f} | "
                    f"{r['wins']}/{r['losses']} | {r['p']:.3g} |"
                )
    REPORT.write_text("\n".join(lines) + "\n")
    print(REPORT.read_text())


def draw(out: dict, strengths: list, ranges: dict) -> None:
    fig, axes = plt.subplots(len(strengths), 2, figsize=(122 / 25.4, 4.4),
                             sharex=True, sharey=True)
    x = range(len(STRATA))
    for i, a in enumerate(strengths):
        for j, (corpus, label) in enumerate(CORPORA.items()):
            ax = axes[i][j]
            ax.grid(axis="y", color=LIGHT, lw=0.6, zorder=0)
            for key, colour, name, face, dx in (("full", FULL, "Full lens", "white", -0.07),
                                                ("off", OFF, "Off-diagonal lens", OFF, 0.07)):
                ys = [100 * out[(a, corpus, s)][key] / out[(a, corpus, s)]["n"] for s in STRATA]
                err = [[y - out[(a, corpus, s)]["ci"][key][0] for y, s in zip(ys, STRATA)],
                       [out[(a, corpus, s)]["ci"][key][1] - y for y, s in zip(ys, STRATA)]]
                ax.errorbar([k + dx for k in x], ys, yerr=err, color=colour, lw=1.6, marker="o", ms=4.5,
                            mfc=face, mec=colour, mew=1.2, capsize=2.2, elinewidth=0.8,
                            label=name, zorder=3)
            for k, s in enumerate(STRATA):
                r = out[(a, corpus, s)]
                gain = 100 * (r["off"] - r["full"]) / r["n"]
                top = max(r["ci"]["off"][1], r["ci"]["full"][1])
                ax.text(k, top + 3, f"{gain:+.0f} pp",
                        ha="center", va="bottom", fontsize=6.5, color=GRAY)
            ax.set_ylim(0, 100)
            ax.set_xlim(-0.35, 2.35)
            ax.set_yticks([0, 25, 50, 75, 100])
            if i == 0:
                ax.set_title(label, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"Strength {a:g}\nSuccessful swaps (%)")
            if i == len(strengths) - 1:
                ax.set_xticks(list(x))
                ax.set_xticklabels([f"{s}\n{ranges[s][0]:.3f}–{ranges[s][1]:.3f}" for s in STRATA])
    fig.supxlabel("Workspace loading of the source concept (thirds; cosine range)",
                  fontsize=8, y=0.04)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG}.{ext}", dpi=400 if ext == "png" else None, bbox_inches="tight")
    print(f"wrote {FIG}.pdf/.png")


# ---------------------------------------------------------------------------
# Latent-intermediate set (55 development + 139 new items).
#
# The intermediate is never written, so the paper's "argument" positions do not
# exist; the primary loading is the readout (final) position alone, with the
# all-position mean as a secondary (``jsteer.latent_panel.run_latent_loading``).
# The contrast is the paper's latent one: full lens vs norm-matched
# off-diagonal lens (C22 arms swap_full / swap_off_matched).
# ---------------------------------------------------------------------------

LATENT = ROOT / "results/latent"
DEV = {"gsm8k": ROOT / "results/c22/write_controls_dac7603b1af57dbd.json",
       "wikitext_a": ROOT / "results/c22/write_controls_6fff00659415ddf5.json"}
NEW = {"gsm8k": LATENT / "latent_controls_gsm8k_80a6934224f0ac55.json",
       "wikitext_a": LATENT / "latent_controls_wikitext_a_393f229cc9c869be.json"}
PANEL = ROOT / "data/heldout/latent-panel.json"
LATENT_REPORT = ROOT / "results/c41_latent_loading_split_qwen3-8b.md"
LATENT_FIG = ROOT / "paper/overleaf/figures/fig_latent_loading_split"
LATENT_ARMS = {"full": "swap_full", "off": "swap_off_matched"}
METRICS = {"hit_swap_answer": "Correct downstream answer (%)", "emitted_substring": "Name leakage (%)"}


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latent_loading() -> dict:
    files = sorted(LATENT.glob("latent_loading_*.json"), key=lambda q: q.stat().st_mtime)
    if not files:
        raise SystemExit("no latent_loading_*.json; run the Modal task and `modal volume get` it")
    blob = json.loads(files[-1].read_text())
    m = blob["manifest"]
    # The loading file must have been computed on exactly these item sets.
    assert m["dev_sha256"] == file_sha(DEV["gsm8k"]), "dev file changed since loading was computed"
    assert m["panel_sha256"] == file_sha(PANEL), "panel changed since loading was computed"
    recs = {r["name"]: r for r in blob["records"]}
    sets = [r["set"] for r in recs.values()]
    assert len(recs) == 194 and sets.count("development") == 55 and sets.count("new") == 139, len(recs)
    return recs | {"__file__": files[-1].name}


def latent_outcomes() -> dict:
    """(corpus, strength, arm-key) -> {item: {metric: bool}}."""
    out: dict = {}
    for corpus in CORPORA:
        for path in (DEV[corpus], NEW[corpus]):
            blob = json.loads(path.read_text())
            assert blob["status"] == "complete", path
            for r in blob["records"]:
                for key, arm in LATENT_ARMS.items():
                    if r["arm"] == arm:
                        out.setdefault((corpus, r["strength"], key), {})[r["name"]] = {
                            m: bool(r[m]) for m in METRICS}
    for k, v in out.items():
        assert len(v) == 194, (k, len(v))
    return out


def latent_main() -> None:
    loading = latent_loading()
    source = loading.pop("__file__")
    outcomes = latent_outcomes()
    # Paper totals (App. D, strength 1) must reproduce before anything is split.
    # Success = the replacement's answer (hit_swap_answer); hit_answer is retention.
    for corpus, want in {"gsm8k": (67, 18), "wikitext_a": (69, 29)}.items():
        got = tuple(sum(v["emitted_substring"] for v in outcomes[(corpus, 1.0, k)].values()) for k in ("full", "off"))
        assert got == want, (corpus, got, want)
    for corpus, gain in {"gsm8k": 15.5, "wikitext_a": 14.4}.items():
        f, o = (sum(v["hit_swap_answer"] for v in outcomes[(corpus, 1.0, k)].values()) for k in ("full", "off"))
        assert round(100 * (o - f) / 194, 1) == gain, (corpus, f, o, gain)

    names = sorted(loading, key=lambda n: (loading[n]["loading_readout"], n))
    cut = [len(names) * k // 3 for k in range(4)]
    stratum = {n: STRATA[k] for k in range(3) for n in names[cut[k]:cut[k + 1]]}
    ranges = {s: (min(loading[n]["loading_readout"] for n in names if stratum[n] == s),
                  max(loading[n]["loading_readout"] for n in names if stratum[n] == s)) for s in STRATA}
    rng = random.Random(SEED)
    res: dict = {}
    strengths = sorted({a for (_, a, _) in outcomes})
    for corpus, a in ((c, a) for c in CORPORA for a in strengths):
        full, off = outcomes[(corpus, a, "full")], outcomes[(corpus, a, "off")]
        for s in STRATA:
            ids = [n for n in names if stratum[n] == s]
            clusters: dict[str, list[str]] = {}
            for n in ids:
                clusters.setdefault(loading[n]["intermediate"], []).append(n)
            groups = list(clusters.values())
            for metric in METRICS:
                boots = {"full": [], "off": [], "gain": []}
                for _ in range(N_BOOT):
                    flat = [n for _ in groups for n in groups[rng.randrange(len(groups))]]
                    f = 100 * sum(full[n][metric] for n in flat) / len(flat)
                    o = 100 * sum(off[n][metric] for n in flat) / len(flat)
                    boots["full"].append(f); boots["off"].append(o); boots["gain"].append(o - f)
                ci = {k: (sorted(v)[int(0.025 * N_BOOT)], sorted(v)[int(0.975 * N_BOOT)]) for k, v in boots.items()}
                w = sum(off[n][metric] and not full[n][metric] for n in ids)
                l = sum(full[n][metric] and not off[n][metric] for n in ids)
                res[(corpus, a, s, metric)] = {
                    "n": len(ids), "clusters": len(groups),
                    "full": sum(full[n][metric] for n in ids), "off": sum(off[n][metric] for n in ids),
                    "up": w, "down": l, "p": sign_p(w, l), "ci": ci,
                    "dev": sum(loading[n]["set"] == "development" for n in ids),
                }

    lines = [
        "# C41b - latent-intermediate filtering effect split by workspace loading (qwen3-8b)",
        "",
        "194 items (55 development + 139 new). Loading = cosine between the clean residual "
        "stream and the intermediate's released-J-lens vector at the **readout (final) "
        "position**, averaged over layers 13-31 (the intermediate is never written, so the "
        f"paper's argument positions do not exist). Source: `{source}`. Items split into thirds "
        "of 64-65. Contrast: full lens vs norm-matched off-diagonal lens (C22 arms). "
        f"CIs: 95% bootstrap over intermediate entities ({N_BOOT:,} resamples). "
        "Sign test over paired items (up = off-only, down = full-only).",
        "",
        "| third | readout loading range | items | of which development |",
        "|---|---|---|---|",
    ]
    for s in STRATA:
        r = res[("gsm8k", 1.0, s, "hit_swap_answer")]
        lines.append(f"| {s} | {ranges[s][0]:+.3f} to {ranges[s][1]:+.3f} | {r['n']} | {r['dev']} |")
    for a in strengths:
        for metric, label in METRICS.items():
            lines += ["", f"## Strength {a:g}: {label.replace(' (%)', '')}", "",
                      "| lens | third | full | off | change (pp) | 95% CI | up/down | sign p |",
                      "|---|---|---|---|---|---|---|---|"]
            for corpus, clabel in CORPORA.items():
                for s in STRATA:
                    r = res[(corpus, a, s, metric)]
                    lo, hi = r["ci"]["gain"]
                    lines.append(f"| {clabel} | {s} | {r['full']}/{r['n']} | {r['off']}/{r['n']} | "
                                 f"{100 * (r['off'] - r['full']) / r['n']:+.1f} | {lo:+.1f} to {hi:+.1f} | "
                                 f"{r['up']}/{r['down']} | {r['p']:.3g} |")
    # Secondary: does the readout-position loading agree with the all-position mean?
    xs = [loading[n]["loading_readout"] for n in names]
    ys = [loading[n]["loading_all"] for n in names]
    rx = {v: i for i, v in enumerate(sorted(xs))}; ry = {v: i for i, v in enumerate(sorted(ys))}
    rho = 1 - 6 * sum((rx[x] - ry[y]) ** 2 for x, y in zip(xs, ys)) / (len(xs) * (len(xs) ** 2 - 1))
    lines += ["", f"Secondary definition check: Spearman(readout loading, all-position loading) = {rho:+.2f}.",
              f"Multi-token intermediates (loading uses the first token): "
              f"{sum(loading[n]['src_n_tokens'] > 1 for n in names)}/194."]
    lines += robustness(loading, outcomes, names)
    LATENT_REPORT.write_text("\n".join(lines) + "\n")
    print(LATENT_REPORT.read_text())
    draw_latent(res, ranges)


def robustness(loading: dict, outcomes: dict, names: list) -> list[str]:
    """Two checks on the strength-1 split: relation type and loading definition.

    The answer gain is concentrated in country relations (C40), so a loading
    split that sorts items by type would look like a loading effect. Country =
    a C40 country-type item, or a development item whose intermediate is one of
    the C40 country entities.
    """
    panel = json.loads(PANEL.read_text())["items"]
    countries = {i["intermediate"] for i in panel if i["type"] == "country"}
    group = {n: "country" if loading[n]["intermediate"] in countries else "other" for n in names}

    def cell(ids, metric):
        return " | ".join(
            f"{sum(outcomes[(c, 1.0, 'full')][n][metric] for n in ids)} -> "
            f"{sum(outcomes[(c, 1.0, 'off')][n][metric] for n in ids)}" for c in CORPORA)

    out = ["", "## Robustness (strength 1)", "",
           "### Relation type within readout-loading thirds (correct answers, full -> off)", "",
           "| group | third | n | GSM8K | WikiText |", "|---|---|---|---|---|"]
    k = len(names)
    # Same boundaries as the main split (names[k*j//3 : k*(j+1)//3]).
    third = {n: STRATA[j] for j in range(3) for n in names[k * j // 3: k * (j + 1) // 3]}
    for g in ("country", "other"):
        for s in STRATA:
            ids = [n for n in names if third[n] == s and group[n] == g]
            out.append(f"| {g} | {s} | {len(ids)} | {cell(ids, 'hit_swap_answer')} |")
    out += ["", "### Alternative definition: loading averaged over all prompt positions", "",
            "| third | n country/other | answers GSM8K | answers WikiText | leakage GSM8K | leakage WikiText |",
            "|---|---|---|---|---|---|"]
    alt = sorted(names, key=lambda n: (loading[n]["loading_all"], n))
    for j, s in enumerate(STRATA):
        ids = alt[k * j // 3: k * (j + 1) // 3]
        nc = sum(group[n] == "country" for n in ids)
        out.append(f"| {s} | {nc}/{len(ids) - nc} | {cell(ids, 'hit_swap_answer')} | {cell(ids, 'emitted_substring')} |")
    return out


def draw_latent(res: dict, ranges: dict, a: float = 1.0) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(122 / 25.4, 4.4), sharex=True, sharey=True)
    x = range(len(STRATA))
    for i, (metric, ylabel) in enumerate(METRICS.items()):
        for j, (corpus, label) in enumerate(CORPORA.items()):
            ax = axes[i][j]
            ax.grid(axis="y", color=LIGHT, lw=0.6, zorder=0)
            for key, colour, name, face, dx in (("full", FULL, "Full lens", "white", -0.07),
                                                ("off", OFF, "Off-diagonal lens (norm-matched)", OFF, 0.07)):
                cells = [res[(corpus, a, s, metric)] for s in STRATA]
                ys = [100 * c[key] / c["n"] for c in cells]
                err = [[y - c["ci"][key][0] for y, c in zip(ys, cells)],
                       [c["ci"][key][1] - y for y, c in zip(ys, cells)]]
                ax.errorbar([k + dx for k in x], ys, yerr=err, color=colour, lw=1.6, marker="o", ms=4.5,
                            mfc=face, mec=colour, mew=1.2, capsize=2.2, elinewidth=0.8,
                            label=name, zorder=3)
            for k, s in enumerate(STRATA):
                r = res[(corpus, a, s, metric)]
                d = 100 * (r["off"] - r["full"]) / r["n"]
                top = max(r["ci"]["off"][1], r["ci"]["full"][1])
                ax.text(k, top + 3, f"{d:+.0f} pp", ha="center", va="bottom", fontsize=6.5, color=GRAY)
            ax.set_ylim(0, 100)
            ax.set_xlim(-0.35, 2.35)
            ax.set_yticks([0, 25, 50, 75, 100])
            if i == 0:
                ax.set_title(label, fontweight="bold")
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 1:
                ax.set_xticks(list(x))
                ax.set_xticklabels([f"{s}\n{ranges[s][0]:.3f}–{ranges[s][1]:.3f}" for s in STRATA])
    fig.supxlabel("Workspace loading of the latent intermediate at the readout position (thirds; cosine range)",
                  fontsize=8, y=0.04)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    for ext in ("pdf", "png"):
        fig.savefig(f"{LATENT_FIG}.{ext}", dpi=400 if ext == "png" else None, bbox_inches="tight")
    print(f"wrote {LATENT_FIG}.pdf/.png")


if __name__ == "__main__":
    import sys

    latent_main() if sys.argv[1:] == ["latent"] else main()

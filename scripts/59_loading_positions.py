"""C41c - why does filtering survive low loading on one set and not the other?

C41 found the filtering gain flat across workspace-loading thirds on flexible
generalization, but absent in the lowest third on the latent-intermediate set.
The two splits were not measured the same way: the paper's definition averages
the argument positions with the readout position, and a latent item has no
argument position, so it was measured at readout only.

This uses per-layer, per-position loading profiles (Modal task
``latent_profiles``) to measure both sets both ways, then

1. re-splits each set by input-position loading (argument / clue) and by
   readout-position loading, **within category** (flexible: the four argument
   categories; latent: country vs other intermediate), because loading is
   confounded with category in both sets (number words load lowest; the latent
   gain is concentrated in country items);
2. tests the low-vs-high difference in gain by permuting loading within
   category;
3. classifies outcomes (success / name leakage / original answer retained /
   other) to see what filtering converts.

No GPU calls.

    python scripts/59_loading_positions.py
"""

from __future__ import annotations

import collections
import glob
import importlib.util
import json
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jsteer.steering import answer_matches  # noqa: E402

spec = importlib.util.spec_from_file_location("s58", ROOT / "scripts/58_loading_split.py")
s58 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s58)

REPORT = ROOT / "results/c41c_loading_positions_qwen3-8b.md"
KB = ROOT / "data/heldout/latent-kb.json"
LAYERS = ("13", "16", "19", "22", "25", "28", "31")
N_PERM = 20_000
SEED = 4102
CORPORA = s58.CORPORA


def mean_at(profile: dict, positions: list[int]) -> float:
    return st.mean(st.mean(profile[l][p] for p in positions) for l in LAYERS)


def clue_positions(tokens: list[str], clue: str) -> list[int] | None:
    text = "".join(tokens)
    start = text.find(clue)
    if start < 0:
        return None
    out, at = [], 0
    for i, tok in enumerate(tokens):
        if at < start + len(clue) and at + len(tok) > start:
            out.append(i)
        at += len(tok)
    return out


def load_profiles() -> tuple[dict, dict, str]:
    files = sorted(glob.glob(str(ROOT / "results/latent/loading_profiles_*.json")))
    if not files:
        raise SystemExit("no loading_profiles_*.json; run the Modal task `latent_profiles`")
    blob = json.loads(Path(files[-1]).read_text())
    assert blob["manifest"]["panel_sha256"] == s58.file_sha(s58.PANEL)
    assert blob["manifest"]["dev_sha256"] == s58.file_sha(s58.DEV["gsm8k"])
    flex, lat = {}, {}
    for r in blob["records"]:
        n = len(r["tokens"])
        if r["set"] == "flexible":
            arg = list(range(*r["arg_span"]))
            r["input"] = mean_at(r["source"], arg)
            r["readout"] = mean_at(r["source"], [n - 1])
            r["paper"] = mean_at(r["source"], sorted(set(arg) | {n - 1}))
            flex[r["key"]] = r
        else:
            cp = clue_positions(r["tokens"], r["clue"]) if r["clue"] else None
            r["input"] = mean_at(r["source"], cp) if cp else None
            r["readout"] = mean_at(r["source"], [n - 1])
            r["all"] = mean_at(r["source"], list(range(1, n)))
            lat[r["name"]] = r
    assert len(flex) == 64 and len(lat) == 194, (len(flex), len(lat))
    assert sum(r["input"] is not None for r in lat.values()) == 139
    return flex, lat, Path(files[-1]).name


def within_group_test(units: list[tuple[str, float, list[int]]], rng: random.Random) -> dict:
    """Median split of loading inside each group; permute loading within group."""
    groups: dict[str, list] = {}
    for u in units:
        groups.setdefault(u[0], []).append(u)

    def halves(shuffle: bool):
        lo, hi = [], []
        for us in groups.values():
            vals = [u[1] for u in us]
            if shuffle:
                rng.shuffle(vals)
            order = sorted(range(len(us)), key=lambda i: vals[i])
            h = len(us) // 2
            lo += [d for i in order[:h] for d in us[i][2]]
            hi += [d for i in order[h:] for d in us[i][2]]
        return lo, hi

    lo, hi = halves(False)
    obs = st.mean(hi) - st.mean(lo)
    null = [st.mean(b) - st.mean(a) for a, b in (halves(True) for _ in range(N_PERM))]
    p = (1 + sum(abs(x) >= abs(obs) - 1e-12 for x in null)) / (N_PERM + 1)
    return {"low": 100 * st.mean(lo), "high": 100 * st.mean(hi), "n_low": len(lo), "n_high": len(hi),
            "diff": 100 * obs, "p": p}


def main() -> None:
    flex, lat, source = load_profiles()
    rng = random.Random(SEED)
    lines = [
        "# C41c - where loading is measured decides whether it gates filtering (qwen3-8b)",
        "",
        f"Profiles: `{source}` (released J-lens, cosine of the clean residual stream with the source "
        "concept's lens vector, mean over layers 13-31). Strength 1. Flexible generalization: full vs "
        "off-diagonal lens (C18 comp shards, 144 trials). Latent: full vs norm-matched off-diagonal "
        "(C22/C40, 194 items). **Input** = argument tokens (flexible) or clue tokens (latent, 139 new "
        "items only; development items do not record their clue). **Readout** = final position. "
        "**Paper** = argument + readout (Gurnee et al. definition).",
    ]

    # Sanity: the paper-definition value reproduces the C2 loading file.
    c2 = json.loads(s58.LOADING.read_text())
    gap = max(abs(flex[k]["paper"] - c2[k]["loading"]) for k in flex if k in c2)
    assert gap < 1e-4, gap
    lines += ["", f"Sanity: paper-definition loading reproduces the C2 file (max abs diff {gap:.1e})."]

    # ---- 1. Within-category split + permutation test
    _, trials = s58.load_trials()
    lines += ["", "## 1. Filtering gain in the low vs high loading half, within category", "",
              "Gain = off minus full success rate (pp). Halves are median splits inside each category "
              f"(flexible: 4 argument categories; latent: country vs other). Permutation p: loading "
              f"shuffled within category, {N_PERM:,} draws, two-sided.", "",
              "| set | lens | loading measured at | gain, low half | gain, high half | difference | perm p |",
              "|---|---|---|---|---|---|---|"]
    fig_data: dict = {}
    kb = json.loads(KB.read_text())
    etype = {e: t for t, v in kb["types"].items() for e in v["entities"]}
    for corpus, label in CORPORA.items():
        full, off = trials[(f"swap_{corpus}_full", 1.0)], trials[(f"swap_{corpus}_off", 1.0)]
        bases = sorted({v[0] for v in full.values()})
        for defn in ("paper", "input", "readout"):
            units = [(b.split("/")[0], flex[b][defn],
                      [int(off[t][1]) - int(full[t][1]) for t in full if full[t][0] == b]) for b in bases]
            r = within_group_test(units, rng)
            fig_data[("flexible", corpus, defn)] = r | {"ci": boot_ci(units, rng)}
            lines.append(f"| flexible | {label} | {defn} | {r['low']:+.1f} (n={r['n_low']}) | "
                         f"{r['high']:+.1f} (n={r['n_high']}) | {r['diff']:+.1f} | {r['p']:.3f} |")
    outcomes = s58.latent_outcomes()
    for corpus, label in CORPORA.items():
        full, off = outcomes[(corpus, 1.0, "full")], outcomes[(corpus, 1.0, "off")]
        for defn, names in (("input", [n for n in lat if lat[n]["input"] is not None]),
                            ("readout", [n for n in lat if lat[n]["input"] is not None]),
                            ("readout", list(lat)), ("all", list(lat))):
            units = [("country" if etype.get(lat[n]["intermediate"]) == "country" else "other", lat[n][defn],
                      [int(off[n]["hit_swap_answer"]) - int(full[n]["hit_swap_answer"])]) for n in names]
            r = within_group_test(units, rng)
            fig_data[("latent", corpus, f"{defn}{len(names)}")] = r | {"ci": boot_ci(units, rng)}
            tag = f"{defn} ({len(names)} items)"
            lines.append(f"| latent | {label} | {tag} | {r['low']:+.1f} (n={r['n_low']}) | "
                         f"{r['high']:+.1f} (n={r['n_high']}) | {r['diff']:+.1f} | {r['p']:.3f} |")

    # ---- 2. Outcome classes by readout-loading third
    lines += ["", "## 2. What filtering converts: outcome classes by readout-loading third", "",
              "success / name leakage / original answer retained / other. Thirds of the pooled set "
              "(not within category).", ""]
    rows = s58.c9.load(s58.PATTERN, 2, 15)
    s58.c9.regrade(rows)
    ok = {t.prompt for t, r in rows if r["arm"] == "baseline" and r["hit_re"]}
    fb = sorted({f"{t.category}/{t.func}/{t.source_arg}" for t, _ in rows if t.prompt in ok},
                key=lambda b: (flex[b]["readout"], b))
    fthird = {b: j for j in range(3) for b in fb[len(fb) * j // 3: len(fb) * (j + 1) // 3]}
    ln = sorted(lat, key=lambda n: (lat[n]["readout"], n))
    lthird = {n: j for j in range(3) for n in ln[len(ln) * j // 3: len(ln) * (j + 1) // 3]}

    def cls(success, leak, retain):
        return "success" if success else "leak" if leak else "retain" if retain else "other"

    lines += ["| set | lens | arm | low third | middle third | high third |", "|---|---|---|---|---|---|"]
    for corpus, label in CORPORA.items():
        for arm, name in ((f"swap_{corpus}_full", "full"), (f"swap_{corpus}_off", "off-diagonal")):
            cnt = [collections.Counter() for _ in range(3)]
            for t, r in rows:
                if r["arm"] != arm or r["strength"] != 1.0 or t.prompt not in ok:
                    continue
                g = r.get("generated") or ""
                cnt[fthird[f"{t.category}/{t.func}/{t.source_arg}"]][cls(
                    r["hit_re"], t.target_arg.casefold() in g.casefold(), answer_matches(g, t.source_answer))] += 1
            lines.append(f"| flexible | {label} | {name} | " + " | ".join(
                f"{c['success']}/{c['leak']}/{c['retain']}/{c['other']}" for c in cnt) + " |")
    recs = {}
    for corpus in CORPORA:
        for path in (s58.DEV[corpus], s58.NEW[corpus]):
            for r in json.loads(path.read_text())["records"]:
                recs[(corpus, r["arm"], r["strength"], r["name"])] = r
    for corpus, label in CORPORA.items():
        for arm, name in (("swap_full", "full"), ("swap_off_matched", "off-diagonal (matched)")):
            cnt = [collections.Counter() for _ in range(3)]
            for n in ln:
                r = recs[(corpus, arm, 1.0, n)]
                cnt[lthird[n]][cls(r["hit_swap_answer"], r["emitted_substring"], r["hit_answer"])] += 1
            lines.append(f"| latent | {label} | {name} | " + " | ".join(
                f"{c['success']}/{c['leak']}/{c['retain']}/{c['other']}" for c in cnt) + " |")

    # ---- 3. Composition
    lines += ["", "## 3. Composition of the readout-loading thirds", "",
              "| set | low third | middle third | high third |", "|---|---|---|---|"]
    lines.append("| flexible | " + " | ".join(
        str(dict(collections.Counter(b.split("/")[0] for b in fb if fthird[b] == j))) for j in range(3)) + " |")
    lines.append("| latent (country/other) | " + " | ".join(
        f"{sum(etype.get(lat[n]['intermediate']) == 'country' for n in ln if lthird[n] == j)}/"
        f"{sum(etype.get(lat[n]['intermediate']) != 'country' for n in ln if lthird[n] == j)}" for j in range(3)) + " |")
    REPORT.write_text("\n".join(lines) + "\n")
    print(REPORT.read_text())
    draw(fig_data)


def boot_ci(units, rng, n_boot: int = 10_000) -> dict:
    """95% CI of each half's gain, resampling units within each (group, half)."""
    groups: dict[str, list] = {}
    for u in units:
        groups.setdefault(u[0], []).append(u)
    cells = {"low": [], "high": []}
    for us in groups.values():
        order = sorted(us, key=lambda u: u[1])
        h = len(us) // 2
        cells["low"].append(order[:h])
        cells["high"].append(order[h:])
    out = {}
    for half, parts in cells.items():
        draws = []
        for _ in range(n_boot):
            ds = [d for part in parts for _ in part for d in part[rng.randrange(len(part))][2]]
            draws.append(100 * st.mean(ds))
        draws.sort()
        out[half] = (draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot)])
    return out


def draw(data: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(s58.plt.rcParams)
    INPUT, READ = "#B85D29", "#007F73"  # diagonal-orange / off-teal from the paper palette
    panels = [("flexible", "Flexible generalization", "paper", "input", "Argument", "readout"),
              ("latent", "Latent intermediates", None, "input139", "Clue (139 new items)", "readout194")]
    fig, axes = plt.subplots(2, 2, figsize=(122 / 25.4, 4.4), sharex=True, sharey=True)
    for i, (key, title, _, inp, inp_label, rd) in enumerate(panels):
        for j, (corpus, clabel) in enumerate(CORPORA.items()):
            ax = axes[i][j]
            ax.grid(axis="y", color=s58.LIGHT, lw=0.6, zorder=0)
            ax.axhline(0, color=s58.GRAY, lw=0.6, zorder=1)
            for dkey, colour, label, dx in ((inp, INPUT, inp_label, -0.04),
                                            (rd, READ, "Readout", 0.04)):
                r = data[(key, corpus, dkey)]
                ys = [r["low"], r["high"]]
                err = [[ys[0] - r["ci"]["low"][0], ys[1] - r["ci"]["high"][0]],
                       [r["ci"]["low"][1] - ys[0], r["ci"]["high"][1] - ys[1]]]
                ax.errorbar([0 + dx, 1 + dx], ys, yerr=err, color=colour, lw=1.6, marker="o", ms=4.5,
                            mec=colour, mfc=colour, capsize=2.2, elinewidth=0.8, zorder=3,
                            label=f"{label}: {r['diff']:+.0f} pp (p = {r['p']:.3f})")
            ax.set_xlim(-0.3, 1.3)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Low-loading half", "High-loading half"])
            ax.set_ylim(-5, 55)
            if i == 0:
                ax.set_title(clabel, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{title}\nGain from filtering (pp)")
            ax.legend(loc="upper left", frameon=False, fontsize=5.8, title="Loading measured at; high − low",
                      title_fontsize=5.8, alignment="left")
    fig.tight_layout()
    dest = ROOT / "paper/overleaf/figures/fig_loading_positions"
    for ext in ("pdf", "png"):
        fig.savefig(f"{dest}.{ext}", dpi=400 if ext == "png" else None, bbox_inches="tight")
    print(f"wrote {dest}.pdf/.png")


if __name__ == "__main__":
    main()

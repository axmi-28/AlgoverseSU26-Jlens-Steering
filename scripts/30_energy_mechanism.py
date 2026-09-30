"""C16 - does a corpus's Jacobian energy profile explain what its lens does?

Joins three things per corpus:

* the energy profile of its documents' prompt-local Jacobians (``run_energy_map``,
  Yan et al. arXiv 2608.25347 Sec. 3.3): the share of energy on the ``t = t'``
  diagonal, and the participation ratio of the off-diagonal mass;
* the read side of its fitted J_bar: present-token recall at argument positions
  (short-horizon) and ICR on probe-swap intermediates (sparse concept);
* the write side: swap successes, copied from the C11/C13 and held-out tables.

Steering scores are transcribed, not recomputed, and their source is named
beside each. Every correlation here is across a handful of corpora, so it is
reported with its n and read as a direction, not a law.

    python scripts/30_energy_mechanism.py
"""

from __future__ import annotations

import collections
import glob
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

#: Flexible-generalization math swaps, of 39. Single 32-document draw, the same
#: draw ``run_energy_map`` streams. Source: results/c11_expanded_corpora_qwen3-8b.md.
MATH39 = {
    "aqua_rat": 14,
    "gsm8k": 11,
    "math_algebra": 8,
    "svamp": 6,
    "openwebmath": 4,
    "ordered_scale": 1,
    "arith_words": 0,
    "wikitext_a": 0,
}
#: Held-out ordered-scale swaps, of 105. Source: implementation.md (C10 panel).
HELDOUT105 = {
    "gsm8k": 35,
    "gsm8k_sol": 26,
    "reasoning_traces": 19,
    "gsm8k_q": 6,
    "equations": 2,
}


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def perm_p(xs, ys, n=20000, seed=0) -> float:
    """Two-sided permutation p for Spearman, exact enough at these n."""
    obs = abs(spearman(xs, ys))
    rng = random.Random(seed)
    ys = list(ys)
    hits = 0
    for _ in range(n):
        rng.shuffle(ys)
        hits += abs(spearman(xs, ys)) >= obs - 1e-12
    return (hits + 1) / (n + 1)


def latest(pattern: str, exclude: str = "") -> Path:
    paths = [Path(p) for p in glob.glob(str(RESULTS / pattern))]
    paths = [p for p in paths if not exclude or exclude not in p.name]
    if not paths:
        raise FileNotFoundError(pattern)
    return max(paths, key=lambda p: p.stat().st_mtime)


def energy_table(path: Path) -> dict[str, dict]:
    recs = json.loads(path.read_text())
    layers = sorted({r["layer"] for r in recs})
    out: dict[str, dict] = {}
    for corpus in sorted({r["corpus"] for r in recs}):
        sub = [r for r in recs if r["corpus"] == corpus]
        docs = sorted({r["doc"] for r in sub})
        # Per-document mean over layers first: documents are the independent
        # unit, layers of one document are not.
        diag_by_doc = [
            statistics.mean(r["share"]["0"] for r in sub if r["doc"] == d) for d in docs
        ]
        pr_by_doc = [
            statistics.mean(
                r["offdiag_pr_frac"]
                for r in sub
                if r["doc"] == d and r["offdiag_pr_frac"] is not None
            )
            for d in docs
        ]
        rng = random.Random(0)
        boot = sorted(
            statistics.mean(rng.choice(diag_by_doc) for _ in diag_by_doc)
            for _ in range(4000)
        )
        out[corpus] = {
            "n_docs": len(docs),
            "diag": statistics.mean(diag_by_doc),
            "diag_ci": (boot[100], boot[3899]),
            "pr_frac": statistics.mean(pr_by_doc),
            "by_layer": {
                ll: statistics.mean(r["share"]["0"] for r in sub if r["layer"] == ll)
                for ll in layers
            },
        }
    return out


def read_side() -> tuple[dict[str, float], dict[str, float]]:
    """Present-token recall at argument positions, and ICR, per lens."""
    sh: dict[str, float] = {}
    icr: dict[str, float] = {}
    try:
        recs = json.loads(latest("readout/readout_*_g*.json", exclude="perm").read_text())
        recs = [r for r in recs if not r["is_last"]]
        for lens in sorted({r["lens"] for r in recs}):
            s = [r for r in recs if r["lens"] == lens]
            sh[lens] = sum(r["rank"] < 10 for r in s) / len(s)
    except FileNotFoundError:
        pass
    try:
        recs = json.loads(
            latest("readout/probe_readout_*_g*.json", exclude="perm").read_text()
        )
        per = collections.defaultdict(lambda: collections.defaultdict(int))
        items = sorted({r["name"] for r in recs})
        for r in recs:
            per[r["lens"]][r["name"]] += r["rank_intermediate"] < 10
        for lens in per:
            icr[lens] = statistics.mean(per[lens][i] for i in items)
    except FileNotFoundError:
        pass
    return sh, icr


def main() -> None:
    path = latest("energy/energy_qwen3-*_energy_*.json", exclude="0b90e7d2")  # 0b90e7d2 = smoke test
    energy = energy_table(path)
    sh, icr = read_side()

    lines = [
        "# C16 - Jacobian energy profile vs lens behaviour (qwen3-8b)",
        "",
        f"Energy: `{path.name}`. Diagonal share = fraction of "
        "``||d h_final,t' / d h_l,t||^2`` at ``t = t'`` (Yan et al. Sec. 3.3's "
        "short-horizon component), Hutchinson-estimated, mean over layers "
        "13-31 then over documents; 95% CI bootstraps documents. "
        "Off-diagonal PR = effective number of non-diagonal source positions "
        "feeding each target, as a fraction of those available (low = stripes, "
        "high = smear).",
        "",
        "| corpus | docs | diagonal share | 95% CI | off-diag PR frac "
        "| math /39 | held-out /105 | present-token top-10 | ICR |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    order = sorted(energy, key=lambda c: energy[c]["diag"])
    for c in order:
        e = energy[c]
        lines.append(
            f"| {c} | {e['n_docs']} | {e['diag']:.3f} | "
            f"[{e['diag_ci'][0]:.3f}, {e['diag_ci'][1]:.3f}] | {e['pr_frac']:.3f} | "
            f"{MATH39.get(c, '')} | {HELDOUT105.get(c, '')} | "
            f"{f'{sh[c]:.3f}' if c in sh else ''} | {f'{icr[c]:.2f}' if c in icr else ''} |"
        )

    lines += ["", "## Diagonal share by layer", ""]
    layers = sorted(next(iter(energy.values()))["by_layer"])
    lines.append("| corpus | " + " | ".join(f"L{ll}" for ll in layers) + " |")
    lines.append("|---|" + "---|" * len(layers))
    for c in order:
        lines.append(
            f"| {c} | "
            + " | ".join(f"{energy[c]['by_layer'][ll]:.3f}" for ll in layers)
            + " |"
        )

    lines += ["", "## Does the energy profile predict the lens?", ""]
    lines.append("| metric | outcome | n | Spearman | perm p |")
    lines.append("|---|---|---|---|---|")
    for metric in ("diag", "pr_frac"):
        for label, table in (
            ("math /39", MATH39),
            ("held-out /105", HELDOUT105),
            ("present-token top-10", sh),
            ("ICR", icr),
        ):
            shared = [c for c in energy if c in table]
            if len(shared) < 4:
                continue
            xs = [energy[c][metric] for c in shared]
            ys = [table[c] for c in shared]
            lines.append(
                f"| {metric} | {label} | {len(shared)} | {spearman(xs, ys):+.2f} "
                f"| {perm_p(xs, ys):.3f} |"
            )

    out = RESULTS / "c16_energy_mechanism_qwen3-8b.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

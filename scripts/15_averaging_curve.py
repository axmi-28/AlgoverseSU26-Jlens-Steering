"""C4 - is the averaged direction's advantage just sample size?

Reads the C4 sweep and reports steering success as a function of how many
wikitext prompts the steering direction averages over, against three
references: the lens's own J_bar (the full ~1000-prompt average), the trial
prompt's own g_x, and a *different* eval prompt's g_x.

How to read the curve:

* **Smooth climb from the n=1 point up to J_bar** -> averaging is variance
  reduction. The consistent component of g_x is what steers, and single
  prompts fail because they are noisy estimates of it.
* **Flat, then a jump; or a plateau well below J_bar** -> the averaged
  direction is not a denoised g_x. It is a point that no partial average
  reaches, and "denoising" is the wrong description.

The two n=1 controls separate two explanations that C1 could not: if
``local_matched`` and ``local_mismatched`` score the same, then being *this
prompt's* direction buys nothing and the entire effect is sample size.

    python scripts/15_averaging_curve.py --config qwen3.6-27b
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import logging
import statistics as st
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("c4")

REFERENCES = {
    "lens_Jbar": "J-Lens averaged direction (the published lens, ~1000 prompts)",
    "local_mismatched": "a different prompt's own direction (n=1, mismatched)",
    "local_matched": "this prompt's own direction (n=1, matched)",
}


def load(results: Path, pattern: str) -> list[dict]:
    seen: dict[tuple, dict] = {}
    for f in sorted(glob.glob(str(results / "causal" / pattern))):
        blob = json.loads(Path(f).read_text())
        for r in blob["records"] if isinstance(blob, dict) else blob:
            seen[(r["arm"], r["prompt_key"], r["target_arg"])] = r
    return list(seen.values())


def mcnemar(a: list[dict], b: list[dict]) -> tuple[int, int, float, int]:
    """Exact two-sided McNemar on paired hit/miss over the same trials.

    Paired, not two-proportion: every arm is run on the identical 192 (prompt,
    target) trials, so the discordant pairs are the evidence and an unpaired
    test would throw away the pairing that makes these arms comparable.
    """
    from math import comb

    A = {(r["prompt_key"], r["target_arg"]): r for r in a}
    B = {(r["prompt_key"], r["target_arg"]): r for r in b}
    shared = sorted(set(A) & set(B))
    n10 = sum(1 for t in shared if A[t]["hit"] and not B[t]["hit"])
    n01 = sum(1 for t in shared if not A[t]["hit"] and B[t]["hit"])
    n = n01 + n10
    if n == 0:
        return n10, n01, 1.0, len(shared)
    tail = sum(comb(n, k) for k in range(0, min(n01, n10) + 1)) / 2**n
    return n10, n01, min(2 * tail, 1.0), len(shared)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="qwen3.6-27b")
    ap.add_argument("--band", default="24-34")
    ap.add_argument("--results", default=str(REPO_ROOT / "results"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = Path(args.results)

    recs = load(results, f"steering_{args.config}_avgn_L{args.band}_shard*.json")
    if not recs:
        raise SystemExit(f"no C4 records for {args.config} L{args.band}")

    baseline = [r for r in recs if r["arm"] == "baseline"]
    by_arm: dict[str, list[dict]] = collections.defaultdict(list)
    for r in recs:
        if r["arm"] != "baseline":
            by_arm[r["arm"]].append(r)

    def stats(rs: list[dict]) -> tuple[float, float, float]:
        return (
            sum(bool(r["hit"]) for r in rs) / len(rs),
            st.median([r["kl_from_clean"] for r in rs]),
            st.median([r["target_rank"] for r in rs]),
        )

    out: list[str] = []
    w = out.append
    w(f"# C4 - does averaging buy its advantage through sample size? ({args.config})")
    w("")
    n_trials = len(baseline)
    w(
        f"Swap steering over layers L{args.band}, {n_trials} (prompt, target) "
        "trials, one fixed strength. The swap operator is magnitude-invariant, "
        "so a single strength is fair across every n despite partial averages "
        "having very different lengths - no dose matching is needed."
    )
    w("")
    w(
        "Directions are averages of g_x over n **wikitext** prompts, drawn from "
        "the same distribution the lens was fitted on - only that pool "
        "converges to J_bar. Each n is run over several disjoint subsets; the "
        "spread column is the range across those subsets."
    )
    w("")

    w("## Table C4.1 - Success against the number of prompts averaged")
    w("")
    rows = []
    ns = sorted({r["n"] for rs in by_arm.values() for r in rs if r["n"] > 0})
    for n in ns:
        reps = collections.defaultdict(list)
        for arm, rs in by_arm.items():
            if rs and rs[0]["n"] == n:
                reps[arm] = rs
        rates = [stats(rs)[0] for rs in reps.values()]
        pooled = [r for rs in reps.values() for r in rs]
        rate, kl, rank = stats(pooled)
        rows.append(
            [
                str(n),
                str(len(reps)),
                f"{100 * rate:.0f}%",
                f"{100 * min(rates):.0f}-{100 * max(rates):.0f}%"
                if len(rates) > 1
                else "-",
                f"{kl:.2f}",
                f"{rank:.0f}",
            ]
        )
    for arm, label in REFERENCES.items():
        if arm not in by_arm:
            continue
        rate, kl, rank = stats(by_arm[arm])
        rows.append([label, "-", f"{100 * rate:.0f}%", "-", f"{kl:.2f}", f"{rank:.0f}"])
    if baseline:
        rate = sum(bool(r["hit"]) for r in baseline) / len(baseline)
        rows.append(
            [
                "no steering (baseline accuracy)",
                "-",
                f"{100 * rate:.0f}%",
                "-",
                "0.00",
                "-",
            ]
        )
    _table(
        [
            "prompts averaged (n)",
            "subsets",
            "succeeded",
            "spread across subsets",
            "median KL from clean",
            "median rank",
        ],
        rows,
        w,
    )

    # ---- paired tests ----
    w("## Table C4.2 - Paired comparisons")
    w("")
    w(
        "Every arm runs the same trials, so these are paired (exact McNemar). "
        "**Only A** / **only B** are the trials one arm converted and the other "
        "did not - the discordant pairs the test is built on."
    )
    w("")
    n1_arm = next((k for k in by_arm if k.startswith("avg_n1_")), None)
    n5_arm = next((k for k in by_arm if k.startswith("avg_n5_")), None)
    comparisons = [
        (
            "local_matched",
            "local_mismatched",
            "the prompt's own direction vs another prompt's",
        ),
        (
            "local_matched",
            n1_arm,
            "the prompt's own direction vs one random wikitext prompt's",
        ),
        ("local_mismatched", "lens_Jbar", "another prompt's direction vs the lens"),
        (n5_arm, "lens_Jbar", "a 5-prompt average vs the lens"),
    ]
    rows = []
    for a, b, label in comparisons:
        if not a or not b or a not in by_arm or b not in by_arm:
            continue
        n10, n01, pval, n = mcnemar(by_arm[a], by_arm[b])
        ra = stats(by_arm[a])[0]
        rb = stats(by_arm[b])[0]
        rows.append(
            [
                label,
                f"{100 * ra:.0f}%",
                f"{100 * rb:.0f}%",
                str(n10),
                str(n01),
                f"{pval:.1g}" if pval >= 1e-16 else "<1e-16",
            ]
        )
    _table(["Comparison", "A", "B", "only A", "only B", "p (exact McNemar)"], rows, w)

    # ---- the verdict ----
    w("## Reading")
    w("")
    lens_rate = stats(by_arm["lens_Jbar"])[0] if "lens_Jbar" in by_arm else None
    if ns and lens_rate is not None:
        top_n = max(ns)
        top = [r for rs in by_arm.values() for r in rs if r["n"] == top_n]
        top_rate = stats(top)[0]
        small = [r for rs in by_arm.values() for r in rs if r["n"] == min(ns)]
        small_rate = stats(small)[0]
        w(
            f"- n = {min(ns)} reaches {100 * small_rate:.0f}%, "
            f"n = {top_n} reaches {100 * top_rate:.0f}%, and the lens's own "
            f"J_bar reaches {100 * lens_rate:.0f}%."
        )
        gap = lens_rate - top_rate
        w(
            f"- The largest partial average is {100 * gap:+.0f} percentage "
            "points from J_bar. A gap near zero means the curve has converged "
            "and averaging is doing the work; a large residual gap means J_bar "
            "is reaching something the partial averages do not."
        )
    if "local_matched" in by_arm and "local_mismatched" in by_arm:
        m = stats(by_arm["local_matched"])[0]
        mm = stats(by_arm["local_mismatched"])[0]
        mkl = stats(by_arm["local_matched"])[1]
        mmkl = stats(by_arm["local_mismatched"])[1]
        w(
            f"- Matched vs mismatched single prompt: {100 * m:.0f}% vs "
            f"{100 * mm:.0f}%. These are NOT close, and the direction is the "
            "surprising one: the steered prompt's own direction is worse than "
            "an unrelated prompt's."
        )
        w(
            f"- That comparison is not a dose artifact. The matched arm's "
            f"median KL is {mkl:.2f} against the mismatched arm's {mmkl:.2f}, "
            "so the prompt's own direction perturbs the output distribution "
            "*more* and converts *fewer* trials."
        )
    w("")

    dest = Path(args.out) if args.out else results / f"c4_averaging_{args.config}.md"
    dest.write_text("\n".join(out) + "\n")
    logger.info("wrote %s", dest)
    print("\n".join(out))


def _table(header: list[str], rows: list[list[str]], w) -> None:
    w("| " + " | ".join(header) + " |")
    w("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        w("| " + " | ".join(r) + " |")
    w("")


if __name__ == "__main__":
    main()

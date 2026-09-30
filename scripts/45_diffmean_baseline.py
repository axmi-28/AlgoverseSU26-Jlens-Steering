#!/usr/bin/env python3
"""C30 - DiffMean/CAA, the supervised baseline, split by template leakage.

The direction is fitted per concept on contrast pairs (same category, same
template, different argument) drawn from two of each category's four templates.
The other two never enter the fit, so the same causal run yields both numbers:

* **seen templates** -- an in-distribution ceiling. The supervised method has
  been shown the exact template it is tested on.
* **unseen templates** -- the fair comparison against an unsupervised J-lens
  direction, which was never fitted on this dataset at all.

Reporting only the first would flatter the baseline; reporting only the second
would flatter us. Both belong in the paper.

    python scripts/45_diffmean_baseline.py
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Written by ``run_diffmean`` into its summary; duplicated here so the split
#: used for scoring is the one used for fitting, and a change to either side
#: shows up as a mismatch rather than as a silently different number.
FIT_TEMPLATES_BY_NFIT = {
    2: {
        ("animals", "class"), ("animals", "group"),
        ("countries", "capital"), ("countries", "continent"),
        ("months", "holiday"), ("months", "next_month"),
        ("numbers", "double"), ("numbers", "first_letter"),
    },
    3: {
        ("animals", "class"), ("animals", "group"), ("animals", "habitat"),
        ("countries", "capital"), ("countries", "continent"), ("countries", "currency"),
        ("months", "holiday"), ("months", "next_month"), ("months", "number"),
        ("numbers", "double"), ("numbers", "first_letter"), ("numbers", "square"),
    },
}
MATH = {"numbers"}


def mcnemar(a: list[bool], b: list[bool]) -> tuple[float, int, int]:
    only_a = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = only_a + only_b
    if n == 0:
        return 1.0, only_a, only_b
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail), only_a, only_b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="dm")
    ap.add_argument("--n-shards", type=int, default=4)
    ap.add_argument("--n-fit", type=int, default=2,
                    help="templates per category used for the fit; selects the "
                         "seen/unseen split, and MUST match the sweep's n_fit")
    ap.add_argument("--out", default="results/c30_diffmean_baseline_qwen3-8b.md")
    args = ap.parse_args()

    if args.n_fit not in FIT_TEMPLATES_BY_NFIT:
        raise SystemExit(f"no recorded template split for n_fit={args.n_fit}")
    fit_templates = FIT_TEMPLATES_BY_NFIT[args.n_fit]

    rows: list[dict] = []
    for i in range(args.n_shards):
        p = ROOT / f"results/causal/steering_qwen3-8b_swap_{args.label}_shard{i}of{args.n_shards}.json"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        rows.extend(json.loads(p.read_text()))

    # Same exclusion rule as every other causal report: a trial counts only if
    # the model produces the source answer unsteered, so a "hit" is a change
    # the intervention caused rather than a coincidence.
    clean: set[str] = {
        r["prompt_key"] for r in rows if r["arm"] == "baseline" and r["hit_generated"]
    }
    trials = [r for r in rows if r["arm"] != "baseline" and r["prompt_key"] in clean]
    arms = sorted({r["arm"] for r in trials})
    strengths = sorted({r["strength"] for r in trials})

    def key(r: dict) -> tuple:
        return (r["prompt_key"], r["target_arg"])

    def vec(arm: str, s: float, pred) -> list[bool]:
        sel = [r for r in trials if r["arm"] == arm and r["strength"] == s and pred(r)]
        sel.sort(key=key)
        return [bool(r["hit_generated"]) for r in sel]

    def kl(arm: str, s: float, pred) -> float:
        sel = [r["kl_from_clean"] for r in trials
               if r["arm"] == arm and r["strength"] == s and pred(r)]
        return statistics.mean(sel) if sel else float("nan")

    seen = lambda r: (r["category"], r["func"]) in fit_templates
    unseen = lambda r: (r["category"], r["func"]) not in fit_templates
    nonmath = lambda r: r["category"] not in MATH

    cells = {
        "seen templates (fitted on)": lambda r: seen(r) and nonmath(r),
        "unseen templates (held out)": lambda r: unseen(r) and nonmath(r),
        "math, unseen only": lambda r: unseen(r) and not nonmath(r),
    }

    lines = [
        "# C30 - DiffMean/CAA as a supervised baseline (qwen3-8b)",
        "",
        "`diffmean_*` is the mass-mean shift behind CAA and the direction ITI "
        "steers along: for each argument `y`, the mean activation of prompts "
        "carrying `y` minus the mean of prompts carrying a different argument "
        "of the same category and the same template. `_arg` reads at the "
        "argument's own token span, `_last` at the final prompt position (the "
        "usual CAA convention), `_shuf` is the same set of vectors assigned to "
        "the wrong arguments -- the specificity control.",
        "",
        f"It is fitted on **{args.n_fit} of each category's four templates**. "
        f"The other {4 - args.n_fit} never enter the fit, so the seen/unseen split below separates an "
        "in-distribution ceiling from a fair comparison. The J-lens arms were "
        "fitted on unlabelled corpus text and have seen none of these prompts, "
        "so *every* cell is held out for them.",
        "",
        "Same 192 trials, same seven band layers, same swap operator, same "
        "exclusion rule (the model must produce the source answer unsteered).",
        "",
    ]

    for name, pred in cells.items():
        n = len(vec(arms[0], strengths[0], pred))
        lines += [
            f"## {name} - hits out of {n} trials",
            "",
            "| arm | " + " | ".join(f"a={s}" for s in strengths) + " | KL at best a |",
            "|---" * (len(strengths) + 2) + "|",
        ]
        for arm in arms:
            counts = [sum(vec(arm, s, pred)) for s in strengths]
            best_s = strengths[counts.index(max(counts))]
            lines.append(
                f"| {arm} | " + " | ".join(str(c) for c in counts)
                + f" | {kl(arm, best_s, pred):.2f} |"
            )
        lines.append("")

    # The comparison the section exists for, at each arm's own best strength.
    lines += [
        "## Supervised baseline against unsupervised J-lens directions",
        "",
        "**Read the seen/unseen drop with care.** The two template halves are "
        "not equally hard: `published` and `uy` were fitted on neither, and both "
        "score HIGHER on the held-out half (6->14 and 4->13). So a fall from "
        "seen to unseen is only partly leakage, and a rise is not evidence of "
        "generalisation. The column that carries the argument is the level on "
        "the held-out half, compared across arms within that half.",
        "",
        "Non-math, at each arm's better strength. `gsm8k_off` is the "
        "horizon-filtered J-lens direction; it is fitted on 32 GSM8K documents "
        "with no labels and no contrast pairs, and it covers the whole "
        "vocabulary rather than these 16 arguments.",
        "",
        "| arm | seen /n | unseen /n | drop | KL (unseen) |",
        "|---|---|---|---|---|",
    ]
    for arm in arms:
        s_counts = [sum(vec(arm, s, cells["seen templates (fitted on)"])) for s in strengths]
        u_counts = [sum(vec(arm, s, cells["unseen templates (held out)"])) for s in strengths]
        s_best, u_best = max(s_counts), max(u_counts)
        n_s = len(vec(arms[0], strengths[0], cells["seen templates (fitted on)"]))
        n_u = len(vec(arms[0], strengths[0], cells["unseen templates (held out)"]))
        drop = "-" if s_best == 0 else f"{100 * (u_best / n_u - s_best / n_s) / (s_best / n_s):+.0f}%"
        bs = strengths[u_counts.index(u_best)]
        lines.append(
            f"| {arm} | {s_best}/{n_s} | {u_best}/{n_u} | {drop} | "
            f"{kl(arm, bs, cells['unseen templates (held out)']):.2f} |"
        )

    lines += ["", "## Paired tests on the held-out templates", "",
              "| contrast | hits | only-A | only-B | p |", "|---|---|---|---|---|"]
    pred = cells["unseen templates (held out)"]

    def best_strength(arm: str) -> float:
        counts = [sum(vec(arm, s, pred)) for s in strengths]
        return strengths[counts.index(max(counts))]

    # Arms are recorded with the operator prefix the causal run gave them.
    # Naming them bare silently produced an EMPTY contrast table rather than an
    # error, which is the failure mode this repo keeps hitting: the report
    # generated fine and said nothing.
    best_dm = max(
        ("swap_diffmean_arg", "swap_diffmean_last"),
        key=lambda a: max(sum(vec(a, s, pred)) for s in strengths),
    )
    pairs = [("swap_diffmean_arg", "swap_diffmean_shuf"),
             ("swap_diffmean_last", "swap_diffmean_shuf"),
             ("swap_gsm8k_off", best_dm), ("swap_gsm8k_full", best_dm),
             ("swap_published", best_dm), ("swap_uy", best_dm)]
    unknown = [a for pair in pairs for a in pair if a not in arms]
    if unknown:
        raise SystemExit(f"contrast names {unknown} are not arms of this run: {arms}")
    for a, b in pairs:
        if a not in arms or b not in arms:
            continue
        va, vb = vec(a, best_strength(a), pred), vec(b, best_strength(b), pred)
        p, only_a, only_b = mcnemar(va, vb)
        flag = "**" if p < 0.05 else ""
        lines.append(
            f"| {a} vs {b} | {flag}{sum(va)} vs {sum(vb)}{flag} | {only_a} | {only_b} | {p:.4f} |"
        )

    lines += ["", f"Written by `scripts/45_diffmean_baseline.py`.", ""]
    out = ROOT / args.out
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    for arm in arms:
        u = max(sum(vec(arm, s, pred)) for s in strengths)
        print(f"  {arm:<20} unseen {u}")


if __name__ == "__main__":
    main()

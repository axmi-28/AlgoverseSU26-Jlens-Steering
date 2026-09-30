#!/usr/bin/env python3
"""C6 - steering local vs averaged directions across dose, at matched collateral.

The concrete counterpart to C5. C5 measured how well each Jacobian *predicts*
the model's response to a probe; this measures what each direction actually
does when used to steer, with the real operator, graded on the real answer
token.

Why matched KL and not matched alpha
-------------------------------------
RQ3 asks when local directions outperform averaged "after matching target
effect or output-distribution change". Matching alpha does not match either:
the two directions have different gains, so the same alpha buys a different
amount of perturbation. Comparing them at equal alpha therefore confounds
"which direction is better" with "which direction is being pushed harder". The
tables below bin by each trial's own measured KL from the clean distribution,
which is the collateral-damage axis the steering literature reports on.

The binned table pools several doses into a bin, so a trial contributes more
than one point and the cells are not independent. It is there to show the
shape. The paired table is the inferential one: it holds the trial fixed and
picks, for each local (trial, dose), the averaged dose for that same trial
whose KL is closest, discarding pairs that are not close enough to count as
matched.

    python scripts/19_dose_response.py --model qwen3-8b --mode additive
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

from jsteer.config import REPO_ROOT

#: Bins for the collateral axis. Two sets, because the two metrics do not live
#: on the same scale: plain KL runs out to ln(vocab) ~ 11.9 and saturates
#: there, while off-target KL is near zero for a clean edit and only grows when
#: the edit damages something it was not aiming at.
BINS = {
    "kl_from_clean": [
        (0, 0.5),
        (0.5, 1),
        (1, 2),
        (2, 4),
        (4, 7),
        (7, 10),
        (10, 13),
        (13, 1e9),
    ],
    "off_target_kl": [
        (0, 0.01),
        (0.01, 0.03),
        (0.03, 0.1),
        (0.1, 0.3),
        (0.3, 1),
        (1, 3),
        (3, 10),
        (10, 1e9),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--mode", default="additive", choices=["additive", "swap"])
    parser.add_argument("--band", default="L14-19")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.25,
        help="max relative KL difference for a pair to count as matched",
    )
    parser.add_argument(
        "--metric",
        default="off_target_kl",
        choices=["off_target_kl", "kl_from_clean"],
        help=(
            "the collateral axis to match on. off_target_kl is FishBack's "
            "concept-merged divergence and is the right one: kl_from_clean "
            "counts the intended change as damage, which biases every matched "
            "comparison against whichever arm produces more on-target effect."
        ),
    )
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def load(model: str, band: str) -> list[dict]:
    """Every merged causal file for this model/band, deduped by trial identity.

    Skips ``_shard`` files: they are also merged into the pooled file, and
    counting both double-counts every trial. That has happened in this repo
    before and produced a 384/384 where only 192 trials exist.
    """
    root = REPO_ROOT / "results" / "causal"
    paths = [
        p
        for p in glob.glob(str(root / f"steering_{model}_*{band}*.json"))
        if "_shard" not in Path(p).name
    ]
    if not paths:
        raise SystemExit(f"no causal files for {model} {band} under {root}")
    seen: dict[tuple, dict] = {}
    for path in paths:
        payload = json.load(open(path))
        records = payload if isinstance(payload, list) else payload.get("records", [])
        for r in records:
            seen[(r["arm"], r["strength"], r["prompt_key"], r["target_arg"])] = r
    print(f"{len(paths)} files, {len(seen)} unique (arm, dose, prompt, target) records")
    return list(seen.values())


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial on the discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def _cell(text) -> str:
    return str(text).replace("|", "\\|")


def table(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(_cell(h) for h in header) + " |"]
    out.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        out.append("| " + " | ".join(_cell(c) for c in row) + " |")
    return "\n".join(out)


def main() -> int:
    args = parse_args()
    records = load(args.model, args.band)
    arms = (f"{args.mode}_averaged", f"{args.mode}_local")
    steered = [r for r in records if r["arm"] in arms]
    if not steered:
        raise SystemExit(f"no records for arms {arms}")
    doses = sorted({r["strength"] for r in steered})
    metric = args.metric
    missing = [r for r in steered if metric not in r]
    if missing:
        raise SystemExit(
            f"{len(missing)}/{len(steered)} records predate the {metric} field. "
            "Re-run the causal grid with the current grader rather than mixing "
            "graded and ungraded records -- a partial column is worse than none."
        )
    bins = BINS[metric]

    # 1. dose-response
    rows = []
    for dose in doses:
        cells = [dose]
        for arm in arms:
            v = [r for r in steered if r["arm"] == arm and r["strength"] == dose]
            if not v:
                cells += ["-", "-", "-", "-", "-"]
                continue
            cells += [
                f"{sum(x['hit'] for x in v)}/{len(v)}",
                f"{st.median(x['margin'] for x in v):+.2f}",
                f"{st.median(x['target_rank'] for x in v):.0f}",
                f"{st.median(x[metric] for x in v):.3f}",
                f"{st.median(x['concept_mass_ratio'] for x in v):.2f}",
            ]
        rows.append(cells)
    t1 = table(
        rows,
        ["alpha"]
        + [
            f"{a.split('_')[1]} {m}"
            for a in arms
            for m in ("hits", "margin", "rank", metric, "mass")
        ],
    )

    # 2. binned by collateral
    rows = []
    for lo, hi in bins:
        cells = [f"[{lo}, {hi if hi < 1e8 else 'inf'})"]
        for arm in arms:
            v = [r for r in steered if r["arm"] == arm and lo <= r[metric] < hi]
            if not v:
                cells += ["0", "-", "-", "-"]
                continue
            cells += [
                len(v),
                f"{st.median(x['margin'] for x in v):+.2f}",
                f"{st.median(x['target_rank'] for x in v):.0f}",
                sum(x["hit"] for x in v),
            ]
        rows.append(cells)
    t2 = table(
        rows,
        [metric]
        + [
            f"{a.split('_')[1]} {m}"
            for a in arms
            for m in ("n", "margin", "rank", "hits")
        ],
    )

    # 3. paired, trial held fixed, nearest matched KL
    by_trial: dict[tuple, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in steered:
        by_trial[(r["prompt_key"], r["target_arg"])][r["arm"]].append(r)

    buckets: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for arms_here in by_trial.values():
        local = arms_here.get(arms[1], [])
        avg = arms_here.get(arms[0], [])
        for lo_rec in local:
            kl = lo_rec[metric]
            if kl <= 0:
                continue
            best = min(avg, key=lambda a: abs(a[metric] - kl), default=None)
            if best is None:
                continue
            if abs(best[metric] - kl) / kl > args.tolerance:
                continue
            label = next(
                (
                    f"[{lo}, {hi if hi < 1e8 else 'inf'})"
                    for lo, hi in bins
                    if lo <= kl < hi
                ),
                "?",
            )
            buckets[label].append((lo_rec["margin"], best["margin"], kl))

    rows = []
    for lo, hi in bins:
        label = f"[{lo}, {hi if hi < 1e8 else 'inf'})"
        pairs = buckets.get(label, [])
        if not pairs:
            rows.append([label, 0, "-", "-", "-", "-"])
            continue
        wins = sum(1 for a, b, _ in pairs if a > b)
        losses = sum(1 for a, b, _ in pairs if a < b)
        rows.append(
            [
                label,
                len(pairs),
                f"{st.median(a - b for a, b, _ in pairs):+.2f}",
                wins,
                losses,
                f"{sign_test(wins, losses):.1e}",
            ]
        )
    t3 = table(
        rows,
        [
            metric,
            "matched pairs",
            "median margin (local - averaged)",
            "local better",
            "averaged better",
            "p (sign test)",
        ],
    )

    report = f"""# C6 - {args.mode} steering, local vs averaged direction, across dose ({args.model})

The real operator ({args.mode} at every band layer, every prompt position,
band {args.band}), graded on the target *answer* token over 192 trials. Only
the direction source differs between arms.

## Table C6.1 - dose-response

{t1}

## Table C6.2 - binned by collateral damage

Every (trial, dose) point placed in a bin by its own measured `{metric}`.
Pooled across doses, so a trial appears more than once and these cells are not
independent -- read the shape, not the significance.

`off_target_kl` is FishBack's concept-merged divergence (S5.3): the target and
source answer tokens are merged into one category before the divergence is
taken, so probability moved *between* them -- the intended edit -- costs
nothing, and only damage elsewhere registers. `mass` is counterfactual mass
preservation, the steered probability on that pair over the clean probability
on it; well below 1 means the edit drained the pair and leaked the mass to
unrelated tokens, a failure that leaves `margin` looking healthy.

{t2}

## Table C6.3 - paired at matched KL

Trial held fixed. For each local (trial, dose), the averaged dose for that same
trial whose `{metric}` is nearest, keeping only pairs within
{args.tolerance:.0%} relative of each other. Positive margin difference means
the local direction achieved more target-vs-source separation for the same
collateral damage.

{t3}
"""
    out = (
        Path(args.out)
        if args.out
        else REPO_ROOT / "results" / f"c6_{args.mode}_{metric}_{args.model}.md"
    )
    out.write_text(report)
    print(f"wrote {out}\n")
    print(t1)
    print()
    print(t3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

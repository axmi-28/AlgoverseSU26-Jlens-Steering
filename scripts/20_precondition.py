#!/usr/bin/env python3
"""C7 - does raising the index rescue the prompt-local direction?

Reads the preconditioning-ladder shards and answers two questions, in order:

1. Does any preconditioner rescue the local direction? (local_* vs local_none)
2. Does a preconditioned local direction beat the raw averaged one? That is
   FishBack's headline claim transplanted onto J-Lens objects, and it is the
   comparison that decides whether "averaging is the crux" is earned.

Everything is graded on the answer token and compared at matched off-target KL,
because matching on dose would just re-run the confound C6 already found: the
arms have different gains, so equal alpha is not equal perturbation.

Mass retention is reported alongside every success count. It is the metric that
found the mechanism in C6 -- the raw local direction drains the {target, source}
answer pair rather than moving probability between its members -- so it is also
the right success criterion here. A rung that lifts hits without lifting mass
retention has not fixed the mechanism, it has found a different way to be loud.

    python scripts/20_precondition.py --model qwen3-8b --mode swap
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

BINS = [(0, 0.01), (0.01, 0.03), (0.03, 0.1), (0.1, 0.3), (0.3, 1), (1, 3), (3, 1e9)]
RUNGS = ["none", "sigma", "gn", "fisher"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--mode", default="swap", choices=["swap", "additive"])
    parser.add_argument("--tau", default=None, help="restrict to one damping level")
    parser.add_argument("--tolerance", type=float, default=0.25)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def load(model: str, tau: str | None) -> dict[str, list[dict]]:
    """Records grouped by tau. Shards are the only artifact; there is no merge."""
    root = REPO_ROOT / "results" / "causal"
    paths = sorted(glob.glob(str(root / f"precond_{model}_precond_*_shard*.json")))
    if not paths:
        raise SystemExit(f"no C7 shards for {model} under {root}")
    by_tau: dict[str, dict[tuple, dict]] = defaultdict(dict)
    for path in paths:
        name = Path(path).stem
        this_tau = name.split("_tau")[1].split("_shard")[0]
        if "smoke" in name or (tau is not None and this_tau != tau):
            continue
        for r in json.load(open(path)):
            by_tau[this_tau][
                (r["arm"], r["strength"], r["prompt_key"], r["target_arg"])
            ] = r
    out = {t: list(v.values()) for t, v in by_tau.items()}
    for t, recs in sorted(out.items()):
        trials = {(r["prompt_key"], r["target_arg"]) for r in recs}
        print(f"tau={t}: {len(recs)} records over {len(trials)} trials")
    return out


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def _cell(text) -> str:
    return str(text).replace("|", "\\|")


def table(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(_cell(h) for h in header) + " |"]
    out.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        out.append("| " + " | ".join(_cell(c) for c in row) + " |")
    return "\n".join(out)


def best_by_arm(records: list[dict], mode: str) -> list[list]:
    """Each arm at its own best strength -- the J-Lens paper's own convention."""
    rows = []
    for direction in ("averaged", "local"):
        for rung in RUNGS:
            arm = f"{mode}_{direction}_{rung}"
            v = [r for r in records if r["arm"] == arm]
            if not v:
                continue
            by_strength = defaultdict(list)
            for r in v:
                by_strength[r["strength"]].append(r)
            best = max(by_strength.items(), key=lambda kv: sum(x["hit"] for x in kv[1]))
            dose, group = best
            rows.append(
                [
                    f"{direction} / {rung}",
                    dose,
                    f"{sum(x['hit'] for x in group)}/{len(group)}",
                    f"{st.median(x['margin'] for x in group):+.2f}",
                    f"{st.median(x['target_rank'] for x in group):.0f}",
                    f"{st.median(x['off_target_kl'] for x in group):.3f}",
                    f"{st.median(x['concept_mass_ratio'] for x in group):.2f}",
                ]
            )
    return rows


def paired(records: list[dict], arm_a: str, arm_b: str, tolerance: float) -> list[list]:
    """Trial held fixed, matched on nearest off-target KL. A minus B."""
    by_trial = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_trial[(r["prompt_key"], r["target_arg"])][r["arm"]].append(r)
    buckets = defaultdict(list)
    for arms in by_trial.values():
        for a in arms.get(arm_a, []):
            kl = a["off_target_kl"]
            if kl <= 0:
                continue
            b = min(
                arms.get(arm_b, []),
                key=lambda x: abs(x["off_target_kl"] - kl),
                default=None,
            )
            if b is None or abs(b["off_target_kl"] - kl) / kl > tolerance:
                continue
            label = next(
                (
                    f"[{lo}, {hi if hi < 1e8 else 'inf'})"
                    for lo, hi in BINS
                    if lo <= kl < hi
                ),
                "?",
            )
            buckets[label].append((a["margin"], b["margin"]))
    rows = []
    for lo, hi in BINS:
        label = f"[{lo}, {hi if hi < 1e8 else 'inf'})"
        pairs = buckets.get(label, [])
        if not pairs:
            rows.append([label, 0, "-", "-", "-", "-"])
            continue
        wins = sum(1 for x, y in pairs if x > y)
        losses = sum(1 for x, y in pairs if x < y)
        rows.append(
            [
                label,
                len(pairs),
                f"{st.median(x - y for x, y in pairs):+.2f}",
                wins,
                losses,
                f"{sign_test(wins, losses):.1e}",
            ]
        )
    return rows


def main() -> int:
    args = parse_args()
    by_tau = load(args.model, args.tau)
    mode = args.mode
    sections = []
    for tau in sorted(by_tau, key=float):
        records = [r for r in by_tau[tau] if r["arm"].startswith(f"{mode}_")]
        if not records:
            continue
        t1 = table(
            best_by_arm(records, mode),
            [
                "direction / rung",
                "best alpha",
                "hits",
                "margin",
                "rank",
                "off-target KL",
                "mass",
            ],
        )
        t2 = table(
            paired(
                records, f"{mode}_local_fisher", f"{mode}_local_none", args.tolerance
            ),
            [
                "off-target KL",
                "pairs",
                "median margin (fisher - raw)",
                "fisher better",
                "raw better",
                "p",
            ],
        )
        t3 = table(
            paired(
                records, f"{mode}_local_fisher", f"{mode}_averaged_none", args.tolerance
            ),
            [
                "off-target KL",
                "pairs",
                "median margin (local fisher - averaged raw)",
                "local better",
                "averaged better",
                "p",
            ],
        )
        sections.append(
            f"""## Damping tau = {tau}

### Each arm at its own best strength

{t1}

### Does the Fisher rescue the local direction? (local fisher vs local raw)

{t2}

### Does a preconditioned local direction beat the raw averaged one?

This is FishBack's headline claim transplanted onto J-Lens objects. Their
result was preconditioned-local > CAA (an averaged direction), median 1.5x at
matched concept probability.

{t3}
"""
        )

    report = f"""# C7 - the preconditioning ladder ({args.mode}, {args.model})

Every steering arm before this one wrote the **raw** pullback `g = J^T u`,
which is FishBack's `G = I` case -- so the project's negative result about
prompt-local directions was a result about the Euclidean metric and said
nothing about any other. This runs the identical trials with the direction
preconditioned four ways: `none` (raw), `sigma` (activation covariance), `gn`
(`J^T J`), and `fisher` (`J^T H J`, FishBack's G exactly), on both the local
and the averaged direction.

Directions are unit-normalized after preconditioning, so dose is matched by
construction and only direction differs. `gn` and `fisher` are built from the
cached rank-64 SVD of `J_x`, so the preconditioner acts as the identity on the
discarded complement -- a damped approximation, defensible because the measured
spectra are concentrated (participation ratio ~14 of 4096) but an approximation
all the same. Damping is `lambda = tau * mean(eigenvalue)`; large tau collapses
every rung back to `none`.

{"".join(sections)}"""
    out = (
        Path(args.out)
        if args.out
        else REPO_ROOT / "results" / f"c7_precondition_{mode}_{args.model}.md"
    )
    out.write_text(report)
    print(f"wrote {out}")
    for s in sections:
        print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

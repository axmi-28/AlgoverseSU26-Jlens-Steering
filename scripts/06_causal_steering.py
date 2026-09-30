#!/usr/bin/env python3
"""The causal arm: "how does this vary with causal effects of steering?"

Produces the dependent variable both research questions end in -- a per-trial
steering outcome that the geometry from scripts 04 and 05 can be regressed
against, joined on ``prompt_key``.

Three modes, and the order matters:

``--mode swap`` -- the **validation gate**, not a result. Reproduces upstream's
rank-2 operator on all 192 trials and checks the hit rate against their
published 76/192 at strength 1 and 101/192 at strength 2. Upstream ships prompt
sets but no intervention code, so this recipe is reconstructed from prose
(``jsteer.steering``); if it does not land near their numbers, the recipe is
wrong and nothing downstream can be trusted. Run this first, and before the
expensive Jacobian sweep -- it is ~2% of that sweep's cost and it is what tells
you whether the causal half of both research questions is answerable at all.

``--mode additive`` -- the **experiment**. Pure rank-1 addition of a pulled-back
direction at every band layer at every prompt position, under the strength
convention upstream states for verbal-introspection. Run with both
``--directions averaged local`` to get the contrast the project exists to make:
same prompt, same target, same magnitude, same positions, only the direction
differs. Both arms run locally rather than sourcing the averaged one from
Neuronpedia's hosted lens, which would confound direction with implementation.

``--mode both`` -- runs the gate and the experiment together.

A risk confirmed on qwen3-1.7b: additive steering may floor at 0 hits, leaving
the contrast no dynamic range. ``--strengths`` sweeps for an intermediate
regime; if none exists, ``target_rank`` and ``margin`` are the outcome and hit
rate is reported alongside. Where the swap succeeds and the addition fails on
the same trial, that gap isolates the contribution of *removal*.

Grading follows upstream: greedy next token at the final position against the
answer for the *injected* argument under the *same* template. The injected token
(``Canada``) and the graded token (``Ottawa``) are different by design, so a hit
cannot be manufactured by boosting the graded token's logit.

The work lives in :func:`jsteer.sweeps.run_causal`, which the Modal entrypoints
call too.

    python scripts/06_causal_steering.py --config qwen3-8b --mode swap
    python scripts/06_causal_steering.py --config qwen3-8b --mode additive \
        --directions averaged local --strengths 0 0.5 1 2 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from jsteer.sweeps import CausalSpec, run_causal

logger = logging.getLogger("causal")

#: Upstream's published flexible-generalization totals.
PUBLISHED = {1.0: 76, 2.0: 101}
N_TRIALS = 192

#: Per-template swap counts out of 12, from the paper's appendix A.13. These
#: sum to exactly 76 and are the real gate.
#:
#: The aggregate is a weak target because the published numbers are **Claude
#: Sonnet 4.5** ("By default, we report results on Claude Sonnet 4.5"), not a
#: Qwen3. Sonnet gets essentially every unsteered baseline right; Qwen3-8B gets
#: 93/192. So matching 76 would mostly measure model strength.
#:
#: The *shape* is the part that transfers. Countries near-perfect, months
#: bimodal, animals weak, numbers exactly zero -- if our distribution looks
#: like that, the operator is right and the gap is capability. If it is flat
#: across categories, something is still wrong with the intervention.
PUBLISHED_BY_TEMPLATE = {
    ("countries", "capital"): 12,
    ("countries", "language"): 10,
    ("countries", "continent"): 12,
    ("countries", "currency"): 8,
    ("months", "season"): 11,
    ("months", "number"): 1,
    ("months", "holiday"): 12,
    ("months", "next_month"): 0,
    ("animals", "habitat"): 2,
    ("animals", "legs"): 0,
    ("animals", "class"): 4,
    ("animals", "group"): 4,
    ("numbers", "double"): 0,
    ("numbers", "square"): 0,
    ("numbers", "successor"): 0,
    ("numbers", "first_letter"): 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="qwen3-8b")
    parser.add_argument("--mode", default="swap", choices=["swap", "additive", "both"])
    parser.add_argument("--strengths", type=float, nargs="*", default=[1.0, 2.0])
    parser.add_argument(
        "--directions",
        nargs="*",
        default=["averaged"],
        choices=["averaged", "local"],
        help="which pullback to write; the contrast the project exists to make",
    )
    parser.add_argument(
        "--swap-mode", default="clamp", choices=["clamp", "replace", "exchange"]
    )
    parser.add_argument(
        "--strict-answers",
        action="store_true",
        help=(
            "require answers to be single tokens rather than grading on the "
            "answer's first token. Drops 22/192 trials under Qwen3 (all in "
            "'animals'), so it changes the denominator the gate compares "
            "against upstream's 76/192 -- see jsteer.loading.first_token_id."
        ),
    )
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--dim-batch", type=int, default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--report",
        default=None,
        help="skip the run and report an existing results JSON (or a merged one)",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    if args.report:
        report(json.loads(Path(args.report).read_text()), partial=False)
        return 0

    spec = CausalSpec(
        config_name=args.config,
        mode=args.mode,
        strengths=args.strengths,
        directions=args.directions,
        swap_mode=args.swap_mode,
        layers=args.layers,
        dim_batch=args.dim_batch,
        strict_answers=args.strict_answers,
        shard=args.shard,
        n_shards=args.n_shards,
        limit=args.limit,
        out_dir=args.out,
        overwrite=args.overwrite,
        device=args.device,
    )
    run_causal(spec)
    path = spec.resolve_out("causal") / f"steering_{spec.tag}.json"
    records = json.loads(path.read_text())
    report(records, partial=spec.n_shards > 1 or bool(spec.limit))
    logger.info("%d records -> %s", len(records), path)
    return 0


def report(records: list[dict], *, partial: bool = False) -> None:
    """Hit rates per arm and strength, with the gate checked against upstream.

    The gate comparison is suppressed unless the run actually covers all 192
    trials. A shard, a ``--limit`` run, or an incompletely merged set has a
    different denominator from the 192 upstream quotes, and "4/8 vs 76/192
    [OFF]" is the kind of line that gets skimmed as a real failure. The check
    is derived from the record count rather than trusted from a flag, so a
    merged file that is quietly missing a shard is caught too.
    """
    graded = [r for r in records if r["arm"] != "baseline"]
    baseline = [r for r in records if r["arm"] == "baseline"]
    if baseline:
        logger.info(
            "\nbaseline greedy-correct: %d/%d",
            sum(r["hit"] for r in baseline),
            len(baseline),
        )
    logger.info(
        "%-22s %9s %7s %9s %8s %11s %9s",
        "arm",
        "strength",
        "hits",
        "hits_excl",
        "n",
        "mean margin",
        "med rank",
    )
    for arm, strength in sorted({(r["arm"], r["strength"]) for r in graded}):
        rows = [r for r in graded if r["arm"] == arm and r["strength"] == strength]
        ranks = sorted(r["target_rank"] for r in rows)
        logger.info(
            "%-22s %9.2f %7d %9d %8d %11.3f %9d",
            arm,
            strength,
            sum(r["hit"] for r in rows),
            sum(r["hit_excluding"] for r in rows),
            len(rows),
            sum(r["margin"] for r in rows) / max(len(rows), 1),
            ranks[len(ranks) // 2] if ranks else -1,
        )

    gate_rows = [r for r in graded if r["arm"] == "swap_averaged"]
    if not gate_rows:
        return
    by_strength = {s: [r for r in gate_rows if r["strength"] == s] for s in PUBLISHED}
    incomplete = partial or any(
        rows and len(rows) != N_TRIALS for rows in by_strength.values()
    )
    if incomplete:
        counts = {s: len(rows) for s, rows in by_strength.items() if rows}
        logger.info(
            "\n(gate not scored: %s trials per strength, not the %d upstream "
            "quotes -- merge every shard and drop --limit before comparing)",
            counts or len(gate_rows),
            N_TRIALS,
        )
        return
    logger.info(
        "\ngate vs upstream (%d trials, published = Claude Sonnet 4.5):", N_TRIALS
    )
    for strength, published in PUBLISHED.items():
        rows = by_strength[strength]
        if not rows:
            continue
        hits = sum(r["hit"] for r in rows)
        logger.info(
            "  strength %.1f: ours %d/%d vs published %d/%d",
            strength,
            hits,
            len(rows),
            published,
            N_TRIALS,
        )
    if by_strength.get(1.0):
        per_template(by_strength[1.0])


def per_template(rows: list[dict]) -> None:
    """The real gate: does our per-template profile match the paper's shape?

    Reports Spearman rank correlation against appendix A.13 rather than
    per-cell agreement, because the published counts are from a stronger model
    -- the ordering of templates is what should transfer, not the levels.
    """
    import torch

    from jsteer.analysis import rank_correlation

    logger.info("\nper-template, strength 1.0 (ours / 12 vs A.13 / 12):")
    ours, theirs = [], []
    for (category, func), published in PUBLISHED_BY_TEMPLATE.items():
        cells = [r for r in rows if r["category"] == category and r["func"] == func]
        hits = sum(r["hit"] for r in cells)
        ours.append(hits)
        theirs.append(published)
        bar = "#" * hits + "." * (12 - hits)
        logger.info(
            "  %-10s %-13s %2d/%-2d  vs %2d/12   %s",
            category,
            func,
            hits,
            len(cells),
            published,
            bar,
        )
    rho = rank_correlation(
        torch.tensor(ours, dtype=torch.float32),
        torch.tensor(theirs, dtype=torch.float32),
    )
    logger.info(
        "  totals %d vs 76 | Spearman rho = %.3f  %s",
        sum(ours),
        rho,
        "[shape matches]" if rho >= 0.6 else "[SHAPE MISMATCH -- check the operator]",
    )


if __name__ == "__main__":
    sys.exit(main())

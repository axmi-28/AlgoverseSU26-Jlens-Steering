#!/usr/bin/env python3
"""Do workspace loading and the prompt-Jacobian predictors flag the same prompts?

The C2 tables report each predictor's correlation with steering success
separately, and 16_c2_strength_sweep.py asks whether geometry adds *variance
explained* on top of loading. Neither answers the question this script does:
are the two measuring the same thing about a prompt?

That matters for how the result is framed. If loading and subspace alignment
are near-duplicates, then "the prompt's Jacobian predicts steerability" is
mostly a restatement of the paper's own predictor and adds little. If they are
close to independent, they are complementary signals and combining them should
separate prompts better than either alone -- which is a claim the paper does
not make and cannot make, since it has no Jacobian-distribution predictor.

Two outputs: the full predictor-predictor Spearman matrix, and a median-split
contingency table of actual swap success, which is the practical version of the
same question.

    python scripts/21_predictor_overlap.py --model qwen3-8b
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import torch

from jsteer.analysis import rank_correlation
from jsteer.config import REPO_ROOT, load_config
from jsteer.data import flexible_generalization_prompts

LOADING = "workspace loading"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--band", default="14-19")
    parser.add_argument("--positions", default="all")
    parser.add_argument(
        "--against",
        default="subspace align k=8 right (to J_bar)",
        help="the geometric predictor to cross-tabulate loading against",
    )
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def _c2_module():
    spec = importlib.util.spec_from_file_location(
        "c2", REPO_ROOT / "scripts" / "11_c2_predictors.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    lo, hi = (int(x) for x in args.band.split("-"))
    band = list(range(lo, hi + 1))
    config = load_config(args.model)
    results = REPO_ROOT / "results"

    predictors = dict(
        _c2_module().build_predictors(
            config,
            results,
            band,
            args.positions,
            list(flexible_generalization_prompts()),
        )
    )
    loading_path = results / "c2" / f"workspace_loading_{config.name}_L{lo}-{hi}.json"
    predictors[LOADING] = {
        k: v["loading"] for k, v in json.loads(loading_path.read_text()).items()
    }
    keys = sorted(set.intersection(*[set(v) for v in predictors.values()]))
    names = list(predictors)

    def col(name: str) -> torch.Tensor:
        return torch.tensor([predictors[name][k] for k in keys])

    matrix = table(
        [
            [a] + [f"{rank_correlation(col(a), col(b)):+.3f}" for b in names]
            for a in names
        ],
        [""] + names,
    )

    # Practical version: median-split both, then look at real swap success.
    swap = results / "causal" / f"steering_{config.name}_both_c1_L{lo}-{hi}.json"
    seen: dict[tuple, dict] = {}
    for r in json.loads(swap.read_text()):
        seen[(r["arm"], r["strength"], r["prompt_key"], r["target_arg"])] = r
    hits: dict[str, list[bool]] = defaultdict(list)
    for r in seen.values():
        if r["arm"] == "swap_averaged" and r["strength"] == args.strength:
            hits[r["prompt_key"]].append(r["hit"])

    keys = [k for k in keys if k in hits]
    med_load = st.median(predictors[LOADING][k] for k in keys)
    med_geo = st.median(predictors[args.against][k] for k in keys)

    def quadrant(k: str) -> tuple[str, str]:
        return (
            "high" if predictors[LOADING][k] > med_load else "low",
            "high" if predictors[args.against][k] > med_geo else "low",
        )

    cells: dict[tuple, list[bool]] = defaultdict(list)
    counts: dict[tuple, int] = defaultdict(int)
    for k in keys:
        cells[quadrant(k)] += hits[k]
        counts[quadrant(k)] += 1
    rows = []
    for q in (("high", "high"), ("high", "low"), ("low", "high"), ("low", "low")):
        v = cells[q]
        rows.append(
            [q[0], q[1], counts[q], f"{sum(v)}/{len(v)}", f"{sum(v) / len(v):.1%}"]
        )
    contingency = table(
        rows, ["workspace loading", "geometry", "prompts", "swap hits", "success rate"]
    )
    agree = sum(1 for k in keys if (quadrant(k)[0] == quadrant(k)[1]))

    report = f"""# Do workspace loading and the Jacobian predictors measure the same thing? ({config.name})

Band L{lo}-{hi}, {len(keys)} prompts, swap success from the averaged direction
at alpha={args.strength}.

## Predictor-predictor Spearman

{matrix}

## Median split: loading vs `{args.against}`

Each prompt is classified high/low on both predictors at their medians, and the
cell reports actual swap success over that cell's trials.

{contingency}

The two predictors put {agree}/{len(keys)} prompts on the same side of their
respective medians.
"""
    out = Path(args.out) if args.out else results / f"c2_overlap_{config.name}.md"
    out.write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

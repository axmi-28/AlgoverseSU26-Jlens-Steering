#!/usr/bin/env python3
"""C2 - does the per-prompt geometry predict which prompts steer well?

RQ1 and RQ2 established that J_x scatters around J_bar and that g_x scatters
around g_bar. C1 then showed that *substituting* the prompt-local direction
does not improve steering. C2 asks the remaining, weaker question: even if the
local direction is not a better thing to write, does the *size of a prompt's
departure from the average* predict how well that prompt responds to the
lens's own averaged direction?

Unit of analysis is the prompt (n = 64). Each prompt contributes 3 steering
trials (its 3 in-category swap targets), aggregated to one score.

**The confound this script is built around.** Prompts the model already answers
correctly are likely to steer better for reasons that have nothing to do with
Jacobian geometry, and baseline correctness is itself correlated with template
difficulty. So every correlation is reported twice: over all 64 prompts, and
over the baseline-correct subset only. A predictor that survives the second is
telling us something about geometry; one that does not is mostly re-measuring
task difficulty.

    python scripts/11_c2_predictors.py --config qwen3-8b
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import torch

from jsteer.analysis import rank_correlation, subspace_alignment
from jsteer.config import REPO_ROOT, load_config
from jsteer.data import category_args, flexible_generalization_prompts

logger = logging.getLogger("c2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="qwen3-8b")
    parser.add_argument("--positions", default="all")
    parser.add_argument("--band", default="14-19", help="layers the causal run used")
    parser.add_argument("--swap-file", default="steering_qwen3-8b_both_c1_L14-19.json")
    parser.add_argument("--swap-strength", type=float, default=1.0)
    parser.add_argument(
        "--additive-file", default="steering_qwen3-8b_additive_c1_L14-19.json"
    )
    parser.add_argument("--additive-strength", type=float, default=0.01)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def spearman(x: list[float], y: list[float]) -> float:
    return rank_correlation(
        torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    )


def p_value(rho: float, n: int) -> float:
    """Two-sided p for a Spearman rho, via the usual t approximation.

    Reported because at n = 64 the 95%% interval on rho is roughly +/-0.25 and
    at n = 31 it is roughly +/-0.35 -- wide enough that several of the
    correlations below are not distinguishable from zero, which is not obvious
    from the point estimate alone.
    """
    import math

    if n < 4 or abs(rho) >= 1:
        return float("nan")
    t = abs(rho) * math.sqrt((n - 2) / max(1 - rho**2, 1e-12))
    # Normal approximation to the t tail; adequate at these n for a rough p.
    return 2 * (1 - 0.5 * (1 + math.erf(t / math.sqrt(2))))


def partial_spearman(x: list[float], y: list[float], z: list[float]) -> float:
    """Spearman correlation of x and y with z partialled out.

    Preferred over subsetting to the baseline-correct prompts: subsetting
    halves n (64 -> 31) and so halves the power exactly where the question is
    hardest. This keeps all 64 and removes the linear-in-ranks contribution of
    the confound instead.
    """
    rxy, rxz, ryz = spearman(x, y), spearman(x, z), spearman(y, z)
    denominator = ((1 - rxz**2) * (1 - ryz**2)) ** 0.5
    return (rxy - rxz * ryz) / denominator if denominator > 1e-9 else float("nan")


def table(headers, rows, out):
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    out.append("")


def build_predictors(config, results, band, positions, prompts):
    """Per-prompt geometric predictors, averaged over ``band``.

    Extracted from ``main`` so the strength sweep (16_c2_strength_sweep.py) uses
    the identical construction rather than a re-implementation that could drift.
    The sweep asserts it reproduces this script's numbers at the same strength.
    """
    import statistics as st
    from collections import defaultdict

    import torch

    from jsteer.loading import load_lens

    args_positions = positions
    # ---- predictors -----------------------------------------------------
    scalars = json.loads(
        (results / "rq1" / f"scalars_{config.name}_{args_positions}.json").read_text()
    )
    by_prompt = defaultdict(list)
    for r in scalars:
        if r["layer"] in band:
            by_prompt[r["prompt"]].append(r)

    vectors = torch.load(
        results / "rq2" / f"vectors_{config.name}_{args_positions}.pt",
        map_location="cpu",
        weights_only=False,
    )
    target_index = {w: i for i, w in enumerate(vectors["targets"])}
    averaged = vectors["averaged"]
    cats = category_args()

    lens = load_lens(config)
    bar_basis = {}
    for layer in band:
        U, _, Vh = torch.linalg.svd(lens.jacobians[layer].float(), full_matrices=False)
        bar_basis[layer] = (U[:, :8].T.contiguous(), Vh[:8].contiguous())

    digest_dir = results / "rq1" / "digests" / f"{config.name}_{args_positions}_k64"

    predictors: dict[str, dict[str, float]] = defaultdict(dict)
    for p in prompts:
        rows = by_prompt.get(p.key, [])
        if rows:
            predictors["rel_frobenius(J_x, J_bar)"][p.key] = st.mean(
                r["rel_frobenius"] for r in rows
            )
            predictors["participation ratio"][p.key] = st.mean(
                r["participation_ratio"] for r in rows
            )

        path = digest_dir / f"{p.key.replace('/', '_')}.pt"
        if path.exists():
            blob = torch.load(path, map_location="cpu", weights_only=False)
            left, right = [], []
            for layer in band:
                if layer in blob:
                    left.append(
                        subspace_alignment(blob[layer]["left"][:8], bar_basis[layer][0])
                    )
                    right.append(
                        subspace_alignment(
                            blob[layer]["right"][:8], bar_basis[layer][1]
                        )
                    )
            if left:
                predictors["subspace align k=8 left (to J_bar)"][p.key] = st.mean(left)
                predictors["subspace align k=8 right (to J_bar)"][p.key] = st.mean(
                    right
                )

        # RQ2: cosine and length ratio for the tokens this prompt is steered
        # toward -- its 3 in-category swap targets, not all 16.
        cos_vals, ratio_vals = [], []
        for layer in band:
            key = f"{p.key}|{layer}"
            if key not in vectors["local"]:
                continue
            local = vectors["local"][key]
            for other in cats[p.category]:
                if other == p.arg:
                    continue
                i = target_index[other]
                a, b = local[i].float(), averaged[layer][i].float()
                cos_vals.append(float((a @ b) / (a.norm() * b.norm())))
                ratio_vals.append(float(a.norm() / b.norm()))
        if cos_vals:
            predictors["cos(g_x, g_bar) on swap targets"][p.key] = st.mean(cos_vals)
            predictors["|g_x| / |g_bar| on swap targets"][p.key] = st.mean(ratio_vals)
    return predictors


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    config = load_config(args.config)
    results = REPO_ROOT / "results"
    lo, hi = (int(x) for x in args.band.split("-"))
    band = list(range(lo, hi + 1))

    # ---- outcomes -------------------------------------------------------
    def scores(filename: str, arm: str, strength: float) -> tuple[dict, dict]:
        path = results / "causal" / filename
        if not path.exists():
            return {}, {}
        recs = json.loads(path.read_text())
        rank, margin = defaultdict(list), defaultdict(list)
        for r in recs:
            if r["arm"] == arm and r["strength"] == strength:
                rank[r["prompt_key"]].append(r["target_rank"])
                margin[r["prompt_key"]].append(r["margin"])
        # Lower rank is better, so negate: every score is "higher = steers better".
        return (
            {k: -st.median(v) for k, v in rank.items()},
            {k: st.mean(v) for k, v in margin.items()},
        )

    swap_rank, swap_margin = scores(args.swap_file, "swap_averaged", args.swap_strength)
    add_rank, add_margin = scores(
        args.additive_file, "additive_averaged", args.additive_strength
    )
    baseline = {}
    for filename in (args.swap_file, args.additive_file):
        path = results / "causal" / filename
        if path.exists():
            for r in json.loads(path.read_text()):
                if r["arm"] == "baseline":
                    baseline[r["prompt_key"]] = r["hit"]

    prompts = [p for p in flexible_generalization_prompts() if p.key in swap_rank]
    logger.info("%d prompts with outcomes", len(prompts))

    predictors = build_predictors(config, results, band, args.positions, prompts)

    # ---- correlate ------------------------------------------------------
    out: list[str] = [
        "# C2 - Does the geometry predict which prompts steer well?",
        "",
        "## Methodology",
        "",
        f"**Unit.** The prompt (n = {len(prompts)}). Each contributes 3 steering "
        "trials (its 3 in-category swap targets), aggregated to one score.",
        "",
        "**Outcome.** Steering with the **lens's own averaged direction g_bar** - "
        "this asks whether geometry predicts where the lens works, not whether "
        "substituting g_x helps (that was C1, and it does not). Two operators at "
        f"their best strengths: swap at alpha = {args.swap_strength}, additive at "
        f"alpha = {args.additive_strength}, both over layers L{lo}-L{hi}. Scores "
        "are negated median target rank, so **higher = steers better** for every "
        "column.",
        "",
        "**Predictors.** Per prompt, averaged over the same band: the Jacobian's "
        "distance from J_bar, its effective rank, its top-8 subspace alignment to "
        "J_bar (left and right separately), and the cosine and length ratio of "
        "its pulled-back directions **for the 3 tokens it is actually steered "
        "toward** (not all 16).",
        "",
        "**Sign convention.** The project's hypothesis predicts **negative** "
        "correlations for the distance-like predictors (further from the average "
        "-> steers worse) and **positive** for the agreement-like ones (better "
        "aligned -> steers better).",
        "",
        "**The confound.** Baseline-correct prompts plausibly steer better for "
        "reasons unrelated to Jacobian geometry, and baseline correctness tracks "
        "template difficulty. Every correlation is therefore reported twice: over "
        "all prompts, and over the baseline-correct subset. A predictor that "
        "survives the second column is about geometry; one that does not is "
        "largely re-measuring difficulty.",
        "",
        "## Table C2.1 - Spearman rank correlation with steerability",
        "",
    ]

    ok = [p.key for p in prompts if baseline.get(p.key)]
    logger.info("%d of %d prompts are baseline-correct", len(ok), len(prompts))

    rows = []
    for name, values in predictors.items():
        keys = [p.key for p in prompts if p.key in values]
        if len(keys) < 10:
            continue
        cell = []
        for outcome in (swap_rank, add_rank):
            sub = [k for k in keys if k in outcome and k in baseline]
            if len(sub) < 10:
                cell += ["-", "-", "-"]
                continue
            xs = [values[k] for k in sub]
            ys = [outcome[k] for k in sub]
            zs = [float(baseline[k]) for k in sub]
            raw = spearman(xs, ys)
            par = partial_spearman(xs, ys, zs)
            star = "*" if p_value(raw, len(sub)) < 0.05 else " "
            star_p = "*" if p_value(par, len(sub) - 1) < 0.05 else " "
            sub_ok = [k for k in sub if k in ok]
            cell.append(f"{raw:+.3f}{star}")
            cell.append(f"{par:+.3f}{star_p}")
            cell.append(
                f"{spearman([values[k] for k in sub_ok], [outcome[k] for k in sub_ok]):+.3f}"
                if len(sub_ok) > 9
                else "-"
            )
        rows.append([name, *cell])

    table(
        [
            "Predictor",
            "swap raw",
            "swap partial",
            f"swap base-ok (n={len(ok)})",
            "additive raw",
            "additive partial",
            "additive base-ok",
        ],
        rows,
        out,
    )
    # n is computed, not written down: the base-ok subset is however many
    # prompts this model answers correctly unsteered, which differs per model
    # (31 on qwen3-8b, 38 on qwen3.6-27b). A hardcoded 31 was right for the
    # first model by coincidence and silently wrong for the second.
    ci = 1.96 / max(len(ok) - 1, 1) ** 0.5
    out += [
        f"`*` = p < 0.05, two-sided. **raw** = Spearman over all {len(prompts)} "
        "prompts. **partial** = the same with baseline correctness partialled "
        "out - the column that isolates geometry from task difficulty, and the "
        f"one to read. **base-ok** = raw correlation within the {len(ok)} "
        "baseline-correct prompts, shown as a consistency check; at "
        f"n = {len(ok)} its 95% interval is roughly +/-{ci:.2f}, so it is "
        "underpowered on its own.",
        "",
    ]

    # Reference: how much does baseline correctness alone explain?
    ref = []
    for outcome in (swap_rank, add_rank):
        keys = [p.key for p in prompts if p.key in outcome and p.key in baseline]
        ref.append(
            f"{spearman([float(baseline[k]) for k in keys], [outcome[k] for k in keys]):+.3f}"
        )
    out += [
        "## Table C2.2 - Reference: baseline correctness alone",
        "",
        "The yardstick. Any geometric predictor should be judged against how much "
        "a single bit - did the model answer the unsteered prompt correctly - "
        "already explains.",
        "",
    ]
    table(
        ["Predictor", "swap", "additive"],
        [["baseline correct (0/1)", ref[0], ref[1]]],
        out,
    )

    destination = Path(args.out) if args.out else results / f"c2_{config.name}.md"
    destination.write_text("\n".join(out))
    logger.info("wrote %s", destination)
    print(
        "\n".join(
            out[
                out.index(
                    "## Table C2.1 - Spearman rank correlation with steerability"
                ) :
            ]
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

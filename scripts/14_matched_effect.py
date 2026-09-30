"""C3 - do local directions win once the *dose* is matched?

C1 compared each arm at its own best strength. That is fair on strength but not
on **effect**: alpha means different things for g_x and g_bar, because the two
directions have different lengths and land in differently-conditioned
subspaces. A local arm that perturbs the model half as hard at the same alpha
is under-dosed, not beaten, and C1 cannot tell those apart.

So re-ask the question with the dose held fixed. Two matching variables:

**Output-distribution change** -- ``kl_from_clean``, the KL between the steered
and unsteered next-token distributions. This is "how much did you disturb the
model at all", the standard denominator for an intervention's cost. Matching on
it asks: *per unit of collateral damage, which direction buys more target
effect?*

**Target effect** -- ``margin`` = target logit minus source logit, measured as a
delta against the same trial's unsteered margin. This is the thing steering is
trying to move.

The headline is the dose-response curve: success rate (and delta-margin) as a
function of median KL, both arms on one axis. Whoever is higher at matched KL
wins, independently of any strength convention. The per-prompt table then asks
the user's actual question -- not "does local win on average" but *when* does
it win, if ever.

    python scripts/14_matched_effect.py --config qwen3.6-27b --band 24-34
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
logger = logging.getLogger("matched")


def load_records(results: Path, patterns: list[str]) -> list[dict]:
    """Merged files only, deduplicated by trial identity (see 13_scale_*)."""
    seen: dict[tuple, dict] = {}
    for pattern in patterns:
        for f in sorted(glob.glob(str(results / "causal" / pattern))):
            if "_shard" in Path(f).name:
                continue
            blob = json.loads(Path(f).read_text())
            for r in blob["records"] if isinstance(blob, dict) else blob:
                seen[(r["arm"], r["strength"], r["prompt_key"], r["target_arg"])] = r
    return list(seen.values())


def trial(r: dict) -> tuple:
    return (r["prompt_key"], r["target_arg"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="qwen3.6-27b")
    ap.add_argument("--band", default="24-34")
    ap.add_argument("--results", default=str(REPO_ROOT / "results"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = Path(args.results)

    recs = load_records(
        results,
        [
            f"steering_{args.config}_swap_c1_L{args.band}*.json",
            f"steering_{args.config}_additive_c1_L{args.band}*.json",
            f"steering_{args.config}_both_c1_L{args.band}*.json",
        ],
    )
    if not recs:
        raise SystemExit(f"no causal records for {args.config} L{args.band}")

    # NOTE: the baseline record is NOT a reference for this margin, and
    # subtracting it is a bug I made once already. run_causal grades the
    # baseline with target_id = the prompt's OWN answer and no source_id, so
    # its `margin` is (that answer's logit - 0). The steered rows' `margin` is
    # (swap-target answer logit - own answer logit). Different tokens, different
    # zero: the difference of the two is not an effect size.
    #
    # `margin` on a steered row is already the effect, and already referenced to
    # the model's own preferred answer: negative means the model still prefers
    # its original answer, 0 is the crossover, positive means the intended
    # answer now beats it. The near-zero-dose rows (KL ~ 0.04) are the de facto
    # clean reading of it.
    clean_kl = [r["kl_from_clean"] for r in recs if r["arm"] == "baseline"]
    assert all(abs(v) < 1e-6 for v in clean_kl), "baseline KL must be 0"

    cells: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in recs:
        if r["arm"] == "baseline":
            continue
        cells[(r["arm"], r["strength"])].append(r)

    def summarise(rs: list[dict]) -> dict:
        return {
            "n": len(rs),
            "kl": st.median([r["kl_from_clean"] for r in rs]),
            "hits": sum(bool(r["hit"]) for r in rs),
            "rate": sum(bool(r["hit"]) for r in rs) / len(rs),
            "dmargin": st.median([r["margin"] for r in rs]),
            "rank": st.median([r["target_rank"] for r in rs]),
        }

    summary = {k: summarise(v) for k, v in cells.items()}

    out: list[str] = []
    w = out.append
    w(f"# C3 - matched-dose comparison ({args.config}, layers L{args.band})")
    w("")
    w(
        "`local` = the prompt's own pulled-back direction g_x = J_x^T u_y. "
        "`averaged` = the J-Lens direction g_bar = J_bar^T u_y. Identical "
        "trials, operator and layers; only the direction source differs."
    )
    w("")
    w("## Table C3.1 - Dose-response: effect as a function of disturbance")
    w("")
    w(
        "**KL from clean** is how far the next-token distribution moved "
        "(0 = untouched). **Margin** is the intended answer's logit minus the "
        "answer the model gives unsteered: negative means the original answer "
        "still wins, 0 is the crossover, positive means the intended answer has "
        "overtaken it. Rows are sorted by KL, so reading down a block is "
        "turning the dose up."
    )
    w("")
    rows = []
    for (arm, strength), s in sorted(
        summary.items(), key=lambda kv: (kv[0][0], kv[1]["kl"])
    ):
        rows.append(
            [
                arm,
                f"{strength:g}",
                f"{s['kl']:.2f}",
                f"{s['dmargin']:+.2f}",
                f"{s['hits']}/{s['n']} ({100 * s['rate']:.0f}%)",
                f"{s['rank']:.0f}",
            ]
        )
    _table(
        [
            "Arm",
            "strength",
            "median KL from clean",
            "median margin (target - original)",
            "succeeded",
            "median rank",
        ],
        rows,
        w,
    )

    # ---- matched-KL pairing ----
    w("## Table C3.2 - Matched on output-distribution change")
    w("")
    w(
        "For each local setting, the averaged setting whose median KL is "
        "closest. Same disturbance to the model's output distribution, so "
        "whoever scores higher is genuinely the better direction rather than "
        "the more strongly dosed one."
    )
    w("")
    rows = []
    for operator in ("swap", "additive"):
        loc = {k: v for k, v in summary.items() if k[0] == f"{operator}_local"}
        avg = {k: v for k, v in summary.items() if k[0] == f"{operator}_averaged"}
        if not loc or not avg:
            continue
        for (_arm, strength), s in sorted(loc.items(), key=lambda kv: kv[1]["kl"]):
            match, ms = min(avg.items(), key=lambda kv: abs(kv[1]["kl"] - s["kl"]))
            rows.append(
                [
                    operator,
                    f"{s['kl']:.2f}",
                    f"a={strength:g}: {100 * s['rate']:.0f}% ({s['dmargin']:+.2f})",
                    f"a={match[1]:g}: {100 * ms['rate']:.0f}% ({ms['dmargin']:+.2f})",
                    f"{ms['kl']:.2f}",
                    "local" if s["rate"] > ms["rate"] else "averaged",
                ]
            )
    _table(
        [
            "Operator",
            "local KL",
            "local: success (margin)",
            "averaged at nearest KL: success (margin)",
            "averaged KL",
            "winner",
        ],
        rows,
        w,
    )

    # ---- per-prompt: when, if ever, does local win? ----
    w("## Table C3.3 - Per-trial: when does local win?")
    w("")
    w(
        "The closest-KL pair for each operator, compared trial by trial on the "
        "same 192 (prompt, target) pairs. This is the 'when' question: an "
        "arm that loses on average could still win on an identifiable subset."
    )
    w("")
    rows = []
    for operator in ("swap", "additive"):
        loc = {k: v for k, v in summary.items() if k[0] == f"{operator}_local"}
        avg = {k: v for k, v in summary.items() if k[0] == f"{operator}_averaged"}
        if not loc or not avg:
            continue
        # Pick the local setting nearest the averaged arm's own best KL.
        best_avg = max(avg.items(), key=lambda kv: kv[1]["rate"])
        best_loc = min(loc.items(), key=lambda kv: abs(kv[1]["kl"] - best_avg[1]["kl"]))
        lr = {trial(r): r for r in cells[best_loc[0]]}
        ar = {trial(r): r for r in cells[best_avg[0]]}
        shared = sorted(set(lr) & set(ar))
        loc_only = sum(1 for t in shared if lr[t]["hit"] and not ar[t]["hit"])
        avg_only = sum(1 for t in shared if ar[t]["hit"] and not lr[t]["hit"])
        both = sum(1 for t in shared if ar[t]["hit"] and lr[t]["hit"])
        neither = len(shared) - loc_only - avg_only - both
        better_margin = sum(1 for t in shared if lr[t]["margin"] > ar[t]["margin"])
        rows.append(
            [
                operator,
                f"a={best_loc[0][1]:g} (KL {best_loc[1]['kl']:.2f})",
                f"a={best_avg[0][1]:g} (KL {best_avg[1]['kl']:.2f})",
                str(both),
                str(loc_only),
                str(avg_only),
                str(neither),
                f"{better_margin}/{len(shared)}",
            ]
        )
    _table(
        [
            "Operator",
            "local setting",
            "averaged setting",
            "both hit",
            "only local hit",
            "only averaged hit",
            "neither",
            "local moved target further",
        ],
        rows,
        w,
    )

    dest = Path(args.out) if args.out else results / f"c3_matched_{args.config}.md"
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

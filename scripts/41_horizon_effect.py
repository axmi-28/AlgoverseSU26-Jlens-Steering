"""C27 - does writing at horizon d move the output d tokens LATER?

C26 showed per-horizon write directions are distinct and smoothly ordered.
Distinct is necessary but not sufficient for positional addressing: the
directions could differ in what they influence rather than in when. This is
the discriminating test.

``v_y^(d)`` is written additively across the band at the prompt's positions,
and we record how much the log-probability of ``y`` rises at each of the next
16 output positions. The continuation is teacher-forced to the model's own
clean greedy text and held fixed across arms, so position ``k`` means the same
thing in every row.

Two hypotheses, with different signatures:

* **addressable in time** -- arm ``d`` peaks at ``k ~ d``; a diagonal band.
* **addressable in duration** -- every arm peaks immediately, but far-horizon
  arms decay more slowly; ordered curves with a common peak.

    python scripts/41_horizon_effect.py
"""

from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["h0", "h1", "h2", "h3_4", "h5_8", "h9_16", "h17plus", "total"]
LABEL = {
    "h0": "d=0",
    "h1": "d=1",
    "h2": "d=2",
    "h3_4": "d=3-4",
    "h5_8": "d=5-8",
    "h9_16": "d=9-16",
    "h17plus": "d>=17",
    "total": "all (J_bar)",
}
DOSES = [("hzfx01", 0.1), ("hzfx025", 0.25), ("hzfx", 1.0)]
N = 16
LATE = range(8, N)


def curves(tag: str) -> dict[str, list[float]]:
    recs = [
        x
        for f in glob.glob(str(ROOT / f"results/horizon/horizon_effect_*_{tag}_shard*.json"))
        for x in json.loads(Path(f).read_text())
    ]
    if not recs:
        raise SystemExit(f"no records for {tag}")
    out = {}
    for a in ARMS:
        full = f"wikitext_a_{a}"
        out[a] = [
            st.mean(x["delta"] for x in recs if x["arm"] == full and x["k"] == k)
            for k in range(N)
        ]
    return out


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main() -> None:
    by_dose = {tag: curves(tag) for tag, _ in DOSES}

    lines = [
        "# C27 - is horizon addressable in time, or in duration? (qwen3-8b)",
        "",
        "96 flexible-generalization prompts, wikitext-fitted horizon lens (64 "
        "documents). `v_y^(d)` written additively at the prompt positions "
        "across the band; the table is the mean rise in log p(y) at each of "
        "the next 16 output positions, with the continuation teacher-forced to "
        "the clean greedy text so `k` is comparable across arms.",
        "",
        "## The answer: duration, not time",
        "",
        "**No diagonal band.** Every arm peaks at k=0 or k=1 regardless of its "
        "horizon, at every dose tested. Writing with a far-horizon direction "
        "does not delay the effect's onset.",
        "",
        "**But the curves are ordered in how fast they decay.** The `d=0` arm "
        "collapses within a couple of tokens while every `d>=1` arm sustains, "
        "and among those the far horizons sustain longest.",
        "",
        "### Mean delta log p(y) by output position, strength 0.1",
        "",
        "| arm | " + " | ".join(f"k={k}" for k in range(0, N, 2)) + " |",
        "|---|" + "---|" * len(range(0, N, 2)),
    ]
    c = by_dose["hzfx01"]
    for a in ARMS:
        lines.append(
            f"| {LABEL[a]} | " + " | ".join(f"{c[a][k]:+.2f}" for k in range(0, N, 2)) + " |"
        )

    lines += [
        "",
        "### Persistence: late window (k=8-15) relative to onset (k=0)",
        "",
        "The onset column is saturated (see the caveat below), so the ratio is "
        "the quantity to read: it measures how much of the initial effect "
        "survives eight or more tokens later.",
        "",
        "| arm | " + " | ".join(f"a={d} late / ratio" for _, d in DOSES) + " |",
        "|---|" + "---|" * len(DOSES),
    ]
    ratios: dict[str, float] = {}
    for a in ARMS:
        cells = []
        for tag, _ in DOSES:
            cur = by_dose[tag][a]
            late = st.mean(cur[k] for k in LATE)
            cells.append(f"{late:.2f} / {late / cur[0]:.3f}")
            if tag == "hzfx01":
                ratios[a] = late / cur[0]
        lines.append(f"| {LABEL[a]} | " + " | ".join(cells) + " |")

    order = ["h0", "h1", "h2", "h3_4", "h5_8", "h9_16", "h17plus"]
    rho = spearman(list(range(len(order))), [ratios[a] for a in order])
    lines += [
        "",
        f"Across the seven buckets, Spearman(horizon rank, persistence) = "
        f"**{rho:+.2f}** at strength 0.1. The ordering is monotone apart from "
        "the near-horizon buckets, which sit together.",
        "",
        "## What this establishes, and what it does not",
        "",
        "**Establishes:** the off-diagonal term is not merely *not-the-"
        "diagonal*. Resolved by distance it behaves as a decay-rate control: "
        "`d=0` writes an effect that is gone within ~2 tokens, near horizons "
        "sustain moderately, and far horizons sustain longest. That is a "
        "direct, positive confirmation of the mechanism C21 and C22 inferred "
        "from geometry -- the diagonal is a *say it now* channel and the "
        "off-diagonal is a *keep it available* channel -- measured here as a "
        "time course rather than as an outcome.",
        "",
        "**Does not establish:** positional addressing. There is no evidence a "
        "write can be aimed at a specific future position, so the multi-token "
        "composition idea does not follow from this and should not be claimed.",
        "",
        "## The caveat that matters",
        "",
        "**k=0 is saturated at every dose tested.** The onset lift is +13 to "
        "+15 log-probs for all arms at strength 0.1, 0.25 and 1.0 -- the "
        "additive edit scales with the residual norm, so even a=0.1 is a large "
        "perturbation and the target token is already at the top of the "
        "distribution. A delayed-onset effect, if one existed, could be hidden "
        "under that ceiling. The persistence result is the robust part, since "
        "it is a *relative* decay and reproduces across a tenfold dose range; "
        "the no-diagonal-band conclusion is weaker and would be firmer after a "
        "run at a dose low enough to leave k=0 unsaturated.",
    ]

    out = ROOT / "results/c27_horizon_effect_qwen3-8b.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

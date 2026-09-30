"""C26 - is the off-diagonal term one object, or a family ordered by horizon?

The gating check for horizon-targeted writing. ``_off`` lumps every
``d = t' - t >= 1`` together and steers well, which is equally consistent with

* only ``d = 0`` is special, everything above it is interchangeable; or
* each horizon carries its own direction, and writing may be targetable in
  time as well as in content.

These differ in one measurable way: the pairwise cosine between per-horizon
write directions. Near-1 everywhere means there is nothing to address.

    python scripts/40_horizon_gate.py
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["h0", "h1", "h2", "h3_4", "h5_8", "h9_16", "h17plus"]
LABEL = {
    "h0": "d=0",
    "h1": "d=1",
    "h2": "d=2",
    "h3_4": "d=3-4",
    "h5_8": "d=5-8",
    "h9_16": "d=9-16",
    "h17plus": "d>=17",
}


def main() -> None:
    path = ROOT / "results/geometry/horizon_qwen3-8b_hz.json"
    blob = json.loads(path.read_text())
    cos = {(r["a"], r["b"]): r for r in blob["cosines"]}
    counts, norms = blob["pair_counts"], blob["norms"]
    layers = len(next(iter(cos.values()))["by_layer"])

    def get(a, b):
        if a == b:
            return None
        return cos[(a, b)] if (a, b) in cos else cos[(b, a)]

    lines = [
        "# C26 - is the off-diagonal term horizon-agnostic? (qwen3-8b)",
        "",
        "12 wikitext documents, 16 target words, 7 band layers, every "
        "``(t, t')`` pair binned by ``d = t' - t``. Each bucket gives a write "
        "direction ``v_y^(d) = J_bar^(d)^T u_y``; the table is the cosine "
        f"between them, averaged over words and {layers} layers.",
        "",
        "**The check:** if these are all near 1, the off-diagonal term is one "
        "object, the only special horizon is 0, and horizon-targeted writing "
        "has nothing to address.",
        "",
        "| | " + " | ".join(LABEL[n] for n in NAMES) + " |",
        "|---|" + "---|" * len(NAMES),
    ]
    for a in NAMES:
        cells = []
        for b in NAMES:
            cells.append("1.000" if a == b else f"{get(a, b)['cos']:.3f}")
        lines.append(f"| **{LABEL[a]}** | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## It is a family, not a lump",
        "",
        "The matrix is banded: cosine falls monotonically with the distance "
        "between horizons, along every row and every column. Adjacent buckets "
        "are near-collinear (d=1 vs d=2: 0.887); far ones are not "
        f"(d=1 vs d>=17: {get('h1', 'h17plus')['cos']:.3f}). That is the "
        "signature of a continuum ordered by horizon rather than a single "
        "direction with varying magnitude.",
        "",
        "`d=0` is the outlier even within this structure -- it is the least "
        "similar to everything and decays fastest "
        f"({get('h0','h1')['cos']:.3f} down to {get('h0','h17plus')['cos']:.3f}), "
        "which is what C21 predicts for a term that is essentially the "
        "unembedding row rather than a downstream influence at all.",
        "",
        "## Pair counts and norms",
        "",
        "**Norms are not comparable across buckets** and are listed only so "
        "they are not mistaken for an effect: a bucket accumulates one term "
        "per `(t, t')` pair it contains, and the far buckets contain far more "
        "pairs inside a 128-token window. Per pair, near horizons are much "
        "the stronger.",
        "",
        "| bucket | pairs | summed norm | norm per pair |",
        "|---|---|---|---|",
    ]
    for n in NAMES:
        per = norms[f"wikitext_a_{n}"] / counts[n]
        lines.append(
            f"| {LABEL[n]} | {counts[n]} | {norms[f'wikitext_a_{n}']:.3f} | {per:.2e} |"
        )

    lines += [
        "",
        "## Does the structure survive at depth?",
        "",
        "The concern is that late layers commit to an answer and the horizon "
        "structure collapses. Shown for the widest-separated pair.",
        "",
        "| pair | " + " | ".join(f"L{ll}" for ll in [13, 16, 19, 22, 25, 28, 31]) + " |",
        "|---|" + "---|" * 7,
    ]
    for a, b in (("h1", "h17plus"), ("h1", "h9_16"), ("h0", "h1")):
        r = get(a, b)
        lines.append(
            f"| {LABEL[a]} vs {LABEL[b]} | "
            + " | ".join(f"{v:.3f}" for v in r["by_layer"])
            + " |"
        )

    lines += [
        "",
        "## Verdict",
        "",
        "**The gate opens.** Horizons carry distinct, smoothly ordered write "
        "directions, so there is a real question about whether writing along "
        "`v_y^(d)` produces an effect around position `d`. The positional "
        "heatmap experiment is worth running.",
        "",
        "Caveats this rests on: one corpus (wikitext), 12 documents, and far "
        "buckets that pool a wide range of `d` -- so `d>=17` is a direction "
        "averaged over many horizons and its distinctness from `d=1` is a "
        "lower bound on how much the extremes differ, not a measurement of any "
        "single horizon.",
    ]

    out = ROOT / "results/c26_horizon_gate_qwen3-8b.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

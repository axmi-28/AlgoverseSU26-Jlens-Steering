"""C24 - the read side of the component split, measured on our own lens.

The read/write asymmetry is the paper's thesis, and until this run it was an
inference across two studies: Yan et al. masked the diagonal and reported that
READOUT degrades; we masked it and reported that WRITING roughly doubles.
Different models, different prompt sets, different metrics. This measures both
sides on one lens, one model and one prompt set.

**Restricted vocabulary, and it matters.** The paper's readout ranks over the
whole vocabulary, which needs the full mean matrices. The component sweep only
materialized per-word pullbacks -- full matrices would cost d_model backward
passes per position pair instead of 16 -- so the rank here is among the 16
fitted argument words. Every arm faces the identical 16-way choice, so the
comparison is sound; the absolute numbers are NOT comparable to C14's pass@k,
and a 16-way task has less headroom to reveal a small readout cost than a
150k-way one does.

    python scripts/37_component_readout.py
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ORDER = [
    "published",
    "gsm8k_full",
    "gsm8k_diag",
    "gsm8k_off",
    "wikitext_a_full",
    "wikitext_a_diag",
    "wikitext_a_off",
]
#: Write side at a=1.0, from results/c9_domain_causal_qwen3-8b_comp.md, so the
#: two sides of the asymmetry sit in one table.
WRITE = {
    "published": 20,
    "gsm8k_full": 35,
    "gsm8k_diag": 16,
    "gsm8k_off": 60,
    "wikitext_a_full": 20,
    "wikitext_a_diag": 17,
    "wikitext_a_off": 50,
}


def main() -> None:
    path = ROOT / "results/readout/component_readout_qwen3-8b_comp.json"
    recs = json.loads(path.read_text())
    layers = sorted({r["layer"] for r in recs})
    arms = [a for a in ORDER if a in {r["arm"] for r in recs}]
    n_prompts = len({r["key"] for r in recs})

    for arm in arms:
        n = len([r for r in recs if r["arm"] == arm])
        if n != n_prompts * len(layers):
            raise SystemExit(f"{arm} has {n} rows, expected {n_prompts * len(layers)}")

    def rows(a):
        return [r for r in recs if r["arm"] == a]

    lines = [
        "# C24 - does dropping the diagonal cost anything on the READ side? (qwen3-8b)",
        "",
        f"`{path.name}`. {n_prompts} eval prompts x {len(layers)} layers "
        "(13-31 by 3), scored at the argument and readout positions of the "
        "clean forward pass -- the same positions and band C14 and the loading "
        "control use.",
        "",
        "**Restricted vocabulary.** The rank is of the prompt's own argument "
        "among the 16 fitted target words, not over the full vocabulary: the "
        "component digest holds per-word pullbacks, not full mean matrices. "
        "Arms are directly comparable to each other; these numbers are **not** "
        "comparable to C14's pass@k, and a 16-way choice has less headroom to "
        "expose a small readout cost than a full-vocabulary rank would.",
        "",
        "| arm | top-1 | MRR | write (non-math /105) |",
        "|---|---|---|---|",
    ]
    for arm in arms:
        rs = rows(arm)
        lines.append(
            f"| {arm} | {st.mean(r['top1'] for r in rs):.3f} | "
            f"{st.mean(r['rr'] for r in rs):.3f} | {WRITE.get(arm, '')} |"
        )

    lines += [
        "",
        "## The asymmetry, stated plainly",
        "",
    ]
    for corpus in ("gsm8k", "wikitext_a"):
        rf = st.mean(r["top1"] for r in rows(f"{corpus}_full"))
        ro = st.mean(r["top1"] for r in rows(f"{corpus}_off"))
        lines.append(
            f"* **{corpus}**: reading {rf:.3f} -> {ro:.3f} "
            f"({ro - rf:+.3f}), writing {WRITE[f'{corpus}_full']} -> "
            f"{WRITE[f'{corpus}_off']} of 105 "
            f"({WRITE[f'{corpus}_off'] - WRITE[f'{corpus}_full']:+d})."
        )
    lines += [
        "",
        "Dropping the diagonal leaves readout where it was and roughly doubles "
        "writing. The diagonal arm alone reads *worst* of all (0.46, 0.55), so "
        "the metric is not saturated and does discriminate -- it simply does "
        "not charge anything for the removal.",
        "",
        "## By layer (top-1)",
        "",
        "| arm | " + " | ".join(f"L{ll}" for ll in layers) + " |",
        "|---|" + "---|" * len(layers),
    ]
    for arm in arms:
        cells = [
            f"{st.mean(r['top1'] for r in rows(arm) if r['layer'] == ll):.3f}"
            for ll in layers
        ]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "The `_diag` arms climb from ~0.25-0.42 early to ~0.80-0.83 at L31 "
        "while every other arm is flat. That is the signature C21 predicts for "
        "an identity/unembedding path: it only becomes predictive of the "
        "answer as the residual approaches the output, which is also why it "
        "reads acceptably at the last layer and writes nothing anywhere.",
    ]

    out = ROOT / "results/c24_component_readout_qwen3-8b.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

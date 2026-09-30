"""C21 - how much of each write direction is just the unembedding row u_y?

The mechanism candidate for C18. The residual stream is a sum, so the
``t' = t`` block of ``d h_L,t / d h_l,t`` carries an identity path. If that
path dominates the diagonal term of Eq. 20, then ``J_diag^T u_y ~ u_y``: the
short-horizon half of a J-lens write direction is approximately *logit-lens
steering*, and the ``_off`` arm is what is left once it is removed.

The prediction, if true:

* ``gsm8k_diag`` / ``wikitext_a_diag`` sit far above the null on cos(v, u_y);
* ``_off`` sits at or near the null;
* ``_full`` sits between them, since it is the sum;
* cos(full, diag) is high -- the diagonal supplied most of the full direction.

Every cosine is read against ``cos_null``, the mean |cos| against 256 random
unembedding rows. The rows are not orthogonal and d_model is 4096, so a raw
cosine has no interpretation without the level it must clear.

    python scripts/34_direction_geometry.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

#: Swap hits at alpha=1.0, transcribed from results/c9_domain_causal_qwen3-8b_comp.md,
#: so the geometry can be read directly against the outcome it is meant to explain.
MATH39 = {
    "published": 0,
    "gsm8k_full": 11,
    "gsm8k_diag": 0,
    "gsm8k_off": 18,
    "wikitext_a_full": 0,
    "wikitext_a_diag": 0,
    "wikitext_a_off": 0,
}
NONMATH105 = {
    "published": 20,
    "gsm8k_full": 35,
    "gsm8k_diag": 16,
    "gsm8k_off": 60,
    "wikitext_a_full": 20,
    "wikitext_a_diag": 17,
    "wikitext_a_off": 50,
}
#: Two-hop latent-concept swap at alpha=1.0, from the C18xC19 cross.
TWOHOP = {
    "gsm8k_full": 0.250,
    "gsm8k_diag": 0.200,
    "gsm8k_off": 0.418,
    "wikitext_a_full": 0.268,
    "wikitext_a_diag": 0.200,
    "wikitext_a_off": 0.455,
    "published": 0.232,
}

ORDER = [
    "published",
    "gsm8k_full",
    "gsm8k_diag",
    "gsm8k_off",
    "wikitext_a_full",
    "wikitext_a_diag",
    "wikitext_a_off",
]


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main() -> None:
    blob = json.loads(
        (RESULTS / "geometry" / "geometry_qwen3-8b_comp.json").read_text()
    )
    records, cross = blob["records"], blob["cross"]
    layers = sorted({r["layer"] for r in records})
    arms = [a for a in ORDER if a in {r["arm"] for r in records}]

    def rows(arm):
        return [r for r in records if r["arm"] == arm]

    lines = [
        "# C21 - is the diagonal half of the write direction the unembedding row? (qwen3-8b)",
        "",
        "`geometry_qwen3-8b_comp.json`. For each arm, layer and target word, the "
        "cosine between that arm's write direction `v_y = J_bar^T u_y` and the "
        "token's own unembedding row `u_y` -- which is exactly the logit-lens "
        "steering direction. 16 words x 7 layers (13-31 by 3) per arm.",
        "",
        "`null` is the mean |cos| of the same vector against 256 random "
        "unembedding rows. It is the level a cosine has to clear: the rows are "
        "not orthogonal and d_model is 4096, so a raw cosine says nothing on "
        "its own. `x null` is cos_self divided by that level.",
        "",
        "| arm | cos(v, u_y) | null | x null | mean norm | math /39 | non-math /105 | two-hop |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm in arms:
        rs = rows(arm)
        cs = statistics.mean(r["cos_self"] for r in rs)
        nl = statistics.mean(r["cos_null"] for r in rs)
        nm = statistics.mean(r["norm"] for r in rs)
        th = TWOHOP.get(arm)
        lines.append(
            f"| {arm} | {cs:+.4f} | {nl:.4f} | {cs / nl:.1f}x | {nm:.3f} | "
            f"{MATH39.get(arm, '')} | {NONMATH105.get(arm, '')} | "
            f"{f'{th:.3f}' if th is not None else ''} |"
        )

    lines += ["", "## By layer: cos(v, u_y)", ""]
    lines.append("| arm | " + " | ".join(f"L{ll}" for ll in layers) + " |")
    lines.append("|---|" + "---|" * len(layers))
    for arm in arms:
        cells = []
        for ll in layers:
            rs = [r for r in rows(arm) if r["layer"] == ll]
            cells.append(f"{statistics.mean(r['cos_self'] for r in rs):+.3f}")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## How much of the full direction did each half supply?",
        "",
        "Cosine between arms, same word and layer, averaged over both.",
        "",
        "| corpus | full vs diag | full vs off | diag vs off |",
        "|---|---|---|---|",
    ]
    for corpus in sorted({c["corpus"] for c in cross}):
        cells = []
        for pair in ("full_vs_diag", "full_vs_off", "diag_vs_off"):
            sub = [c["cos"] for c in cross if c["corpus"] == corpus and c["pair"] == pair]
            cells.append(f"{statistics.mean(sub):+.3f}" if sub else "")
        lines.append(f"| {corpus} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Does alignment with u_y track steering failure across arms?",
        "",
        "Seven arms, so this is a direction and not a law -- and the arms are "
        "algebraically related (`full = diag + off`), which is exactly why the "
        "projection control below is the experiment that would settle it.",
        "",
        "| outcome | n | Spearman vs cos(v, u_y) |",
        "|---|---|---|",
    ]
    for label, table in (
        ("math /39", MATH39),
        ("non-math /105", NONMATH105),
        ("two-hop", TWOHOP),
    ):
        shared = [a for a in arms if a in table]
        xs = [statistics.mean(r["cos_self"] for r in rows(a)) for a in shared]
        ys = [table[a] for a in shared]
        lines.append(f"| {label} | {len(shared)} | {spearman(xs, ys):+.2f} |")

    lines += [
        "",
        "The `_off` arms' low cosine is partly algebraic: if `v_diag ~ c*u_y` "
        "then subtracting it must lower the cosine. What is *not* algebraic is "
        "that `v_diag` aligns with `u_y` at all -- it could have sat at the "
        "0.012 null and does not -- nor that the arms with the least `u_y` "
        "content steer best, which is the independent C18 result.",
        "",
        "**The control this needs:** explicitly project `u_y` out of the FULL "
        "lens direction (Gram-Schmidt, keeping the diagonal otherwise intact) "
        "and re-run the swap. If that alone recovers the `_off` gain, the "
        "mechanism is 'the write direction was contaminated by the unembedding "
        "row', and the horizon framing is a way of arriving at it rather than "
        "the operative fact. If it does not, the off-diagonal term carries "
        "something beyond the absence of `u_y`.",
        "",
        "## Per-word cos(v, u_y), middle of the band (L22)",
        "",
    ]
    words = sorted({r["word"] for r in records})
    lines.append("| word | " + " | ".join(arms) + " |")
    lines.append("|---|" + "---|" * len(arms))
    for word in words:
        cells = []
        for arm in arms:
            hit = [r for r in rows(arm) if r["word"] == word and r["layer"] == 22]
            cells.append(f"{hit[0]['cos_self']:+.3f}" if hit else "")
        lines.append(f"| {word} | " + " | ".join(cells) + " |")

    out = RESULTS / "c21_direction_geometry_qwen3-8b.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

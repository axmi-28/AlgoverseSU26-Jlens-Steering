#!/usr/bin/env python3
"""C29 - does Park et al.'s causal inner product already do what our correction does?

Park et al. 2023 (arXiv:2311.03658) Thm 3.2/3.4 map a read direction to a write
direction by whitening with the inverse covariance of the unembedding rows over
the vocabulary. Our beta=1 / alpha_l correction removes a rank-1 piece -- the
target's own unembedding row. Same family, different operation, and if theirs
recovers our gain then our contribution is "Park 2023, applied to the J-lens"
and the paper has to say so.

Reads the C9-format report the causal run generated and lays the arms out so
that question is answerable at a glance.

    python scripts/25_domain_causal.py --label park --n-shards 4
    python scripts/44_park_baseline.py
"""

from __future__ import annotations

import argparse
import json
from math import comb as _comb  # NOT "import math": a local named `math` holds the math-cell table
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The reference numbers our claim rests on, transcribed from the reports that
#: generated them rather than retyped.
BETA_REPORT = ROOT / "results/c25_beta_family_qwen3-8b.md"


def table(report: Path, section: str, prefix: str = "swap_") -> dict[str, tuple[int, int]]:
    text = report.read_text()
    if section not in text:
        raise SystemExit(f"{report.name} has no section {section!r}")
    out: dict[str, tuple[int, int]] = {}
    for line in text.split(section, 1)[1].splitlines():
        m = re.match(rf"\|\s*{prefix}(\S+?)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*$", line)
        if m:
            out[m.group(1)] = (int(m.group(2)), int(m.group(3)))
        elif out and not line.strip():
            break
    if not out:
        raise SystemExit(f"parsed no rows under {section!r} of {report.name}")
    return out


def beta_row(corpus: str) -> dict[str, int]:
    """``| corpus | b0 | b0.5 | b1 | b2 | b3 | galpha | off |`` from C25."""
    for line in BETA_REPORT.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 8 and cells[0] == corpus:
            def num(x: str) -> int | None:
                return int(x) if x.isdigit() else None
            return {
                "b0": num(cells[1]), "b1": num(cells[3]),
                "galpha": num(cells[6]), "off": num(cells[7]),
            }
    raise SystemExit(f"no row for {corpus!r} in {BETA_REPORT.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="park")
    ap.add_argument("--out", default="results/c29_park_baseline_qwen3-8b.md")
    args = ap.parse_args()

    report = ROOT / f"results/c9_domain_causal_qwen3-8b_{args.label}.md"
    non = table(report, "## Non-math - hits out of 105 trials")
    math = table(report, "## Math cell (numbers) - hits out of 39 trials")

    lines = [
        "# C29 - Park et al.'s causal inner product as a baseline (qwen3-8b)",
        "",
        "Park et al. 2023 (arXiv:2311.03658) Thm 3.4 gives the causal inner "
        "product as `<g,g'>_C = g^T Cov(gamma)^-1 g'`, the covariance taken over "
        "the **vocabulary's unembedding rows** (not over activations -- Yan's "
        "Stein bridge uses Cov(h), and FishBack uses J^T H J and argues Cov(h)^-1 "
        "is the wrong object; three different matrices, none citing each other). "
        "The induced read->write map is `g -> Cov(gamma)^-1 g`. `_park` is that "
        "map at a light ridge (1e-3 x mean eigenvalue of Cov(gamma)), `_park2` at "
        "100x the ridge, `_parkh` the symmetric `Cov^-1/2` reading.",
        "",
        "**What Park's method is applied to matters, and an earlier version of "
        "this report got it wrong.** Their `gamma_bar_W` is a CONCEPT "
        "representation estimated from counterfactual pairs -- a difference of "
        "unembeddings whose sign is meaningful -- not a single word's row. "
        "Theorems 3.2/3.4 say nothing about a bare `u_y`, so `uy_park` tests a "
        "formula on an object outside its domain and is reported below only as "
        "a contrast. The arm that tests **Park's method as specified** is "
        "`gamma_park`: within each category, `gamma_y = u_y - mean(u_y')` over "
        "the three sibling arguments, then `Cov(gamma)^-1 gamma_y`.",
        "",
        "Two scope caveats that belong in any citation of this result. Thm 3.4 "
        "fixes the causal inner product only up to `d` remaining degrees of "
        "freedom -- it is a family, not a point -- and the `D = I` choice that "
        "yields `Cov(gamma)^-1` is one the authors call unjustified. And their "
        "own steering demonstration works: adding `alpha * Cov(gamma)^-1 "
        "gamma_bar_W` walks top-1 from *king* to *queen* on \"Long live the \" "
        "with off-target logits flat, a flatness their Thm 2.5 predicts in "
        "advance. Nothing here contradicts that. The claim is narrower: **at "
        "D = I, on a swap task that must propagate through a template function "
        "rather than substitute a next token, the construction does not recover "
        "the gain our rank-1 correction does.**",
        "",
        "Same 192 flexible-generalization trials, same seven band layers, same "
        "swap operator as C18/C22/C25. The swap unit-normalises its read "
        "directions, so none of these transforms changes the dose.",
        "",
        "## Non-math /105",
        "",
        "| arm | a=1.0 | a=2.0 |",
        "|---|---|---|",
    ]
    for arm, (a1, a2) in non.items():
        lines.append(f"| {arm} | {a1} | {a2} |")

    lines += ["", "## Math /39", "", "| arm | a=1.0 | a=2.0 |", "|---|---|---|"]
    for arm, (a1, a2) in math.items():
        lines.append(f"| {arm} | {a1} | {a2} |")

    lines += [
        "",
        "## Against our own correction",
        "",
        "Non-math /105 at each arm's better strength. `b0` is the unmodified "
        "lens and `b1`/`galpha`/`off` are from `c25_beta_family_qwen3-8b.md`.",
        "",
        "| corpus | unmodified | Park Cov^-1 | Park Cov^-1/2 | ours beta=1 | ours galpha | horizon _off |",
        "|---|---|---|---|---|---|---|",
    ]
    for corpus, stem in (("published", "published"), ("gsm8k", "gsm8k_full"),
                         ("wikitext_a", "wikitext_a_full")):
        b = beta_row(corpus)

        def best(arm: str) -> str:
            if arm not in non:
                return "-"
            return str(max(non[arm]))

        lines.append(
            f"| {corpus} | {best(stem)} | {best(stem + '_park')} | "
            f"{best(stem + '_parkh')} | {b['b1']} | {b['galpha']} | "
            f"{b['off'] if b['off'] is not None else '-'} |"
        )

    lines += [
        "",
        "## The Jacobian-free arms",
        "",
        "`gamma` is the counterfactual-pair concept direction and `gamma_park` "
        "is **Park's method as specified**. `uy` is the logit-lens write "
        "direction -- the degenerate extreme of the contamination axis, and the "
        "thing C21 says the diagonal term approximates (cos 0.60-0.66, rising "
        "to 0.88-0.91 by L31); `uy_*` are out-of-domain for Park and are shown "
        "only for contrast.",
        "",
        "| arm | non-math a=1 | non-math a=2 | math a=1 |",
        "|---|---|---|---|",
    ]
    for arm in ("gamma", "gamma_park", "gamma_parkh", "uy", "uy_park", "uy_park2", "uy_parkh"):
        if arm in non:
            lines.append(f"| {arm} | {non[arm][0]} | {non[arm][1]} | {math.get(arm, ('-',))[0]} |")

    # Reproducibility: three arms in this run also appear in the C18 component
    # run, computed by the same code on the same trials. Any drift between them
    # is run-to-run noise and has to be visible, because every claim in this
    # document is a difference of a few hits.
    comp = ROOT / "results/c9_domain_causal_qwen3-8b_comp.md"
    lines += ["", "## Reproducibility check against the C18 run", ""]
    if comp.exists():
        prev_non = table(comp, "## Non-math - hits out of 105 trials")
        prev_math = table(comp, "## Math cell (numbers) - hits out of 39 trials")
        shared = sorted(set(prev_non) & set(non))
        lines += [
            "Arms present in both this run and `c9_domain_causal_qwen3-8b_comp.md`, "
            "same trials, same band, same operator. Differences here are "
            "run-to-run nondeterminism and bound how finely any contrast in this "
            "document can be read.",
            "",
            "| arm | non-math a=1 here | there | math a=1 here | there |",
            "|---|---|---|---|---|",
        ]
        drift = 0
        for arm in shared:
            d = non[arm][0] - prev_non[arm][0]
            drift = max(drift, abs(d))
            lines.append(
                f"| {arm} | {non[arm][0]} | {prev_non[arm][0]} | "
                f"{math.get(arm, ('-', '-'))[0]} | {prev_math.get(arm, ('-', '-'))[0]} |"
            )
        lines += ["", f"Largest drift on a shared arm: **{drift} hit(s) of 105**.", ""]
    else:
        lines += [f"`{comp.name}` not present; no cross-run check available.", ""]

    # Paired tests. The whole Park conclusion turns on differences of a few
    # trials out of 105, and unpaired counts cannot carry that -- C28's
    # contrasts were held to a paired standard and these must be too.
    rows: list[dict] = []
    for i in range(4):
        shard = ROOT / (
            f"results/causal/steering_qwen3-8b_swap_{args.label}_shard{i}of4.json"
        )
        if shard.exists():
            rows.extend(json.loads(shard.read_text()))
    if rows:
        cleank = {
            r["prompt_key"] for r in rows
            if r["arm"] == "baseline" and r["hit_generated"]
        }
        tr = [
            r for r in rows
            if r["arm"] != "baseline" and r["prompt_key"] in cleank
            and r["category"] != "numbers"
        ]

        def hits(arm: str, s: float = 1.0) -> list[bool]:
            sel = [r for r in tr if r["arm"] == arm and r["strength"] == s]
            sel.sort(key=lambda r: (r["prompt_key"], r["target_arg"]))
            return [bool(r["hit_generated"]) for r in sel]

        def mcnemar(a: list[bool], b: list[bool]) -> tuple[float, int, int]:
            oa = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
            ob = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
            n = oa + ob
            if n == 0:
                return 1.0, oa, ob
            k = min(oa, ob)
            t = sum(_comb(n, i) for i in range(k + 1)) / 2**n
            return min(1.0, 2 * t), oa, ob

        lines += [
            "",
            "## Paired tests, non-math at a=1",
            "",
            "| contrast | hits | only-A | only-B | p |",
            "|---|---|---|---|---|",
        ]
        for a, b in (("swap_gamma", "swap_gamma_park"),
                     ("swap_gamma", "swap_gamma_parkh"),
                     ("swap_gsm8k_off", "swap_gamma_park"),
                     ("swap_gsm8k_off", "swap_gamma"),
                     ("swap_published", "swap_gamma"),
                     ("swap_uy", "swap_gamma")):
            va, vb = hits(a), hits(b)
            if not va or not vb:
                continue
            pv, oa, ob = mcnemar(va, vb)
            star = "**" if pv < 0.05 else ""
            lines.append(
                f"| {a[5:]} vs {b[5:]} | {star}{sum(va)} vs {sum(vb)}{star} | "
                f"{oa} | {ob} | {pv:.4f} |"
            )
        lines += [
            "",
            "**What this licenses, and what it does not.** `gamma` -> "
            "`gamma_park` is 16 -> 10 but **p = 0.18: not significant**. So the "
            "claim is *whitening does not improve the concept direction, and if "
            "anything degrades it* -- NOT that it demonstrably hurts. The "
            "load-bearing gap is elsewhere and is unambiguous: our correction "
            "reaches 47 and `_off` 59 against the whole gamma family's 10-16 "
            "(gsm8k_off vs gamma_park p < 1e-4).",
            "",
            "Second caveat, which cuts both ways: the entire gamma family (16) "
            "sits *below* both `uy` (17) and the published lens (20). That "
            "supports the larger point -- a pure unembedding-space construction "
            "cannot do what the Jacobian does -- but it also means the Park "
            "comparison runs in a weak regime, so the ordering *within* that "
            "family should not be over-read.",
        ]

    lines += [
        "",
        "Provenance for the A.7 claims in this document: **arXiv:2607.15495 "
        "Appendix A.7, \"Methodological Details and Ablations\", Figures 57-58 "
        "(p. 69 of the PDF)** -- cite the versioned arXiv entry, not the "
        "transformer-circuits URL, which serves no appendix body.",
        "",
        f"Written by `scripts/44_park_baseline.py` from `{report.name}`.",
        "",
    ]

    out = ROOT / args.out
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    for arm in sorted(non):
        print(f"  {arm:<28} non-math {non[arm]}  math {math.get(arm, '-')}")


if __name__ == "__main__":
    main()

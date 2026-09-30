"""C25 - the unembedding correction as a tunable family, and its closed form.

Two questions in one run.

**Is beta = 1 the optimum?** C22 removed exactly 100% of the ``u_y`` component.
``v_y(beta) = v_y - beta <v_y, u_hat_y> u_hat_y`` makes that a dial: beta=0 is
the standard lens, beta=1 is C22, beta>1 over-corrects into anti-alignment,
actively suppressing the "say y next" channel.

**Can the expensive estimator be replaced by a scalar?** ``_galpha`` subtracts
``alpha_l * u_y`` with a single per-layer scalar
``alpha_l = sum_y <v_y,u_y> / sum_y ||u_y||^2``, read off the standard lens
with no refit -- against ``_off``, which needs ~111 backward passes per prompt.

    python scripts/39_beta_family.py
"""

from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results/c9_domain_causal_qwen3-8b_beta.md"
GEOM = ROOT / "results/geometry/beta_family_qwen3-8b_beta.json"

BETAS = ["b0", "b0.5", "b1", "b2", "b3"]
CORPORA = ["published", "gsm8k", "wikitext_a"]


def table(section: str) -> dict[str, tuple[int, int]]:
    text = REPORT.read_text()
    if section not in text:
        raise SystemExit(f"{REPORT.name} has no section {section!r}")
    out: dict[str, tuple[int, int]] = {}
    for line in text.split(section, 1)[1].splitlines():
        m = re.match(r"\|\s*swap_(\S+?)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*$", line)
        if m:
            out[m.group(1)] = (int(m.group(2)), int(m.group(3)))
        elif out and not line.strip():
            break
    if not out:
        raise SystemExit(f"parsed no rows under {section!r}")
    return out


def main() -> None:
    math = table("## Math cell (numbers) - hits out of 39 trials")
    non = table("## Non-math - hits out of 105 trials")
    geo = json.loads(GEOM.read_text())
    alphas = geo["alphas"]

    lines = [
        "# C25 - tuning the unembedding correction (qwen3-8b)",
        "",
        "`v_y(beta) = v_y - beta * <v_y, u_hat_y> * u_hat_y`, on the 192 "
        "flexible-generalization trials at the same seven band layers. "
        "`b0` is the unmodified lens, `b1` is C22's `_perp`, `b2`/`b3` "
        "over-correct into anti-alignment with the token's unembedding row. "
        "`galpha` subtracts one per-layer scalar instead of a per-word "
        "coefficient. `_off` is the expensive horizon-filtered estimator, for "
        "reference.",
        "",
        "## Non-math /105",
        "",
        "| corpus | b0 | b0.5 | b1 | b2 | b3 | galpha | off |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in CORPORA:
        cells = [str(non.get(f"{c}_{b}", ("", ""))[0]) for b in BETAS]
        cells.append(str(non.get(f"{c}_galpha", ("", ""))[0]))
        cells.append(str(non.get(f"{c}_off", ("", ""))[0]) if f"{c}_off" in non else "-")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")

    lines += ["", "## Math /39", "", "| corpus | b0 | b0.5 | b1 | b2 | b3 | galpha | off |", "|---|---|---|---|---|---|---|---|"]
    for c in CORPORA:
        cells = [str(math.get(f"{c}_{b}", ("", ""))[0]) for b in BETAS]
        cells.append(str(math.get(f"{c}_galpha", ("", ""))[0]))
        cells.append(str(math.get(f"{c}_off", ("", ""))[0]) if f"{c}_off" in math else "-")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## The closed form against the per-word projection",
        "",
        "`galpha` uses ONE scalar per layer where `b1` solves for a coefficient "
        "per word. If they agree, the expensive per-position fit has a "
        "no-cost surrogate that applies to the published lens as shipped.",
        "",
        "| corpus | b1 non-math | galpha non-math | b1 math | galpha math |",
        "|---|---|---|---|---|",
    ]
    for c in CORPORA:
        lines.append(
            f"| {c} | {non.get(f'{c}_b1', ('', ''))[0]} | "
            f"{non.get(f'{c}_galpha', ('', ''))[0]} | "
            f"{math.get(f'{c}_b1', ('', ''))[0]} | "
            f"{math.get(f'{c}_galpha', ('', ''))[0]} |"
        )

    lines += [
        "",
        "### The scalar itself",
        "",
        "`alpha_l = sum_y <v_y, u_y> / sum_y ||u_y||^2`, the least-squares "
        "identity coefficient of the averaged Jacobian, estimated from the "
        "pullbacks alone.",
        "",
        "| corpus | " + " | ".join(f"L{ll}" for ll in sorted(map(int, alphas["gsm8k"]))) + " |",
        "|---|" + "---|" * len(alphas["gsm8k"]),
    ]
    for c in CORPORA:
        per = alphas[c]
        lines.append(
            f"| {c} | " + " | ".join(f"{per[k]:.3f}" for k in sorted(per, key=int)) + " |"
        )
    lines += [
        "",
        "The coefficient rises with depth and **converges across corpora** "
        "(L31: gsm8k 1.154, wikitext 1.144, published 1.153). The identity "
        "component is a property of the model, not of the fitting corpus -- "
        "which is why the correction transfers to a lens it was not fitted on.",
    ]

    lines += [
        "",
        "## Reading",
        "",
        "**beta = 1 is the optimum, and it is not a tuned hyperparameter.** The "
        "curve is unimodal and peaks exactly at the value that makes the write "
        "direction orthogonal to the token's unembedding row. Over-correcting "
        "is not merely useless but actively harmful: beta=2 gives back most of "
        "the gain and beta=3 falls *below* the unmodified lens on every corpus "
        "(13, 16, 12 against 20, 35, 20). So the prescription is not "
        "\"suppress the say-y-next channel as hard as possible\" -- it is "
        "\"remove it exactly\", which is a statement about geometry rather than "
        "a dial to tune per model.",
        "",
        "**The closed form is as good as the per-word projection, and needs no "
        "fit.** `galpha` matches or beats `b1` everywhere (47/47, 54 vs 51, "
        "47/47), while costing one scalar per layer against `_off`'s ~111 "
        "backward passes per prompt. On the published lens -- which we did not "
        "fit and cannot refit -- it goes from 20 to 47 of 105.",
        "",
        "It does not fully replace horizon filtering: `gsm8k_off` still reaches "
        "60, so `galpha` recovers about 76% of the gain "
        "((54-35)/(60-35)). The remaining quarter is what the off-diagonal "
        "term carries beyond the absence of `u_y`, and is the part that would "
        "justify keeping the expensive estimator.",
    ]

    out = ROOT / "results/c25_beta_family_qwen3-8b.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

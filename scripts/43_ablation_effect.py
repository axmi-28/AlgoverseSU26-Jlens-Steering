#!/usr/bin/env python3
"""C28 - the workspace paper's causal measure, applied to our arms.

The paper (App. A.7) already tested the diagonal/off-diagonal recipe choice
causally and reported that "our qualitative results are robust to these
choices". Our swap numbers separate the same arms by a factor of four. Either
one of us is wrong, or the two measures ask different questions.

This reports their measure -- delete the direction from the residual stream,
record the induced output KL, higher = more causally important -- on the same
arms, the same trials and the same band as our swap runs, and puts the two
columns side by side.

    python scripts/43_ablation_effect.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Swap success at a=1, transcribed from the generated C18/C22 reports rather
#: than retyped, so the two documents cannot drift apart. Keyed by arm.
SWAP_REPORT = ROOT / "results/c9_domain_causal_qwen3-8b_comp.md"
PERP_REPORT = ROOT / "results/c9_domain_causal_qwen3-8b_perp.md"


def swap_table(report: Path, section: str) -> dict[str, int]:
    """The ``| swap_arm | a=1.0 | a=2.0 |`` rows under ``section``, a=1 only."""
    text = report.read_text()
    if section not in text:
        raise SystemExit(f"{report.name} has no section {section!r}")
    out: dict[str, int] = {}
    for line in text.split(section, 1)[1].splitlines():
        m = re.match(r"\|\s*swap_(\S+?)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*$", line)
        if m:
            out[m.group(1)] = int(m.group(2))
        elif out and not line.strip():
            break
    if not out:
        raise SystemExit(f"parsed no arm rows under {section!r} of {report.name}")
    return out


def load(label: str, n_shards: int) -> list[dict]:
    records: list[dict] = []
    for i in range(n_shards):
        path = ROOT / f"results/causal/ablation_qwen3-8b_swap_{label}_shard{i}of{n_shards}.json"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        records.extend(json.loads(path.read_text()))
    return records


def boot(values: list[float], n: int = 10000, seed: int = 7) -> tuple[float, float]:
    rng = random.Random(seed)
    k = len(values)
    means = []
    for _ in range(n):
        means.append(sum(values[rng.randrange(k)] for _ in range(k)) / k)
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def paired_sign(a: list[float], b: list[float]) -> tuple[int, int, float]:
    """Two-sided sign test on paired differences; ties dropped."""
    wins = sum(1 for x, y in zip(a, b) if x > y)
    losses = sum(1 for x, y in zip(a, b) if x < y)
    n = wins + losses
    if n == 0:
        return wins, losses, 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return wins, losses, min(1.0, 2 * tail)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="abl")
    ap.add_argument("--n-shards", type=int, default=2)
    ap.add_argument("--out", default="results/c28_ablation_effect_qwen3-8b.md")
    args = ap.parse_args()

    records = load(args.label, args.n_shards)
    by_arm: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in records:
        by_arm[r["arm"]][r["prompt_key"]] = r
    arms = sorted(by_arm)

    # Every arm must cover the same trials, or the means are not comparable.
    keys = set.intersection(*(set(v) for v in by_arm.values()))
    sizes = {a: len(v) for a, v in by_arm.items()}
    if len(set(sizes.values())) != 1:
        print(f"WARNING: arms cover different trial counts {sizes}; "
              f"intersecting to {len(keys)}")
    order = sorted(keys)
    if not order:
        raise SystemExit("no trials shared by every arm")

    swap_non = swap_table(SWAP_REPORT, "## Non-math - hits out of 105 trials")
    swap_math = swap_table(SWAP_REPORT, "## Math cell (numbers) - hits out of 39 trials")
    try:
        swap_non |= swap_table(PERP_REPORT, "## Non-math - hits out of 105 trials")
        swap_math |= swap_table(PERP_REPORT, "## Math cell (numbers) - hits out of 39 trials")
    except SystemExit:
        pass

    lines = [
        "# C28 - ablation effect: the workspace paper's own causal measure (qwen3-8b)",
        "",
        "The paper's App. A.7 tested the present-only / future-only recipe choice "
        "and reported the results robust to it; its main text says so verbatim "
        '("computing only present and not future token effects ... our qualitative '
        'results are robust to these choices"). Our swap numbers separate the same '
        "arms by a factor of four. This runs **their** measure on **our** arms.",
        "",
        "Measure: delete the direction from the residual stream at every prompt "
        "position of all seven band layers (`h -> h - <h,g_hat>g_hat`, coefficients "
        "read off the clean pass, exactly as the swap arms do), then take "
        "KL(clean || ablated) over the full vocabulary at the final prompt "
        "position. **Higher = the direction was more load-bearing.** This is the "
        "opposite sign convention to the collateral-KL columns elsewhere in "
        "`results/`; the two are not comparable and should never be put in the "
        "same table.",
        "",
        f"{len(order)} prompts shared by all {len(arms)} arms. Ablation is a "
        "property of the prompt and its source argument, not of a "
        "(prompt, target) pair, so the 192 swap trials collapse to the 64 base "
        "prompts here; the swap columns below are still out of 105/39 trials. "
        "`frac_norm` is the mean fraction of the residual's norm the projection "
        "removed, and it is the guard against a fake null: a flat KL over arms "
        "that removed very different amounts is a different claim from a flat KL "
        "over arms that removed the same amount. `rand_uy` ablates a *different* "
        "token's unembedding row and is the floor.",
        "",
        "## Ablation effect by arm",
        "",
        "| arm | ablation KL | 95% CI | d log p(answer) | frac_norm |",
        "|---|---|---|---|---|",
    ]

    stats: dict[str, dict[str, float]] = {}
    for arm in arms:
        kl = [by_arm[arm][k]["kl"] for k in order]
        dl = [by_arm[arm][k]["dlogp_answer"] for k in order]
        fr = [by_arm[arm][k]["frac_norm"] for k in order]
        lo, hi = boot(kl)
        stats[arm] = {"kl": statistics.mean(kl), "frac": statistics.mean(fr)}
        lines.append(
            f"| {arm} | {statistics.mean(kl):.3f} | [{lo:.3f}, {hi:.3f}] | "
            f"{statistics.mean(dl):+.3f} | {statistics.mean(fr):.4f} |"
        )

    lines += [
        "",
        "## The dissociation: what ablation says vs what writing says",
        "",
        "Swap-success columns are transcribed from `c9_domain_causal_qwen3-8b_comp.md` "
        "and `_perp.md`, at a=1.0, same band, same trial set. A `-` means the arm "
        "was not run there.",
        "",
        "| arm | ablation KL (presence) | swap non-math /105 (control) | swap math /39 |",
        "|---|---|---|---|",
    ]
    for arm in arms:
        n = swap_non.get(arm)
        m = swap_math.get(arm)
        lines.append(
            f"| {arm} | {stats[arm]['kl']:.3f} | "
            f"{n if n is not None else '-'} | {m if m is not None else '-'} |"
        )

    # A.7's actual comparison, isolated. The global rank correlation is a poor
    # summary here: it is dragged positive by _diag, which is last on both
    # measures. The contrast the paper made -- keep the diagonal or drop it --
    # is the one that inverts, and it has to be shown on its own.
    lines += [
        "",
        "## The contrast A.7 actually made: keep the diagonal, or drop it?",
        "",
        "This is the recipe choice the workspace paper tested (\"computing only "
        "present and not future token effects\") and scored by ablation. Read "
        "each row left to right: dropping the diagonal makes the direction "
        "**less** ablatable and **more** steerable.",
        "",
        "| corpus | KL full | KL off | change | swap full | swap off | change |",
        "|---|---|---|---|---|---|---|",
    ]
    for corpus in ("gsm8k", "wikitext_a"):
        f, o = f"{corpus}_full", f"{corpus}_off"
        if f not in stats or o not in stats or f not in swap_non:
            continue
        dk = 100 * (stats[o]["kl"] - stats[f]["kl"]) / stats[f]["kl"]
        ds = 100 * (swap_non[o] - swap_non[f]) / max(swap_non[f], 1)
        lines.append(
            f"| {corpus} | {stats[f]['kl']:.2f} | {stats[o]['kl']:.2f} | "
            f"{dk:+.0f}% | {swap_non[f]} | {swap_non[o]} | {ds:+.0f}% |"
        )
    lines += [
        "",
        "So a recipe sweep scored by ablation effect does not merely fail to "
        "prefer the off-diagonal direction -- it prefers the **full** one.",
        "",
        "**This is a local inversion, not a global one, and the distinction is "
        "the whole claim.** The rank correlations below are computed over three "
        "nested arm sets; they disagree in sign. Restricted to the "
        "full/off contrast the two metrics anti-correlate; over all arms they "
        "agree, because at the bottom of the range (`uy`, `rand_uy`, `_diag`) "
        "both metrics are low together. The defensible claim is therefore "
        "**ablation effect is insensitive to the diagonal contaminant and "
        "orders that contrast backwards** -- NOT that ablation anti-ranks write "
        "directions in general.",
        "",
        "Nor can we say A.7's own sweep *would have* misranked. Their Fig. 53 "
        "ablates the intermediate token's lens vector on multihop tasks on "
        "Sonnet 4.5; ours ablates the argument's vector at all prompt positions "
        "of seven band layers on Qwen3-8B. Different task, model, band and "
        "target. The finding is that these two metrics order this contrast "
        "oppositely on this setup.",
    ]

    # The headline. If the two measures agreed, this would be strongly positive.
    paired_arms = [a for a in arms if a in swap_non]
    if len(paired_arms) >= 4:
        xs = [stats[a]["kl"] for a in paired_arms]
        ys = [float(swap_non[a]) for a in paired_arms]
        rho = spearman(xs, ys)
        # Three nested arm sets, because the sign depends on which you take and
        # reporting only one of them would be choosing the answer.
        subsets = {
            "all arms scored by both": paired_arms,
            "lens arms only (no uy)": [a for a in paired_arms if a != "uy"],
            "full/off contrast only": [
                a for a in paired_arms if a.endswith("_full") or a.endswith("_off")
            ],
        }
        lines += [
            "",
            "## Do the two measures agree on the ranking?",
            "",
            "| arm set | n | Spearman(ablation KL, swap hits) |",
            "|---|---|---|",
        ]
        for label, sub in subsets.items():
            if len(sub) < 3:
                continue
            r = spearman(
                [stats[a]["kl"] for a in sub], [float(swap_non[a]) for a in sub]
            )
            lines.append(f"| {label} | {len(sub)} | **{r:+.2f}** |")
        lines += [
            "",
            "The sign flips between the first row and the last. That is the "
            "result: the metrics agree about weak directions and disagree about "
            "the diagonal.",
            "",
            "| arm | ablation KL | rank by KL | swap /105 | rank by swap |",
            "|---|---|---|---|---|",
        ]
        kl_rank = {a: i + 1 for i, a in enumerate(sorted(paired_arms, key=lambda a: -stats[a]["kl"]))}
        sw_rank = {a: i + 1 for i, a in enumerate(sorted(paired_arms, key=lambda a: -swap_non[a]))}
        for a in sorted(paired_arms, key=lambda a: -stats[a]["kl"]):
            lines.append(
                f"| {a} | {stats[a]['kl']:.2f} | {kl_rank[a]} | "
                f"{swap_non[a]} | {sw_rank[a]} |"
            )

        # The obvious confound: an arm that removes more of the residual should
        # induce more KL whatever it is pointing at. Normalise by the norm the
        # projection actually took out before concluding anything about the
        # direction.
        lines += [
            "",
            "### Controlling for how much was removed",
            "",
            "Ablation KL should grow with the size of the component deleted, so a "
            "raw KL ordering partly reports `frac_norm`. The third column divides "
            "by it; the ordering that survives is the one about direction.",
            "",
            "| arm | ablation KL | frac_norm | KL / frac_norm | swap /105 |",
            "|---|---|---|---|---|",
        ]
        for a in sorted(paired_arms, key=lambda x: -stats[x]["kl"]):
            f = stats[a]["frac"]
            lines.append(
                f"| {a} | {stats[a]['kl']:.2f} | {f:.4f} | "
                f"{stats[a]['kl'] / f if f else float('nan'):.0f} | {swap_non[a]} |"
            )
        norm_rho = spearman(
            [stats[a]["kl"] / stats[a]["frac"] for a in paired_arms],
            [float(swap_non[a]) for a in paired_arms],
        )
        lines += [
            "",
            f"Spearman(KL / frac_norm, swap hits) = **{norm_rho:+.2f}**.",
        ]

    lines += ["", "## Paired contrasts on the ablation measure", ""]
    pairs = [
        ("gsm8k_diag", "gsm8k_off"),
        ("wikitext_a_diag", "wikitext_a_off"),
        ("gsm8k_full", "gsm8k_off"),
        ("gsm8k_off", "rand_uy"),
        ("gsm8k_diag", "rand_uy"),
        ("uy", "rand_uy"),
        ("published", "rand_uy"),
    ]
    lines += ["| contrast | mean delta | wins | losses | sign p |", "|---|---|---|---|---|"]
    for a, b in pairs:
        if a not in by_arm or b not in by_arm:
            continue
        xa = [by_arm[a][k]["kl"] for k in order]
        xb = [by_arm[b][k]["kl"] for k in order]
        w, l, p = paired_sign(xa, xb)
        delta = statistics.mean(x - y for x, y in zip(xa, xb))
        lines.append(f"| {a} vs {b} | {delta:+.3f} | {w} | {l} | {p:.4f} |")

    lines += [
        "",
        "## Reading",
        "",
        "Written by `scripts/43_ablation_effect.py` from "
        f"`results/causal/ablation_qwen3-8b_swap_{args.label}_shard*of{args.n_shards}.json`. "
        "Interpretation belongs in the summary, not here -- but the shape to check "
        "is whether the arms are close together in the first table and far apart in "
        "the second. That, and not a disagreement about the facts, is what would "
        "reconcile A.7's null with our swap result.",
        "",
    ]

    out = ROOT / args.out
    out.write_text("\n".join(lines))
    print(f"wrote {out} ({len(order)} trials, {len(arms)} arms)")
    for arm in arms:
        print(f"  {arm:<26} KL={stats[arm]['kl']:.3f}  frac={stats[arm]['frac']:.4f}")


if __name__ == "__main__":
    main()

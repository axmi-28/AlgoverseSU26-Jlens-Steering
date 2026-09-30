"""C19 - writing a latent intermediate, and whether the second hop follows.

Every other causal arm in this repo swaps a token that is in the prompt and
grades the token that comes next: the thing written is the thing read. Here the
swapped concept (Brazil -> Mexico) never appears in the prompt, and success is
the *downstream* answer (Spanish), which the model can only produce by using
the swapped intermediate in a computation it runs afterwards.

Three outcomes are kept apart, because the interesting one is the gap between
them: the second hop ran; the injected token merely surfaced; neither.

    python scripts/32_two_hop.py
"""

from __future__ import annotations

import glob
import json
from math import comb
from pathlib import Path
from statistics import mean

from jsteer.steering import answer_matches

ROOT = Path(__file__).resolve().parents[1]
STRENGTHS = (1.0, 2.0, 4.0)


def sign_p(diffs: list[float]) -> tuple[int, int, float]:
    pos = sum(d > 0 for d in diffs)
    neg = sum(d < 0 for d in diffs)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    tail = sum(comb(n, i) for i in range(min(pos, neg) + 1))
    return pos, neg, min(1.0, 2 * tail / 2**n)


#: Pinned, NOT globbed by mtime. Two two-hop runs exist -- this one (the six
#: whole-corpus lenses) and the later component-arm run that scripts/35 reads.
#: The component run is the newer file, so a most-recent glob would silently
#: rebuild this report from the wrong arms. That is the repo's recurring bug
#: class and it had already happened once, to the energy sweep.
SHARD = "two_hop_qwen3-8b_all_g63451680_twohop71db20_shard0of1.json"

#: Arms this report is about. Anything else in the file is a different run.
EXPECTED = {
    "baseline",
    "logit_lens",
    "published",
    "gsm8k",
    "wikitext_a",
    "openwebmath",
    "ultrachat",
    "arith_words",
}


def main() -> None:
    path = ROOT / "results/causal" / SHARD
    recs = json.loads(path.read_text())
    found = {r["arm"] for r in recs}
    if found != EXPECTED:
        raise SystemExit(
            f"{SHARD} holds arms {sorted(found)}, expected {sorted(EXPECTED)}. "
            "Check counts, not exit codes -- this is the wrong shard."
        )
    idx = {(r["arm"], r["strength"], r["name"]): r for r in recs}
    usable = sorted({r["name"] for r in recs if r["baseline_ok"]})
    arms = sorted({r["arm"] for r in recs if r["arm"] not in ("baseline", "logit_lens")})

    base = {r["name"]: r for r in recs if r["arm"] == "baseline"}
    # Stored as False by construction; recomputed here because it is the
    # control the whole eval rests on.
    b_swap = mean(
        answer_matches(base[n]["generated"], base[n]["swap_answer"]) for n in usable
    )
    b_said = mean(base[n]["said_swap_to"] for n in usable)

    lines = [
        "# C19 - two-hop concept write (qwen3-8b)",
        "",
        f"`{path.name}`. {len(usable)} of 90 probe-swap items, kept where the "
        "unsteered model already answers correctly. The swap replaces the "
        "*latent* intermediate, which never appears in the prompt. "
        "**2hop** = the downstream answer the swap implies; **said** = the "
        "injected token appears in the continuation; **kept** = the original "
        "answer survives.",
        "",
        f"Baseline with no edit: 2hop {b_swap:.3f}, said {b_said:.3f}. Every hit "
        "below is caused by the write.",
        "",
        "| arm | " + " | ".join(f"a={s} 2hop / said / kept" for s in STRENGTHS) + " |",
        "|---|" + "---|" * len(STRENGTHS),
    ]
    for arm in [*arms, "logit_lens"]:
        cells = []
        for s in STRENGTHS:
            sub = [idx[(arm, s, n)] for n in usable]
            cells.append(
                f"{mean(r['hit_swap_answer'] for r in sub):.3f} / "
                f"{mean(r['said_swap_to'] for r in sub):.3f} / "
                f"{mean(r['hit_answer'] for r in sub):.3f}"
            )
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")

    pos, neg, p = sign_p(
        [
            mean(idx[(a, 1.0, n)]["hit_swap_answer"] for a in arms)
            - idx[("logit_lens", 1.0, n)]["hit_swap_answer"]
            for n in usable
        ]
    )
    lines += [
        "",
        "## Does the Jacobian transport matter?",
        "",
        f"Per item, the mean of the {len(arms)} J_bar arms against J = I: "
        f"**{pos} items better, {neg} worse, sign p={p:.4f}**. Items are the "
        "independent unit; arms on one item are not.",
        "",
        "## The two outcomes are near-exclusive, and strength picks which",
        "",
        "| strength | concept used only | token surfaced only | both | neither |",
        "|---|---|---|---|---|",
    ]
    for s in STRENGTHS:
        counts = {(t, u): 0 for t in (0, 1) for u in (0, 1)}
        for a in arms:
            for n in usable:
                r = idx[(a, s, n)]
                counts[(int(r["hit_swap_answer"]), int(r["said_swap_to"]))] += 1
        total = sum(counts.values())
        lines.append(
            f"| {s} | {counts[(1, 0)] / total:.3f} | {counts[(0, 1)] / total:.3f} | "
            f"{counts[(1, 1)] / total:.3f} | {counts[(0, 0)] / total:.3f} |"
        )

    lines += ["", "## Examples (a=1.0)", "", "| item | swap | clean | steered | wanted |", "|---|---|---|---|---|"]
    shown = 0
    for n in usable:
        r = idx[("gsm8k", 1.0, n)]
        if r["hit_swap_answer"] and shown < 6:
            clean = base[n]["generated"].strip().split("\n")[0][:28]
            got = r["generated"].strip().split("\n")[0][:28]
            lines.append(
                f"| {n} | {r['intermediate']} -> {r['swap_to']} | {clean!r} | "
                f"**{got!r}** | {r['swap_answer']} |"
            )
            shown += 1

    out = ROOT / "results/c19_two_hop_qwen3-8b.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

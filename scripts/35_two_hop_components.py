"""C18 x C19 - does dropping the diagonal also improve the LATENT-concept write?

C18 showed that removing Eq. 20's ``t' = t`` term roughly doubles swap success
on tokens that are present in the prompt. C19 showed that a J-lens direction
can swap a latent intermediate that never appears in the prompt at all. This
crosses them: the component arms, on the two-hop eval.

It is the sharper test of the two, because the two-hop eval separates the
outcome the component split is supposed to change from the one it is supposed
to suppress:

* ``2hop`` -- the model used the swapped concept in a downstream computation;
* ``said`` -- the injected token merely surfaced in the text (leakage).

If the diagonal term is the "say y next" channel, dropping it should raise the
first and cut the second. Those are separate columns here rather than one
success rate, which is what makes the prediction falsifiable.

Both are per-item means over the same usable set, so arms are paired and the
sign tests below take the item as the independent unit.

    python scripts/35_two_hop_components.py
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path
from statistics import mean

from jsteer.steering import answer_matches

ROOT = Path(__file__).resolve().parents[1]

#: This run carries a=1 and a=2 only; C19's six-lens run is the one that also
#: has a=4. Asserted against the file below rather than assumed.
STRENGTHS = (1.0, 2.0)

#: Pinned. The other two_hop shard is the six-whole-corpus run behind C19, and
#: it is the OLDER file, so an mtime glob here would pick this one and an
#: mtime glob in scripts/32 would also pick this one. Both are now pinned.
SHARD = "two_hop_qwen3-8b_all_g595bfbbf_twohop89d9c5_shard0of1.json"

COMPONENT = [
    "gsm8k_full",
    "gsm8k_diag",
    "gsm8k_off",
    "wikitext_a_full",
    "wikitext_a_diag",
    "wikitext_a_off",
]
REFERENCE = ["published", "gsm8k", "wikitext_a", "logit_lens"]


def sign_p(diffs: list[float]) -> tuple[int, int, float]:
    pos = sum(d > 0 for d in diffs)
    neg = sum(d < 0 for d in diffs)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    tail = sum(comb(n, i) for i in range(min(pos, neg) + 1))
    return pos, neg, min(1.0, 2 * tail / 2**n)


def main() -> None:
    path = ROOT / "results/causal" / SHARD
    recs = json.loads(path.read_text())

    found = {r["arm"] for r in recs}
    missing = [a for a in COMPONENT + REFERENCE if a not in found]
    if missing:
        raise SystemExit(f"{SHARD} is missing arms {missing}; holds {sorted(found)}")

    idx = {(r["arm"], r["strength"], r["name"]): r for r in recs}
    base = {r["name"]: r for r in recs if r["arm"] == "baseline"}

    # The component arms cover fewer items than the whole-corpus arms: the
    # component digest was fitted over a narrower target-word list, so items
    # whose swapped-in token is outside it were never run. Taking each arm's
    # own mean would compare 55-item arms against 56-item arms and read as a
    # real difference. Intersect first, then assert the grid is complete.
    strengths_present = sorted({r["strength"] for r in recs if r["arm"] != "baseline"})
    if tuple(strengths_present) != STRENGTHS:
        raise SystemExit(f"{SHARD} has strengths {strengths_present}, not {STRENGTHS}")

    covered = {
        n
        for n in {r["name"] for r in recs if r["baseline_ok"]}
        if all((a, s, n) in idx for a in COMPONENT + REFERENCE for s in STRENGTHS)
    }
    usable = sorted(covered)
    dropped = len({r["name"] for r in recs if r["baseline_ok"]}) - len(usable)
    if not usable:
        raise SystemExit("no item is covered by every arm at every strength")

    b_swap = mean(
        answer_matches(base[n]["generated"], base[n]["swap_answer"]) for n in usable
    )
    b_said = mean(base[n]["said_swap_to"] for n in usable)

    def cell(arm, s, key):
        return mean(idx[(arm, s, n)][key] for n in usable)

    lines = [
        "# C18 x C19 - the component split on the latent-concept write (qwen3-8b)",
        "",
        f"`{path.name}`. **{len(usable)} items**, every arm on the same set: "
        "of 90 probe-swap items, those the unsteered model already answers "
        "correctly AND that every arm ran. The component digest was fitted "
        f"over a narrower target-word list than the whole-corpus lenses, so "
        f"{dropped} otherwise-usable item{'s' if dropped != 1 else ''} "
        f"{'are' if dropped != 1 else 'is'} dropped to keep the denominator "
        "identical across arms. The swapped concept is the *latent* "
        "intermediate and never appears in the prompt.",
        "",
        "**2hop** = the model used the swapped concept downstream (the outcome "
        "that matters). **said** = the injected token merely surfaced in the "
        "text (leakage -- a write that reached the surface but not the "
        "computation). The component split predicts the first should rise and "
        "the second should fall.",
        "",
        f"Baseline with no edit: 2hop {b_swap:.3f}, said {b_said:.3f}.",
        "",
        "| arm | " + " | ".join(f"a={s} 2hop / said" for s in STRENGTHS) + " |",
        "|---|" + "---|" * len(STRENGTHS),
    ]
    for arm in COMPONENT + REFERENCE:
        cells = [
            f"{cell(arm, s, 'hit_swap_answer'):.3f} / {cell(arm, s, 'said_swap_to'):.3f}"
            for s in STRENGTHS
        ]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## off vs full, paired per item (a=1.0)",
        "",
        "The item is the independent unit. `2hop` should go up, `said` down.",
        "",
        "| corpus | 2hop up / down / p | said up / down / p |",
        "|---|---|---|",
    ]
    for corpus in ("gsm8k", "wikitext_a"):
        row = [corpus]
        for key in ("hit_swap_answer", "said_swap_to"):
            d = [
                idx[(f"{corpus}_off", 1.0, n)][key] - idx[(f"{corpus}_full", 1.0, n)][key]
                for n in usable
            ]
            p, m, pv = sign_p(d)
            row.append(f"{p} / {m} / {pv:.4f}")
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "### Pooled over both corpora", ""]
    for key, label in (("hit_swap_answer", "2hop"), ("said_swap_to", "said")):
        d = [
            mean(
                idx[(f"{c}_off", 1.0, n)][key] - idx[(f"{c}_full", 1.0, n)][key]
                for c in ("gsm8k", "wikitext_a")
            )
            for n in usable
        ]
        p, m, pv = sign_p(d)
        lines.append(f"* **{label}**: {p} items up, {m} down, sign p={pv:.4f}")

    lines += [
        "",
        "## Examples: what the full arm says where the off arm computes (a=1.0)",
        "",
        "| item | swap | gsm8k_full said | gsm8k_off said | wanted |",
        "|---|---|---|---|---|",
    ]
    shown = 0
    for n in usable:
        full, off = idx[("gsm8k_full", 1.0, n)], idx[("gsm8k_off", 1.0, n)]
        if off["hit_swap_answer"] and not full["hit_swap_answer"] and shown < 8:
            f_txt = full["generated"].strip().split("\n")[0][:26]
            o_txt = off["generated"].strip().split("\n")[0][:26]
            lines.append(
                f"| {n} | {off['intermediate']} -> {off['swap_to']} | {f_txt!r} | "
                f"**{o_txt!r}** | {off['swap_answer']} |"
            )
            shown += 1

    out = ROOT / "results/c18x19_two_hop_components_qwen3-8b.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

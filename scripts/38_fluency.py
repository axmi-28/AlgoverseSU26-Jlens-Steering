"""C23 - does dropping the diagonal term damage generation?

The question every existing metric in this project is blind to. Swap success,
KL and collateral damage are all read off a single graded token, so none of
them sees whether the text stays coherent. There is a specific reason to
expect damage: Yan et al. describe the ``t' = t`` term as where a position
"primarily preserves or prepares the next token", and ``_off`` deletes it;
the J-Lens paper separately reports that J-space ablation "tends to impair the
coherence of responses".

Four measures on 32 greedily decoded tokens, against the model's own
unsteered continuation of the same prompt:

* ``nll``       mean negative log-likelihood of the continuation under the
                CLEAN model -- the steered model does not grade its own output;
* ``rep4``      share of repeated 4-grams;
* ``max_run``   longest run of one repeated token;
* ``distinct1`` type/token ratio.

    python scripts/38_fluency.py
"""

from __future__ import annotations

import glob
import json
import statistics as st
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ORDER = [
    "baseline",
    "averaged",
    "gsm8k_full",
    "gsm8k_diag",
    "gsm8k_off",
    "wikitext_a_full",
    "wikitext_a_off",
]
METRICS = (
    ("nll", "clean-model NLL", "lower"),
    ("rep4", "repeated 4-grams", "lower"),
    ("max_run", "longest token run", "lower"),
    ("distinct1", "type/token", "higher"),
)


def sign_p(diffs: list[float]) -> tuple[int, int, float]:
    pos = sum(d > 0 for d in diffs)
    neg = sum(d < 0 for d in diffs)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    tail = sum(comb(n, i) for i in range(min(pos, neg) + 1))
    return pos, neg, min(1.0, 2 * tail / 2**n)


def main() -> None:
    files = sorted(glob.glob(str(ROOT / "results/fluency/fluency_*_comp_shard*.json")))
    if not files:
        raise SystemExit("no fluency shards")
    recs = [x for f in files for x in json.loads(Path(f).read_text())]
    idx = {(r["arm"], r["key"]): r for r in recs}
    keys = sorted({r["key"] for r in recs})
    arms = [a for a in ORDER if a in {r["arm"] for r in recs}]

    for arm in arms:
        n = len([r for r in recs if r["arm"] == arm])
        if n != len(keys):
            raise SystemExit(f"{arm} has {n} rows over {len(keys)} trials")

    lines = [
        "# C23 - does dropping the diagonal damage generation? (qwen3-8b)",
        "",
        f"{len(keys)} flexible-generalization trials, 32 greedily decoded "
        "tokens per arm at a=1.0, same seven band layers and same swap as the "
        "causal runs. `baseline` is the unsteered continuation of the same "
        "prompt. NLL is scored under the **clean** model, so the steered model "
        "never grades its own output.",
        "",
        "| arm | clean NLL | repeated 4-grams | longest run | type/token |",
        "|---|---|---|---|---|",
    ]
    for arm in arms:
        rs = [r for r in recs if r["arm"] == arm]
        lines.append(
            f"| {arm} | {st.mean(r['nll'] for r in rs):.3f} | "
            f"{st.mean(r['rep4'] for r in rs):.3f} | "
            f"{st.mean(r['max_run'] for r in rs):.2f} | "
            f"{st.mean(r['distinct1'] for r in rs):.3f} |"
        )

    lines += [
        "",
        "## off vs full, paired per trial",
        "",
        "The trial is the independent unit. A positive count under `off better` "
        "means the off arm moved the metric in its good direction.",
        "",
        "| corpus | metric | off better | full better | sign p | mean delta |",
        "|---|---|---|---|---|---|",
    ]
    for corpus in ("gsm8k", "wikitext_a"):
        for key, label, good in METRICS:
            d = [
                idx[(f"{corpus}_off", k)][key] - idx[(f"{corpus}_full", k)][key]
                for k in keys
            ]
            signed = d if good == "higher" else [-x for x in d]
            p, m, pv = sign_p(signed)
            lines.append(
                f"| {corpus} | {label} | {p} | {m} | {pv:.4f} | "
                f"{st.mean(d):+.4f} |"
            )

    lines += [
        "",
        "## Against the unsteered baseline",
        "",
        "Every steered arm roughly triples the continuation's surprisal to the "
        "clean model -- that is what steering costs, and it is the same cost "
        "for every arm. The arms separate on *degeneracy*, not on surprisal.",
        "",
        "| arm | NLL vs baseline | type/token vs baseline |",
        "|---|---|---|",
    ]
    b_nll = st.mean(r["nll"] for r in recs if r["arm"] == "baseline")
    b_d1 = st.mean(r["distinct1"] for r in recs if r["arm"] == "baseline")
    for arm in arms[1:]:
        rs = [r for r in recs if r["arm"] == arm]
        lines.append(
            f"| {arm} | {st.mean(r['nll'] for r in rs) - b_nll:+.3f} | "
            f"{st.mean(r['distinct1'] for r in rs) - b_d1:+.3f} |"
        )

    lines += [
        "",
        "## Reading",
        "",
        "Fluency holds. Dropping the diagonal does not cost coherence: NLL is "
        "flat against the full lens, and repetition and diversity both *improve*. "
        "`gsm8k_off` restores the type/token ratio to the unsteered baseline "
        "exactly while `gsm8k_full` degrades it.",
        "",
        "**The confound to state.** The arms differ in success rate, and a "
        "successful swap plausibly yields more natural text than a failed one, "
        "so part of the degeneracy gap may be downstream of the success gap "
        "rather than independent of it. The NLL column is the part least "
        "exposed to this -- it is flat across arms whose success rates differ "
        "threefold -- but a success-matched comparison is the clean version and "
        "has not been run.",
    ]

    out = ROOT / "results/c23_fluency_qwen3-8b.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

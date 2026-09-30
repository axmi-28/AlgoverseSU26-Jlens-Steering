"""C9 -- does the corpus a J-Lens direction is averaged over change what it steers?

Reads the domain causal shards and writes ``results/c9_domain_causal_<config>.md``.
Laptop-only; the shard JSONs are the sole input.

Grading is redone here rather than trusted from the run
-------------------------------------------------------
Every record carries ``generated`` (four greedily decoded tokens), and this
script re-scores it with :func:`jsteer.steering.answer_matches`. That is
deliberate: the grading rule changed three times while this experiment was
being built, twice because a bug manufactured hits, and re-scoring from stored
text means a rule fix never requires GPU time and never leaves two runs graded
by different rules.

The two bugs, because both are easy to reintroduce:

* Accepting a variant whose *first token* is whitespace. ``" 10"`` tokenizes to
  ``[' ', '1']``, and Qwen3-8B emits a bare space before every number, so every
  numbers trial scored as a hit whatever followed. This produced a stable
  9/48 that did not respond to the intervention at all.
* Matching a prefix without a boundary. ``" 90,"`` scored as ``"nine"`` because
  ``"90"`` starts with ``"9"``; ``" ninety"`` would too.

Why first-token grading cannot be used on this eval at all
-----------------------------------------------------------
Qwen3-8B answers "Two times five equals" with " 10." -- a bare space, then the
digits. Graded on the first token the model scores **1/16 on numbers
unsteered**; graded on generated text it scores **13/16**. The category was
never hard for the model, and the paper's "number relations never succeed" is
reproduced here for a different reason than on their model (per the J-Lens
grids, Sonnet 4.5 answers in words and its diagonals are correct).
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics as st
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jsteer.config import REPO_ROOT  # noqa: E402
from jsteer.data import flexible_generalization_trials  # noqa: E402
from jsteer.steering import answer_matches  # noqa: E402
from jsteer.sweeps import _digit_form, take_shard  # noqa: E402

MATH = ("numbers",)
NON_MATH = ("countries", "months", "animals")

#: Every (category, template) cell classified on two axes, because the naive
#: read of this experiment -- "gsm8k helps math" -- is not what the cells say.
#:
#: ``arg`` is what gets swapped, ``ans`` is what the model must then produce.
#: The categories confound them: every numbers template has a numeric argument
#: *and* (except first_letter) a numeric answer, so the headline cannot tell
#: which axis carries the effect. Two cells break the confound --
#: ``numbers/first_letter`` (numeric argument, letter answer) and
#: ``animals/legs`` (categorical argument, number-word answer) -- and they
#: point opposite ways from the "gsm8k is good at numbers" story.
#:
#: ``ordered`` marks arguments lying on a scale the model can traverse
#: (integers, months); countries and animals are unordered sets.
CELL_AXES: dict[tuple[str, str], tuple[str, str, bool]] = {
    ("numbers", "double"): ("number", "number", True),
    ("numbers", "successor"): ("number", "number", True),
    ("numbers", "square"): ("number", "number", True),
    ("numbers", "first_letter"): ("number", "letter", True),
    ("months", "number"): ("month", "number", True),
    ("months", "next_month"): ("month", "month", True),
    ("months", "season"): ("month", "season", True),
    ("months", "holiday"): ("month", "holiday", True),
    ("animals", "legs"): ("animal", "number", False),
    ("animals", "class"): ("animal", "class", False),
    ("animals", "group"): ("animal", "group", False),
    ("animals", "habitat"): ("animal", "habitat", False),
    ("countries", "capital"): ("country", "city", False),
    ("countries", "continent"): ("country", "continent", False),
    ("countries", "currency"): ("country", "currency", False),
    ("countries", "language"): ("country", "language", False),
}


def load(pattern: str, n_shards: int, per_trial: int) -> list[tuple]:
    """Pair each record with its trial, recovering the swap target.

    Records store ``source_arg`` but not ``target_arg``, so the three targets
    of one prompt are indistinguishable in the file. They are recovered from
    append order -- ``per_trial`` records per trial, trials strided by shard --
    and the recovery is *asserted* against the fields that are stored, so a
    change to the write order fails here instead of silently mislabelling every
    target. (Recording ``target_arg`` directly would be better; this is the
    reader for runs that predate it.)
    """
    trials = flexible_generalization_trials()
    rows: list[tuple] = []
    for shard in range(n_shards):
        path = pattern.format(shard=shard, n=n_shards)
        records = json.loads(Path(path).read_text())
        mine = take_shard(trials, shard, n_shards)
        if len(records) != len(mine) * per_trial:
            raise SystemExit(
                f"{path}: {len(records)} records for {len(mine)} trials; "
                f"expected {per_trial} per trial. Was --strengths or --arms changed?"
            )
        for i, record in enumerate(records):
            trial = mine[i // per_trial]
            if not record["prompt_key"].endswith(trial.source_arg):
                raise SystemExit(f"{path}: record {i} does not match its trial")
            rows.append((trial, record))
    return rows


def regrade(rows: list[tuple]) -> None:
    for trial, record in rows:
        want = (
            trial.source_answer if record["arm"] == "baseline" else trial.target_answer
        )
        record["hit_re"] = bool(record.get("generated")) and answer_matches(
            record["generated"], want, digit=_digit_form(want)
        )


def mcnemar(a: list[bool], b: list[bool]) -> tuple[float, int, int]:
    """Exact two-sided McNemar on the discordant pairs."""
    only_a = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = only_a + only_b
    if n == 0:
        return 1.0, only_a, only_b
    tail = sum(comb(n, i) for i in range(min(only_a, only_b) + 1))
    return min(1.0, 2 * tail / 2**n), only_a, only_b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="qwen3-8b")
    parser.add_argument("--label", default="gen")
    parser.add_argument("--n-shards", type=int, default=4)
    parser.add_argument("--results", default=str(REPO_ROOT / "results"))
    parser.add_argument(
        "--reference",
        default="swap_averaged",
        help="arm every other arm is McNemar-tested against",
    )
    args = parser.parse_args()

    pattern = (
        f"{args.results}/causal/steering_{args.config}_swap_{args.label}"
        "_shard{shard}of{n}.json"
    )
    found = glob.glob(pattern.format(shard="*", n=args.n_shards))
    if len(found) != args.n_shards:
        raise SystemExit(f"expected {args.n_shards} shards, found {len(found)}")

    probe = json.loads(Path(pattern.format(shard=0, n=args.n_shards)).read_text())
    arms = [a for a in dict.fromkeys(r["arm"] for r in probe) if a != "baseline"]
    strengths = sorted({r["strength"] for r in probe if r["arm"] != "baseline"})
    per_trial = 1 + len(arms) * len(strengths)

    rows = load(pattern, args.n_shards, per_trial)
    regrade(rows)

    # The screen. A trial whose source answer the model never produces cannot
    # show a *swap*: there is nothing to swap away from. Leaving these in is
    # what made the numbers cell look impossible.
    ok = {t.prompt for t, r in rows if r["arm"] == "baseline" and r["hit_re"]}

    def sel(arm: str, strength: float, cats: tuple[str, ...]) -> list[tuple]:
        return [
            (t, r)
            for t, r in rows
            if r["arm"] == arm
            and r["strength"] == strength
            and t.category in cats
            and t.prompt in ok
        ]

    def vec(arm: str, strength: float, cats: tuple[str, ...]) -> list[bool]:
        return [
            r["hit_re"]
            for t, r in sorted(
                sel(arm, strength, cats), key=lambda x: (x[0].prompt, x[0].target_arg)
            )
        ]

    def best(arm: str, cats: tuple[str, ...]) -> float:
        return max(strengths, key=lambda s: sum(vec(arm, s, cats)))

    out: list[str] = []
    w = out.append
    w(f"# C9 - domain-averaged directions, causal ({args.config})\n")
    w(
        "Swap steering over the 192 flexible-generalization trials, every arm "
        "at the same seven band layers the domain Jacobians were fitted at. "
        "Graded on four greedily decoded tokens, not the first token.\n"
    )

    # --- grading, first ---------------------------------------------------
    w("## Grading: why the first-token rule cannot be used here\n")
    w("| category | unsteered, first-token | unsteered, generated text |")
    w("|---|---|---|")
    tally: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0])
    for t, r in rows:
        if r["arm"] != "baseline":
            continue
        cell = tally[t.category]
        cell[0] += bool(r["hit"])
        cell[1] += bool(r["hit_re"])
        cell[2] += 1
    for category, (first, generated, n) in sorted(tally.items()):
        w(f"| {category} | {first // 3}/{n // 3} | **{generated // 3}/{n // 3}** |")
    w("")
    w(
        "The model answers in digits where the key writes words, so first-token "
        "grading reports it cannot do arithmetic it performs correctly. Every "
        "number below is on the generated-text rule, and trials whose source "
        f"answer the model does not produce unsteered are excluded ({len(ok)} of "
        "64 prompts survive).\n"
    )

    # --- the result -------------------------------------------------------
    for name, cats in (("Math cell (numbers)", MATH), ("Non-math", NON_MATH)):
        n = len(sel(arms[0], strengths[0], cats))
        w(f"## {name} - hits out of {n} trials\n")
        w("| arm | " + " | ".join(f"a={s}" for s in strengths) + " |")
        w("|---" * (len(strengths) + 1) + "|")
        for arm in arms:
            cells = " | ".join(str(sum(vec(arm, s, cats))) for s in strengths)
            w(f"| {arm} | {cells} |")
        w("")
        reference = args.reference
        if reference not in arms:
            raise SystemExit(
                f"reference arm {reference!r} is not in this run (arms: {arms}). "
                "Pass --reference to name the arm this run should be compared "
                "against; a missing reference used to fall through to an empty "
                "vector and die inside the McNemar zip."
            )
        ref = vec(reference, best(reference, cats), cats)
        w(
            "| vs published lens (each at own best strength) | hits | only-arm | only-lens | p |"
        )
        w("|---|---|---|---|---|")
        for arm in arms:
            if arm == reference:
                continue
            got = vec(arm, best(arm, cats), cats)
            p, only_a, only_r = mcnemar(got, ref)
            flag = " **" if p < 0.05 else " "
            w(
                f"| {arm} (a={best(arm, cats)}) |{flag}{sum(got)} vs {sum(ref)}{flag.strip()} "
                f"| {only_a} | {only_r} | {p:.4f} |"
            )
        w("")

    # --- is a failure real, or unreadable? --------------------------------
    w("## Is the lens's math failure real, or a grading miss?\n")
    w(
        "Classifying what each arm actually *says*. **source** means the model "
        "gave its original, un-swapped answer -- the grader reads it fine, the "
        "intervention simply did not take. That is the check that rules out an "
        "unreadable-format explanation for a zero.\n"
    )
    w("| arm | a | target (hit) | source (no effect) | other |")
    w("|---|---|---|---|---|")
    for arm in arms:
        for strength in strengths:
            counts = collections.Counter()
            for t, r in sel(arm, strength, MATH):
                text = r["generated"]
                if answer_matches(
                    text, t.target_answer, digit=_digit_form(t.target_answer)
                ):
                    counts["t"] += 1
                elif answer_matches(
                    text, t.source_answer, digit=_digit_form(t.source_answer)
                ):
                    counts["s"] += 1
                else:
                    counts["o"] += 1
            w(f"| {arm} | {strength} | {counts['t']} | {counts['s']} | {counts['o']} |")
    w("")

    # --- template breakdown ----------------------------------------------
    w("## Math cell by template\n")
    w(
        "`first_letter` is the control: same argument tokens, but the downstream "
        "operation is orthographic rather than arithmetic. A gain there as large "
        "as on `double`/`successor` means the effect is about making the argument "
        "swap land, not about arithmetic. `square` is unusable -- its answer key "
        'holds truncations ("twenty" for 25), so the model\'s correct "25" scores '
        "as a miss.\n"
    )
    funcs = ("double", "square", "successor", "first_letter")
    w("| arm | " + " | ".join(funcs) + " |")
    w("|---" * (len(funcs) + 1) + "|")
    for arm in arms:
        strength = best(arm, MATH)
        cells = []
        for func in funcs:
            sub = [(t, r) for t, r in sel(arm, strength, MATH) if t.func == func]
            cells.append(f"{sum(r['hit_re'] for _, r in sub)}/{len(sub)}")
        w(f"| {arm} (a={strength}) | " + " | ".join(cells) + " |")
    w("")

    # --- every cell, both axes -------------------------------------------
    w("## Every cell, at each arm's best strength for that category\n")
    w(
        "Categories confound *what gets swapped* with *what must be produced*. "
        "`arg`/`ans` name the two axes; `numbers/first_letter` (numeric "
        "argument, letter answer) and `animals/legs` (categorical argument, "
        "number-word answer) are the cells that separate them.\n"
    )
    w("| cell | arg | ans | " + " | ".join(a.replace("swap_", "") for a in arms) + " |")
    w("|---" * (len(arms) + 3) + "|")
    cells_by_cat: dict[str, list[str]] = collections.OrderedDict()
    for t, _ in rows:
        cells_by_cat.setdefault(t.category, [])
        if t.func not in cells_by_cat[t.category]:
            cells_by_cat[t.category].append(t.func)
    for category in ("numbers", "months", "animals", "countries"):
        for func in sorted(cells_by_cat.get(category, [])):
            arg, ans, _ = CELL_AXES[(category, func)]
            cells = []
            for arm in arms:
                strength = best(arm, (category,))
                sub = [
                    (t, r) for t, r in sel(arm, strength, (category,)) if t.func == func
                ]
                cells.append(f"{sum(r['hit_re'] for _, r in sub)}/{len(sub)}")
            w(f"| {category}/{func} | {arg} | {ans} | " + " | ".join(cells) + " |")
    w("")

    # --- the two discriminating cells -------------------------------------
    w("## The two cells that break the confound\n")
    w(
        "Read against `swap_averaged`. If the effect were about producing "
        "number-word *answers* (gsm8k is dense in them, so their directions "
        "would be better estimated), `animals/legs` should gain -- its answers "
        "are number words. If it were about *arithmetic*, `numbers/first_letter` "
        "should not gain -- its operation is orthographic. Each strength is "
        "shown separately because the arms peak at different ones.\n"
    )
    for category, func in (("numbers", "first_letter"), ("animals", "legs")):
        arg, ans, _ = CELL_AXES[(category, func)]
        w(f"**{category}/{func}** (argument: {arg}, answer: {ans})\n")
        w("| arm | " + " | ".join(f"a={s}" for s in strengths) + " |")
        w("|---" * (len(strengths) + 1) + "|")
        for arm in arms:
            cells = []
            for strength in strengths:
                sub = [
                    (t, r) for t, r in sel(arm, strength, (category,)) if t.func == func
                ]
                cells.append(f"{sum(r['hit_re'] for _, r in sub)}/{len(sub)}")
            w(f"| {arm} | " + " | ".join(cells) + " |")
        w("")

    # --- matched perturbation budget --------------------------------------
    w("## Math hits against perturbation budget\n")
    w(
        "The arms are not equally aggressive at equal alpha -- gsm8k's "
        "directions perturb less -- so a fixed-alpha comparison is confounded "
        "by magnitude. Pairing each cell's hit count with the KL it cost "
        "removes that: an arm that wins at *lower* KL wins on the merits.\n"
    )
    w("| arm | " + " | ".join(f"a={s}" for s in strengths) + " |")
    w("|---" * (len(strengths) + 1) + "|")
    for arm in arms:
        cells = []
        for strength in strengths:
            sub = sel(arm, strength, MATH)
            kl = st.mean(
                r["kl_from_clean"] for _, r in sel(arm, strength, MATH + NON_MATH)
            )
            cells.append(f"{sum(r['hit_re'] for _, r in sub)}/{len(sub)} @ KL {kl:.2f}")
        w(f"| {arm} | " + " | ".join(cells) + " |")
    w("")

    # --- collateral -------------------------------------------------------
    w("## Collateral damage\n")
    w("| arm | " + " | ".join(f"a={s} KL / mass" for s in strengths) + " |")
    w("|---" * (len(strengths) + 1) + "|")
    for arm in arms:
        cells = []
        for strength in strengths:
            sub = [r for _, r in sel(arm, strength, MATH + NON_MATH)]
            cells.append(
                f"{st.median(r['off_target_kl'] for r in sub):.3f} / "
                f"{st.median(r['concept_mass_ratio'] for r in sub):.2f}"
            )
        w(f"| {arm} | " + " | ".join(cells) + " |")
    w("")

    # --- audit ------------------------------------------------------------
    w("## Every math hit, for manual audit\n")
    w("| arm | a | prompt | injected | wanted | model said |")
    w("|---|---|---|---|---|---|")
    for arm in arms:
        for strength in strengths:
            for t, r in sel(arm, strength, MATH):
                if r["hit_re"]:
                    w(
                        f"| {arm} | {strength} | {t.prompt} | {t.target_arg} | "
                        f"{t.target_answer} | `{r['generated']}` |"
                    )
    w("")

    text = "\n".join(out) + "\n"
    # The label has to reach the filename. Without it, analysing a second
    # panel overwrites the first's report and says nothing -- which is how the
    # component run clobbered C9. "gen" -- the label whose shards carry the
    # generated text this script re-grades, and which produced the original
    # report -- keeps the original path so existing references stay valid;
    # every other label gets its own file.
    leaf = f"c9_domain_causal_{args.config}"
    if args.label != "gen":
        leaf += f"_{args.label}"
    path = Path(args.results) / f"{leaf}.md"
    path.write_text(text)
    print(text)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

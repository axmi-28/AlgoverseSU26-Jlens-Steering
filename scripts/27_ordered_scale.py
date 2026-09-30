"""Held-out test of the ordered-scale hypothesis.

On flexible-generalization the gsm8k-averaged lens gained in every cell whose
ARGUMENT lay on an ordered scale and in none whose argument was an unordered
category -- regardless of what the answer looked like. That reading came from
the same data that suggested it, and the cells are 9-12 trials each. This runs
it on a set built to separate the two axes, with the prediction recorded in
``data/heldout/ordered-scale.json`` before the run.

Stages mirror script 26::

    python scripts/27_ordered_scale.py --stage fit     # GPU, new cotangents
    python scripts/27_ordered_scale.py --stage causal  # GPU
    python scripts/27_ordered_scale.py --stage report  # laptop

``fit`` is NOT optional and NOT a repeat of 26's. The stored pullbacks are
indexed by argument, so a digest fitted against the upstream 16 has no row for
``saturday``; steering this eval needs its own digest against its own 16.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import statistics as st
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jsteer.config import REPO_ROOT, load_config  # noqa: E402
from jsteer.data import heldout_args, heldout_trials  # noqa: E402
from jsteer.steering import answer_matches  # noqa: E402
from jsteer.sweeps import _digit_form  # noqa: E402

SET = "ordered-scale"  # overridden by --set
#: The two corpora the claim is about, plus the published lens as the baseline.
CORPORA = ["gsm8k", "wikitext_a"]

#: Panels addressable with --panel. The ablation panel runs the within-corpus
#: controls against the held-out eval, which is where the effect is largest.
PANELS = {
    "default": CORPORA,
    "ablations": [
        "gsm8k",
        "wikitext_a",
        "gsm8k_q",
        "gsm8k_sol",
        "gsm8k_shuffled",
        "gsm8k_noent",
        "gsm8k_nonum",
        "reasoning_traces",
    ],
    # Trimmed to the five arms the question actually needs: gsm8k and
    # wikitext_a as the two poles, gsm8k_noeq (operators removed, prose kept)
    # and equations (operators only, no prose) as the two halves of the split,
    # and arith_words as the matched control that states the same arithmetic in
    # words. Dropping the rest is a wall-clock decision, not a scientific one.
    "notation": [
        "gsm8k",
        "wikitext_a",
        "gsm8k_noeq",
        "equations",
        "arith_words",
    ],
}


def spec_kwargs(args) -> dict:
    band = load_config(args.config).band
    return {
        "config_name": args.config,
        "domains": PANELS[args.panel],
        "n_per": args.n_per,
        "min_tokens": 128,
        "layers": list(band)[:: args.layer_stride] if args.layer_stride > 1 else None,
        "dim_batch": args.dim_batch,
        "out_dir": args.out_dir,
        "target_words": heldout_args(SET),
    }


def main() -> None:
    global SET
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("fit", "causal", "report"), required=True)
    ap.add_argument("--config", default="qwen3-8b")
    ap.add_argument("--n-per", type=int, default=32)
    ap.add_argument("--layer-stride", type=int, default=3)
    ap.add_argument("--dim-batch", type=int, default=4)
    ap.add_argument("--strengths", default="0.5,1.0,2.0")
    ap.add_argument("--label", default="oscale")
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--panel", choices=tuple(PANELS), default="default")
    ap.add_argument("--set", dest="trial_set", default="ordered-scale")
    ap.add_argument("--results", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()
    SET = args.trial_set
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.stage == "fit":
        from jsteer.sweeps import DomainPullbackSpec, run_domain_pullbacks

        print(
            json.dumps(
                run_domain_pullbacks(DomainPullbackSpec(**spec_kwargs(args))),
                indent=2,
                default=str,
            )
        )
        return

    if args.stage == "causal":
        from jsteer.sweeps import CausalSpec, DomainPullbackSpec, run_causal

        probe = DomainPullbackSpec(**spec_kwargs(args))
        root = Path(args.out_dir) if args.out_dir else REPO_ROOT / "results"
        digests = root / "rq1" / f"group_mean_digests_{probe.namespace}.pt"
        if not digests.exists():
            raise SystemExit(f"run --stage fit first; missing {digests}")
        spec = CausalSpec(
            config_name=args.config,
            mode="swap",
            strengths=[float(s) for s in args.strengths.split(",")],
            directions=["averaged"] + PANELS[args.panel],
            directions_paths=[str(digests)],
            layers=spec_kwargs(args)["layers"],
            trial_set=SET,
            label=args.label,
            out_dir=args.out_dir,
        )
        print(json.dumps(run_causal(spec), indent=2, default=str))
        return

    report(args)


def _mcnemar(a: list[bool], b: list[bool]) -> tuple[float, int, int]:
    only_a = sum(x and not y for x, y in zip(a, b, strict=True))
    only_b = sum(y and not x for x, y in zip(a, b, strict=True))
    n = only_a + only_b
    if n == 0:
        return 1.0, 0, 0
    tail = sum(comb(n, i) for i in range(min(only_a, only_b) + 1))
    return min(1.0, 2 * tail / 2**n), only_a, only_b


def _key(r: dict) -> tuple:
    """Identity of one trial, for pairing arms row by row."""
    return (r["category"], r["func"], r["source_arg"], r["target_arg"])


def report(args) -> None:
    path = (
        Path(args.results)
        / "causal"
        / (f"steering_{args.config}_swap_{args.label}_shard0of{args.n_shards}.json")
    )
    records = json.loads(path.read_text())
    trials = {
        (t.category, t.func, t.source_arg, t.target_arg): t for t in heldout_trials(SET)
    }
    arms = [a for a in dict.fromkeys(r["arm"] for r in records) if a != "baseline"]
    strengths = sorted({r["strength"] for r in records if r["arm"] != "baseline"})

    # Re-grade from stored text, as C9 does: a rule fix must never need a GPU.
    for r in records:
        key = (r["category"], r["func"], r["source_arg"], r["target_arg"])
        t = trials[key]
        want = t.source_answer if r["arm"] == "baseline" else t.target_answer
        r["hit_re"] = bool(r.get("generated")) and answer_matches(
            r["generated"], want, digit=_digit_form(want)
        )
        r["axes"] = (t.arg_scale, t.answer_type)
        r["depth"] = t.depth

    # The screen: a trial whose own answer the model never produces cannot show
    # a swap, because there is nothing to swap away from.
    ok = {
        (r["category"], r["func"], r["source_arg"])
        for r in records
        if r["arm"] == "baseline" and r["hit_re"]
    }

    def sel(arm, strength, pred):
        return [
            r
            for r in records
            if r["arm"] == arm
            and r["strength"] == strength
            and pred(r)
            and (r["category"], r["func"], r["source_arg"]) in ok
        ]

    def best(arm, pred):
        return max(strengths, key=lambda s: sum(r["hit_re"] for r in sel(arm, s, pred)))

    out: list[str] = []
    w = out.append
    w(f"# Ordered-scale held-out eval ({args.config})\n")
    w(
        "Prediction registered before the run: the gsm8k-averaged lens lifts the "
        "ordered-argument categories on BOTH answer types, and lifts neither "
        "unordered category on either.\n"
    )

    w("## Baseline screen\n")
    w("Cells where the model cannot produce its own answer are excluded.\n")
    w("| category | func | arg_scale | answer | usable / 12 |")
    w("|---|---|---|---|---|")
    base = collections.Counter()
    tot = collections.Counter()
    for r in records:
        if r["arm"] != "baseline":
            continue
        k = (r["category"], r["func"])
        tot[k] += 1
        base[k] += bool(r["hit_re"])
    for (cat, func), n in sorted(tot.items()):
        t = next(t for t in heldout_trials(SET) if t.category == cat and t.func == func)
        flag = "" if base[(cat, func)] else "  <- AT FLOOR, uninformative"
        w(
            f"| {cat} | {func} | {t.arg_scale} | {t.answer_type} | "
            f"{base[(cat, func)]}/{n}{flag} |"
        )
    w("")

    w("## The 2x2\n")
    w("| arg_scale | answer | " + " | ".join(arms) + " |")
    w("|---" * (len(arms) + 2) + "|")
    for scale in ("ordered", "unordered"):
        for answer in ("number", "letter"):

            def pred(r, _s=scale, _a=answer):
                return r["axes"] == (_s, _a)

            cells = []
            for arm in arms:
                sub = sel(arm, best(arm, pred), pred)
                cells.append(f"{sum(r['hit_re'] for r in sub)}/{len(sub)}")
            w(f"| {scale} | {answer} | " + " | ".join(cells) + " |")
    w("")

    w("## Matched-operation controls\n")
    w(
        "Same operation, same answer type, differing only in the argument. "
        "These carry the claim: a gain on the ordered row and none on the "
        "unordered row cannot be explained by what the model must produce.\n"
    )
    for func in ("first_letter", "letter_count"):
        w(f"**{func}**\n")
        w("| arg_scale | " + " | ".join(arms) + " |")
        w("|---" * (len(arms) + 1) + "|")
        for scale in ("ordered", "unordered"):

            def pred(r, _f=func, _s=scale):
                return r["func"] == _f and r["axes"][0] == _s

            cells = []
            for arm in arms:
                sub = sel(arm, best(arm, pred), pred)
                cells.append(f"{sum(r['hit_re'] for r in sub)}/{len(sub)}")
            w(f"| {scale} | " + " | ".join(cells) + " |")
        w("")

    w("## By template\n")
    w(
        "Added after the fact, because neither pre-registered axis explains the "
        "result and this one does. `first_letter` reads the argument's identity "
        "off its surface form with no retrieval and no computation; the other "
        "templates need a stored fact (players), a definition (cardinal, "
        "position) or a count (letter_count).\n"
    )
    w("| template | arg_scale | answer | " + " | ".join(arms) + " |")
    w("|---" * (len(arms) + 3) + "|")
    seen: set[tuple[str, str]] = set()
    for t in heldout_trials(SET):
        if (t.category, t.func) in seen:
            continue
        seen.add((t.category, t.func))

        def pred(r, _c=t.category, _f=t.func):
            return r["category"] == _c and r["func"] == _f

        cells = []
        for arm in arms:
            sub = sel(arm, best(arm, pred), pred)
            cells.append(f"{sum(r['hit_re'] for r in sub)}/{len(sub)}")
        w(
            f"| {t.category}/{t.func} | {t.arg_scale} | {t.answer_type} | "
            + " | ".join(cells)
            + " |"
        )
    w("")

    w("## Paired tests, gsm8k against the published lens\n")
    w("| slice | gsm8k | averaged | won | lost | p (exact McNemar) |")
    w("|---|---|---|---|---|---|")
    for label, pred in (
        ("ordered args", lambda r: r["axes"][0] == "ordered"),
        ("unordered args", lambda r: r["axes"][0] == "unordered"),
        ("first_letter only", lambda r: r["func"] == "first_letter"),
        ("all other templates", lambda r: r["func"] != "first_letter"),
        ("everything", lambda r: True),
    ):
        if "swap_gsm8k" not in arms:
            continue
        s_g, s_a = best("swap_gsm8k", pred), best("swap_averaged", pred)
        g = {_key(r): r["hit_re"] for r in sel("swap_gsm8k", s_g, pred)}
        a = {_key(r): r["hit_re"] for r in sel("swap_averaged", s_a, pred)}
        shared = sorted(set(g) & set(a))
        gv = [g[k] for k in shared]
        av = [a[k] for k in shared]
        p, only_g, only_a = _mcnemar(gv, av)
        w(
            f"| {label} | {sum(gv)}/{len(gv)} | {sum(av)}/{len(av)} | "
            f"{only_g} | {only_a} | {p:.4f} |"
        )
    w("")

    depths = sorted({t.depth for t in heldout_trials(SET) if t.depth >= 0})
    if depths:
        w("## Gain against computation depth\n")
        w(
            "Registered prediction: the gain shrinks monotonically with depth, "
            "because the corpus buys a cleaner rebinding and every downstream "
            "step attenuates it.\n"
        )
        w(
            "| depth | template | "
            + " | ".join(arms)
            + " | median target_rank (gsm8k) |"
        )
        w("|---" * (len(arms) + 3) + "|")
        for d in depths:
            name = next(t.func for t in heldout_trials(SET) if t.depth == d)

            def pred(r, _d=d):
                return r.get("depth", -1) == _d

            cells = []
            for arm in arms:
                sub = sel(arm, best(arm, pred), pred)
                cells.append(f"{sum(r['hit_re'] for r in sub)}/{len(sub)}")
            g = sel("swap_gsm8k", best("swap_gsm8k", pred), pred)
            rank = f"{st.median(r['target_rank'] for r in g):.0f}" if g else "-"
            w(f"| {d} | {name} | " + " | ".join(cells) + f" | {rank} |")
        w("")

    w("## Is the target ARGUMENT installed better?\n")
    w(
        "`target_rank` is the rank of the swap target's own token, which is "
        "what the intervention writes -- not the answer, which is what the "
        "model then computes. Separating them tests whether a corpus buys a "
        "cleaner rebinding or a better downstream computation. Lower is "
        "better.\n"
    )
    w("| arm | " + " | ".join(f"a={s} rank" for s in strengths) + " |")
    w("|---" * (len(strengths) + 1) + "|")
    for arm in arms:
        cells = []
        for strength in strengths:
            sub = sel(arm, strength, lambda r: True)
            cells.append(
                f"{st.median(r['target_rank'] for r in sub):.0f}" if sub else "-"
            )
        w(f"| {arm} | " + " | ".join(cells) + " |")
    w("")
    w(
        "Per template, at each arm's own best strength. The claim this tests: "
        "the templates whose steering improves are the templates whose target "
        "rank improves.\n"
    )
    w("| template | " + " | ".join(f"{a} rank / hits" for a in arms) + " |")
    w("|---" * (len(arms) + 1) + "|")
    seen2: set[tuple[str, str]] = set()
    for t in heldout_trials(SET):
        if (t.category, t.func) in seen2:
            continue
        seen2.add((t.category, t.func))

        def pred(r, _c=t.category, _f=t.func):
            return r["category"] == _c and r["func"] == _f

        cells = []
        for arm in arms:
            sub = sel(arm, best(arm, pred), pred)
            if not sub:
                cells.append("-")
                continue
            rank = st.median(r["target_rank"] for r in sub)
            cells.append(f"{rank:.0f} / {sum(r['hit_re'] for r in sub)}")
        w(f"| {t.category}/{t.func} | " + " | ".join(cells) + " |")
    w("")

    w("## Perturbation budget\n")
    w("| arm | " + " | ".join(f"a={s}" for s in strengths) + " |")
    w("|---" * (len(strengths) + 1) + "|")
    for arm in arms:
        cells = []
        for s in strengths:
            sub = sel(arm, s, lambda r: True)
            kl = st.mean(r["kl_from_clean"] for r in sub) if sub else float("nan")
            cells.append(f"{sum(r['hit_re'] for r in sub)}/{len(sub)} @ KL {kl:.2f}")
        w(f"| {arm} | " + " | ".join(cells) + " |")
    w("")

    text = "\n".join(out) + "\n"
    suffix = "" if args.panel == "default" else f"_{args.panel}"
    dest = Path(args.results) / f"c10_ordered_scale{suffix}_{args.config}.md"
    dest.write_text(text)
    print(text)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()

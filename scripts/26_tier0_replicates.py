"""Tier 0 -- an error bar for the domain result.

Every corpus number so far is a single draw of 32 documents, so "gsm8k beats
the lens 11/39 to 1/39" and "one lucky sample of gsm8k beat one unlucky sample
of wikitext" are the same measurement. This fits ``N_REPLICATES`` disjoint
draws of each and reports the spread, which is the null every later comparison
has to clear.

Three stages, so the GPU work and the analysis can run in different places::

    python scripts/26_tier0_replicates.py --stage fit     # GPU, ~4 s/prompt
    python scripts/26_tier0_replicates.py --stage causal  # GPU
    python scripts/26_tier0_replicates.py --stage report  # laptop

``fit`` uses :func:`run_domain_pullbacks`, not the full-Jacobian sweep: the
causal arm reads only the target pullbacks, and averaging commutes with the
pullback, so a replicate costs one backward pass per prompt instead of
``d_model``. Ten corpora at n=32 is ~20 GPU-minutes rather than ~4.7 hours,
which is what makes replicating affordable at all.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jsteer.config import REPO_ROOT, load_config  # noqa: E402
from jsteer.domains import N_REPLICATES  # noqa: E402

GSM = [f"gsm8k_r{i}" for i in range(N_REPLICATES)]
WIKI = [f"wikirep_r{i}" for i in range(N_REPLICATES)]
PANEL = GSM + WIKI

#: The expanded corpus panel, run on the same eval as Tier 0 so the numbers
#: are directly comparable to the replicate spread that calibrates them.
#: Each corpus moves one variable against gsm8k -- see jsteer.domains.
#: The ablation panel: GSM8K with one structural property destroyed at a time,
#: anchored by untouched gsm8k and wikitext_a, plus the cross-dataset corpora
#: that test derivation without mathematics.
AQUA = [f"aqua_rat_r{i}" for i in range(N_REPLICATES)] + ["gsm8k", "wikitext_a"]

ABLATIONS = [
    "gsm8k",
    "wikitext_a",
    "gsm8k_q",
    "gsm8k_sol",
    "gsm8k_shuffled",
    "gsm8k_noent",
    "gsm8k_nonum",
    "reasoning_traces",
    "physics_worked",
    "code_python",
    "tinystories",
]

EXPANDED = [
    "gsm8k",
    "wikitext_a",
    "svamp",
    "aqua_rat",
    "math_algebra",
    "openwebmath",
    "arith_words",
    "ordered_scale",
]


def spec_kwargs(args) -> dict:
    band = load_config(args.config).band
    return {
        "config_name": args.config,
        "domains": {
            "replicates": GSM + WIKI,
            "expanded": EXPANDED,
            "ablations": ABLATIONS,
            "aqua": AQUA,
        }[args.panel],
        "n_per": args.n_per,
        "min_tokens": 128,
        "layers": list(band)[:: args.layer_stride] if args.layer_stride > 1 else None,
        "dim_batch": args.dim_batch,
        "out_dir": args.out_dir,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("fit", "causal", "report"), required=True)
    ap.add_argument("--config", default="qwen3-8b")
    ap.add_argument("--n-per", type=int, default=32)
    ap.add_argument("--layer-stride", type=int, default=3)
    ap.add_argument("--dim-batch", type=int, default=16)
    ap.add_argument("--strengths", default="0.5,1.0,2.0")
    ap.add_argument("--label", default="tier0")
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument(
        "--panel",
        choices=("replicates", "expanded", "ablations", "aqua"),
        default="replicates",
        help="replicates = the Tier-0 spread; expanded = the corpus comparison",
    )
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--results", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()
    # Without this the sweep runs completely silently: the library logs through
    # `logger.info` and nothing configures a handler, so a 30-minute run shows
    # three lines of HuggingFace chatter and no progress at all.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.stage == "fit":
        from jsteer.sweeps import DomainPullbackSpec, run_domain_pullbacks

        spec = DomainPullbackSpec(**spec_kwargs(args))
        print(json.dumps(run_domain_pullbacks(spec), indent=2, default=str))
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
            # Every arm must be NAMED here, not merely present in the digest:
            # run_causal iterates spec.directions, so a group that is loaded but
            # unrequested is silently absent from the output. That produced a
            # complete-looking run with one arm and no replicates.
            directions=["averaged"]
            + {
                "replicates": PANEL,
                "expanded": EXPANDED,
                "ablations": ABLATIONS,
                "aqua": AQUA,
            }[args.panel],
            directions_paths=[str(digests)],
            layers=spec_kwargs(args)["layers"],
            label=args.label,
            shard=args.shard,
            n_shards=args.n_shards,
            out_dir=args.out_dir,
        )
        print(json.dumps(run_causal(spec), indent=2, default=str))
        return

    report(args)


def report(args) -> None:
    """Spread across replicates, per corpus -- the actual deliverable."""
    import importlib.util

    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("c9", here / "25_domain_causal.py")
    c9 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c9)

    pattern = (
        f"{args.results}/causal/steering_{args.config}_swap_{args.label}"
        "_shard{shard}of{n}.json"
    )
    probe = json.loads(Path(pattern.format(shard=0, n=args.n_shards)).read_text())
    arms = [a for a in dict.fromkeys(r["arm"] for r in probe) if a != "baseline"]
    strengths = sorted({r["strength"] for r in probe if r["arm"] != "baseline"})
    rows = c9.load(pattern, args.n_shards, 1 + len(arms) * len(strengths))
    c9.regrade(rows)
    ok = {t.prompt for t, r in rows if r["arm"] == "baseline" and r["hit_re"]}

    def hits(arm: str, strength: float, cats) -> tuple[int, int]:
        sub = [
            (t, r)
            for t, r in rows
            if r["arm"] == arm
            and r["strength"] == strength
            and t.category in cats
            and t.prompt in ok
        ]
        return sum(r["hit_re"] for _, r in sub), len(sub)

    def best(arm: str, cats) -> float:
        return max(strengths, key=lambda s: hits(arm, s, cats)[0])

    if args.panel in ("expanded", "ablations", "aqua"):
        expanded_report(args, rows, ok, hits, best, arms, strengths, c9)
        return

    out: list[str] = []
    w = out.append
    w(f"# Tier 0 - replicate spread ({args.config})\n")
    w(
        f"{N_REPLICATES} disjoint draws of {args.n_per} documents per corpus. "
        "A corpus gap is evidence only if it clears the spread *within* each "
        "corpus. Strength is chosen per arm on the category shown, which is "
        "the same selection C9 makes.\n"
    )
    for label, cats in (("math (numbers)", c9.MATH), ("non-math", c9.NON_MATH)):
        w(f"## {label}\n")
        header = ["corpus"] + [f"r{i}" for i in range(N_REPLICATES)]
        header += ["median", "min-max", "spread"]
        w("| " + " | ".join(header) + " |")
        w("|---" * len(header) + "|")
        for corpus, members in (("gsm8k", GSM), ("wikitext", WIKI)):
            counts, cells = [], []
            for name in members:
                arm = f"swap_{name}"
                if arm not in arms:
                    cells.append("-")
                    continue
                n, total = hits(arm, best(arm, cats), cats)
                counts.append(n)
                cells.append(f"{n}/{total}")
            if counts:
                w(
                    f"| {corpus} | " + " | ".join(cells) + " | "
                    f"{st.median(counts):.0f} | {min(counts)}-{max(counts)} | "
                    f"{max(counts) - min(counts)} |"
                )
        if "swap_averaged" in arms:
            n, total = hits("swap_averaged", best("swap_averaged", cats), cats)
            w(
                "| published lens | "
                + " | ".join(["-"] * N_REPLICATES)
                + f" | {n}/{total} | - | - |"
            )
        w("")
        # The verdict, stated rather than left to the reader.
        g = [
            hits(f"swap_{m}", best(f"swap_{m}", cats), cats)[0]
            for m in GSM
            if f"swap_{m}" in arms
        ]
        k = [
            hits(f"swap_{m}", best(f"swap_{m}", cats), cats)[0]
            for m in WIKI
            if f"swap_{m}" in arms
        ]
        if g and k:
            verdict = (
                "SEPARATED - every gsm8k draw beats every wikitext draw"
                if min(g) > max(k)
                else "OVERLAPPING - the corpora are not separated by these draws"
            )
            w(f"gsm8k {min(g)}-{max(g)} vs wikitext {min(k)}-{max(k)}: **{verdict}**\n")

    text = "\n".join(out) + "\n"
    path = Path(args.results) / f"tier0_replicates_{args.config}.md"
    path.write_text(text)
    print(text)
    print(f"wrote {path}")


def expanded_report(args, rows, ok, hits, best, arms, strengths, c9) -> None:
    """The corpus comparison, read against Tier 0's within-corpus spread.

    Tier 0 measured that spread on the same eval: gsm8k draws span 8-10 on
    math and wikitext draws 0-2, so a corpus is meaningfully different from
    gsm8k only if it falls outside 8-10, and meaningfully better than the
    control only if it clears 2. Without those two numbers this table is a
    ranking of noise.
    """
    import statistics as st

    out: list[str] = []
    w = out.append
    w(f"# Expanded corpus panel ({args.config})\n")
    w(
        f"{args.n_per} documents per corpus, same eval and same grading as C9 "
        "and Tier 0. **Read every number against the Tier-0 spread**: five "
        "disjoint gsm8k draws scored 8-10 on math and five wikitext draws 0-2, "
        "so differences inside those bands are draws, not effects.\n"
    )
    # Each column takes its OWN best strength. Scoring non-math at the
    # math-optimal alpha understates every arm whose optimum differs -- it put
    # the published lens at 10/105 when its own best on that slice is 42/105.
    w("| corpus | math (of 39) | a | KL | non-math (of 105) | a | KL |")
    w("|---|---|---|---|---|---|---|")
    panel = {"expanded": EXPANDED, "ablations": ABLATIONS, "aqua": AQUA}[args.panel]
    order = ["swap_averaged"] + [f"swap_{c}" for c in panel]

    def mean_kl(arm, strength):
        vals = [
            r["kl_from_clean"]
            for t, r in rows
            if r["arm"] == arm and r["strength"] == strength and t.prompt in ok
        ]
        return st.mean(vals) if vals else float("nan")

    for arm in [a for a in order if a in arms]:
        cells = []
        for cats in (c9.MATH, c9.NON_MATH):
            a_star = best(arm, cats)
            h, tot = hits(arm, a_star, cats)
            cells.append(f"{h}/{tot} | {a_star} | {mean_kl(arm, a_star):.2f}")
        w(f"| {arm.replace('swap_', '')} | " + " | ".join(cells) + " |")
    w("")
    text = "\n".join(out) + "\n"
    stem = {
        "expanded": "c11_expanded_corpora",
        "ablations": "c12_ablations",
        "aqua": "c13_aqua_replicates",
    }[args.panel]
    path = Path(args.results) / f"{stem}_{args.config}.md"
    path.write_text(text)
    print(text)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

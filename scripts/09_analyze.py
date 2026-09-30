#!/usr/bin/env python3
"""Preliminary distribution tables for RQ1 and RQ2. Local, no GPU.

Emits a Markdown report whose tables paste cleanly into Google Docs. The two
research questions are reported in fully separate sections because they are
different objects: RQ1 compares *matrices* (J_x against J_bar), RQ2 compares
*directions* (g_x = J_x^T u_y against g_bar = J_bar^T u_y).

    python scripts/09_analyze.py --config qwen3-8b --positions all
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import torch

from jsteer.analysis import leave_one_out_influence, subspace_alignment
from jsteer.config import REPO_ROOT, load_config

logger = logging.getLogger("analyze")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="qwen3-8b")
    parser.add_argument("--positions", default="all")
    parser.add_argument("--results", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=200,
        help="pairwise subspace alignments sampled per layer (all pairs is O(n^2))",
    )
    return parser.parse_args()


def med(xs: list[float]) -> float:
    return st.median(xs) if xs else float("nan")


def quantile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def _cell(value) -> str:
    r"""Escape ``|`` so a cell like ``median |g_x| / |g_bar|`` stays one cell.

    Several statistics here are written in norm bars, and an unescaped pipe
    inside a cell silently splits it: the row grows extra columns, every later
    value shifts left, and both the Markdown renderer and anything parsing the
    table read the wrong numbers out of it.
    """
    return str(value).replace("|", r"\|")


def table(headers: list[str], rows: list[list], out: list[str]) -> None:
    """Emit a Markdown pipe table -- pastes into Google Docs as a real table."""
    out.append("| " + " | ".join(_cell(h) for h in headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(_cell(c) for c in row) + " |")
    out.append("")


# --------------------------------------------------------------------------


def rq1(args, results: Path, out: list[str]) -> None:
    config = load_config(args.config)
    scalars_path = results / "rq1" / f"scalars_{config.name}_{args.positions}.json"
    if not scalars_path.exists():
        logger.warning("no RQ1 scalars at %s -- skipping", scalars_path)
        return
    records = json.loads(scalars_path.read_text())
    layers = sorted({r["layer"] for r in records})
    n_eval = len({r["prompt"] for r in records if r["group"] == "eval"})
    n_fit = len({r["prompt"] for r in records if r["group"] == "fit"})

    out += [
        "# RQ1 - Distribution of prompt-local Jacobians J_x around the averaged J_bar",
        "",
        "## Methodology",
        "",
        f"**Model.** {config.hf_model_id} ({config.n_layers} layers, d_model "
        f"{config.d_model}), with the pre-fitted Jacobian lens from "
        f"`neuronpedia/jacobian-lens`. That lens supplies **J_bar**: one averaged "
        f"Jacobian per layer, fitted by Anthropic on ~{config.fit.prompts_fitted} "
        "wikitext-103 prompts. J_bar is the object the lens uses to *read* the "
        "residual stream, and the object steering re-uses as a direction to *write*.",
        "",
        "**J_x, the prompt-local Jacobian.** For a single prompt x we compute the "
        "Jacobian of the final-layer residual with respect to layer l, at that "
        "prompt only, using the same estimator the lens averages (verified "
        "bit-for-bit against `jlens.fitting.jacobian_for_prompt`). Where the lens "
        "has one J_bar per layer, we have one J_x per (prompt, layer).",
        "",
        "**Prompts.** Two groups:",
        "",
        f"- **eval** (n = {n_eval}): the *flexible-generalization* set from "
        "`anthropics/jacobian-lens` - 4 categories x 4 templates x 4 arguments = 64 "
        'prompts, e.g. "The capital of France is the city of". Out of the lens\'s '
        "fitting distribution.",
        f"- **fit** (n = {n_fit}): wikitext-103 windows drawn from the *same* "
        "distribution the lens was fitted on. These are a **calibrated null**: "
        "because J_bar is itself a mean of per-prompt Jacobians over this "
        "distribution, the fit group's cloud is centred on J_bar by construction, "
        "so whatever spread we measure there is the finite-sample floor for the "
        "same measurement on eval.",
        "",
        f"**Layers.** L{layers[0]}-L{layers[-1]} ({len(layers)} layers), the "
        "workspace band = 38%-92% of depth per the J-Lens paper.",
        "",
        "**Position convention.** `"
        + args.positions
        + "` - J_x is averaged over these token positions. (The fitting estimator "
        "averages over positions; the steering intervention writes at particular "
        "ones, so this is a stated choice, not an inherited default.)",
        "",
        "**Storage.** A full J_x is 4096x4096 fp32 = 67 MB per layer, so none is "
        "kept. Each is reduced in memory to: the top-64 left and right singular "
        "vectors, the pullbacks of the 16 target tokens, and the scalars below.",
        "",
        "## Table 1.1 - How far is a prompt-local Jacobian from the average?",
        "",
        "Relative Frobenius distance `||J_x - J_bar||_F / ||J_bar||_F`. "
        "0 = identical to the lens's averaged Jacobian; 1 = as far from J_bar as "
        "J_bar is from zero.",
        "",
    ]

    rows = []
    for layer in layers:
        ev = [
            r["rel_frobenius"]
            for r in records
            if r["group"] == "eval" and r["layer"] == layer
        ]
        ft = [
            r["rel_frobenius"]
            for r in records
            if r["group"] == "fit" and r["layer"] == layer
        ]
        rows.append(
            [
                f"L{layer}",
                f"{med(ev):.3f}",
                f"{quantile(ev, 0.10):.3f}-{quantile(ev, 0.90):.3f}",
                f"{med(ft):.3f}",
                f"{med(ev) / med(ft):.2f}x" if med(ft) else "-",
            ]
        )
    table(
        ["Layer", "eval median", "eval p10-p90", "fit median (null)", "eval / fit"],
        rows,
        out,
    )

    out += [
        "## Table 1.2 - Spectrum: how many directions does J_x actually use?",
        "",
        "Participation ratio `(sum s^2)^2 / sum s^4` over the singular values - the "
        "effective number of directions carrying weight, out of 4096. Reported "
        "because a subspace-agreement number is only interpretable next to how many "
        "directions there are to agree about.",
        "",
    ]
    rows = []
    for layer in layers:
        ev = [
            r["participation_ratio"]
            for r in records
            if r["group"] == "eval" and r["layer"] == layer
        ]
        ft = [
            r["participation_ratio"]
            for r in records
            if r["group"] == "fit" and r["layer"] == layer
        ]
        ident = [
            r["identity_distance"]
            for r in records
            if r["group"] == "eval" and r["layer"] == layer
        ]
        rows.append(
            [f"L{layer}", f"{med(ev):.0f}", f"{med(ft):.0f}", f"{med(ident):.2f}"]
        )
    table(
        ["Layer", "eval PR (eff. rank)", "fit PR", "eval ||J_x - I||_F / sqrt(d)"],
        rows,
        out,
    )

    # ---- subspace alignment from the digests ----
    digest_dir = results / "rq1" / "digests" / f"{config.name}_{args.positions}_k64"
    digests = sorted(digest_dir.glob("*.pt")) if digest_dir.exists() else []
    if digests:
        out += [
            "## Table 1.3 - Do different prompts share a Jacobian subspace?",
            "",
            "Subspace alignment in [0, 1]: mean squared cosine of the principal "
            "angles between two top-64 singular subspaces (`||Q_A^T Q_B||_F^2 / 64`). "
            "1 = identical subspace, **0.016 = chance** at k=64, d=4096.",
            "",
            "- **left** = output side: which final-layer directions this Jacobian "
            "can write into, hence which target tokens have a stable readout.",
            "- **right** = input side: where in the residual stream a steering "
            "vector must be written to land anywhere.",
            "- **eval-eval** = agreement between two different eval prompts.",
            "- **eval-J_bar** = agreement between one eval prompt and the lens.",
            "",
        ]
        loaded = {}
        for path in digests:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            loaded[path.stem] = blob
        eval_keys = [k for k in loaded if not k.startswith("wikitext")]
        fit_keys = [k for k in loaded if k.startswith("wikitext")]

        from jsteer.loading import load_lens

        lens = load_lens(config)
        rows = []
        generator = torch.Generator().manual_seed(0)
        for layer in layers:
            J_bar = lens.jacobians[layer].float()
            U, _, Vh = torch.linalg.svd(J_bar, full_matrices=False)
            bar_left, bar_right = U[:, :64].T.contiguous(), Vh[:64].contiguous()

            # `layer` bound as a default: these close over the loop variable.
            def pair_scores(keys, side, layer=layer):
                scores = []
                n = len(keys)
                pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
                if len(pairs) > args.max_pairs:
                    index = torch.randperm(len(pairs), generator=generator)[
                        : args.max_pairs
                    ]
                    pairs = [pairs[i] for i in index.tolist()]
                for i, j in pairs:
                    a, b = loaded[keys[i]][layer][side], loaded[keys[j]][layer][side]
                    scores.append(subspace_alignment(a, b))
                return scores

            def bar_scores(keys, side, basis, layer=layer):
                return [subspace_alignment(loaded[k][layer][side], basis) for k in keys]

            # Also at k=8. Table 1.2 shows the effective rank is ~9-13 in the
            # early band, so a top-64 basis there is ~50 directions of numerical
            # noise and the k=64 alignment is dominated by them. k=8 sits inside
            # the effective rank everywhere in the band and is the number to
            # read for the early layers.
            def pair8(keys, side, layer=layer):
                sc = []
                n = len(keys)
                pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
                if len(pairs) > args.max_pairs:
                    idx = torch.randperm(len(pairs), generator=generator)[
                        : args.max_pairs
                    ]
                    pairs = [pairs[i] for i in idx.tolist()]
                for i, j in pairs:
                    a = loaded[keys[i]][layer][side][:8]
                    b = loaded[keys[j]][layer][side][:8]
                    sc.append(subspace_alignment(a, b))
                return sc

            rows.append(
                [
                    f"L{layer}",
                    f"{med(pair_scores(eval_keys, 'left')):.3f}",
                    f"{med(bar_scores(eval_keys, 'left', bar_left)):.3f}",
                    f"{med(pair_scores(eval_keys, 'right')):.3f}",
                    f"{med(bar_scores(eval_keys, 'right', bar_right)):.3f}",
                    f"{med(bar_scores(fit_keys, 'left', bar_left)):.3f}"
                    if fit_keys
                    else "-",
                    f"{med(pair8(eval_keys, 'left')):.3f}",
                    f"{med(pair8(eval_keys, 'right')):.3f}",
                ]
            )
        table(
            [
                "Layer",
                "left: eval-eval",
                "left: eval-J_bar",
                "right: eval-eval",
                "right: eval-J_bar",
                "left: fit-J_bar (null)",
                "left k=8: eval-eval",
                "right k=8: eval-eval",
            ],
            rows,
            out,
        )

    # ---- bias vs variance from the pooled group means ----
    pooled = results / "rq1" / f"group_mean_digests_{config.name}_{args.positions}.pt"
    if pooled.exists():
        blob = torch.load(pooled, map_location="cpu", weights_only=False)
        out += [
            "## Table 1.4 - Bias or variance? Where the cloud's centre sits",
            "",
            "The decisive question for steering. Average the prompt-local Jacobians "
            "within a group and ask how far *that mean* is from J_bar.",
            "",
            "- If the eval mean lands on J_bar, the spread is **variance**: the "
            "averaged direction is right on average and steering failures are luck.",
            "- If the eval mean is displaced, the spread is **bias**: the averaged "
            "direction is systematically wrong for this distribution and failures "
            "are predictable. This is the stronger result.",
            "",
            "The fit column is the calibrated null - it *must* go to 0 as n grows, "
            "so its value here is the finite-sample floor the eval column is read "
            "against.",
            "",
        ]
        rows = []
        for layer in layers:
            entry = {}
            for key, d in blob["digests"].items():
                group, lay = key.split("|")
                if int(lay) == layer:
                    entry[group] = d
            if "eval" not in entry:
                continue
            observed = entry["eval"]["scalars"]["rel_frobenius"]
            individual = med(
                [
                    r["rel_frobenius"]
                    for r in records
                    if r["group"] == "eval" and r["layer"] == layer
                ]
            )
            # If the cloud were centred on J_bar and merely scattered, averaging
            # n independent draws would shrink the distance by sqrt(n). This is
            # that prediction; the ratio against it is the bias.
            predicted = individual / (entry["eval"]["n"] ** 0.5)
            rows.append(
                [
                    f"L{layer}",
                    f"{observed:.3f}",
                    f"{predicted:.3f}",
                    f"{observed / predicted:.1f}x",
                    f"{entry['fit']['scalars']['rel_frobenius']:.3f}"
                    if "fit" in entry
                    else "-",
                ]
            )
        out += [
            "The **variance-only prediction** is what the centre's distance would "
            "be if the cloud were genuinely centred on J_bar and merely scattered: "
            "averaging n independent draws shrinks the distance by sqrt(n), so it "
            "is the Table 1.1 median divided by sqrt(64). The **bias ratio** is "
            "how many times further out the centre actually sits. ~1x would mean "
            "pure variance; large means bias.",
            "",
        ]
        table(
            [
                "Layer",
                "observed ||mean(J_x) - J_bar||_F / ||J_bar||_F",
                "variance-only prediction",
                "bias ratio",
                "fit null (n=16)",
            ],
            rows,
            out,
        )


# --------------------------------------------------------------------------


def rq2(args, results: Path, out: list[str]) -> None:
    config = load_config(args.config)
    path = results / "rq2" / f"pullbacks_{config.name}_{args.positions}.json"
    if not path.exists():
        logger.warning("no RQ2 records at %s -- skipping", path)
        return
    records = json.loads(path.read_text())
    layers = sorted({r["layer"] for r in records})
    targets = sorted({r["target"] for r in records})
    n_eval = len({r["prompt"] for r in records if r["group"] == "eval"})
    n_fit = len({r["prompt"] for r in records if r["group"] == "fit"})

    out += [
        "",
        "---",
        "",
        "# RQ2 - Distribution of prompt-local pulled-back directions g_x around g_bar",
        "",
        "## Methodology",
        "",
        "**The object.** To steer a model toward token y, J-Lens style steering "
        "uses the *pullback* of that token's unembedding row through the Jacobian:",
        "",
        "    g_bar = J_bar^T u_y      (what the lens gives you - one direction per token)",
        "    g_x   = J_x^T u_y        (the same thing computed at prompt x only)",
        "",
        "RQ2 asks how far g_x scatters around g_bar. This is the question that "
        "matters operationally, because steering writes a *direction*, not a matrix - "
        "so a large matrix-level difference is only relevant insofar as it moves the "
        "directions anyone actually uses.",
        "",
        "**Cost note.** A pullback costs one backward pass per batch of target "
        "tokens and yields every band layer at once, so RQ2 runs on far more "
        "prompts than RQ1 (which needs one backward pass per residual dimension).",
        "",
        f"**Prompts.** eval n = {n_eval} (the 64 flexible-generalization prompts), "
        f"fit n = {n_fit} (wikitext-103, the calibrated null - see RQ1 methodology).",
        "",
        f"**Target tokens (n = {len(targets)}).** The **argument** words of the "
        "flexible-generalization set: " + ", ".join(targets) + ".",
        "",
        "These are the arguments, *not* the answers. Upstream's task injects the "
        'argument ("Canada") and grades the answer ("Ottawa"); the two are '
        "different tokens by design, so a hit cannot be manufactured by boosting "
        "the graded token's logit. The Jacobian is therefore only ever asked about "
        "arguments. 16 targets is small enough to pull **every** target back "
        "through **every** prompt, which makes relevance a measured variable:",
        "",
        "- **self** - the target is this prompt's own argument.",
        "- **swap_target** - one of the 3 other arguments in the same category "
        "(the tokens upstream's swap trials actually use).",
        "- **foreign** - an argument from a different category, unrelated to the "
        "prompt.",
        "",
        f"**Layers.** L{layers[0]}-L{layers[-1]}. **Positions.** `{args.positions}`.",
        "",
        "**Metrics.** Per (prompt, layer, target): the cosine between g_x and "
        "g_bar, and the length ratio |g_x| / |g_bar|. Cosine is the primary "
        "measure - it is the part of the comparison that survives the "
        "normalisation every steering recipe applies. Length is reported "
        "separately because normalising discards it and it turns out to be large.",
        "",
        "## Table 2.1 - HEADLINE: is the averaged direction the right direction?",
        "",
        "`cos(g_x, g_bar)` by layer. 1.0 = the lens's averaged direction is exactly "
        "the prompt's own; 0 = orthogonal; < 0 = points the wrong way.",
        "",
    ]

    by = defaultdict(list)
    for r in records:
        by[(r["group"], r["layer"])].append(r)

    rows = []
    for layer in layers:
        ev = [r["cos"] for r in by[("eval", layer)]]
        ft = [r["cos"] for r in by[("fit", layer)]]
        rows.append(
            [
                f"L{layer}",
                f"{med(ev):.3f}",
                f"{quantile(ev, 0.10):.3f}",
                f"{med(ft):.3f}",
                f"{med(ft) - med(ev):+.3f}",
            ]
        )
    table(
        [
            "Layer",
            "eval median",
            "eval p10 (worst 10%)",
            "fit median (null)",
            "null - eval",
        ],
        rows,
        out,
    )

    out += [
        "## Table 2.2 - Length: the part normalisation throws away",
        "",
        "`|g_x| / |g_bar|`. A value of 2.0 means the prompt's own direction is "
        "twice as long as the averaged one, so steering at 'matched magnitude' "
        "using g_bar under-shoots by 2x at that layer.",
        "",
    ]
    rows = []
    for layer in layers:
        ev = [r["norm_ratio"] for r in by[("eval", layer)]]
        ft = [r["norm_ratio"] for r in by[("fit", layer)]]
        rows.append(
            [
                f"L{layer}",
                f"{med(ev):.2f}",
                f"{quantile(ev, 0.10):.2f}-{quantile(ev, 0.90):.2f}",
                f"{med(ft):.2f}",
            ]
        )
    table(["Layer", "eval median", "eval p10-p90", "fit median (null)"], rows, out)

    out += [
        "## Table 2.3 - The four dispositions RQ2 asks about",
        "",
        "Summarised over the whole band, eval prompts only.",
        "",
    ]
    ev_all = [r for r in records if r["group"] == "eval"]
    ft_all = [r for r in records if r["group"] == "fit"]
    cos_ev = [r["cos"] for r in ev_all]
    rows = [
        [
            "Aligned?",
            "median cos(g_x, g_bar)",
            f"{med(cos_ev):.3f}",
            f"fit null {med([r['cos'] for r in ft_all]):.3f}",
        ],
        [
            "Sign-inconsistent?",
            "fraction with cos < 0",
            f"{sum(c < 0 for c in cos_ev) / len(cos_ev):.4f}",
            f"fit null {sum(r['cos'] < 0 for r in ft_all) / len(ft_all):.4f}",
        ],
        [
            "Weakly aligned tail",
            "fraction with cos < 0.3",
            f"{sum(c < 0.3 for c in cos_ev) / len(cos_ev):.3f}",
            f"fit null {sum(r['cos'] < 0.3 for r in ft_all) / len(ft_all):.3f}",
        ],
        [
            "Wrong magnitude?",
            "median |g_x| / |g_bar|",
            f"{med([r['norm_ratio'] for r in ev_all]):.2f}",
            f"fit null {med([r['norm_ratio'] for r in ft_all]):.2f}",
        ],
    ]
    table(["Question", "Statistic", "eval", "reference"], rows, out)

    out += [
        "## Table 2.4 - Does relevance help or hurt?",
        "",
        "The same cosine, split by whether the target token has anything to do "
        "with the prompt. Every target is pulled back through every prompt, so "
        "this is a fully crossed comparison, not a selection.",
        "",
    ]
    rows = []
    for relevance in ("self", "swap_target", "foreign"):
        for band_name, lo, hi in (
            ("early", layers[0], layers[len(layers) // 3]),
            ("mid", layers[len(layers) // 3], layers[2 * len(layers) // 3]),
            ("late", layers[2 * len(layers) // 3], layers[-1]),
        ):
            xs = [
                r["cos"]
                for r in ev_all
                if r["relevance"] == relevance and lo <= r["layer"] <= hi
            ]
            if xs:
                rows.append(
                    [
                        relevance,
                        f"{band_name} (L{lo}-L{hi})",
                        len(xs),
                        f"{med(xs):.3f}",
                        f"{quantile(xs, 0.10):.3f}",
                    ]
                )
    table(["Relevance", "Layer band", "n", "median cos", "p10"], rows, out)

    # ---- rare-prompt domination ----
    vectors_path = results / "rq2" / f"vectors_{config.name}_{args.positions}.pt"
    if vectors_path.exists():
        blob = torch.load(vectors_path, map_location="cpu", weights_only=False)
        out += [
            "## Table 2.5 - Is the spread driven by a few rare prompts?",
            "",
            "Leave-one-out influence: how far the mean of the eval prompts' g_x "
            "moves when one prompt is removed, as a fraction of the mean's own "
            "length. If a handful of prompts each move it a lot, the 'average "
            "direction' is an artifact of those prompts rather than a summary.",
            "",
        ]
        rows = []
        for layer in layers[:: max(1, len(layers) // 6)]:
            stacked = []
            for key, vec in blob["local"].items():
                prompt, lay = key.rsplit("|", 1)
                if int(lay) == layer and not prompt.startswith("wikitext"):
                    stacked.append(vec.mean(dim=0))
            if len(stacked) < 3:
                continue
            influence = leave_one_out_influence(torch.stack(stacked))
            rows.append(
                [
                    f"L{layer}",
                    len(stacked),
                    f"{influence.median():.4f}",
                    f"{influence.max():.4f}",
                    f"{(influence.max() / influence.median()):.1f}x",
                ]
            )
        table(
            ["Layer", "n prompts", "median influence", "max influence", "max / median"],
            rows,
            out,
        )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    results = Path(args.results) if args.results else REPO_ROOT / "results"

    out: list[str] = [
        f"# Preliminary distribution results - {args.config}",
        "",
        "Prompt-local Jacobians and pulled-back directions against the J-Lens "
        "averaged Jacobian. Generated by `scripts/09_analyze.py`.",
        "",
        "All numbers are medians over the stated group unless labelled otherwise. "
        "Medians rather than means throughout: several of these distributions are "
        "skewed and a mean would be moved by a handful of prompts.",
        "",
    ]
    rq1(args, results, out)
    rq2(args, results, out)

    destination = (
        Path(args.out)
        if args.out
        else REPO_ROOT / "results" / f"preliminary_{args.config}_{args.positions}.md"
    )
    destination.write_text("\n".join(out))
    logger.info("wrote %s (%d lines)", destination, len(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

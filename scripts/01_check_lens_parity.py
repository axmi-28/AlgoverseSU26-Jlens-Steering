#!/usr/bin/env python3
"""Sanity check 1: does our J_bar reproduce the lens Neuronpedia serves?

If it does not, we are characterizing a different lens from the one the paper's
causal experiments were run against, and every downstream number is about the
wrong object.

Why this is not a top-1 string comparison
-----------------------------------------
The obvious check -- "our top-1 token equals theirs" -- fails at ~44% of cells
on gemma-2-2b, and it fails for a reason that has nothing to do with the lens.
Neuronpedia's readout applies a *display-layer* vocabulary filter: it drops
``<bos>``, bare punctuation, and tokens already present in the prompt (the
usual trick to stop a lens view from being dominated by "the current token").
The same filter suppresses ~44% of our rank-1 readouts on the plain LOGIT_LENS
too -- and the logit lens involves no Jacobian at all, which is what pins the
cause on the filter rather than on J_bar.

So the check runs on a statistic that is invariant to any filter *they* apply:

    rank of THEIR top-1 token within OUR full-vocabulary ranking

A filter can only remove candidates from their list, so every token they do
show must sit near the top of our ranking if the two lenses agree. Measured on
gemma-2-2b, "The capital of France is the city of":

    rank 0 for 157/225 cells, <= 3 for 219/225, and within our top-64 for
    225/225 -- with LOGIT_LENS behaving the same way.

The LOGIT_LENS control is the discriminating comparison: it shares our
activations and unembedding but uses no Jacobian, so if J_bar were wrong the
JACOBIAN numbers would be markedly worse than the LOGIT ones.

Layers are scored separately inside and outside the workspace band
(``ModelConfig.band``), because measurement says the two regimes behave very
differently. Per-layer medians / p90 on gemma-2-2b:

    L1-L9   JACOBIAN median 0-5, p90 9-38   vs LOGIT p90 1-6   -- divergent
    L10-L24 JACOBIAN median 0,   p90 1-8    vs LOGIT p90 1-5   -- matched

That is the early-layer regime the R-lens work is about: J_bar is far from the
identity there (identity_distance 8.7 at L0 against 0.52 at L26 on qwen3-1.7b),
so the transport is ill-conditioned and small differences in J_bar move the
readout a lot. The paper reports over the band, and so do the pass criteria
here; out-of-band numbers are printed as diagnostics, not gates.

Only models with ``neuronpedia_served: true`` can be checked -- unserved models
return ``{"error": "No server host found"}``. As of 2026-08-29 that means
gemma-2-2b locally (qwen3.6-27b is served but too large to run here), so the
Qwen3 pipeline is validated transitively: same code path, same lens files,
parity established on the one model where both sides are available.

    python scripts/01_check_lens_parity.py --config gemma-2-2b

Note google/gemma-2-2b is a gated HF repo: accept the license and log in with
``huggingface-cli login``.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

import torch

from jsteer.config import REPO_ROOT, load_config
from jsteer.loading import load_lens, load_model
from jsteer.neuronpedia import (
    JACOBIAN_LENS,
    LOGIT_LENS,
    NeuronpediaError,
    fetch_lens_prompt,
)

logger = logging.getLogger("parity")

DEFAULT_PROMPTS = [
    "The capital of France is the city of",
    "Fact: The currency used in the country shaped like a boot is",
    "Most people in Canada speak",
]

#: How deep into our ranking we look before giving up on a cell.
SEARCH_DEPTH = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="gemma-2-2b")
    parser.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument(
        "--max-median-rank",
        type=int,
        default=0,
        help="fail if the median rank of their top-1 in our ranking exceeds this",
    )
    parser.add_argument(
        "--max-p95-rank",
        type=int,
        default=8,
        help="fail if the 95th-percentile rank exceeds this",
    )
    parser.add_argument(
        "--max-missing",
        type=float,
        default=0.01,
        help="fail if more than this fraction of cells miss our top-SEARCH_DEPTH",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def rank_of(tokenizer, our_logits: torch.Tensor, target: str, depth: int) -> int | None:
    """0-based rank of the token string ``target`` in our ranking, or None."""
    indices = torch.topk(our_logits, depth).indices.tolist()
    for rank, token_id in enumerate(indices):
        if tokenizer.decode([token_id]) == target:
            return rank
    return None


def summarize(ranks: list[int], n_missing: int, label: str) -> dict[str, float]:
    total = len(ranks) + n_missing
    stats = {
        "n_cells": total,
        "frac_rank0": sum(r == 0 for r in ranks) / total,
        "frac_top3": sum(r < 3 for r in ranks) / total,
        "median_rank": statistics.median(ranks) if ranks else float("inf"),
        "p95_rank": (
            sorted(ranks)[min(len(ranks) - 1, int(0.95 * len(ranks)))]
            if ranks
            else float("inf")
        ),
        "max_rank": max(ranks) if ranks else float("inf"),
        "frac_missing": n_missing / total,
    }
    logger.info(
        "  %-26s rank0 %.3f | top3 %.3f | median %.0f | p95 %.0f | max %.0f | "
        "missing %.3f  (%d cells)",
        label,
        stats["frac_rank0"],
        stats["frac_top3"],
        stats["median_rank"],
        stats["p95_rank"],
        stats["max_rank"],
        stats["frac_missing"],
        total,
    )
    return stats


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    config = load_config(args.config)

    if not config.neuronpedia_served:
        logger.error(
            "config %s is marked neuronpedia_served: false -- the hosted API has "
            "no lens server for it, so this check cannot run. Try gemma-2-2b.",
            config.name,
        )
        return 2

    model = load_model(config, device=args.device)
    lens = load_lens(config)
    tokenizer = model.tokenizer
    logger.info("%s | %s", config, lens)

    band = config.band
    logger.info("workspace band: L%d-L%d", band.start, band.stop - 1)
    keys = [
        (t, in_band) for t in (JACOBIAN_LENS, LOGIT_LENS) for in_band in (True, False)
    ]
    ranks: dict[tuple[str, bool], list[int]] = {k: [] for k in keys}
    missing: dict[tuple[str, bool], int] = {k: 0 for k in keys}
    failures: list[str] = []

    for prompt in args.prompts:
        logger.info("\nprompt: %r", prompt)
        try:
            remote = fetch_lens_prompt(
                config.np_model_id, prompt, num_completion_tokens=0
            )
        except NeuronpediaError as exc:
            logger.error("  neuronpedia: %s", exc)
            return 2

        local_by_type = {}
        for lens_type, use_jacobian in ((JACOBIAN_LENS, True), (LOGIT_LENS, False)):
            local_by_type[lens_type], _, input_ids = lens.apply(
                model, prompt, max_seq_len=args.max_seq_len, use_jacobian=use_jacobian
            )
        local_ids = input_ids[0].tolist()

        # Tokenization must match before any readout comparison is meaningful.
        if local_ids != remote.prompt_token_ids:
            logger.error(
                "  TOKENIZATION MISMATCH\n    local : %s\n    remote: %s",
                [tokenizer.decode([i]) for i in local_ids],
                remote.prompt_tokens,
            )
            failures.append(f"{prompt!r}: tokenization")
            continue
        logger.info("  %d tokens, tokenization matches", len(local_ids))

        for lens_type in (JACOBIAN_LENS, LOGIT_LENS):
            local_logits = local_by_type[lens_type]
            # The server streams a continuation even when asked for none; only
            # the prompt positions have a local counterpart.
            for position in remote.prompt_positions(lens_type):
                if position >= len(local_ids):
                    continue
                remote_by_layer = remote.top_tokens[lens_type][position]
                for layer in lens.source_layers:
                    if layer >= len(remote_by_layer):
                        continue
                    rank = rank_of(
                        tokenizer,
                        local_logits[layer][position],
                        remote_by_layer[layer][0],
                        SEARCH_DEPTH,
                    )
                    key = (lens_type, layer in band)
                    if rank is None:
                        missing[key] += 1
                    else:
                        ranks[key].append(rank)

    if not ranks[(JACOBIAN_LENS, True)]:
        logger.error("no in-band cells compared")
        return 2

    logger.info("\n=== %s ===", config.name)
    stats = {}
    for lens_type, in_band in keys:
        key = (lens_type, in_band)
        if not ranks[key] and not missing[key]:
            continue
        label = f"{'in ' if in_band else 'out'}-band {lens_type}"
        stats[f"{lens_type}|{'band' if in_band else 'outside'}"] = summarize(
            ranks[key], missing[key], label
        )

    jacobian = stats[f"{JACOBIAN_LENS}|band"]
    control = stats[f"{LOGIT_LENS}|band"]
    if jacobian["median_rank"] > args.max_median_rank:
        failures.append(
            f"median rank {jacobian['median_rank']} > {args.max_median_rank}"
        )
    if jacobian["p95_rank"] > args.max_p95_rank:
        failures.append(f"p95 rank {jacobian['p95_rank']} > {args.max_p95_rank}")
    if jacobian["frac_missing"] > args.max_missing:
        failures.append(
            f"{jacobian['frac_missing']:.3f} of cells miss our top-{SEARCH_DEPTH}"
        )
    # The control: J_bar being wrong would show up as the Jacobian lens
    # tracking the hosted readout markedly worse than the plain logit lens does.
    if jacobian["frac_top3"] < control["frac_top3"] - 0.10:
        failures.append(
            f"in-band JACOBIAN top-3 {jacobian['frac_top3']:.3f} trails LOGIT "
            f"control {control['frac_top3']:.3f} by >0.10"
        )

    out = (
        Path(args.out)
        if args.out
        else REPO_ROOT / "results" / f"parity_{config.name}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=1))
    logger.info("stats -> %s", out)

    if failures:
        logger.error("FAIL: %s", "; ".join(str(f) for f in failures))
        return 1
    logger.info("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Capability demo: g_x = J_x^T u_y at scale, and the cost it saves.

Takes one flexible-generalization trial, pulls back the unembedding rows for a
batch of target tokens through the prompt-local Jacobian, and reports how far
each local direction sits from the averaged one the lens would have used.

This is the shape of the research-question-2 measurement, run on a single
prompt so the pipeline can be eyeballed:

    python scripts/03_pullback_demo.py --config qwen3-1.7b --layer 13

Cost, for the record it prints: a full J_x is ceil(d_model / dim_batch)
backward passes and 4 * d_model^2 bytes per layer; K target tokens through
:func:`pullback_for_prompt` is ceil(K / dim_batch) passes and 4 * K * d_model
bytes.
"""

from __future__ import annotations

import argparse
import logging
import sys

from jsteer.config import load_config
from jsteer.data import flexible_generalization_trials
from jsteer.jacobian import averaged_pullback, pullback_for_prompt
from jsteer.loading import load_lens, load_model, single_token_id, unembedding_rows
from jsteer.metrics import pullback_summary

logger = logging.getLogger("pullback")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="qwen3-1.7b")
    parser.add_argument("--layer", type=int, default=None, help="default: mid-network")
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--dim-batch", type=int, default=None)
    parser.add_argument(
        "--positions",
        default="all",
        choices=["all", "last", "fit"],
        help=(
            "source positions to average J_x over. 'fit' is the lens's own "
            "convention (skip_first=16), which the steering prompts are too "
            "short for; 'all' uses every position, 'last' only the final token."
        ),
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def source_positions_for(mode: str, seq_len: int) -> list[int] | None:
    """The lens's fitting convention, or an explicit set that short prompts allow."""
    if mode == "fit":
        return None
    if mode == "last":
        return [-1]
    return list(range(seq_len))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    config = load_config(args.config)

    model = load_model(config, device=args.device)
    lens = load_lens(config)
    layer = args.layer if args.layer is not None else config.n_layers // 2
    dim_batch = args.dim_batch or config.dim_batch

    trials = flexible_generalization_trials()
    trial = trials[args.trial]
    logger.info(
        "trial %d/%d: %s/%s", args.trial, len(trials), trial.category, trial.func
    )
    logger.info("  prompt      : %r", trial.prompt)
    logger.info("  swap        : %s -> %s", trial.source_arg, trial.target_arg)
    logger.info("  answer      : %s -> %s", trial.source_answer, trial.target_answer)

    # The J-Lens swap acts on the token directions for the two args, so those
    # are the cotangents that matter; the answers come along to watch what the
    # readout does downstream.
    words = [
        trial.source_arg,
        trial.target_arg,
        trial.source_answer,
        trial.target_answer,
    ]
    token_ids, labels = [], []
    for word in words:
        try:
            token_ids.append(single_token_id(model, word))
            labels.append(word)
        except ValueError as exc:
            logger.warning("  skipping %r: %s", word, exc)
    if not token_ids:
        logger.error("no single-token targets in this trial")
        return 2

    cotangents = unembedding_rows(model, token_ids)  # [K, d_model]
    seq_len = model.encode(trial.prompt, max_length=config.fit.max_seq_len).shape[1]
    positions = source_positions_for(args.positions, seq_len)
    logger.info("  tokens      : %d (positions=%s)", seq_len, args.positions)

    result = pullback_for_prompt(
        model,
        trial.prompt,
        [layer],
        cotangents,
        target_layer=config.fit.target_layer,
        source_positions=positions,
        dim_batch=dim_batch,
        max_seq_len=config.fit.max_seq_len,
        skip_first=config.fit.skip_first,
    )

    g_local = result[layer]
    g_avg = averaged_pullback(lens, cotangents, layer)
    summary = pullback_summary(g_local, g_avg)

    logger.info(
        "\nlayer %d, %d cotangents, %d backward pass(es), seq_len=%d n_valid=%d",
        layer,
        len(token_ids),
        result.n_backward_passes,
        result.seq_len,
        result.n_valid_positions,
    )
    logger.info(
        "%-14s %9s %11s %11s %10s", "token", "cos", "|g_x|", "|g_bar|", "rel_err"
    )
    for i, label in enumerate(labels):
        logger.info(
            "%-14s %9.4f %11.3f %11.3f %10.4f",
            label,
            summary["cos"][i].item(),
            summary["norm_local"][i].item(),
            summary["norm_avg"][i].item(),
            summary["rel_error"][i].item(),
        )

    d = config.d_model
    logger.info(
        "\ncost: full J_x = %d passes, %.1f MB/layer | K=%d pullbacks = %d passes, %.2f MB",
        -(-d // dim_batch),
        4 * d * d / 1e6,
        len(token_ids),
        result.n_backward_passes,
        4 * len(token_ids) * d / 1e6,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

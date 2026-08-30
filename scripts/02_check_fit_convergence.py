#!/usr/bin/env python3
"""Sanity check 2: does averaging our own J_x converge to the downloaded J_bar?

This is the check that licenses the whole project. We claim to compute the same
per-prompt object the lens averages over; if our running mean does not walk
toward the published J_bar, then our J_x is not the lens's J_x and the
"variation around the average" we go on to measure is measuring our own bug.

Two things are compared against the published artifacts:

1. The per-prompt convergence trace. Each pre-fitted lens ships a
   ``*_convergence.csv`` with one row per prompt (``identity_distance``,
   ``mean_rel_change``). If our reconstruction of the fitting corpus is right,
   our trace should track it row for row. A mismatch here means we have the
   right estimator on the wrong prompts -- still fine for the science, but the
   convergence claim then has to be made on our own corpus rather than
   inherited.

2. The endpoint. ``rel_frobenius(our_mean, J_bar)`` should fall steadily with
   the number of prompts. It will not reach 0: the published lens averaged
   ~450-480 prompts and is stored in fp16.

    python scripts/02_check_fit_convergence.py --config qwen3-1.7b --n-prompts 32

Cost warning: a full J_x is ceil(d_model / dim_batch) backward passes per
prompt per layer set. Start with --source-layers a few layers and --n-prompts
small; the default fits every layer, which is the expensive path.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
import time
from pathlib import Path

import requests
import torch

from jsteer.config import REPO_ROOT, load_config
from jsteer.corpus import load_fitting_prompts
from jsteer.jacobian import jacobian_for_prompt
from jsteer.loading import load_lens, load_model
from jsteer.metrics import identity_distance, relative_frobenius

logger = logging.getLogger("convergence")

LENS_REPO_URL = "https://huggingface.co/neuronpedia/jacobian-lens/resolve/main"


def fetch_convergence_csv(config) -> list[dict[str, str]] | None:
    """The published per-prompt trace for this lens, if it has one."""
    url = f"{LENS_REPO_URL}/{config.lens_filename.replace('_jacobian_lens.pt', '_convergence.csv')}"
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("no published convergence csv (%s)", exc)
        return None
    return list(csv.DictReader(io.StringIO(response.text)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="qwen3-1.7b")
    parser.add_argument("--n-prompts", type=int, default=32)
    parser.add_argument(
        "--source-layers",
        type=int,
        nargs="*",
        default=None,
        help="default: every fitted layer (expensive). Try a handful first.",
    )
    parser.add_argument("--dim-batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None, help="write the trace as JSON here")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    config = load_config(args.config)

    model = load_model(config, device=args.device)
    lens = load_lens(config)
    source_layers = args.source_layers or lens.source_layers
    dim_batch = args.dim_batch or config.dim_batch
    logger.info("%s | %s", config, lens)
    logger.info(
        "fitting %d prompts over layers %s, dim_batch=%d",
        args.n_prompts,
        source_layers,
        dim_batch,
    )

    published = fetch_convergence_csv(config)
    if published:
        logger.info(
            "published trace: %d rows, final identity_distance=%s",
            len(published),
            published[-1]["identity_distance"],
        )

    cache = REPO_ROOT / "data" / "corpora" / f"{config.name}_fit_prompts.json"
    prompts = load_fitting_prompts(
        config.fit, args.n_prompts, tokenizer=model.tokenizer, cache_path=cache
    )
    logger.info("corpus: %d prompts cached at %s", len(prompts), cache)

    running_sum = {
        layer: torch.zeros(config.d_model, config.d_model, dtype=torch.float32)
        for layer in source_layers
    }
    # The published trace reports identity_distance at the last source layer.
    trace_layer = max(source_layers)
    trace: list[dict[str, float]] = []
    n_done = 0

    for index, prompt in enumerate(prompts):
        start = time.perf_counter()
        result = jacobian_for_prompt(
            model,
            prompt,
            source_layers,
            target_layer=config.fit.target_layer,
            dim_batch=dim_batch,
            max_seq_len=config.fit.max_seq_len,
            skip_first=config.fit.skip_first,
        )
        previous_mean = (
            {l: running_sum[l] / n_done for l in source_layers} if n_done else None
        )
        for layer in source_layers:
            running_sum[layer] += result[layer]
        n_done += 1
        mean = {layer: running_sum[layer] / n_done for layer in source_layers}

        # Same statistic jlens.fit logs, so it is comparable with the csv.
        if previous_mean is not None:
            mean_rel_change = max(
                (
                    (result[l] - previous_mean[l]).norm()
                    / (n_done * previous_mean[l].norm())
                ).item()
                for l in source_layers
            )
        else:
            mean_rel_change = float("nan")

        row = {
            "n_done": n_done,
            "seq_len": result.seq_len,
            "n_valid_positions": result.n_valid_positions,
            "elapsed_s": round(time.perf_counter() - start, 3),
            "identity_distance": identity_distance(mean[trace_layer]),
            "mean_rel_change": mean_rel_change,
            # The endpoint check: distance from the downloaded lens.
            "rel_frobenius_to_published": max(
                relative_frobenius(mean[l], lens.jacobians[l]) for l in source_layers
            ),
        }
        trace.append(row)

        note = ""
        if published and index < len(published):
            ref = published[index]
            note = (
                f"  [published: seq_len={ref['seq_len']} "
                f"n_valid={ref['n_valid_positions']} "
                f"id_dist={float(ref['identity_distance']):.6f}]"
            )
        logger.info(
            "  %3d/%d  seq_len=%d n_valid=%d  %.1fs  id_dist=%.6f  "
            "d_mean=%.2e  rel_F_to_J_bar=%.4f%s",
            n_done,
            len(prompts),
            row["seq_len"],
            row["n_valid_positions"],
            row["elapsed_s"],
            row["identity_distance"],
            row["mean_rel_change"],
            row["rel_frobenius_to_published"],
            note,
        )

    logger.info("\n=== summary ===")
    logger.info(
        "rel_frobenius to published J_bar: %.4f after 1 prompt -> %.4f after %d",
        trace[0]["rel_frobenius_to_published"],
        trace[-1]["rel_frobenius_to_published"],
        n_done,
    )
    monotone = (
        trace[-1]["rel_frobenius_to_published"] < trace[0]["rel_frobenius_to_published"]
    )
    logger.info("converging toward published lens: %s", monotone)
    if published:
        matches = sum(
            1
            for i, row in enumerate(trace)
            if i < len(published)
            and int(published[i]["seq_len"]) == row["seq_len"]
            and int(published[i]["n_valid_positions"]) == row["n_valid_positions"]
        )
        logger.info(
            "corpus reconstruction: %d/%d rows match published (seq_len, n_valid)",
            matches,
            min(len(trace), len(published)),
        )

    out = (
        Path(args.out)
        if args.out
        else REPO_ROOT / "results" / f"convergence_{config.name}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trace, indent=1))
    logger.info("trace -> %s", out)
    return 0 if monotone else 1


if __name__ == "__main__":
    sys.exit(main())

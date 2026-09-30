#!/usr/bin/env python3
"""Research question 1: the distribution of prompt-local Jacobians about J_bar.

Materializes the full ``J_x`` for each prompt and band layer, reduces it to a
:class:`~jsteer.reduce.JacobianDigest` while it is still in memory, and drops
the matrix. Nothing 4096x4096 is ever written to disk -- see ``jsteer.reduce``
for the budget that forces this.

The comparison is matrix against matrix, not direction against direction (that
is research question 2). Three things come out of it:

- **Subspace alignment.** How much of one prompt's top-k singular subspace the
  next prompt's shares, and how much each shares with ``J_bar``. The left basis
  is the output side -- which final-layer directions this Jacobian can write
  into, hence which ``u_y`` have a stable readout at all. The right basis is the
  input side -- where in the residual stream a steering vector has to be written
  to land anywhere. They answer different questions and both are kept.
- **Displacement of the cloud's center.** ``J_bar`` is a mean of per-prompt
  Jacobians over wikitext, so on wikitext the center is ``J_bar`` by
  construction and any measured offset is finite-sample error. That makes the
  wikitext arm a null with a known answer, and makes an offset measured on the
  eval prompts readable as genuine distribution shift.
- **Structure within the eval set.** Whether the spread organizes by category,
  by template, or by argument -- tested with label permutations rather than
  clustering, because the labels are known in advance and the crossed grid only
  has n=16 inside a category.

Cost. One ``J_x`` is ``ceil(d_model / dim_batch)`` backward passes: 32 on
Qwen3-8B at ``dim_batch=128``. Those passes yield every band layer at once, so
the pass count is per *prompt*, not per (prompt, layer). The SVDs are the other
half of the bill -- one 4096x4096 SVD per (prompt, layer), ~4.8 s each on CPU --
and belong on the GPU; ``--svd-device`` controls that.

The work lives in :func:`jsteer.sweeps.run_jacobian_sweep`, which the Modal
entrypoints call too. This script is the local front end for it.

    python scripts/04_jacobian_distribution.py --config qwen3-8b --n-fit 32 --dim-batch 128
    python scripts/04_jacobian_distribution.py --dry-run --config qwen3-8b
"""

from __future__ import annotations

import argparse
import logging
import sys

from jsteer.config import load_config
from jsteer.reduce import digest_bytes
from jsteer.run import POSITION_MODES
from jsteer.sweeps import JacobianSweepSpec, run_jacobian_sweep

logger = logging.getLogger("jacobian-sweep")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="qwen3-8b")
    parser.add_argument("--n-fit", type=int, default=32, help="wikitext controls")
    parser.add_argument("--positions", default="all", choices=POSITION_MODES)
    parser.add_argument("--dim-batch", type=int, default=None)
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument(
        "--layer-chunk",
        type=int,
        default=26,
        help=(
            "band layers per backward sweep. Bounds peak host memory at "
            "chunk * 4 * d_model^2 bytes; a chunk smaller than the band repeats "
            "the forward/backward work once per chunk."
        ),
    )
    parser.add_argument("--k", type=int, default=64, help="singular directions kept")
    parser.add_argument("--svd-device", default=None, help="default: the model's")
    parser.add_argument(
        "--lowrank",
        action="store_true",
        help=(
            "randomized top-k SVD instead of an exact one. ~3 h of exact "
            "4096^2 SVDs on CPU for a full qwen3-8b sweep; check "
            "jsteer.reduce.lowrank_agreement on real J_x first."
        ),
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--corpus-cache", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the cost budget and exit without loading a model",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    spec = JacobianSweepSpec(
        config_name=args.config,
        positions=args.positions,
        layers=args.layers,
        n_fit=args.n_fit,
        dim_batch=args.dim_batch,
        shard=args.shard,
        n_shards=args.n_shards,
        limit=args.limit,
        out_dir=args.out,
        corpus_cache=args.corpus_cache,
        overwrite=args.overwrite,
        device=args.device,
        k=args.k,
        layer_chunk=args.layer_chunk,
        lowrank=args.lowrank,
        svd_device=args.svd_device,
    )
    if args.dry_run:
        return budget(spec)
    run_jacobian_sweep(spec)
    return 0


def budget(spec: JacobianSweepSpec) -> int:
    """Print what a run would cost. No model load, no GPU."""
    from jsteer.data import all_args, flexible_generalization_prompts

    config = load_config(spec.config_name)
    dim_batch = spec.dim_batch or config.dim_batch
    n_layers = len(spec.layers or list(config.band))
    n_prompts = len(flexible_generalization_prompts()) + spec.n_fit
    if spec.limit:
        n_prompts = min(n_prompts, spec.limit)
    n_prompts = len(range(spec.shard, n_prompts, spec.n_shards))
    n_chunks = -(-n_layers // spec.layer_chunk)
    passes = -(-config.d_model // dim_batch)
    digest = digest_bytes(config.d_model, spec.k, len(all_args()))

    logger.info("config           : %s (d_model=%d)", config.name, config.d_model)
    logger.info(
        "this shard       : %d/%d -> %d prompts x %d layers",
        spec.shard,
        spec.n_shards,
        n_prompts,
        n_layers,
    )
    logger.info(
        "backward passes  : %d per prompt-chunk x %d chunks x %d prompts = %d",
        passes,
        n_chunks,
        n_prompts,
        passes * n_chunks * n_prompts,
    )
    logger.info(
        "peak host memory : %.1f GB of J_x per chunk (%d layers x %.0f MB)",
        spec.layer_chunk * 4 * config.d_model**2 / 1e9,
        spec.layer_chunk,
        4 * config.d_model**2 / 1e6,
    )
    logger.info(
        "SVDs             : %d of %dx%d",
        n_prompts * n_layers,
        config.d_model,
        config.d_model,
    )
    if n_chunks > 1:
        logger.info(
            "  note: --layer-chunk %d splits the band into %d chunks, so the "
            "forward/backward work is done %dx. Raise it to %d (needs %.1f GB) "
            "to pay it once.",
            spec.layer_chunk,
            n_chunks,
            n_chunks,
            n_layers,
            n_layers * 4 * config.d_model**2 / 1e9,
        )
    logger.info(
        "digest storage   : %.1f MB each, %.1f GB for this shard",
        digest / 1e6,
        n_prompts * n_layers * digest / 1e9,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

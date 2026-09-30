#!/usr/bin/env python3
"""Research question 2: the distribution of local pulled-back directions.

    g_x = J_x^T u_y      against      g_bar = J_bar^T u_y

For every prompt in the set, every band layer, and every target token, this
pulls the token's unembedding row back through the prompt-local Jacobian and
through the lens's averaged one, and stores both. That is the whole measurement;
the dispositions research question 2 names -- aligned, clustered,
sign-inconsistent, dominated by rare prompts -- are then read off the stored
cloud by ``jsteer.analysis`` without re-running the model.

Why this sweep is cheap and the research-question-1 sweep is not: a pullback
needs one backward pass per *batch of cotangents*, not one per residual
dimension, and a single backward pass yields the pullback at every band layer
at once. Sixteen targets is therefore one backward pass per prompt, and the
output is 16 x d_model numbers rather than a 4096 x 4096 matrix per layer.

Targets are the 16 **argument** words, not the answer words -- see
``jsteer.run.TargetTokens``. Every target is pulled back through every prompt,
including the 15 that are foreign to it, so "is this token relevant to this
prompt?" becomes a column in the results rather than an assumption.

Cheap enough to run at every position convention; since which one is right is
itself an open question, do that rather than choosing.

    python scripts/05_pullback_distribution.py --config qwen3-8b --n-fit 100
"""

from __future__ import annotations

import argparse
import logging
import sys

from jsteer.run import POSITION_MODES
from jsteer.sweeps import PullbackSweepSpec, run_pullback_sweep

logger = logging.getLogger("pullback-sweep")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="qwen3-8b")
    parser.add_argument("--n-fit", type=int, default=100, help="wikitext controls")
    parser.add_argument("--positions", default="all", choices=POSITION_MODES)
    parser.add_argument("--dim-batch", type=int, default=None)
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--corpus-cache", default=None)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    run_pullback_sweep(
        PullbackSweepSpec(
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
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

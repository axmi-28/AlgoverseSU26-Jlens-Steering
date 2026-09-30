#!/usr/bin/env python3
"""Everything checkable before renting a GPU.

The expensive runs are scripts 04 and 06. Both have failure modes that only
show up hours in and are entirely knowable in advance -- a target word that is
two tokens under this tokenizer, an argument span that cannot be located, a
band layer the downloaded lens was never fitted at. This script checks all of
them using the **tokenizer and lens metadata only**: a few MB of downloads, no
model weights, no accelerator.

It prints the run budget last, so the machine can be sized from its output.

    python scripts/00_preflight.py --config qwen3-8b
    python scripts/00_preflight.py --config qwen3-8b --n-fit 32 --with-model
"""

from __future__ import annotations

import argparse
import logging
import sys

from jsteer.config import available_configs, load_config
from jsteer.data import (
    all_answers,
    all_args,
    flexible_generalization_prompts,
    flexible_generalization_trials,
)
from jsteer.reduce import digest_bytes

logger = logging.getLogger("preflight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="qwen3-8b")
    parser.add_argument("--n-fit", type=int, default=32)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--dim-batch", type=int, default=128)
    parser.add_argument("--layer-chunk", type=int, default=26)
    parser.add_argument(
        "--with-model",
        action="store_true",
        help="also load weights and run one steered forward pass end to end",
    )
    return parser.parse_args()


def check(label: str, ok: bool, detail: str = "", *, fatal: bool = True) -> bool:
    """``fatal=False`` reports a condition worth knowing that does not block a run."""
    status = "PASS" if ok else ("FAIL" if fatal else "NOTE")
    logger.info("[%s] %-38s %s", status, label, detail)
    return ok


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = parse_args()
    failures = 0

    logger.info("=== config ===")
    logger.info("available: %s", ", ".join(available_configs()))
    config = load_config(args.config)
    logger.info("%s", config)
    band = list(config.band)
    failures += not check("band is non-empty", bool(band), f"L{band[0]}-L{band[-1]}")

    logger.info("\n=== prompt sets ===")
    prompts = flexible_generalization_prompts()
    trials = flexible_generalization_trials()
    failures += not check("64 base prompts", len(prompts) == 64, str(len(prompts)))
    failures += not check("192 swap trials", len(trials) == 192, str(len(trials)))
    failures += not check("16 argument words", len(all_args()) == 16, str(all_args()))
    logger.info(
        "       %d distinct answer words (grader only, never a cotangent)",
        len(all_answers()),
    )

    logger.info("\n=== tokenizer ===")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.hf_model_id)

    def n_tokens(word: str) -> int:
        return len(tokenizer.encode(f" {word}", add_special_tokens=False))

    bad_args = {w: n_tokens(w) for w in all_args() if n_tokens(w) != 1}
    bad_answers = {w: n_tokens(w) for w in all_answers() if n_tokens(w) != 1}
    # Multi-token args are fatal for a trial: the intervention has no single
    # unembedding row to pull back. Multi-token answers are fatal for grading.
    failures += not check(
        "all 16 args are single tokens", not bad_args, str(bad_args) if bad_args else ""
    )
    check(
        "all answers are single tokens",
        not bad_answers,
        f"{len(bad_answers)} multi-token: {list(bad_answers)[:6]} -> first-token grading",
        fatal=False,
    )
    # Two denominators, because grading is ambiguous for multi-token answers.
    # Default (first-token grading) keeps all trials whose *args* tokenize
    # singly -- that is the denominator upstream's 76/192 must be quoted
    # against, since 192 is their count. --strict-answers uses the smaller one.
    gradeable = [
        t for t in trials if n_tokens(t.source_arg) == 1 and n_tokens(t.target_arg) == 1
    ]
    strict = [
        t
        for t in gradeable
        if n_tokens(t.source_answer) == 1 and n_tokens(t.target_answer) == 1
    ]
    usable = gradeable
    logger.info(
        "       denominator: %d/192 with first-token grading, %d/192 with "
        "--strict-answers",
        len(gradeable),
        len(strict),
    )
    if len(strict) < len(gradeable):
        by_category = {}
        for t in gradeable:
            if t not in strict:
                by_category[t.category] = by_category.get(t.category, 0) + 1
        logger.info(
            "       %d trials need first-token grading, all in: %s",
            len(gradeable) - len(strict),
            by_category,
        )

    logger.info("\n=== lens ===")
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    try:
        meta = get_hf_file_metadata(hf_hub_url(config.lens_repo, config.lens_filename))
        failures += not check(
            "lens file on the hub",
            True,
            f"{meta.size / 1e6:.0f} MB {config.lens_filename}",
        )
    except Exception as exc:  # noqa: BLE001 -- report, don't crash the preflight
        failures += not check("lens file on the hub", False, str(exc)[:80])

    logger.info("\n=== budget for the remote run ===")
    n_prompts_rq1 = 64 + args.n_fit
    n_chunks = -(-len(band) // args.layer_chunk)
    passes = -(-config.d_model // args.dim_batch)
    digest = digest_bytes(config.d_model, args.k, len(all_args()))
    logger.info(
        "04 (RQ1)  %d prompts x %d layers | %d backward passes | %d SVDs of %d^2 | "
        "%.1f GB peak host RAM | %.1f GB digests",
        n_prompts_rq1,
        len(band),
        passes * n_chunks * n_prompts_rq1,
        n_prompts_rq1 * len(band),
        config.d_model,
        args.layer_chunk * 4 * config.d_model**2 / 1e9,
        n_prompts_rq1 * len(band) * digest / 1e9,
    )
    logger.info(
        "05 (RQ2)  %d prompts x %d layers x %d targets | 1 backward pass/prompt | "
        "%.0f MB vectors",
        64 + 100,
        len(band),
        len(all_args()),
        (64 + 100) * len(band) * len(all_args()) * config.d_model * 4 / 1e6,
    )
    n_strengths = 5
    n_arms = 4  # {swap, additive} x {averaged, local}
    logger.info(
        "06 (causal) %d trials x %d strengths x %d arms = %d forward passes "
        "(+1 pullback pass/trial)",
        len(usable),
        n_strengths,
        n_arms,
        len(usable) * n_strengths * n_arms,
    )
    logger.info(
        "\nweights: %s in %s ~ %.0f GB, plus ~%.1f GB activations at seq_len=%d "
        "dim_batch=%d",
        config.hf_model_id,
        config.dtype,
        _param_bytes(config) / 1e9,
        args.dim_batch
        * config.fit.max_seq_len
        * config.d_model
        * 2
        * config.n_layers
        / 1e9,
        config.fit.max_seq_len,
        args.dim_batch,
    )

    if args.with_model:
        failures += _end_to_end(config, band)

    logger.info(
        "\n%s",
        "PREFLIGHT PASS" if not failures else f"PREFLIGHT: {failures} FAILURE(S)",
    )
    return 1 if failures else 0


def _param_bytes(config) -> float:
    """Rough parameter footprint from the model id, for sizing the machine."""
    billions = {"qwen3-1.7b": 1.7, "qwen3-4b": 4.0, "qwen3-8b": 8.2, "gemma-2-2b": 2.6}
    return billions.get(config.name, 8.0) * 1e9 * 2  # bf16


def _end_to_end(config, band: list[int]) -> int:
    """One steered forward pass, to prove the whole chain runs before it runs long."""
    import torch

    from jsteer.jacobian import averaged_pullback, pullback_for_prompt
    from jsteer.loading import load_lens, load_model, single_token_id, unembedding_rows
    from jsteer.run import arg_span, build_prompt_set
    from jsteer.steering import (
        additive_edit,
        grade,
        mean_residual_norms,
        next_token_logits,
    )

    logger.info("\n=== end to end (loads weights) ===")
    model = load_model(config)
    lens = load_lens(config)
    layers = [l for l in band if l in lens.source_layers]

    prompts = build_prompt_set(config, n_fit=0)
    missing = [p.key for p in prompts if arg_span(model, p, max_seq_len=128) is None]
    failed = not check(
        "arg span located in all 64 prompts",
        not missing,
        f"{len(missing)} missing: {missing[:4]}",
    )

    prompt = prompts[0]
    target_id = single_token_id(model, "Canada")
    answer_id = single_token_id(model, "Ottawa")
    cotangents = unembedding_rows(model, [target_id])
    result = pullback_for_prompt(
        model,
        prompt.text,
        layers,
        cotangents,
        source_positions=list(
            range(model.encode(prompt.text, max_length=128).shape[1])
        ),
        dim_batch=config.dim_batch,
        max_seq_len=128,
    )
    norms = mean_residual_norms(model, prompt.text, layers)
    for label, directions in (
        ("averaged", {l: averaged_pullback(lens, cotangents, l)[0] for l in layers}),
        ("local", {l: result[l][0] for l in layers}),
    ):
        for strength in (0.0, 1.0, 2.0):
            edit = additive_edit(directions, norms, strength=strength)
            outcome = grade(
                model,
                next_token_logits(model, prompt.text, edit=edit, max_seq_len=128),
                target_id=answer_id,
            )
            logger.info(
                "  %-8s strength %.1f -> greedy %-12r Ottawa rank %d",
                label,
                strength,
                outcome.greedy_token,
                outcome.target_rank,
            )
    cos = torch.nn.functional.cosine_similarity(
        result[layers[len(layers) // 2]][0],
        averaged_pullback(lens, cotangents, layers[len(layers) // 2])[0],
        dim=0,
    )
    logger.info("  cos(g_x, g_bar) at L%d: %.4f", layers[len(layers) // 2], cos.item())
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())

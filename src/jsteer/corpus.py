"""The fitting distribution.

Sanity check 2 -- "does averaging our own ``J_x`` converge to the downloaded
``J_bar``?" -- only means anything if we average over the same distribution the
downloaded lens was fitted on. From each lens's ``config.yaml`` that is:

    Salesforce/wikitext, wikitext-103-raw-v1, split=train, text_field=text,
    max_chars=2000, max_seq_len=128, dim_batch=128, dtype=bfloat16

with early stopping once the running mean's relative change fell below 0.002
(hence ``prompts_fitted`` ~450-480, not the requested 1000).

One caveat we cannot fully close: Neuronpedia's ``fit_lens.py`` is not public,
so the exact document filter is inferred rather than known. The published
``*_convergence.csv`` for each lens reports ``seq_len=128`` and
``n_valid_positions=111`` on *every* row with ``prompt_idx == n_done - 1``,
i.e. no prompt was ever skipped and every prompt filled the 128-token window.
That pins the filter down to "keep documents long enough to tokenize to >= 128
tokens, in dataset order" -- which is what :func:`fitting_prompts` implements.
``scripts/02_check_fit_convergence.py`` verifies the reconstruction against
that CSV row by row before trusting it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from jsteer.config import FitSpec

logger = logging.getLogger(__name__)

#: Documents shorter than this many characters cannot reach 128 tokens under
#: any tokenizer worth using, so they are dropped without tokenizing. Kept
#: deliberately loose; the real filter is the token-count check below.
_MIN_CHARS = 256


def fitting_prompts(
    spec: FitSpec,
    *,
    tokenizer=None,
    limit: int | None = None,
    min_tokens: int | None = None,
) -> Iterator[str]:
    """Yield the fitting corpus in dataset order, truncated to ``max_chars``.

    Args:
        spec: The :class:`~jsteer.config.FitSpec` transcribed from the lens's
            ``config.yaml``.
        tokenizer: If given, documents that tokenize to fewer than
            ``min_tokens`` tokens are skipped. Required to reproduce the
            published convergence trace, where every prompt is a full window.
        limit: Stop after yielding this many prompts.
        min_tokens: Defaults to ``spec.max_seq_len``.

    Yields:
        Prompt strings.
    """
    from datasets import load_dataset

    min_tokens = spec.max_seq_len if min_tokens is None else min_tokens
    dataset = load_dataset(
        spec.dataset, spec.dataset_config, split=spec.dataset_split, streaming=True
    )

    n_yielded = 0
    for record in dataset:
        text = record[spec.text_field]
        if len(text) < _MIN_CHARS:
            continue
        text = text[: spec.max_chars]
        if tokenizer is not None:
            n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            if n_tokens < min_tokens:
                continue
        yield text
        n_yielded += 1
        if limit is not None and n_yielded >= limit:
            return


def load_fitting_prompts(
    spec: FitSpec, n: int, *, tokenizer=None, cache_path=None
) -> list[str]:
    """Materialize ``n`` fitting prompts, optionally caching them to JSON.

    Caching matters for reproducibility across machines: the streamed dataset
    is stable, but pinning the exact prompt list to a file removes any doubt
    about which prompts a given ``J_bar`` reconstruction averaged over.
    """
    import json
    from pathlib import Path

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            if len(cached) >= n:
                return cached[:n]

    prompts = list(fitting_prompts(spec, tokenizer=tokenizer, limit=n))
    if len(prompts) < n:
        logger.warning("only found %d prompts, asked for %d", len(prompts), n)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(prompts, indent=1))
    return prompts

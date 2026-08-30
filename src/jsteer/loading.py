"""Loading models and pre-fitted lenses.

Thin wrappers over ``jlens`` + ``transformers`` so every script in this repo
gets the model in the same state the lens was fitted in. Nothing here is
clever; it exists so that the device/dtype/BOS decisions are made in one place
and can be pointed at when a number looks wrong.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import torch

from jsteer.config import ModelConfig

logger = logging.getLogger(__name__)

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def resolve_device(spec: str = "auto") -> torch.device:
    """``"auto"`` -> cuda, else mps, else cpu."""
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(spec: str) -> torch.dtype:
    if spec not in _DTYPES:
        raise ValueError(f"unknown dtype {spec!r}; expected one of {sorted(_DTYPES)}")
    return _DTYPES[spec]


def load_model(config: ModelConfig, *, device: str | None = None):
    """Load ``config.hf_model_id`` and wrap it as a ``jlens`` ``LensModel``.

    ``jlens.from_hf`` mutates the model in place (``requires_grad_(False)`` on
    every parameter, ``eval()``, and ``tokenizer.add_bos_token = True``), which
    is what the Jacobian estimator assumes -- so do not reuse the returned
    model for anything that needs parameter gradients.
    """
    import jlens
    import transformers

    dev = resolve_device(device or config.device)
    dtype = resolve_dtype(config.dtype)
    logger.info("loading %s on %s (%s)", config.hf_model_id, dev, config.dtype)

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        config.hf_model_id, dtype=dtype
    ).to(dev)
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.hf_model_id)
    model = jlens.from_hf(hf_model, tokenizer)

    if (model.n_layers, model.d_model) != (config.n_layers, config.d_model):
        raise ValueError(
            f"config {config.name} says n_layers={config.n_layers} "
            f"d_model={config.d_model} but the loaded model reports "
            f"n_layers={model.n_layers} d_model={model.d_model}"
        )
    return model


@lru_cache(maxsize=4)
def _load_lens_cached(repo: str, filename: str):
    import jlens

    return jlens.JacobianLens.from_pretrained(repo, filename=filename)


def load_lens(config: ModelConfig):
    """Download (and cache) the pre-fitted ``J_bar`` for ``config``.

    Returns a ``jlens.JacobianLens``. Its ``n_prompts`` is the *early-stopped*
    count, not 1000 -- see ``config.fit.prompts_fitted``.
    """
    lens = _load_lens_cached(config.lens_repo, config.lens_filename)
    if lens.d_model != config.d_model:
        raise ValueError(
            f"lens {config.lens_filename} has d_model={lens.d_model}, "
            f"config says {config.d_model}"
        )
    expected = config.fit.prompts_fitted
    if expected is not None and lens.n_prompts != expected:
        logger.warning(
            "lens n_prompts=%d but config.fit.prompts_fitted=%d",
            lens.n_prompts,
            expected,
        )
    return lens


def unembedding_rows(model, token_ids: torch.Tensor | list[int]) -> torch.Tensor:
    """The raw unembedding rows ``u_y = W_U[y]``, shape ``[K, d_model]``, fp32.

    This is the ``u_y`` in ``v_y = J^T u_y``. It is the *raw* ``lm_head`` row,
    which is what the paper's steering direction uses ("the unit-normalized
    transpose row for that token"); it deliberately ignores the final norm that
    sits between the residual stream and the logits. ``pullback`` accepts any
    cotangent, so a norm-linearized variant can be swapped in later without
    touching the Jacobian code.
    """
    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.tensor(token_ids, dtype=torch.long)
    weight = unembedding_matrix(model)
    return weight[token_ids.to(weight.device)].detach().float().cpu()


def unembedding_matrix(model) -> torch.Tensor:
    """``W_U``, shape ``[vocab, d_model]``.

    ``jlens.hf.HFLensModel`` keeps the head on a private attribute; anything
    else implementing ``LensModel`` (the tiny test decoder, say) is expected to
    expose a plain ``lm_head``. Note that Qwen3-1.7B and Qwen3-4B tie the
    unembedding to the input embedding, so these rows are the embedding rows.
    """
    for attribute in ("_lm_head", "lm_head"):
        head = getattr(model, attribute, None)
        if head is not None:
            return head.weight
    raise AttributeError(f"{type(model).__name__} exposes no lm_head")


def single_token_id(model, text: str, *, with_space: bool = True) -> int:
    """Token id for ``text`` as a single token, raising if it is not one.

    Most target words in the paper's prompt sets appear mid-sentence, so the
    leading-space form is the one that matters; ``with_space=False`` checks the
    bare form instead.
    """
    candidate = f" {text}" if with_space else text
    ids = model.tokenizer.encode(candidate, add_special_tokens=False)
    if len(ids) != 1:
        pieces = [model.tokenizer.decode([i]) for i in ids]
        raise ValueError(f"{candidate!r} is {len(ids)} tokens, not 1: {pieces}")
    return ids[0]

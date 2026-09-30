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


def load_model(
    config: ModelConfig, *, device: str | None = None, dtype: str | None = None
):
    """Load ``config.hf_model_id`` and wrap it as a ``jlens`` ``LensModel``.

    ``jlens.from_hf`` mutates the model in place (``requires_grad_(False)`` on
    every parameter, ``eval()``, and ``tokenizer.add_bos_token = True``), which
    is what the Jacobian estimator assumes -- so do not reuse the returned
    model for anything that needs parameter gradients.

    ``dtype`` overrides ``config.dtype`` for the callers that need more
    precision than the config's default. The configs run bfloat16, which keeps
    ~8 mantissa bits: fine for grading an argmax, useless for measuring a
    response to a perturbation two orders of magnitude below the activation
    scale, because the change is quantized away before it can be read. Any
    sweep that subtracts two forward passes should ask for float32 and pay the
    memory. Whatever is chosen must reach the output path -- it changes the
    numbers.
    """
    import jlens
    import transformers

    dev = resolve_device(device or config.device)
    dtype_name = dtype or config.dtype
    dtype = resolve_dtype(dtype_name)
    logger.info("loading %s on %s (%s)", config.hf_model_id, dev, dtype_name)

    hf_model = load_hf_model(config, dtype=dtype).to(dev)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.hf_model_id, revision=config.model_revision
    )
    model = jlens.from_hf(hf_model, tokenizer)

    if (model.n_layers, model.d_model) != (config.n_layers, config.d_model):
        raise ValueError(
            f"config {config.name} says n_layers={config.n_layers} "
            f"d_model={config.d_model} but the loaded model reports "
            f"n_layers={model.n_layers} d_model={model.d_model}"
        )
    return model


def load_hf_model(config: ModelConfig, *, dtype=None, **kwargs):
    """``from_pretrained`` via ``config.hf_auto_class``, with a loud guard.

    The guard exists because the interesting failure mode here is not an
    exception. ``transformers`` will happily instantiate an architecture whose
    parameter names do not match a single key in the checkpoint, warn on
    stderr, and hand back a randomly initialised model. On a 27B multimodal
    checkpoint that is a $50 sweep producing noise. So: ask for the loading
    info, and refuse to continue if any *decoder* parameter was left
    newly-initialised.

    Vision-tower and multi-token-prediction keys are exempt on both sides --
    we never read them, and a text-only class legitimately ignores them.
    """
    import transformers

    auto = getattr(transformers, config.hf_auto_class, None)
    if auto is None:
        raise ValueError(
            f"transformers has no {config.hf_auto_class!r} "
            f"(config {config.name}); check hf_auto_class in its YAML"
        )
    if dtype is None:
        dtype = resolve_dtype(config.dtype)

    hf_model, info = auto.from_pretrained(
        config.hf_model_id,
        dtype=dtype,
        output_loading_info=True,
        revision=config.model_revision,
        **kwargs,
    )

    def _decoder_keys(names) -> list[str]:
        skip = ("visual", "vision_tower", "vision_model", "mtp", "rotary_emb")
        return [n for n in names if not any(s in n for s in skip)]

    missing = _decoder_keys(info.get("missing_keys", ()))
    if missing:
        raise RuntimeError(
            f"{config.hf_auto_class} for {config.hf_model_id} left "
            f"{len(missing)} decoder parameters uninitialised, e.g. "
            f"{missing[:3]} -- the checkpoint's key layout does not match this "
            f"architecture. Set hf_auto_class in configs/{config.name}.yaml to "
            f"the class matching architectures[] in the model's config.json."
        )
    unexpected = _decoder_keys(info.get("unexpected_keys", ()))
    if unexpected:
        logger.warning(
            "%d unexpected checkpoint keys ignored, e.g. %s",
            len(unexpected),
            unexpected[:3],
        )
    return hf_model


@lru_cache(maxsize=4)
def _load_lens_cached(repo: str, filename: str, revision: str | None = None):
    import jlens

    if revision is None:
        return jlens.JacobianLens.from_pretrained(repo, filename=filename)
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, filename=filename, revision=revision)
    return jlens.JacobianLens.load(path)


def load_lens(config: ModelConfig):
    """Download (and cache) the pre-fitted ``J_bar`` for ``config``.

    Returns a ``jlens.JacobianLens``. Its ``n_prompts`` is the *early-stopped*
    count, not 1000 -- see ``config.fit.prompts_fitted``.
    """
    lens = _load_lens_cached(
        config.lens_repo, config.lens_filename, config.lens_revision
    )
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


def first_token_id(model, text: str, *, with_space: bool = True) -> int:
    """Token id of the *first* token of ``text``.

    Upstream grades by "the greedy next token matches ``funcs[*].answers[arg]``",
    which is only literally satisfiable when the answer is one token. Under
    Qwen3 four flexible-generalization answers are not (``savanna``,
    ``arachnid``, ``convocation``, ``shiver``), and requiring single tokens
    would silently shrink the ``animals`` category from 48 trials to 26 and so
    move the denominator the published 76/192 is quoted against.

    Reading "matches" as "is the first token of" keeps all 192 and is the only
    reading under which the published count is reproducible. The cost is a
    small false-positive risk -- a greedy token that begins the answer but
    continues into a different word -- so :func:`single_token_id` stays
    available for a strict re-scoring of the subset where both apply.
    """
    candidate = f" {text}" if with_space else text
    ids = model.tokenizer.encode(candidate, add_special_tokens=False)
    if not ids:
        raise ValueError(f"{candidate!r} encodes to nothing")
    return ids[0]


#: Number words the flexible-generalization answer keys use, for digit-form
#: grading. Values are the integer the *word* names, not the arithmetic result.
_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}


def answer_variant_ids(model, text: str) -> list[int]:
    """First-token ids for every surface form of an answer word.

    Upstream grades "the greedy next token matches ``funcs[*].answers[arg]``".
    Comparing token ids exactly makes that stricter than the words: the answer
    key writes ``Euro``, ``Dollar``, ``Yuan``, but a model completing
    ``"...the currency now used in France is the"`` mid-sentence emits
    ``" euro"``. Measured on Qwen3-8B, exact-id grading scored the whole
    ``countries/currency`` category 0/4 while the model was right every time --
    ``" euro"`` and ``" Euro"`` were its top two tokens.

    So a hit is the greedy token being any of {leading-space, bare} x
    {as written, lowercase, capitalized, uppercase}. This is more lenient than
    a literal reading of upstream's sentence and is recorded as such; it is
    also the only reading under which "matches the answer" means what the
    words say rather than what the tokenizer happens to do with capitalization.
    """
    forms = [text, text.lower(), text.capitalize(), text.upper()]
    # Digits as well as words. Qwen3-8B answers "Two times five equals" with a
    # digit, so word-only grading scored the whole numbers category 1/16
    # *unsteered* -- the same failure mode as countries/currency (" euro" vs
    # "Euro"), one class further out. Upstream grades words only, and does the
    # digit-or-word thing in dual-task but not here.
    #
    # Note the square template stays broken regardless: its key holds
    # "twenty"/"forty"/"eighty" as first-token truncations of 25/49/81, so the
    # word maps to 20/40/80 and neither form is the true answer. Report square
    # separately rather than trusting it.
    # Digits, but ONLY for single-digit answers. First-token grading cannot do
    # better: Qwen splits "10" into ['1', '0'], so "ten", "fourteen" and
    # "eighteen" would all accept '1' and become mutually indistinguishable.
    # A two-digit answer is simply not gradeable from one token, and pretending
    # otherwise manufactures hits.
    digit = _NUMBER_WORDS.get(text.lower())
    if digit is not None and 0 <= digit <= 9:
        forms.append(str(digit))

    ids: list[int] = []
    for form in dict.fromkeys(forms):
        for candidate in (f" {form}", form):
            encoded = model.tokenizer.encode(candidate, add_special_tokens=False)
            if not encoded:
                continue
            first = encoded[0]
            # Never accept a whitespace-only token. " 10" tokenizes to
            # [' ', '1'], so a naive first-token rule would add the bare space
            # -- which the model emits before *any* number -- and every trial
            # with a leading space would score as a hit whatever followed.
            if not model.tokenizer.decode([first]).strip():
                continue
            if first not in ids:
                ids.append(first)
    if not ids:
        raise ValueError(f"{text!r} encodes to nothing")
    return ids

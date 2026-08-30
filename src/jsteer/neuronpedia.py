"""Client for Neuronpedia's hosted J-lens.

Used for one thing: sanity check 1. If our locally loaded ``J_bar`` does not
reproduce the readout that neuronpedia.org serves, then we are characterizing
a different lens from the one the causal experiments were run against, and
every downstream number is about the wrong object.

``POST /api/lens/prompt`` streams newline-delimited JSON:

    {"kind": "meta",   "model": ..., "types": [...], "layers_by_type": {...},
     "top_n": 8, "prompt_len": N, "prepend_bos": true, ...}
    {"kind": "prompt", "tokens": [{"position": 0, "token": "<bos>", "id": 2, ...}]}
    {"kind": "token",  "position": p, "token": ..., "id": ..., "is_generated": bool,
     "results": [{"type": "JACOBIAN_LENS", "top_tokens": [[...8 strings...] per layer]}]}
    {"kind": "done"}

Only a few models are actually served -- unserved ones return
``{"error": "No server host found"}``. As of 2026-08-29 ``gemma-2-2b`` and
``qwen3.6-27b`` are up while ``qwen3-1.7b`` / ``qwen3-4b`` / ``qwen3-8b`` are
not, which is why the parity check runs on gemma-2-2b. Check with
:func:`is_served` before assuming.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://www.neuronpedia.org"
LENS_PROMPT_ENDPOINT = f"{BASE_URL}/api/lens/prompt"

JACOBIAN_LENS = "JACOBIAN_LENS"
LOGIT_LENS = "LOGIT_LENS"


class NeuronpediaError(RuntimeError):
    """The API returned an error payload (e.g. the model is not served)."""


@dataclass
class LensResponse:
    """A parsed ``/api/lens/prompt`` stream."""

    meta: dict[str, Any]
    tokens: list[dict[str, Any]] = field(default_factory=list)
    #: ``{lens_type: {position: [ [top-n token strings] per layer ]}}``
    top_tokens: dict[str, dict[int, list[list[str]]]] = field(default_factory=dict)
    #: Positions carrying a *generated* token. The server does not always
    #: honour ``num_completion_tokens: 0`` and will happily stream a full
    #: continuation, so readouts have to be filtered back to the prompt.
    generated_positions: set[int] = field(default_factory=set)

    @property
    def prompt_tokens(self) -> list[str]:
        return [t["token"] for t in self.tokens if not t.get("is_generated")]

    @property
    def prompt_token_ids(self) -> list[int]:
        return [t["id"] for t in self.tokens if not t.get("is_generated")]

    def prompt_positions(self, lens_type: str = JACOBIAN_LENS) -> list[int]:
        """Positions of the prompt itself, in order, excluding the continuation."""
        return sorted(
            p
            for p in self.top_tokens.get(lens_type, {})
            if p not in self.generated_positions
        )

    def jacobian_top_tokens(self, position: int, layer: int) -> list[str]:
        """Top-n J-lens tokens at ``(position, layer)``, best first."""
        return self.top_tokens[JACOBIAN_LENS][position][layer]


def fetch_lens_prompt(
    np_model_id: str,
    prompt: str,
    *,
    num_completion_tokens: int = 0,
    timeout: float = 300.0,
    session: requests.Session | None = None,
) -> LensResponse:
    """Run the hosted lens on ``prompt`` and parse the stream.

    Args:
        np_model_id: Neuronpedia model id (``config.np_model_id``).
        prompt: Input text. The server prepends BOS itself (``prepend_bos``
            in the meta record) -- do not add one.
        num_completion_tokens: Generated tokens to also read out. 0 keeps the
            response to the prompt itself, which is all the parity check needs
            and is much faster.
        timeout: Seconds.
        session: Optional requests session for connection reuse.

    Raises:
        NeuronpediaError: If the model is not served or the stream carries an
            error record.
    """
    post = (session or requests).post
    response = post(
        LENS_PROMPT_ENDPOINT,
        json={
            "modelId": np_model_id,
            "prompt": prompt,
            "num_completion_tokens": num_completion_tokens,
        },
        stream=True,
        timeout=timeout,
    )
    response.raise_for_status()

    parsed = LensResponse(meta={})
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        record = json.loads(raw_line)
        if "error" in record:
            raise NeuronpediaError(
                f"{np_model_id}: {record['error']} "
                "(the model may not have a live lens server)"
            )
        kind = record.get("kind")
        if kind == "meta":
            parsed.meta = record
        elif kind == "prompt":
            parsed.tokens = record["tokens"]
        elif kind == "token":
            position = record["position"]
            if record.get("is_generated"):
                parsed.generated_positions.add(position)
            for result in record.get("results", []):
                bucket = parsed.top_tokens.setdefault(result["type"], {})
                bucket[position] = result["top_tokens"]
        elif kind == "done":
            break
    if not parsed.meta:
        raise NeuronpediaError(f"{np_model_id}: no meta record in response")
    return parsed


def is_served(np_model_id: str, *, timeout: float = 60.0) -> bool:
    """Whether Neuronpedia currently has a live lens server for this model."""
    try:
        fetch_lens_prompt(
            np_model_id,
            "The capital of France is the city of",
            num_completion_tokens=0,
            timeout=timeout,
        )
    except NeuronpediaError:
        return False
    except requests.RequestException as exc:
        logger.warning("network error probing %s: %s", np_model_id, exc)
        return False
    return True

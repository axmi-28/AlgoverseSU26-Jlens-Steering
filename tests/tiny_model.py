"""A tiny fp32 CPU decoder implementing ``jlens.protocol.LensModel``.

``jlens`` ships an equivalent at ``tests/tiny.py``, but its ``pyproject.toml``
packages only the ``jlens`` package, so a git install does not provide it.
Ours is fp32 throughout, which matters: the pullback identity
``J_x^T c == pullback(c)`` is exact in exact arithmetic, so testing it in fp32
turns a "roughly agrees" assertion into a tight one.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn


class _ResidualBlock(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.linear.weight.mul_(0.1)  # keeps J well-conditioned

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + torch.tanh(self.linear(hidden))


class _ByteTokenizer:
    bos_token_id = 0

    def __call__(self, text, *, return_tensors="pt", truncation=True, max_length=128):
        ids = [self.bos_token_id] + [1 + (b % 30) for b in text.encode()][
            : max_length - 1
        ]
        return SimpleNamespace(input_ids=torch.tensor([ids]))

    def encode(self, text, add_special_tokens=True):
        ids = [1 + (b % 30) for b in text.encode()]
        return ([self.bos_token_id] if add_special_tokens else []) + ids

    def decode(self, ids, **_kw) -> str:
        return "".join(chr(96 + int(i)) for i in ids)


class TinyDecoder(nn.Module):
    """A ``LensModel`` small enough to materialize the full ``J_x``."""

    def __init__(
        self, n_layers: int = 4, d_model: int = 8, vocab_size: int = 32, seed: int = 0
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.n_layers = n_layers
        self.d_model = d_model
        self.tokenizer = _ByteTokenizer()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([_ResidualBlock(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @property
    def input_device(self) -> torch.device:
        return self.embed_tokens.weight.device

    def encode(self, text: str, *, max_length: int = 128) -> torch.Tensor:
        return self.tokenizer(text, max_length=max_length).input_ids.to(
            self.input_device
        )

    def forward(self, input_ids: torch.Tensor):
        hidden = self.embed_tokens(input_ids)
        for block in self.layers:
            hidden = block(hidden)
        return hidden

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(residual))

"""Model / lens configuration.

Everything model-specific lives in a YAML file under ``configs/``. The one
non-obvious field is ``lens_filename``: the pre-fitted lenses on
``neuronpedia/jacobian-lens`` are laid out as

    {np_model_id}/jlens/{dataset_dir}/{stem}_jacobian_lens.pt

where ``stem`` follows the *HuggingFace* model id, not the directory name
(``qwen3-8b/.../Qwen3-8B_jacobian_lens.pt`` but
``gemma-2-2b/.../gemma-2-2b_jacobian_lens.pt``). There is no rule that derives
one from the other, so it is written down per model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

#: The HF repo hosting the pre-fitted lenses.
LENS_REPO = "neuronpedia/jacobian-lens"


@dataclass(frozen=True)
class FitSpec:
    """How the *downloaded* lens was fitted, transcribed from its config.yaml.

    These are not our choices -- they are the conditions we have to match if a
    per-prompt Jacobian is to be comparable with the downloaded average.
    ``prompts_fitted`` is the early-stopped count, which is well short of
    ``n_prompts``; see the note in the README.
    """

    dataset: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    dataset_split: str = "train"
    text_field: str = "text"
    max_chars: int = 2000
    max_seq_len: int = 128
    skip_first: int = 16
    target_layer: int | None = None
    dtype: str = "bfloat16"
    n_prompts_requested: int = 1000
    prompts_fitted: int | None = None
    final_identity_distance: float | None = None


@dataclass(frozen=True)
class ModelConfig:
    """A model plus the lens fitted on it."""

    name: str
    hf_model_id: str
    np_model_id: str
    n_layers: int
    d_model: int
    lens_filename: str
    #: Whether neuronpedia.org currently serves a live lens for this model.
    #: Only served models can be used for the API parity check.
    neuronpedia_served: bool = False
    dtype: str = "bfloat16"
    device: str = "auto"
    #: Output dims (or cotangents) per backward pass. Trades memory for passes.
    dim_batch: int = 8
    #: Inclusive mid-network layer range the paper reports over ("workspace
    #: band"). Outside it -- especially below it -- the J-lens readout is known
    #: to be unreliable: J_bar is far from the identity in early layers
    #: (identity_distance 8.7 at L0 vs 0.52 at L26 on qwen3-1.7b), the readout
    #: degenerates into "trash tokens", and R-lens (Sec. "Relevant Past Papers")
    #: argues much of that is a measurement artifact. Our own parity against
    #: the hosted lens reproduces this split: essentially exact at L10+,
    #: divergent at L1-L9. ``None`` falls back to the middle half.
    band_start: int | None = None
    band_end: int | None = None
    fit: FitSpec = field(default_factory=FitSpec)

    @property
    def lens_repo(self) -> str:
        return LENS_REPO

    @property
    def band(self) -> range:
        """The workspace band as a ``range`` over layer indices."""
        start = self.n_layers // 4 if self.band_start is None else self.band_start
        end = self.n_layers - 2 if self.band_end is None else self.band_end
        return range(start, end + 1)

    def __repr__(self) -> str:
        return (
            f"ModelConfig({self.name}: {self.hf_model_id}, "
            f"n_layers={self.n_layers}, d_model={self.d_model})"
        )


def load_config(name_or_path: str | os.PathLike[str]) -> ModelConfig:
    """Load a :class:`ModelConfig` by config name (``"qwen3-1.7b"``) or path."""
    path = Path(name_or_path)
    if not path.exists():
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"no config {name_or_path!r}; available: {available}")
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    fit = FitSpec(**raw.pop("fit", {}))
    return ModelConfig(**raw, fit=fit)


def available_configs() -> list[str]:
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))

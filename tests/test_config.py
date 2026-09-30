"""Configs must describe the models and lens files that actually exist.

The lens filenames are the trap: the stem on ``neuronpedia/jacobian-lens``
follows the HF model id for Qwen (``Qwen3-8B_jacobian_lens.pt``) but the
directory name for Gemma (``gemma-2-2b_jacobian_lens.pt``), so a plausible-
looking guess silently 404s.
"""

from __future__ import annotations

import re

import pytest
import requests
import transformers

from jsteer.config import LENS_REPO, available_configs, load_config


def test_configs_load() -> None:
    names = available_configs()
    assert {"qwen3-1.7b", "qwen3-4b", "qwen3-8b", "gemma-2-2b"} <= set(names)
    for name in names:
        config = load_config(name)
        assert config.name == name
        assert config.lens_filename.startswith(f"{config.np_model_id}/")
        # Some lenses carry a fit-size suffix (..._jacobian_lens_n1000.pt).
        assert re.fullmatch(r".*_jacobian_lens(_n\d+)?\.pt", config.lens_filename), (
            config.lens_filename
        )
        # A wrong Auto* class does not raise -- it random-inits the decoder.
        assert hasattr(transformers, config.hf_auto_class)


def test_unknown_config_lists_alternatives() -> None:
    with pytest.raises(FileNotFoundError, match="qwen3-1.7b"):
        load_config("no-such-model")


@pytest.mark.network
@pytest.mark.parametrize("name", available_configs())
def test_lens_file_exists_on_hub(name: str) -> None:
    config = load_config(name)
    url = f"https://huggingface.co/{LENS_REPO}/resolve/main/{config.lens_filename}"
    response = requests.head(url, allow_redirects=True, timeout=60)
    assert response.status_code == 200, (
        f"{config.lens_filename} -> {response.status_code}"
    )

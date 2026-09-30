"""The load guard.

Qwen3.6-27B is a ``*ForConditionalGeneration`` checkpoint whose decoder weights
are named ``model.language_model.*``. ``AutoModelForCausalLM`` maps it to a
class expecting ``model.layers.*`` and defines no conversion mapping, so it
matches nothing and returns a *randomly initialised* 27B model. That does not
raise -- it runs, emits logits, and would have produced an entire sweep of
noise. These tests pin the guard that turns it into an exception.
"""

from __future__ import annotations

import dataclasses
import importlib

import pytest

from jsteer.config import load_config
from jsteer.loading import load_hf_model


class _FakeAuto:
    """Stands in for a ``transformers.Auto*`` class."""

    def __init__(self, missing: list[str], unexpected: list[str] | None = None):
        self.missing = missing
        self.unexpected = unexpected or []

    def from_pretrained(self, model_id, **kwargs):
        assert kwargs["output_loading_info"] is True
        return object(), {
            "missing_keys": self.missing,
            "unexpected_keys": self.unexpected,
        }


@pytest.fixture
def config():
    return load_config("qwen3-8b")


def test_missing_decoder_keys_raise(monkeypatch, config) -> None:
    fake = _FakeAuto(["model.layers.0.mlp.gate_proj.weight"] * 3)
    monkeypatch.setattr(
        importlib.import_module("transformers"), "_FakeAuto", fake, raising=False
    )
    config = dataclasses.replace(config, hf_auto_class="_FakeAuto")
    with pytest.raises(RuntimeError, match="uninitialised"):
        load_hf_model(config)


def test_vision_and_mtp_keys_are_exempt(monkeypatch, config) -> None:
    """A text-only load legitimately ignores the vision tower and MTP head."""
    fake = _FakeAuto(
        missing=["model.visual.blocks.0.attn.qkv.weight", "mtp.fc.weight"],
        unexpected=["model.visual.merger.mlp.0.weight"],
    )
    monkeypatch.setattr(
        importlib.import_module("transformers"), "_FakeAuto", fake, raising=False
    )
    config = dataclasses.replace(config, hf_auto_class="_FakeAuto")
    assert load_hf_model(config) is not None


def test_unknown_auto_class_names_the_config(config) -> None:
    config = dataclasses.replace(config, hf_auto_class="AutoModelForNothing")
    with pytest.raises(ValueError, match="hf_auto_class"):
        load_hf_model(config)


def test_27b_config_uses_the_multimodal_class() -> None:
    """Pinned because reverting this field is silent, not loud."""
    config = load_config("qwen3.6-27b")
    assert config.hf_auto_class == "AutoModelForImageTextToText"
    assert (config.band[0], config.band[-1]) == (24, 58)


def test_model_load_uses_configured_revision(monkeypatch, config):
    observed = {}

    class FakeAuto:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            observed.update(kwargs)
            return object(), {"missing_keys": [], "unexpected_keys": []}

    monkeypatch.setattr(
        importlib.import_module("transformers"),
        "_RevisionFakeAuto",
        FakeAuto,
        raising=False,
    )
    load_hf_model(dataclasses.replace(config, hf_auto_class="_RevisionFakeAuto"))
    assert observed["revision"] == config.model_revision
    assert len(config.model_revision) == 40


def test_lens_cache_distinguishes_revisions(monkeypatch):
    import huggingface_hub
    import jlens

    from jsteer.loading import _load_lens_cached

    seen = []

    def download(repo, *, filename, revision):
        seen.append(revision)
        return revision

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(jlens.JacobianLens, "load", lambda path: path)
    _load_lens_cached.cache_clear()
    try:
        assert _load_lens_cached("repo", "lens.pt", "one") == "one"
        assert _load_lens_cached("repo", "lens.pt", "two") == "two"
        assert _load_lens_cached("repo", "lens.pt", "one") == "one"
        assert seen == ["one", "two"]
    finally:
        _load_lens_cached.cache_clear()

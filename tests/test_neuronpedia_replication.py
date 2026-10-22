from pathlib import Path

import pytest
import torch

from jsteer import neuronpedia_replication as runner
from jsteer.cross_model import sha
from jsteer.write_controls import file_hash


def test_published_baseline_is_unchanged_and_filter_uses_its_diagonal_estimate():
    full = {2: torch.tensor([[2., 3.], [4., 5.]])}
    before = full[2].clone()
    diagonal = {2: torch.tensor([[.5, 1.], [1., 1.5]])}
    half = {2: diagonal[2]*1.1}
    out = runner.derived_digests(full, diagonal, half)
    assert torch.equal(full[2], before)
    assert torch.equal(out["published_full|2"]["pullbacks"], before)
    torch.testing.assert_close(out["published_off|2"]["pullbacks"], before-diagonal[2])
    torch.testing.assert_close(out["published_off32|2"]["pullbacks"], before-half[2])
    # Re-admitting the complete estimated diagonal returns the published map.
    torch.testing.assert_close(out["published_off|2"]["pullbacks"]+out["published_diag|2"]["pullbacks"], before)


def test_invalid_estimate_cannot_silently_create_a_filtered_lens():
    with pytest.raises(ValueError, match="shapes differ"):
        runner.derived_digests({1: torch.ones(2, 3)}, {1: torch.ones(2, 2)}, {1: torch.ones(2, 3)})
    with pytest.raises(ValueError, match="Nonfinite"):
        runner.derived_digests({1: torch.ones(2, 3)}, {1: torch.full((2, 3), float("nan"))}, {1: torch.ones(2, 3)})


def test_missing_published_lens_is_rejected_before_model_load(monkeypatch):
    name = "mistral-7b-instruct-v0.3"  # Deliberately a prior self-fitted configuration.
    spec = runner.NeuronpediaSpec(config_name=name,
        expected_source_sha=file_hash(Path(runner.__file__)),
        expected_inputs_sha=sha(runner.input_hashes(name)))
    monkeypatch.setattr(runner, "load_model", lambda _: pytest.fail("Model loaded without a matching released lens"))
    with pytest.raises(ValueError, match="repository/revision differs"):
        runner.run_neuronpedia(spec)


def test_gemma2_wrapper_pullback_and_cached_intervention_agree():
    """Exercise Gemma 2's softcapped logits and sliding-attention cache."""
    from jlens.hf import HFLensModel
    from jlens.hooks import ActivationRecorder
    from transformers import Gemma2Config, Gemma2ForCausalLM

    from jsteer.concept_use import evaluate
    from jsteer.jacobian import component_pullbacks_for_prompt, pullback_for_prompt
    from jsteer.steering import ResidualEdit, greedy_continuation
    from tests.tiny_model import TinyDecoder

    torch.manual_seed(42)
    hf = Gemma2ForCausalLM(Gemma2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=128, sliding_window=64,
    )).float().eval()
    model = HFLensModel(hf, TinyDecoder().tokenizer)
    text = "abcdefghijklmnopqrstuvwxyz012345"
    ids = model.encode(text)
    with torch.no_grad(), ActivationRecorder(model.layers, at=[3]) as rec:
        model.forward(ids)
        wrapped = model.unembed(rec.activations[3][:, -1])
        native = hf(input_ids=ids, use_cache=False).logits[:, -1]
        torch.testing.assert_close(wrapped, native)
    cot = torch.randn(2, 32)
    parts = component_pullbacks_for_prompt(model, text, [1, 2], cot, dim_batch=2,
                                          max_seq_len=32, skip_first=16)
    direct = pullback_for_prompt(model, text, [1, 2], cot, dim_batch=2,
                                 max_seq_len=32, skip_first=16)
    for layer in (1, 2):
        torch.testing.assert_close(parts[layer]["total"], direct[layer], atol=1e-5, rtol=1e-4)
    delta = torch.randn(2, 5, 32)*.1
    cached = evaluate(model, ids, {1: delta}, gen_tokens=4)["texts"]
    for b in range(2):
        edit = ResidualEdit({1: delta[b]}, positions=list(range(5)))
        assert cached[b] == greedy_continuation(model, text, edit=edit, n_tokens=4)

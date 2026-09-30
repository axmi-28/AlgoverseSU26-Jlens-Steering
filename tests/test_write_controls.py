"""Scientific invariants for C22, rather than expected steering outcomes."""

import pytest
import torch

from jsteer.steering import swap_edit
from jsteer.write_controls import (
    ARMS,
    layer_controls,
    match_row_norms,
    random_projection_axis,
    remove_projection,
    swap_delta,
)
from tests.tiny_model import TinyDecoder


@pytest.fixture
def sample():
    g = torch.Generator().manual_seed(123)
    return [
        torch.randn(shape, generator=g)
        for shape in [(7, 16), (16, 2), (16, 2), (16, 2)]
    ]


def test_matching_preserves_every_position_budget(sample):
    edits = layer_controls(*sample)
    assert set(edits) == set(ARMS)
    for arm in (
        "swap_off_matched",
        "swap_projected_matched",
        "swap_random_projection_matched",
        "add_full",
        "add_off",
    ):
        torch.testing.assert_close(
            edits[arm].norm(dim=-1), edits["swap_full"].norm(dim=-1)
        )


def test_frozen_full_exactly_recovers_full_swap(sample):
    edits = layer_controls(*sample)
    torch.testing.assert_close(edits["frozen_full"], edits["swap_full"])


def test_changing_writer_changes_only_target_term(sample):
    h, full, off, u = sample
    edits = layer_controls(*sample)
    # Difference must be rank one over positions: all rows parallel to the
    # difference between the two norm-matched target directions.
    expected_axis = full[:, 1].norm() * off[:, 1] / off[:, 1].norm() - full[:, 1]
    difference = edits["frozen_off"] - edits["frozen_full"]
    orthogonal = (
        difference
        - (difference @ expected_axis)[:, None]
        * expected_axis
        / expected_axis.square().sum()
    )
    torch.testing.assert_close(
        orthogonal, torch.zeros_like(orthogonal), atol=2e-6, rtol=0
    )


def test_projection_null_removes_the_same_amount(sample):
    _, full, _, u = sample
    g = torch.Generator().manual_seed(8)
    for j in range(2):
        q, z = full[:, j], u[:, j]
        randomized = random_projection_axis(q, z, g)
        actual, null = remove_projection(q, z), remove_projection(q, randomized)
        torch.testing.assert_close(actual.norm(), null.norm())
        assert abs(float(actual @ z)) < 2e-6


def test_zero_dose_cannot_silently_be_rescaled():
    with pytest.raises(ValueError, match="zero edit"):
        match_row_norms(torch.zeros(2, 8), torch.ones(2, 8))


def test_randomness_is_reproducible_and_local(sample):
    state = torch.random.get_rng_state()
    a = layer_controls(*sample, seed=77)
    b = layer_controls(*sample, seed=77)
    assert torch.equal(state, torch.random.get_rng_state())
    for arm in ARMS:
        assert torch.equal(a[arm], b[arm])


def test_control_full_matches_existing_intervention():
    from jlens.hooks import ActivationRecorder

    model = TinyDecoder(n_layers=4, d_model=8)
    prompt = "a short synthetic prompt"
    g = torch.Generator().manual_seed(81)
    v = torch.randn(8, 2, generator=g)
    edit = swap_edit(model, prompt, {1: v[:, 0]}, {1: v[:, 1]}, mode="clamp")
    with torch.no_grad(), ActivationRecorder(model.layers, at=[1]) as rec:
        model.forward(model.encode(prompt))
        h = rec.activations[1][0]
    torch.testing.assert_close(swap_delta(h, v), edit.vectors[1])


def test_pilot_runner_checkpoints_resumes_and_rejects_partial_items(
    tmp_path, monkeypatch
):
    import json

    from jsteer import write_controls as wc
    from jsteer.config import ModelConfig
    from jsteer.data import ProbeSwapItem

    model = TinyDecoder(n_layers=4, d_model=8)
    config = ModelConfig(
        name="tiny",
        hf_model_id="tiny",
        np_model_id="tiny",
        n_layers=4,
        d_model=8,
        lens_filename="unused",
    )
    item = ProbeSwapItem("test-item", "test", "a prompt", "b", "d", "c", "e")
    monkeypatch.setattr(wc, "load_config", lambda _: config)
    monkeypatch.setattr(wc, "load_model", lambda _: model)
    monkeypatch.setattr(wc, "probe_swap_items", lambda: [item])
    monkeypatch.setattr(wc, "first_token_id", lambda _, word: {"b": 2, "c": 3}[word])
    monkeypatch.setattr(
        wc, "unembedding_rows", lambda _, ids: model.lm_head.weight[ids]
    )
    artifact = tmp_path / "directions.pt"
    g = torch.Generator().manual_seed(94)
    torch.save(
        {
            "targets": ["b", "c"],
            "digests": {
                f"gsm8k_{part}|1": {
                    "n": 32,
                    "pullbacks": torch.randn(2, 8, generator=g),
                }
                for part in ("full", "off")
            },
        },
        artifact,
    )
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps([{"name": item.name, "arm": "baseline", "baseline_ok": True}])
    )
    spec = wc.WriteControlSpec(
        str(artifact),
        str(reference),
        config_name="tiny",
        layers=(1,),
        strengths=(1.0,),
        gen_tokens=1,
        limit=1,
        out_dir=str(tmp_path / "out"),
    )
    result = wc.run_write_controls(spec)
    assert result["status"] == "complete"
    assert result["records"] == 1 + len(ARMS)
    assert all(not layer._forward_hooks for layer in model.layers)
    # A complete exact-manifest run must not load the model or repeat work.
    monkeypatch.setattr(wc, "load_model", lambda _: pytest.fail("loaded on resume"))
    assert wc.run_write_controls(spec)["records"] == result["records"]
    path = wc.Path(result["path"])
    payload = json.loads(path.read_text())
    payload["records"].pop()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        wc.run_write_controls(spec)


def test_stale_deployment_fails_before_loading_inputs():
    from jsteer.write_controls import WriteControlSpec, run_write_controls

    with pytest.raises(ValueError, match="Deployed source differs"):
        run_write_controls(
            WriteControlSpec(
                "does-not-exist", "does-not-exist", expected_sources={"old": "code"}
            )
        )

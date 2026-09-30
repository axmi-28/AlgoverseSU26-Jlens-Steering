"""Offline invariants for the frozen expanded panel and paired uncertainty."""

import importlib
import json

import pytest

followup = importlib.import_module("scripts.42_write_controls_followup")


def test_bootstrap_preserves_constant_paired_difference():
    assert followup.category_bootstrap([1, 1, 1], ["a", "a", "b"]) == (1, 1, 1)
    assert followup.category_bootstrap([0, 0], ["a", "b"]) == (0, 0, 0)


def test_bootstrap_point_estimate_is_item_weighted_and_reproducible():
    result = followup.category_bootstrap([1, 1, -1], ["a", "a", "b"])
    assert result[0] == pytest.approx(1 / 3)
    assert result == followup.category_bootstrap([1, 1, -1], ["a", "a", "b"])
    assert result[1] <= result[0] <= result[2]


def test_bootstrap_rejects_empty_and_mismatched_pairs():
    with pytest.raises(ValueError):
        followup.category_bootstrap([], [])
    with pytest.raises(ValueError):
        followup.category_bootstrap([1], ["a", "b"])
    assert followup.category_bootstrap([1, 0], ["a", "a"]) == (0.5, None, None)


def test_freeze_is_idempotent_but_never_silently_overwrites(tmp_path, monkeypatch):
    path = tmp_path / "plan.json"
    monkeypatch.setattr(followup, "make_plan", lambda: {"version": 1})
    followup.freeze_plan(path)
    original = path.read_bytes()
    followup.freeze_plan(path)
    assert original == path.read_bytes()
    monkeypatch.setattr(followup, "make_plan", lambda: {"version": 2})
    with pytest.raises(ValueError, match="Frozen plan differs"):
        followup.freeze_plan(path)
    assert original == path.read_bytes()


def test_verify_rejects_changed_manifest_and_partial_result(tmp_path, monkeypatch):
    from scripts import report_write_controls

    payload = {
        "manifest": {"spec": {"corpus": "gsm8k", "seed": 1}},
        "status": "complete",
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload))
    # Panel completeness/regrading is tested by the imported reader. Here test
    # only the extra frozen-manifest/status contract without external artifacts.
    monkeypatch.setattr(
        report_write_controls,
        "read_panel",
        lambda p: (json.loads(p.read_text()), {}, ["example"]),
    )
    plan = {"manifests": {"gsm8k": payload["manifest"]}}
    corpus, _, names = followup.verify_result(plan, path)
    assert corpus == "gsm8k" and names == {"example"}
    changed = json.loads(json.dumps(payload))
    changed["manifest"]["spec"]["seed"] += 1
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="manifest differs"):
        followup.verify_result(plan, path)
    payload["status"] = "time_limit"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Incomplete run"):
        followup.verify_result(plan, path)

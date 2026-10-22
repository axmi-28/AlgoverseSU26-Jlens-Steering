"""A restarted CPU supervisor must never repeat recorded GPU invocations."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import modal_replication as runner
from jsteer.cross_model import input_hashes, sha
from jsteer.write_controls import file_hash


def test_slot_limit_counts_existing_calls_before_a_new_gpu_submission(monkeypatch):
    state = {"models": {"old_a": {"stage": "submitted"}, "old_b": {"stage": "submitted"}}}
    collected = []
    def get(name):
        collected.append(name)
        return {"seconds": 10}
    calls = {name: SimpleNamespace(get=lambda n=name: get(n)) for name in state["models"]}
    monkeypatch.setattr(runner, "_save_campaign", lambda _: None)
    runner._wait_for_gpu_slot(calls, state)
    assert collected == ["old_a"]
    assert state["models"]["old_a"]["stage"] == "complete"
    assert state["models"]["old_b"]["stage"] == "submitted"


@pytest.fixture
def campaign_state(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / "data/protocols/filtering_replication.json"
    protocol = json.loads(protocol_path.read_text())
    expected = {"protocol_sha": file_hash(protocol_path),
                "source_sha": file_hash(root / "src/jsteer/cross_model.py"),
                "inputs_sha": {name: sha(input_hashes(name)) for name in protocol["models"]}}
    state = {"app": runner.APP_NAME, "stage": "running", "started": 1,
             "protocol": protocol, "expected": expected,
             "models": {name: {"stage": "submitted", "gpu_call_id": f"saved-{name}"}
                        for name in protocol["models"]}}
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "results_volume", SimpleNamespace(reload=lambda: None, commit=lambda: None))
    monkeypatch.setattr(runner, "_save_campaign", lambda value: (tmp_path / "campaign.json").write_text(json.dumps(value)))
    (tmp_path / "campaign.json").write_text(json.dumps(state))
    return state, expected, tmp_path


def test_recovery_gets_saved_calls_and_never_spawns(campaign_state, monkeypatch):
    import subprocess

    state, expected, directory = campaign_state
    collected, reports = [], []

    def saved_call(call_id):
        def get():
            collected.append(call_id)
            return {"status": "complete", "path": "saved", "seconds": 10}
        return SimpleNamespace(get=get)

    def no_spawn(*args, **kwargs):
        pytest.fail("Recovery attempted a new model invocation")

    monkeypatch.setattr(runner.modal.FunctionCall, "from_id", saved_call)
    monkeypatch.setattr(runner.modal.Function, "from_name", lambda *args: SimpleNamespace(spawn=no_spawn))
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs: reports.append(args))
    result = runner.campaign.local(expected)
    assert result["stage"] == "complete"
    assert set(collected) == {r["gpu_call_id"] for r in state["models"].values()}
    assert len(collected) == len(state["models"])
    assert len(reports) == 2
    assert json.loads((directory / "campaign.json").read_text())["stage"] == "complete"


def test_completed_campaign_returns_without_any_function_lookup(campaign_state, monkeypatch):
    state, expected, directory = campaign_state
    state["stage"] = "complete"
    (directory / "campaign.json").write_text(json.dumps(state))

    def no_lookup(*args, **kwargs):
        pytest.fail("Completed campaign tried to resolve remote functions")

    monkeypatch.setattr(runner.modal.Function, "from_name", no_lookup)
    assert runner.campaign.local(expected)["stage"] == "complete"


def test_published_campaign_recovers_in_a_separate_namespace_without_new_calls(tmp_path, monkeypatch):
    import subprocess

    from jsteer.neuronpedia_replication import PROTOCOL
    from jsteer.neuronpedia_replication import input_hashes as published_hashes

    protocol = json.loads(PROTOCOL.read_text())
    expected = {"protocol_sha": file_hash(PROTOCOL),
                "source_sha": file_hash(PROTOCOL.parents[2]/"src/jsteer/neuronpedia_replication.py"),
                "inputs_sha": {name: sha(published_hashes(name)) for name in protocol["models"]}}
    old = tmp_path/"filtering_replication"
    old.mkdir()
    (old/"campaign.json").write_text('{"historical":true}')
    root = tmp_path/"neuronpedia_replication"
    root.mkdir()
    state = {"app": runner.APP_NAME, "stage": "running", "started": 1,
             "protocol": protocol, "expected": expected,
             "models": {name: {"stage": "submitted", "gpu_call_id": f"saved-{name}"} for name in protocol["models"]}}
    (root/"campaign.json").write_text(json.dumps(state))
    monkeypatch.setattr(runner, "ROOT", old)
    monkeypatch.setattr(runner, "results_volume", SimpleNamespace(reload=lambda: None, commit=lambda: None))
    monkeypatch.setattr(runner, "_save_campaign", lambda value: (Path(value["out_dir"])/"campaign.json").write_text(json.dumps(value)))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    collected, reports = [], []

    def saved_call(call_id):
        def get():
            collected.append(call_id)
            return {"status": "complete", "path": "saved", "seconds": 10}
        return SimpleNamespace(get=get)

    def no_spawn(*args, **kwargs):
        pytest.fail("Published recovery attempted fresh GPU/prepare work")

    monkeypatch.setattr(runner.modal.FunctionCall, "from_id", saved_call)
    monkeypatch.setattr(runner.modal.Function, "from_name", lambda *args: SimpleNamespace(spawn=no_spawn))
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs: reports.append(args))
    result = runner.campaign.local(expected, campaign_name="neuronpedia_replication")
    assert result["stage"] == "complete"
    assert result["out_dir"] == str(root)
    assert len(collected) == 6
    assert len(reports) == 1 and reports[0][1].endswith("73_neuronpedia_report.py")
    assert json.loads((old/"campaign.json").read_text()) == {"historical": True}

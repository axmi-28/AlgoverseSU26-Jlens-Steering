import importlib.util
from pathlib import Path


def test_currency_expansion_requires_the_complete_alias_and_preserves_raw_rows():
    spec = importlib.util.spec_from_file_location(
        "currency_sensitivity", Path(__file__).resolve().parents[1] / "scripts/69_filtering_replication_sensitivity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidates = {"currency": {"prompt": "The currency is", "swap_to": "India"},
                  "capital": {"prompt": "The capital is", "swap_to": "India"}}
    rows = [{"name": "currency", "text": text, "success": False}
            for text in ("Indian Rupee.", "Indian elephant.", "France.")]
    rows.append({"name": "capital", "text": "Indian Rupee.", "success": False})
    result, audit = module.rescore(rows, candidates, {"India": ["Indian rupee"]})
    assert [r["success"] for r in result] == [True, False, False, False]
    assert len(audit) == 1
    assert not any(r["success"] for r in rows)

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "cross_model_report", Path(__file__).resolve().parents[1] / "scripts/60_cross_model_report.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_correct_answer_containing_name_is_not_incorrect_emission():
    rows = [
        {"target": "Russia", "text": "Russian.", "success": True, "leakage": True},
        {"target": "Australia", "text": "Australia.", "success": True, "leakage": True},
        {"target": "France", "text": "Paris is in France.", "success": True, "leakage": True},
    ]
    scored = report.emission_metrics(rows)
    assert all(r["name_emission"] and r["correct_name_emission"] for r in scored)
    assert not any(r["incorrect_name_emission"] for r in scored)
    assert all("name_emission" not in r for r in rows)


def test_incorrect_emission_requires_both_name_and_wrong_answer():
    scored = report.emission_metrics([
        {"success": False, "leakage": True},
        {"success": False, "leakage": False},
    ])
    assert [r["incorrect_name_emission"] for r in scored] == [True, False]

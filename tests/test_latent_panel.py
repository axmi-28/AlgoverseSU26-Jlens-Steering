"""C40 panel invariants: no item may name its entities or answers, and the
candidate set is fixed by the fact table and seed alone."""

import json

import pytest

from jsteer.latent_panel import (
    KB_PATH,
    PER_ENTITY_CAP,
    _hits,
    freeze_panel,
    generate_candidates,
    item_problems,
)


@pytest.fixture(scope="module")
def kb():
    return json.loads(KB_PATH.read_text())


@pytest.fixture(scope="module")
def covered(kb):
    return {e for t in kb["types"].values() for e in t["entities"]}


def test_every_candidate_passes_every_check(kb, covered):
    items = generate_candidates(kb, covered)
    assert len(items) > 150
    for item in items:
        assert item_problems(item, kb, covered) == [], item["name"]
        assert item["answer"] in item["answer_forms"]


def test_generation_is_deterministic_and_capped(kb, covered):
    a, b = generate_candidates(kb, covered), generate_candidates(kb, covered)
    assert a == b
    assert len({i["name"] for i in a}) == len(a)
    per = {}
    for i in a:
        per[i["intermediate"]] = per.get(i["intermediate"], 0) + 1
    assert max(per.values()) <= PER_ENTITY_CAP


def test_uncovered_entities_are_never_used(kb, covered):
    items = generate_candidates(kb, covered - {"France"})
    assert all("France" not in (i["intermediate"], i["swap_to"]) for i in items)


def _item(**kw):
    base = {
        "type": "country",
        "prompt": "Fact: The capital of the country where the city of Lyon is located is",
        "intermediate": "France",
        "swap_to": "Italy",
        "answer_forms": ["Paris"],
        "swap_answer_forms": ["Rome"],
    }
    return {**base, **kw}


def test_leak_checks_fire(kb, covered):
    assert item_problems(_item(), kb, covered) == []
    # Demonym of the intermediate, and a prefix form of an answer.
    assert item_problems(_item(prompt="Fact: The capital of the French-speaking country is"), kb, covered)
    assert item_problems(_item(prompt="Fact: The capital of the country near Parisian suburbs is"), kb, covered)
    # Replacement named in the prompt.
    assert item_problems(_item(prompt="Fact: The capital of the country next to Italy is"), kb, covered)
    # Non-disjoint answers.
    assert item_problems(_item(swap_answer_forms=["Paris"]), kb, covered)
    # Start-of-word only: "elephant" must not count as naming "ant".
    ok = {"type": "animal", "prompt": "Fact: The number of legs on the elephant-like beast is",
          "intermediate": "spider", "swap_to": "ant", "answer_forms": ["eight"], "swap_answer_forms": ["six"]}
    assert item_problems(ok, kb, covered) == []
    bad = {**ok, "prompt": "Fact: The number of legs on the creature that builds anthills is"}
    assert item_problems(bad, kb, covered)


def test_scorer_accepts_any_form_at_a_word_boundary():
    assert _hits(" Canadian dollar", ["Canadian", "dollar"])
    assert _hits(" the United States.", ["the United States", "USA"])
    assert not _hits(" Canadians", ["Canadian"])


def test_freeze_applies_the_prespecified_rule():
    cands = {"sha256": "x", "items": [{"name": n} for n in "abcd"]}
    rec = lambda n, a, s, e: {"name": n, "hit_answer": a, "hit_swap_answer": s, "emitted_substring": e,
                              "one_hop_source_ok": True, "one_hop_target_ok": False}
    screen = {"records": [rec("a", True, False, False), rec("b", False, False, False),
                          rec("c", True, True, False), rec("d", True, False, True)]}
    panel = freeze_panel(cands, screen, "y")
    assert [i["name"] for i in panel["items"]] == ["a"]
    with pytest.raises(ValueError):
        freeze_panel(cands, {"records": screen["records"][:3]}, "y")

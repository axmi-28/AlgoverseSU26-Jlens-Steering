from types import SimpleNamespace

from jsteer.cross_model import select_units


def test_selection_balances_sources_and_uses_only_clean_eligible_units():
    units = [SimpleNamespace(type=t, source=s, key=f"{t}/{s}/{i}")
             for t in ("country", "person", "element")
             for s in ("a", "b", "c") for i in range(8)]
    eligible = {u.key: ["express", "use"] for u in units if not u.key.endswith("/7")}
    selected = select_units(eligible, units, per_type=6)
    assert len(selected) == 18
    assert set(selected) <= eligible.keys()
    assert selected == select_units(eligible, list(reversed(units)), per_type=6)
    for t in ("country", "person", "element"):
        for s in ("a", "b", "c"):
            assert sum(k.startswith(f"{t}/{s}/") for k in selected) == 2


def test_selection_does_not_duplicate_when_a_type_is_small():
    u = SimpleNamespace(type="person", source="a", key="one")
    assert select_units({"one": ["express", "use"]}, [u]) == ["one"]

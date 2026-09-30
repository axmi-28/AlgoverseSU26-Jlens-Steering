"""Invariants of the held-out ordered-scale eval.

Each of these encodes a way the set could silently produce a result-shaped
artifact rather than fail, which is the failure mode this repo keeps hitting.
"""

from __future__ import annotations

import json

import pytest

from jsteer.config import REPO_ROOT
from jsteer.data import (
    flexible_generalization_trials,
    heldout_args,
    heldout_trials,
)

NAME = "ordered-scale"


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads((REPO_ROOT / "data" / "heldout" / f"{NAME}.json").read_text())


def test_answers_within_a_func_are_distinct(spec):
    """A swap must not be scorable by landing on another argument's answer."""
    for category in spec["categories"]:
        for func in category["funcs"]:
            values = list(func["answers"].values())
            assert len(set(values)) == len(values), (category["name"], func["name"])


def test_every_arg_has_an_answer_in_every_func(spec):
    for category in spec["categories"]:
        for func in category["funcs"]:
            assert set(func["answers"]) == set(category["args"])


def test_arguments_are_held_out_from_the_upstream_set(spec):
    """Held out means the ARGUMENTS are new; first_letter reuses its template
    on purpose, as the matched control."""
    upstream = {t.source_arg for t in flexible_generalization_trials()}
    ours = {arg for c in spec["categories"] for arg in c["args"]}
    assert not (upstream & ours)


def test_both_axes_are_populated_on_both_sides(spec):
    """The 2x2 must have every cell filled, or it cannot separate the axes."""
    cells = {
        (c["arg_scale"], f["answer_type"])
        for c in spec["categories"]
        for f in c["funcs"]
    }
    for scale in ("ordered", "unordered"):
        for answer in ("number", "letter"):
            assert (scale, answer) in cells, (scale, answer)


def test_matched_operation_control_spans_both_scales(spec):
    """first_letter and letter_count each run over ordered AND unordered args.

    These are what make the design more than four one-sided comparisons: the
    operation and the answer type are identical across the pair, so only the
    argument differs.
    """
    by_func: dict[str, set[str]] = {}
    for c in spec["categories"]:
        for f in c["funcs"]:
            by_func.setdefault(f["name"], set()).add(c["arg_scale"])
    for name in ("first_letter", "letter_count"):
        assert by_func[name] == {"ordered", "unordered"}, (name, by_func[name])


def test_trials_carry_their_axis_labels():
    trials = heldout_trials(NAME)
    assert len(trials) == 120
    assert all(t.arg_scale in ("ordered", "unordered") for t in trials)
    assert all(t.answer_type for t in trials)


def test_heldout_args_are_the_cotangent_list():
    args = heldout_args(NAME)
    assert len(args) == len(set(args)) == 16
    assert set(args) == {t.source_arg for t in heldout_trials(NAME)}


# --------------------------------------------------------------------------
# within-corpus ablations
# --------------------------------------------------------------------------


def test_shuffle_preserves_tokens_and_changes_order():
    """The whole point: identical content, different order."""
    from jsteer.domains import shuffle_sentences

    text = "One. Two. Three. Four. Five. Six. Seven. Eight."
    out = shuffle_sentences(text)
    assert sorted(out.split()) == sorted(text.split())
    assert out != text


def test_scramble_breaks_coreference_but_keeps_numbers_and_calendar():
    """Entities must be rebound; arithmetic and eval arguments must not move."""
    from jsteer.domains import scramble_entities

    text = "Natalia sold 48 clips in April. She sold 24 more. Natalia has 72."
    out = scramble_entities(text)
    # Every number survives.
    digits = [w.strip(".,") for w in out.split() if w.strip(".,").isdigit()]
    assert digits == ["48", "24", "72"]
    # Months are eval arguments and are exempt.
    assert "April" in out
    # The repeated referent is no longer a single referent.
    assert "Natalia" not in out


def test_strip_numbers_removes_every_digit():
    from jsteer.domains import strip_numbers

    out = strip_numbers("She sold 48 clips, then 24, totalling 72.")
    assert not any(ch.isdigit() for ch in out)
    assert "sold" in out and "clips" in out


def test_ablation_specs_differ_in_identity():
    """Distinct transforms must not share a cache key.

    The transform is invisible in dataset/config/split, so if it were left out
    of identity() every ablation would resolve to the same materialised corpus
    and the panel would silently compare a corpus against itself.
    """
    from jsteer.domains import DOMAINS_BY_NAME

    names = ["gsm8k", "gsm8k_shuffled", "gsm8k_noent", "gsm8k_nonum"]
    ids = [DOMAINS_BY_NAME[n].identity() for n in names]
    assert len(set(ids)) == len(ids)

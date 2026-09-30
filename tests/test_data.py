"""The prompt set's shape, and the arg/answer distinction the design turns on.

Upstream injects the **argument** and grades the **answer**. Getting that
backwards silently changes which unembedding rows become cotangents -- 16 rows
rather than 57 -- and makes the task trivial, since boosting the graded token's
own logit would then count as a success. These tests exist so that confusion
cannot recur.
"""

from __future__ import annotations

import pytest

from jsteer.data import (
    all_answers,
    all_args,
    category_args,
    flexible_generalization_prompts,
    flexible_generalization_trials,
)

pytestmark = pytest.mark.network  # the prompt set is fetched from a pinned commit


def test_sixty_four_base_prompts() -> None:
    """4 categories x 4 templates x 4 args."""
    prompts = flexible_generalization_prompts()
    assert len(prompts) == 64
    assert len({p.key for p in prompts}) == 64
    assert len({p.category for p in prompts}) == 4


def test_one_hundred_ninety_two_trials() -> None:
    """The 64 prompts crossed with the 3 *other* args in the same category."""
    trials = flexible_generalization_trials()
    assert len(trials) == 64 * 3 == 192
    assert all(t.source_arg != t.target_arg for t in trials)
    args = category_args()
    assert all(t.target_arg in args[t.category] for t in trials), (
        "swaps stay in-category"
    )


def test_sixteen_argument_tokens() -> None:
    """The cotangent budget: 16 args, not 57 answers."""
    assert len(all_args()) == 16
    assert set(all_args()) == {a for args in category_args().values() for a in args}
    assert len(all_answers()) > len(all_args())


def test_the_injected_token_is_not_the_graded_token() -> None:
    """What makes the task non-trivial: inject Canada, grade Ottawa."""
    trials = flexible_generalization_trials()
    assert not any(t.target_arg == t.target_answer for t in trials)


def test_the_worked_example() -> None:
    trials = flexible_generalization_trials()
    example = next(
        t
        for t in trials
        if t.func == "capital" and t.source_arg == "France" and t.target_arg == "Canada"
    )
    assert example.prompt == "The capital of France is the city of"
    assert example.source_answer == "Paris"
    assert example.target_answer == "Ottawa"

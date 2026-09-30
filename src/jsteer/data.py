"""The paper's prompt sets.

``jlens`` ships its evaluation and experiment prompt sets under ``data/`` in the
GitHub repo, but its ``pyproject.toml`` only packages ``jlens/data/*`` -- so
installing ``jlens`` as a dependency does *not* give you
``data/experiments/*.json``. They are fetched from a pinned commit and cached
locally instead.

The two that matter for this project:

- ``flexible-generalization`` -- 4 categories x 4 args x 4 templates = 64 base
  prompts; each arg is swapped for the 3 others in its category, giving the
  192 trials behind the paper's headline steering result (76/192 at ordinary
  strength, 101/192 at double).
- ``probe-swap`` -- 90 two-hop factual items, ``prompt`` / ``intermediate`` /
  ``answer`` / ``swap_to`` / ``swap_answer``.
"""

from __future__ import annotations

import functools
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import requests

from jsteer.config import REPO_ROOT

logger = logging.getLogger(__name__)

#: Pinned so the prompt sets cannot shift under us. Upstream is archived
#: ("not maintained, not accepting contributions"), but pin anyway.
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
JLENS_RAW = f"https://raw.githubusercontent.com/anthropics/jacobian-lens/{JLENS_COMMIT}"

DATA_DIR = REPO_ROOT / "data" / "jlens-upstream"

EXPERIMENTS = (
    "capacity",
    "directed-modulation",
    "dual-task",
    "flexible-generalization",
    "ignition",
    "probe-swap",
    "selectivity-language",
    "selectivity-linecount",
    "top-down-summoning",
    "verbal-introspection",
    "verbal-report",
)


def fetch_experiment(slug: str, *, force: bool = False) -> Path:
    """Download ``data/experiments/{slug}.json`` from the pinned commit, cached."""
    if slug not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {slug!r}; expected one of {EXPERIMENTS}")
    local = DATA_DIR / "experiments" / f"{slug}.json"
    if local.exists() and not force:
        return local
    url = f"{JLENS_RAW}/data/experiments/{slug}.json"
    logger.info("fetching %s", url)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(response.content)
    return local


@functools.cache
def load_experiment(slug: str) -> dict:
    """Parsed experiment JSON. Cached: the accessors below are called per record."""
    return json.loads(fetch_experiment(slug).read_text())


@dataclass(frozen=True)
class SwapTrial:
    """One flexible-generalization trial.

    ``prompt`` is ``template.format(arg=source_arg)``. The intervention swaps
    the lens coordinate of ``source_arg`` for ``target_arg``; success is the
    greedy next token matching ``target_answer`` instead of ``source_answer``.
    """

    category: str
    func: str
    template: str
    source_arg: str
    target_arg: str
    source_answer: str
    target_answer: str
    #: Held-out sets tag each trial with the two axes they cross. Empty for the
    #: upstream set, which does not separate them -- that is why the held-out
    #: set exists.
    arg_scale: str = ""
    answer_type: str = ""
    #: How much computation stands between the swapped argument and the graded
    #: answer. -1 where the set does not grade the axis.
    depth: int = -1

    @property
    def prompt(self) -> str:
        return self.template.format(arg=self.source_arg)


def flexible_generalization_trials() -> list[SwapTrial]:
    """The 192 trials: 4 categories x 4 args x 4 templates x 3 swap targets."""
    data = load_experiment("flexible-generalization")
    trials: list[SwapTrial] = []
    for category in data["categories"]:
        args = category["args"]
        for func in category["funcs"]:
            for source_arg in args:
                for target_arg in args:
                    if target_arg == source_arg:
                        continue
                    trials.append(
                        SwapTrial(
                            category=category["name"],
                            func=func["name"],
                            template=func["template"],
                            source_arg=source_arg,
                            target_arg=target_arg,
                            source_answer=func["answers"][source_arg],
                            target_answer=func["answers"][target_arg],
                        )
                    )
    return trials


def heldout_trials(name: str = "ordered-scale") -> list[SwapTrial]:
    """Swap trials from ``data/heldout/<name>.json``.

    Same schema as the upstream experiment file plus two labels: ``arg_scale``
    on each category and ``answer_type`` on each func. Those exist because the
    upstream set confounds them -- its numeric-argument templates almost all
    have numeric answers too -- so no split of it can say which axis a steering
    gain rides on.
    """
    path = REPO_ROOT / "data" / "heldout" / f"{name}.json"
    data = json.loads(path.read_text())
    trials: list[SwapTrial] = []
    for category in data["categories"]:
        args = category["args"]
        for func in category["funcs"]:
            for source_arg in args:
                for target_arg in args:
                    if target_arg == source_arg:
                        continue
                    trials.append(
                        SwapTrial(
                            category=category["name"],
                            func=func["name"],
                            template=func["template"],
                            source_arg=source_arg,
                            target_arg=target_arg,
                            source_answer=func["answers"][source_arg],
                            target_answer=func["answers"][target_arg],
                            arg_scale=category["arg_scale"],
                            answer_type=func["answer_type"],
                            depth=func.get("depth", -1),
                        )
                    )
    return trials


def heldout_args(name: str = "ordered-scale") -> list[str]:
    """The held-out set's arguments, in file order.

    These are the cotangents a domain digest must be fitted against to steer
    this set: the stored pullbacks are indexed by argument, so a digest built
    for the upstream 16 cannot supply a write direction for ``saturday``.
    """
    data = json.loads((REPO_ROOT / "data" / "heldout" / f"{name}.json").read_text())
    return [arg for category in data["categories"] for arg in category["args"]]


@dataclass(frozen=True)
class ProbeSwapItem:
    """One two-hop item from ``probe-swap``."""

    name: str
    category: str
    prompt: str
    intermediate: str
    answer: str
    swap_to: str
    swap_answer: str


def probe_swap_items() -> list[ProbeSwapItem]:
    """The 90 two-hop items."""
    data = load_experiment("probe-swap")
    return [ProbeSwapItem(**item) for item in data["items"]]


@dataclass(frozen=True)
class BasePrompt:
    """One of the 64 base prompts: a template filled with one of its args.

    The 192 trials are these 64 prompts crossed with the 3 other args in the
    same category. Which unit of analysis you want depends on the question:
    ``J_x`` is a property of the *prompt* (64 of them), while a pullback
    ``g_x = J_x^T u_y`` and a steering outcome are properties of a
    ``(prompt, target)`` pair (192 of them, or 256 counting each prompt's own
    arg as a self-control).
    """

    category: str
    func: str
    template: str
    arg: str
    answer: str

    @property
    def prompt(self) -> str:
        return self.template.format(arg=self.arg)

    @property
    def key(self) -> str:
        return f"{self.category}/{self.func}/{self.arg}"


def flexible_generalization_prompts() -> list[BasePrompt]:
    """The 64 base prompts: 4 categories x 4 templates x 4 args."""
    data = load_experiment("flexible-generalization")
    return [
        BasePrompt(
            category=category["name"],
            func=func["name"],
            template=func["template"],
            arg=arg,
            answer=func["answers"][arg],
        )
        for category in data["categories"]
        for func in category["funcs"]
        for arg in category["args"]
    ]


def category_args() -> dict[str, list[str]]:
    """``{category: [4 args]}`` -- the 16 tokens the intervention writes.

    The distinction that governs the whole target-token budget: the
    intervention injects the **arg** ("Canada"), while grading reads the
    **answer** ("Ottawa"). Upstream's README is explicit -- the test "swaps the
    lens representation of one arg for another from the same category ... and
    scores the next token against the new arg's answer". So the cotangents
    ``u_y`` are the 16 arg rows, not the ~50 answer rows; the answers never
    enter the Jacobian computation at all, only the grader.

    That asymmetry is the point of the dataset. Injecting X and grading Y means
    a hit cannot be manufactured by boosting the graded token's logit -- the
    model has to re-run the template's function on the substituted argument.
    """
    data = load_experiment("flexible-generalization")
    return {category["name"]: list(category["args"]) for category in data["categories"]}


def all_args() -> list[str]:
    """The 16 arg words, in dataset order."""
    return [arg for args in category_args().values() for arg in args]


def all_answers() -> list[str]:
    """Every distinct answer word, for the grader (never as a cotangent)."""
    data = load_experiment("flexible-generalization")
    seen: dict[str, None] = {}
    for category in data["categories"]:
        for func in category["funcs"]:
            for answer in func["answers"].values():
                seen[answer] = None
    return list(seen)

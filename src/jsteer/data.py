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


def load_experiment(slug: str) -> dict:
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

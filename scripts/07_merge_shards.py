#!/usr/bin/env python3
"""Stitch a fanned-out run back together. Runs locally, no GPU.

Every shard writes only files nobody else writes (see ``jsteer.sweeps``), so
merging is the step that turns N shard-local files into the one artifact the
analysis reads. Cheap, and safe to re-run.

The one subtlety is the research-question-1 group means. Each shard stores a
*sum* and a *count* rather than a mean, so the merge is an exact pooled mean
rather than a mean-of-means -- which would be wrong whenever shards differ in
size, and they do differ whenever a resumed shard skipped completed prompts.
The pooled ``n`` is written alongside so a mean over fewer prompts than expected
is visible rather than silent.

    python scripts/07_merge_shards.py --results results
    python scripts/07_merge_shards.py --results results --what causal
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

from jsteer.config import REPO_ROOT

logger = logging.getLogger("merge")

SHARD_RE = re.compile(r"_shard(\d+)of(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=None, help="default: results/")
    parser.add_argument(
        "--what",
        nargs="*",
        default=["rq1", "rq2", "causal"],
        choices=["rq1", "rq2", "causal"],
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=None,
        help="merge only the run with this shard count (disambiguates re-runs)",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="leave the per-shard files in place (default: they are kept anyway)",
    )
    return parser.parse_args()


def merged_name(path: Path) -> str:
    """``scalars_qwen3-8b_all_shard3of8.json`` -> ``scalars_qwen3-8b_all.json``."""
    return SHARD_RE.sub("", path.name)


def shard_count(path: Path) -> int:
    """The ``M`` in ``_shardNofM``."""
    return int(SHARD_RE.search(path.name).group(2))


def group_by_target(
    paths: list[Path], *, shards: int | None = None
) -> tuple[dict[str, list[Path]], list[str]]:
    """Group shard files by their merged name, refusing to mix separate runs.

    Two runs of the same sweep at different shard counts produce filenames that
    strip to the *same* merged name -- a 3-prompt probe at ``shard0of1`` and the
    80-prompt run at ``shard0of8`` both become ``scalars_qwen3-8b_all.json``.
    Concatenating them duplicates every prompt the probe touched, and the result
    looks like a slightly-too-large but otherwise normal results file.

    So files are keyed by ``(merged name, M)`` and a target with more than one
    ``M`` is refused rather than guessed at; ``shards`` picks one explicitly.
    """
    by_key: dict[tuple[str, int], list[Path]] = defaultdict(list)
    for path in sorted(paths):
        by_key[(merged_name(path), shard_count(path))].append(path)

    groups: dict[str, list[Path]] = {}
    problems: list[str] = []
    for name in {key[0] for key in by_key}:
        counts = sorted(m for (n, m) in by_key if n == name)
        if shards is not None:
            if shards not in counts:
                problems.append(f"{name}: no run with {shards} shards (have {counts})")
                continue
            groups[name] = by_key[(name, shards)]
        elif len(counts) > 1:
            problems.append(
                f"{name}: {len(counts)} separate runs present "
                f"(shard counts {counts}) -- pass --shards to choose one"
            )
        else:
            groups[name] = by_key[(name, counts[0])]
    return groups, problems


class ShardReadError(RuntimeError):
    """A shard file exists but could not be parsed."""


def merge_json_records(paths: list[Path], out: Path) -> tuple[int, list[str]]:
    """Concatenate shard records, reporting rather than crashing on bad files.

    ``modal volume get`` on a directory has been observed to write **0-byte
    files** for entries it downloaded fine individually -- three of eight
    shards, silently, exit code 0. Merging must therefore treat an unreadable
    shard as a first-class outcome: crashing loses the good shards, and
    skipping quietly produces a merged file that is short by a shard nobody
    notices. Bad paths are returned so the caller can refuse to write a
    partial merge.
    """
    records: list[dict] = []
    bad: list[str] = []
    for path in paths:
        try:
            text = path.read_text()
            if not text.strip():
                raise ShardReadError("empty file")
            records.extend(json.loads(text))
        except (json.JSONDecodeError, ShardReadError, OSError) as exc:
            bad.append(f"{path.name}: {exc}")
    if not bad:
        out.write_text(json.dumps(records))
    return len(records), bad


def merge_group_means(paths: list[Path], out: Path) -> dict[str, int]:
    """Pool per-shard ``{sum, n}`` into one exact mean per group and layer."""
    totals: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = defaultdict(int)
    for path in paths:
        for key, state in torch.load(
            path, map_location="cpu", weights_only=False
        ).items():
            totals[key] = (
                state["sum"] if key not in totals else totals[key] + state["sum"]
            )
            counts[key] += state["n"]
    torch.save(
        {
            "mean": {key: (totals[key] / counts[key]).float() for key in totals},
            "n": dict(counts),
        },
        out,
    )
    return dict(counts)


def merge_vectors(paths: list[Path], out: Path) -> int:
    """Concatenate the per-shard ``g_x`` stores; ``g_bar`` is shard-invariant."""
    local: dict[str, torch.Tensor] = {}
    header: dict = {}
    for path in paths:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        local.update(blob.pop("local"))
        if not header:
            header = blob
        elif blob["targets"] != header["targets"] or blob["layers"] != header["layers"]:
            raise ValueError(
                f"{path.name} disagrees with the other shards on targets/layers"
            )
    torch.save({"local": local, **header}, out)
    return len(local)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    root = Path(args.results) if args.results else REPO_ROOT / "results"
    if not root.exists():
        logger.error("no results directory at %s", root)
        return 2

    failures: list[str] = []
    for what in args.what:
        directory = root / what
        if not directory.exists():
            continue
        logger.info("\n=== %s ===", what)

        groups, problems = group_by_target(
            [p for p in directory.glob("*.json") if SHARD_RE.search(p.name)],
            shards=args.shards,
        )
        for problem in problems:
            failures.append(problem)
            logger.error("  AMBIGUOUS %s", problem)
        for name, paths in groups.items():
            n, bad = merge_json_records(paths, directory / name)
            if bad:
                failures.extend(bad)
                logger.error(
                    "  %-46s NOT MERGED -- %d/%d shards unreadable:",
                    name,
                    len(bad),
                    len(paths),
                )
                for reason in bad:
                    logger.error("      %s", reason)
                logger.error(
                    "      re-fetch them individually: "
                    "modal volume get jsteer-results /%s/<file> ./results/%s/",
                    what,
                    what,
                )
                continue
            logger.info("  %-46s %6d records from %d shards", name, n, len(paths))

        groups, problems = group_by_target(
            [p for p in directory.glob("group_means_*.pt") if SHARD_RE.search(p.name)],
            shards=args.shards,
        )
        for problem in problems:
            failures.append(problem)
            logger.error("  AMBIGUOUS %s", problem)
        for name, paths in groups.items():
            counts = merge_group_means(paths, directory / name)
            pooled = sorted({v for v in counts.values()})
            logger.info(
                "  %-46s %d groups from %d shards, n per group %s",
                name,
                len(counts),
                len(paths),
                pooled if len(pooled) < 4 else f"{min(pooled)}-{max(pooled)}",
            )

        groups, problems = group_by_target(
            [p for p in directory.glob("vectors_*.pt") if SHARD_RE.search(p.name)],
            shards=args.shards,
        )
        for problem in problems:
            failures.append(problem)
            logger.error("  AMBIGUOUS %s", problem)
        for name, paths in groups.items():
            n = merge_vectors(paths, directory / name)
            logger.info("  %-46s %6d (prompt, layer) entries", name, n)

        if what == "rq1":
            for run_dir in sorted((directory / "digests").glob("*/")):
                n = len(list(run_dir.glob("*.pt")))
                logger.info("  %-46s %6d prompts", f"digests/{run_dir.name}/", n)

    logger.info(
        "\nper-shard files are left in place; delete them once the merge is checked"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

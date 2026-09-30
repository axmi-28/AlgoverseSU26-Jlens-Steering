#!/usr/bin/env python3
"""Pull the results volume down, one file at a time, verifying each.

``modal volume get`` on a *directory* has been observed to write **0-byte
files** for entries that download fine individually -- on one 80-prompt RQ1
run it silently truncated 5 of 8 ``group_means`` files and dozens of digests,
and exited 0. A truncated ``.json`` at least fails loudly at parse time; a
truncated ``.pt`` fails as ``EOFError: Ran out of input`` somewhere much later,
and a *partially* truncated one might not fail at all.

So this fetches through the Python API file by file, checks the byte count
against the volume's own listing, and retries. It is slower than the bulk CLI
and it is the only version whose output can be trusted.

    python scripts/08_fetch_results.py                    # everything
    python scripts/08_fetch_results.py --prefix /rq1      # one sweep
    python scripts/08_fetch_results.py --verify-only      # check what is local
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from jsteer.config import REPO_ROOT

logger = logging.getLogger("fetch")

VOLUME = "jsteer-results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", default=VOLUME)
    parser.add_argument("--prefix", default="/", help="remote subtree to fetch")
    parser.add_argument("--dest", default=None, help="default: results/")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="report size mismatches without downloading",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-fetch even if the size matches"
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    import modal

    volume = modal.Volume.from_name(args.volume)
    dest = Path(args.dest) if args.dest else REPO_ROOT / "results"

    entries = [
        entry
        for entry in volume.listdir(args.prefix.rstrip("/") or "/", recursive=True)
        if getattr(entry, "type", None) != 2  # 2 == directory
    ]
    logger.info("%d files under %s", len(entries), args.prefix)

    fetched = skipped = 0
    bad: list[str] = []

    for entry in sorted(entries, key=lambda e: e.path):
        remote_size = getattr(entry, "size", None) or 0
        local = dest / entry.path
        if local.exists() and local.stat().st_size == remote_size and not args.force:
            skipped += 1
            continue
        if args.verify_only:
            bad.append(
                f"{entry.path}: local "
                f"{local.stat().st_size if local.exists() else 'missing'} "
                f"!= remote {remote_size}"
            )
            continue

        local.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, args.retries + 1):
            try:
                data = b"".join(volume.read_file(entry.path))
                if len(data) != remote_size:
                    raise OSError(f"got {len(data)} bytes, expected {remote_size}")
                local.write_bytes(data)
                fetched += 1
                break
            except Exception as exc:  # noqa: BLE001 -- retry anything transient
                if attempt == args.retries:
                    bad.append(f"{entry.path}: {exc}")
                    logger.error("  FAILED %s: %s", entry.path, exc)
        if fetched and fetched % 25 == 0:
            logger.info("  %d fetched, %d already current", fetched, skipped)

    logger.info(
        "\nfetched %d, already current %d, failed %d", fetched, skipped, len(bad)
    )
    if bad:
        for reason in bad[:20]:
            logger.error("  %s", reason)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

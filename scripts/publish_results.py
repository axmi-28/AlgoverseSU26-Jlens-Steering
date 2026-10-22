"""Publish or verify the curated CPU-reproducible results snapshot.

Run the numbered analysis/figure scripts first, then `build`. Verification
checks both published-file integrity and all recorded analysis input hashes.
No model downloads, cloud calls or GPU runs are performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/core-results"
DESTINATION = ROOT / "docs/results"
TABLE_FILES = ("full_table.json", "full_table.md", "index.html", "provenance.json")
FIGURES = (
    "cross_model_deduplicated",
    "self_fitted_across_models",
    "neuronpedia_across_models",
)
FILES = (
    *TABLE_FILES,
    "figures/figure_data.json",
    "figures/captions.md",
    *(f"figures/{name}.{ext}" for name in FIGURES for ext in ("pdf", "svg", "png")),
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_path(base, relative):
    path = base / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(base.resolve()):
        raise ValueError(f"Path escapes snapshot root: {relative}")
    return path


def verify_inputs(directory):
    provenance = json.loads((directory / "provenance.json").read_text())
    for name, expected in provenance["sources_sha256"].items():
        path = checked_path(ROOT, name)
        if digest(path) != expected:
            raise ValueError(f"Analysis input changed: {name}")
    payload = json.loads((directory / "full_table.json").read_text())
    rows = {row["id"]: row for row in payload["core"]}
    if len(rows) != 16 or len({row["model"] for row in rows.values()}) != 13:
        raise ValueError("Expected 16 conditions on 13 model checkpoints")
    figures = json.loads((directory / "figures/figure_data.json").read_text())
    if figures["source_sha256"] != digest(directory / "full_table.json"):
        raise ValueError("Figures do not match the published result table")
    if figures["script_sha256"] != digest(ROOT / "scripts/76_core_results_figures.py"):
        raise ValueError("Figure generator differs from the recorded version")
    for group in ("combined", "self_fitted", "neuronpedia"):
        for point in figures[group]:
            original = rows[point["id"]]["contrasts"][point["task"]][point["metric"]]
            if any(point[key] != value for key, value in original.items()):
                raise ValueError(f"Plotted statistic differs: {point['id']}")
    shown = {point["id"] for point in figures["combined"]}
    if len(shown) != 11 or len({rows[key]["model"] for key in shown}) != 11:
        raise ValueError("Combined figure must display 11 distinct checkpoints")
    return len(provenance["sources_sha256"])


def build():
    verify_inputs(SOURCE)
    for name in FILES:
        destination = DESTINATION / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / name, destination)
    manifest = {
        "description": "Curated public results; regenerate from checksummed saved outputs, not new GPU runs.",
        "files_sha256": {name: digest(DESTINATION / name) for name in FILES},
        "rebuild": [
            "python scripts/release.py restore",
            "python scripts/75_all_core_results.py",
            "python scripts/76_core_results_figures.py",
            "python scripts/publish_results.py build",
        ],
    }
    (DESTINATION / "snapshot.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify()


def verify(rebuilt=False):
    manifest = json.loads((DESTINATION / "snapshot.json").read_text())
    if set(manifest["files_sha256"]) != set(FILES):
        raise ValueError("Unexpected public snapshot file list")
    for name, expected in manifest["files_sha256"].items():
        if digest(checked_path(DESTINATION, name)) != expected:
            raise ValueError(f"Published snapshot changed: {name}")
    count = verify_inputs(DESTINATION)
    if rebuilt:
        for name in TABLE_FILES:
            if digest(SOURCE / name) != digest(DESTINATION / name):
                raise ValueError(f"Rebuilt table differs from public snapshot: {name}")
        rebuilt_figures = json.loads((SOURCE / "figures/figure_data.json").read_text())
        published_figures = json.loads((DESTINATION / "figures/figure_data.json").read_text())
        if rebuilt_figures != published_figures:
            raise ValueError("Rebuilt plotted statistics differ from the public snapshot")
    print(f"Verified {len(FILES)} published files, {count} analysis inputs, 16 conditions, 11 plotted checkpoints")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--rebuilt", action="store_true", help="Also compare freshly regenerated tables and plotting statistics")
    args = parser.parse_args()
    if args.command == "build":
        build()
    else:
        verify(rebuilt=args.rebuilt)


if __name__ == "__main__":
    main()

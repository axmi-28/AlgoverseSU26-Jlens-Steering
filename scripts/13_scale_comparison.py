"""Qwen3-8B against Qwen3.6-27B: does model size change the conclusions?

Deliberately a *parser*, not a second implementation. Every distribution number
here is lifted out of the two ``preliminary_{model}_all.md`` reports that
``scripts/09_analyze.py`` already wrote, so the comparison cannot drift from
the reports it claims to summarise. Only the C1 steering table is recomputed,
from the raw causal records, because no report holds it.

    python scripts/13_scale_comparison.py
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import logging
import re
import statistics as st
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("scale")

#: (config name, human label, causal band, C1 file glob)
MODELS = [
    ("qwen3-8b", "Qwen3-8B", "L14-19", "steering_qwen3-8b_*_c1_L14-19*.json"),
    ("qwen3.6-27b", "Qwen3.6-27B", "L24-34", "steering_qwen3.6-27b_*_c1_L24-34*.json"),
]


# --------------------------------------------------------------------------
# markdown table parsing
# --------------------------------------------------------------------------


def read_tables(path: Path) -> dict[str, list[list[str]]]:
    """``{"1.4": [[cell, ...], ...]}`` -- body rows only, header dropped."""
    tables: dict[str, list[list[str]]] = {}
    current: str | None = None
    for line in path.read_text().splitlines():
        heading = re.match(r"^## Table ([\d.]+)", line)
        if heading:
            current = heading.group(1)
            tables[current] = []
            continue
        if current and line.startswith("|"):
            # Cells can legitimately contain "|" -- Table 2.3's statistic column
            # is literally "median |g_x| / |g_bar|". Splitting naively silently
            # shifts every later column left by two.
            cells = _split_row(line)
            # Skip the header row and the |---|---| separator.
            if set("".join(cells)) <= set("- ") or cells[0] in ("Layer", "Question"):
                continue
            tables[current].append(cells)
    return tables


def _split_row(line: str) -> list[str]:
    r"""Split a markdown row, honouring ``\|`` escapes inside cells."""
    parts = re.split(r"(?<!\\)\|", line.strip().strip("|"))
    return [c.strip().replace(r"\|", "|") for c in parts]


def by_layer(rows: list[list[str]]) -> dict[int, list[str]]:
    out = {}
    for cells in rows:
        m = re.match(r"L(\d+)", cells[0])
        if m:
            out[int(m.group(1))] = cells
    return out


def depth_percent(layer: int, n_layers: int) -> float:
    return 100.0 * layer / (n_layers - 1)


# --------------------------------------------------------------------------
# C1, recomputed from the raw records
# --------------------------------------------------------------------------


def c1_best(results: Path, pattern: str) -> dict[str, tuple]:
    """Best-of-arm: each arm at *its own* best strength, plus the baseline.

    Two deduplications, both load-bearing:

    * **Merged files only.** ``results/causal`` holds the per-shard writes
      *and* the merged file that 07_merge_shards built from them, so a naive
      glob counts every trial twice and reports 384/384 where 192 exist.
    * **By trial identity.** Separate strength grids over the same band each
      re-record the unsteered baseline, and the 8B's ``both`` and ``additive``
      runs overlap on some strengths. Keying on
      (arm, strength, prompt, target) collapses those to one row each.
    """
    seen: dict[tuple, dict] = {}
    for f in sorted(glob.glob(str(results / "causal" / pattern))):
        if "_shard" in Path(f).name:
            continue
        blob = json.loads(Path(f).read_text())
        for r in blob["records"] if isinstance(blob, dict) else blob:
            seen[(r["arm"], r["strength"], r["prompt_key"], r["target_arg"])] = r
    if not seen:
        return {}
    cells = collections.defaultdict(list)
    for r in seen.values():
        cells[(r["arm"], r["strength"])].append(r)
    best: dict[str, tuple] = {}
    for (arm, strength), rs in cells.items():
        hits = sum(bool(r["hit"]) for r in rs)
        rate = hits / len(rs)
        row = (rate, hits, len(rs), strength, st.median([r["target_rank"] for r in rs]))
        if arm not in best or row[0] > best[arm][0]:
            best[arm] = row
    return best


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", default="all")
    ap.add_argument("--results", default=str(REPO_ROOT / "results"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    results = Path(args.results)
    reports, c1s = {}, {}
    for name, _label, _band, pattern in MODELS:
        path = results / f"preliminary_{name}_{args.positions}.md"
        reports[name] = read_tables(path) if path.exists() else {}
        if not path.exists():
            logger.warning("missing %s -- its columns will be blank", path)
        c1s[name] = c1_best(results, pattern)

    out: list[str] = []
    w = out.append

    w("# Does model size change the conclusions? Qwen3-8B vs Qwen3.6-27B")
    w("")
    w(
        "Every experiment re-run unchanged on a model 3.4x larger. Layers are "
        "matched by **depth percentage**, not by index: the workspace band is "
        "38%-92% of depth in both cases, which is L13-32 of 36 layers on the 8B "
        "and L24-58 of 64 layers on the 27B. Percentages are given so the rows "
        "line up."
    )
    w("")

    # ---- C1 ----
    w("## 1. Steering: does the prompt's own direction beat the lens's average?")
    w("")
    w(
        "Identical trials, four ways: two intervention methods x two direction "
        "sources. Each arm is shown at *its own* best strength, so no arm is "
        "handicapped by a shared setting."
    )
    w("")
    order = [
        ("swap_averaged", "Swap, lens averaged direction"),
        ("swap_local", "Swap, prompt's own direction"),
        ("additive_averaged", "Additive, lens averaged direction"),
        ("additive_local", "Additive, prompt's own direction"),
    ]
    header = ["Arm"] + [f"{label} succeeded" for _n, label, _b, _p in MODELS]
    rows = []
    for name, label in [("baseline", "No steering (baseline accuracy)"), *order]:
        cells = [label]
        for mname, _lab, _b, _p in MODELS:
            hit = c1s[mname].get(name)
            cells.append(f"{hit[1]}/{hit[2]} ({100 * hit[0]:.0f}%)" if hit else "-")
        rows.append(cells)
    _table(header, rows, w)
    w("")
    for mname, label, _b, _p in MODELS:
        best = c1s[mname]
        if "swap_averaged" in best and "swap_local" in best:
            ratio = best["swap_averaged"][1] / max(best["swap_local"][1], 1)
            w(
                f"- **{label}**: averaged beats prompt-local by "
                f"{ratio:.1f}x on swap "
                f"(best strengths {best['swap_averaged'][3]:g} and "
                f"{best['swap_local'][3]:g})."
            )
    w("")

    # ---- RQ1 bias/variance ----
    w("## 2. Is the averaged Jacobian systematically wrong, or just noisy?")
    w("")
    w(
        "Average all 64 test prompts' own Jacobians and ask how far *that "
        "average* sits from the lens's J_bar. Pure scatter would shrink by "
        "sqrt(64); the bias ratio is how many times further out it actually is."
    )
    w("")
    _depth_table(
        "1.4", [(1, "observed"), (2, "variance-only"), (3, "bias ratio")], reports, w
    )

    # ---- RQ1 single-prompt distance ----
    w("## 3. How far is a single prompt's Jacobian from the average?")
    w("")
    w(
        "Relative Frobenius distance. 0 = identical to the lens; 1 = as far from "
        "J_bar as J_bar is from zero."
    )
    w("")
    _depth_table(
        "1.1", [(1, "eval median"), (3, "wikitext null"), (4, "ratio")], reports, w
    )

    # ---- RQ2 ----
    w("## 4. How do the pulled-back steering directions disagree?")
    w("")
    w(
        "For every prompt, layer and target token, the prompt's own steering "
        "direction against the lens's averaged one. Cosine 1 = identical, "
        "0 = perpendicular, negative = opposite."
    )
    w("")
    rq2_rows = []
    labels = {}
    for mname, _lab, _b, _p in MODELS:
        for cells in reports[mname].get("2.3", []):
            labels[cells[0]] = cells[1]
    for question, stat in labels.items():
        cells = [question, stat]
        for mname, _lab, _b, _p in MODELS:
            hit = [c for c in reports[mname].get("2.3", []) if c[0] == question]
            cells += (
                [hit[0][2], hit[0][3].replace("fit null ", "")] if hit else ["-", "-"]
            )
        rq2_rows.append(cells)
    header = ["Question", "Statistic"]
    for _n, label, _b, _p in MODELS:
        header += [f"{label} test prompts", f"{label} wikitext null"]
    _table(header, rq2_rows, w)

    dest = Path(args.out) if args.out else results / "scale_comparison.md"
    dest.write_text("\n".join(out) + "\n")
    logger.info("wrote %s (%d lines)", dest, len(out))


def _table(header: list[str], rows: list[list[str]], w) -> None:
    w("| " + " | ".join(header) + " |")
    w("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        w("| " + " | ".join(r) + " |")


def _depth_table(table_id: str, columns, reports, w) -> None:
    """One row per depth decile, so two different layer counts line up."""
    per_model = {}
    for name, _label, _b, _p in MODELS:
        per_model[name] = by_layer(reports[name].get(table_id, []))
    n_layers = {"qwen3-8b": 36, "qwen3.6-27b": 64}
    if not any(per_model.values()):
        w("_(not yet computed)_")
        w("")
        return
    header = ["Depth"]
    for _n, label, _b, _p in MODELS:
        header += [f"{label} {c[1]}" for c in columns]
    rows = []
    for pct in (40, 50, 60, 70, 80, 90):
        cells = [f"{pct}% of depth"]
        for name, _label, _b, _p in MODELS:
            table = per_model[name]
            if not table:
                cells += ["-"] * len(columns)
                continue
            layer = min(
                table, key=lambda ll: abs(depth_percent(ll, n_layers[name]) - pct)
            )
            cells[0] = f"{pct}% of depth"
            cells += [table[layer][i] for i, _ in columns]
        rows.append(cells)
    _table(header, rows, w)
    w("")
    w(
        "_Rows are matched by depth, so the two models' layers differ: "
        + "; ".join(
            f"{label} uses "
            + ", ".join(
                f"L{min(per_model[name], key=lambda ll: abs(depth_percent(ll, n_layers[name]) - pct))}"
                for pct in (40, 50, 60, 70, 80, 90)
            )
            for name, label, _b, _p in MODELS
            if per_model[name]
        )
        + "._"
    )
    w("")


if __name__ == "__main__":
    main()

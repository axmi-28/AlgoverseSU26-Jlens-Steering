"""C2 across every steering strength, plus the paper's own predictor as a baseline.

Two things our mentor asked for, both of which C2 as originally run does not do.

**1. Control for activation strength.** C2 correlated the geometric predictors
against steerability at a *single* alpha. A predictor that only works at one
dose is a predictor of dose sensitivity, not of direction quality, so every
correlation is recomputed at every alpha in the grid and read for stability.

**2. Compare against workspace loading.** The J-Lens paper's own per-prompt
predictor of swap success (section 3.4): the cosine between the residual stream
and the *source* concept's lens vector, on the clean forward pass. It is
computed by ``jsteer.sweeps.workspace_loading``; this script only consumes it.

The comparison is deliberately **nested, not a race**. Loading asks "is the
concept present in this activation"; the geometric predictors ask "is the
averaged direction the right thing to write on this prompt". Both can be true,
so the question worth answering is whether geometry adds anything *over*
loading -- reported as incremental R^2 on rank-transformed variables.

One caveat inherited from the paper: its evidence for loading is category-level
("Country arguments have the highest loading and swap most reliably; number-word
arguments have the lowest"), with no correlation coefficient or test reported.
With four categories, loading is near-collinear with category, which also
carries baseline correctness and task difficulty. So loading's raw correlation
is the citable replication and its partial is the honest number, and they are
expected to differ.

    python scripts/16_c2_strength_sweep.py --config qwen3.6-27b --band 24-34
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

from jsteer.analysis import rank_correlation
from jsteer.config import REPO_ROOT, load_config
from jsteer.data import flexible_generalization_prompts

logger = logging.getLogger("c2sweep")


def _load_c2_module():
    """Import the C2 script so the predictors are built by the same code path."""
    path = Path(__file__).with_name("11_c2_predictors.py")
    spec = importlib.util.spec_from_file_location("c2_predictors", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def spearman(a, b) -> float:
    import torch

    return rank_correlation(
        torch.tensor(list(a), dtype=torch.float),
        torch.tensor(list(b), dtype=torch.float),
    )


def partial(a, b, control) -> float:
    """Spearman of a and b with `control` partialled out of both."""
    ra, rb, rc = (_ranks(x) for x in (a, b, control))
    ea = _resid(ra, rc)
    eb = _resid(rb, rc)
    return float(np.corrcoef(ea, eb)[0, 1]) if len(ea) > 2 else float("nan")


def _ranks(x) -> np.ndarray:
    """Tie-averaged ranks -- the same treatment as jsteer.analysis.average_ranks.

    Shares that function rather than re-deriving it, so the partial correlations
    and the R^2 models cannot drift from the plain Spearman numbers.
    """
    import torch

    from jsteer.analysis import average_ranks

    return average_ranks(torch.tensor(list(x), dtype=torch.float)).numpy()


def _resid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


def r2(y: np.ndarray, X: np.ndarray) -> float:
    """Adjusted R^2 of y on X (both already rank-transformed), intercept added."""
    A = np.column_stack([np.ones(len(y)), X]) if X.size else np.ones((len(y), 1))
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot == 0:
        return float("nan")
    n, k = len(y), A.shape[1] - 1
    raw = 1 - ss_res / ss_tot
    return 1 - (1 - raw) * (n - 1) / max(n - k - 1, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="qwen3.6-27b")
    ap.add_argument("--band", default="24-34")
    ap.add_argument("--positions", default="all")
    ap.add_argument("--swap-file", default=None)
    ap.add_argument("--additive-file", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(args.config)
    results = REPO_ROOT / "results"
    lo, hi = (int(x) for x in args.band.split("-"))
    band = list(range(lo, hi + 1))

    # Comma-separated, because a strength grid can be spread over several runs:
    # the qwen3.6-27b swap grid was extended downward in a second run
    # (`_lowalpha`), and reading only the first file silently truncates the
    # sweep to the strengths above the optimum -- which is exactly the range
    # where a strength-stability claim is least convincing.
    swap_file = args.swap_file or ",".join(
        [
            f"steering_{config.name}_swap_c1_L{lo}-{hi}.json",
            f"steering_{config.name}_swap_c1_L{lo}-{hi}_lowalpha.json",
            f"steering_{config.name}_both_c1_L{lo}-{hi}.json",
        ]
    )
    add_file = args.additive_file or (
        f"steering_{config.name}_additive_c1_L{lo}-{hi}.json"
    )

    # ---- outcomes at every strength -------------------------------------
    def all_scores(filenames: str, arm: str) -> dict[float, dict[str, float]]:
        per: dict[float, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        found = False
        for filename in filenames.split(","):
            path = results / "causal" / filename
            if not path.exists():
                continue
            found = True
            for r in json.loads(path.read_text()):
                if r["arm"] == arm:
                    per[r["strength"]][r["prompt_key"]].append(r["target_rank"])
        if not found:
            logger.warning("none of %s exist", filenames)
        return {
            s: {k: -st.median(v) for k, v in d.items()} for s, d in sorted(per.items())
        }

    baseline: dict[str, bool] = {}
    for filename in f"{swap_file},{add_file}".split(","):
        path = results / "causal" / filename
        if path.exists():
            for r in json.loads(path.read_text()):
                if r["arm"] == "baseline":
                    baseline[r["prompt_key"]] = bool(r["hit"])

    outcomes = {
        "swap": all_scores(swap_file, "swap_averaged"),
        "additive": all_scores(add_file, "additive_averaged"),
    }
    any_scores = next(iter(outcomes["swap"].values()), {})
    prompts = [p for p in flexible_generalization_prompts() if p.key in any_scores]

    # ---- predictors, from the C2 script's own code path ------------------
    c2 = _load_c2_module()
    predictors = dict(
        c2.build_predictors(config, results, band, args.positions, prompts)
    )

    loading_path = results / "c2" / f"workspace_loading_{config.name}_L{lo}-{hi}.json"
    if loading_path.exists():
        blob = json.loads(loading_path.read_text())
        predictors["workspace loading (paper, sec 3.4)"] = {
            k: v["loading"] for k, v in blob.items()
        }
    else:
        logger.warning(
            "no workspace loading at %s -- baseline column omitted", loading_path
        )

    out: list[str] = []
    w = out.append
    w(f"# C2 across steering strengths, with the paper's baseline ({config.name})")
    w("")
    w(
        f"Layers L{lo}-L{hi}, n = {len(prompts)} prompts. Outcome is negated "
        "median target rank under steering with the lens's averaged direction, "
        "so **higher = steers better** everywhere. Every cell is a partial "
        "Spearman with baseline correctness partialled out, which is the column "
        "that isolates geometry from task difficulty."
    )
    w("")

    for operator, grid in outcomes.items():
        if not grid:
            continue
        w(f"## Table C2S.1{operator[0]} - {operator}: correlation at each strength")
        w("")
        strengths = sorted(grid)
        header = ["Predictor"] + [f"a={s:g}" for s in strengths]
        rows = []
        for name, values in predictors.items():
            cells = [name]
            for s in strengths:
                score = grid[s]
                keys = [p.key for p in prompts if p.key in values and p.key in score]
                if len(keys) < 10:
                    cells.append("-")
                    continue
                rho = partial(
                    [values[k] for k in keys],
                    [score[k] for k in keys],
                    [1.0 if baseline.get(k) else 0.0 for k in keys],
                )
                cells.append(f"{rho:+.3f}")
            rows.append(cells)
        _table(header, rows, w)
        hits = {s: sum(1 for k in grid[s] if grid[s][k] == 0) for s in strengths}
        w(
            "_Strengths where steering barely works make every correlation "
            "unstable, so read these next to the success rates: "
            + ", ".join(f"a={s:g} -> {hits[s]} prompts at rank 0" for s in strengths)
            + "._"
        )
        w("")

    # ---- nested model comparison ----------------------------------------
    w("## Table C2S.2 - Does the geometry add anything over the paper's predictor?")
    w("")
    w(
        "Adjusted R^2 on rank-transformed variables, at each operator's best "
        "strength. **Loading** is the paper's predictor alone; **geometry** is "
        "the five J_x / g_x predictors; **both** is all six. The column that "
        "matters is the increment from loading to both."
    )
    w("")
    geo_names = [n for n in predictors if not n.startswith("workspace loading")]
    load_name = next((n for n in predictors if n.startswith("workspace loading")), None)
    rows = []
    for operator, grid in outcomes.items():
        if not grid or load_name is None:
            continue
        best = max(grid, key=lambda s: sum(1 for v in grid[s].values() if v == 0))
        score = grid[best]
        keys = [
            p.key
            for p in prompts
            if p.key in score
            and all(p.key in predictors[n] for n in [*geo_names, load_name])
        ]
        if len(keys) < 15:
            continue
        y = _ranks([score[k] for k in keys])
        L = _ranks([predictors[load_name][k] for k in keys]).reshape(-1, 1)
        G = np.column_stack(
            [_ranks([predictors[n][k] for k in keys]) for n in geo_names]
        )
        rows.append(
            [
                operator,
                f"a={best:g}",
                str(len(keys)),
                f"{r2(y, L):.3f}",
                f"{r2(y, G):.3f}",
                f"{r2(y, np.column_stack([L, G])):.3f}",
                f"{r2(y, np.column_stack([L, G])) - r2(y, L):+.3f}",
            ]
        )
    _table(
        [
            "Operator",
            "strength",
            "n",
            "loading alone",
            "geometry alone",
            "both",
            "increment over loading",
        ],
        rows,
        w,
    )

    dest = Path(args.out) if args.out else results / f"c2_strength_{config.name}.md"
    dest.write_text("\n".join(out) + "\n")
    logger.info("wrote %s", dest)
    print("\n".join(out))


def _table(header, rows, w) -> None:
    w("| " + " | ".join(header) + " |")
    w("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        w("| " + " | ".join(r) + " |")
    w("")


if __name__ == "__main__":
    main()

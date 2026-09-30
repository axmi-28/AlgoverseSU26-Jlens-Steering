#!/usr/bin/env python3
"""C5 - is the prompt-local Jacobian the better local model, and out to what dose?

Reads the shards written by ``jsteer.sweeps.run_linear_response`` and reports
how well each Jacobian predicts the model's measured response to a
single-layer, single-position perturbation.

Why the headline metric is a cosine and not a relative error
------------------------------------------------------------
``||actual - predicted|| / ||actual||`` conflates two different failures: a
prediction that points the wrong way, and one that points the right way with
the wrong gain. That distinction is load-bearing here, because J_bar was fitted
as an average over positions on wikitext and there is no reason its scale
should transfer to a single-position perturbation on a 8-token eval prompt. So
every table reports the cosine (direction only), the gain ratio
``||predicted|| / ||actual||``, and ``sqrt(1 - cos^2)`` -- the relative error
that would survive rescaling the prediction optimally. If a Jacobian's raw
error is large but its scaled error is small, the criticism is calibration; if
both are large, the direction itself is wrong.

The floor, and why the leftmost doses are not automatically evidence
--------------------------------------------------------------------
The measured quantity is a difference of two forward passes, so below some dose
the response is smaller than the arithmetic's resolution. The sweep runs fp32
to push that floor down, but it is still there. ``|actual|/alpha`` is reported
for exactly this reason: it must be flat across the grid. Where it stops being
flat, the measurement -- not the model -- has run out.

    python scripts/18_linear_response.py --model qwen3-8b
"""

from __future__ import annotations

import argparse
import statistics as st
from collections import defaultdict
from pathlib import Path

import torch

from jsteer.config import REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--label", default="L14-19")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def load(model: str, label: str) -> tuple[list[dict], dict]:
    """Every shard for this (model, label). Shards, not the merged file.

    Counts are checked rather than exit codes: a shard set that quietly lost a
    prompt is this repo's recurring failure mode, and it always looks like
    success. Every prompt must carry the same number of records.
    """
    root = REPO_ROOT / "results" / "c5"
    paths = sorted(root.glob(f"linresp_{model}_linresp_{label}_*_shard*.pt"))
    if not paths:
        raise SystemExit(f"no C5 shards for {model}/{label} under {root}")
    stamps = {p.name.split("_")[-2] for p in paths}
    if len(stamps) > 1:
        raise SystemExit(
            f"shards disagree about the run: stamps {sorted(stamps)}. The stamp "
            "hashes the layer and alpha grids, so two of these are different "
            "experiments -- pass a --label that separates them."
        )
    records: list[dict] = []
    meta: dict = {}
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        records.extend(payload["records"])
        meta = payload
    per_prompt = defaultdict(int)
    for r in records:
        per_prompt[r["prompt_key"]] += 1
    sizes = set(per_prompt.values())
    if len(sizes) != 1:
        raise SystemExit(f"prompts have different record counts: {sorted(sizes)}")
    print(
        f"{len(paths)} shards, {len(per_prompt)} prompts, {len(records)} records "
        f"({sizes.pop()} each)"
    )
    return records, meta


def quality(
    actual: torch.Tensor, predicted: torch.Tensor
) -> tuple[float, float, float]:
    """(cosine, gain, scaled error) for one predicted response vector."""
    an = actual.norm().clamp_min(1e-30)
    pn = predicted.norm().clamp_min(1e-30)
    cos = float((actual @ predicted) / (an * pn))
    return cos, float(pn / an), float(max(0.0, 1 - cos * cos) ** 0.5)


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|")


def table(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(_cell(h) for h in header) + " |"]
    out.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        out.append("| " + " | ".join(_cell(c) for c in row) + " |")
    return "\n".join(out)


def _convention_control(model: str, layers: list[int]) -> str:
    """cos(g_x, g_bar) under the last-position and all-position conventions."""
    root = REPO_ROOT / "results" / "rq2"
    rows = []
    for positions in ("last", "all"):
        path = root / f"vectors_{model}_{positions}.pt"
        if not path.exists():
            rows.append([positions, "not fetched", "-", "-"])
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        local, averaged = payload["local"], payload["averaged"]
        cosines, gains = [], []
        for key, v in local.items():
            name, layer = key.rsplit("|", 1)
            # Eval prompts only: "category/func/arg". Wikitext controls have no
            # slashes and are not what the causal arm steers.
            if int(layer) not in set(layers) or name.count("/") != 2:
                continue
            g_bar = averaged[int(layer)]
            for y in range(v.shape[0]):
                a, b = v[y], g_bar[y]
                cosines.append(float(a @ b / (a.norm() * b.norm() + 1e-12)))
                gains.append(float(b.norm() / a.norm().clamp_min(1e-12)))
        rows.append(
            [
                positions,
                len(cosines),
                f"{st.median(cosines):.3f}",
                f"{st.median(gains):.3f}",
            ]
        )
    return table(
        rows,
        ["convention", "pairs", "median cos(g_x, g_bar)", "median |g_bar| / |g_x|"],
    )


def main() -> int:
    args = parse_args()
    records, meta = load(args.model, args.label)
    alphas = sorted({r["alpha"] for r in records})
    layers = sorted({r["layer"] for r in records})

    # Random probes only: they are not derived from either Jacobian, so they
    # are the one arm that favours neither.
    probes = [r for r in records if r["arm"].startswith("random")]

    by_alpha: dict[float, list] = defaultdict(list)
    for r in probes:
        by_alpha[r["alpha"]].append(
            (
                quality(r["actual"], r["pred_local"]),
                quality(r["actual"], r["pred_averaged"]),
                float(r["actual"].norm()),
                r["kl"],
            )
        )

    med = lambda xs: st.median(xs)  # noqa: E731
    rows = []
    for a in alphas:
        v = by_alpha[a]
        loc = [x[0] for x in v]
        avg = [x[1] for x in v]
        norms = [x[2] for x in v]
        rows.append(
            [
                a,
                f"{med([x[0] for x in loc]):.3f}",
                f"{med([x[2] for x in loc]):.3f}",
                f"{med([x[1] for x in loc]):.3f}",
                f"{med([x[0] for x in avg]):.3f}",
                f"{med([x[2] for x in avg]):.3f}",
                f"{med([x[1] for x in avg]):.3f}",
                f"{med(norms) / a:.1f}",
                f"{med([x[3] for x in v]):.3f}",
            ]
        )
    t1 = table(
        rows,
        [
            "alpha",
            "local cos",
            "local scaled err",
            "local gain",
            "averaged cos",
            "averaged scaled err",
            "averaged gain",
            "|actual| / alpha",
            "median KL",
        ],
    )

    # Where the local Jacobian stops being the better model.
    cross = None
    for a in alphas:
        v = by_alpha[a]
        if med([x[0][0] for x in v]) <= med([x[1][0] for x in v]):
            cross = a
            break

    # Per layer: does the crossover move with depth?
    rows = []
    for layer in layers:
        cells = [layer]
        for a in alphas:
            v = [
                quality(r["actual"], r["pred_local"])[0]
                for r in probes
                if r["layer"] == layer and r["alpha"] == a
            ]
            cells.append(f"{med(v):.2f}")
        rows.append(cells)
    t2 = table(rows, ["layer", *[str(a) for a in alphas]])

    rows = []
    for layer in layers:
        cells = [layer]
        for a in alphas:
            v = [
                quality(r["actual"], r["pred_averaged"])[0]
                for r in probes
                if r["layer"] == layer and r["alpha"] == a
            ]
            cells.append(f"{med(v):.2f}")
        rows.append(cells)
    t3 = table(rows, ["layer", *[str(a) for a in alphas]])

    # The steering-direction arms. NOTE what this measures and does not.
    arg_index = {a: i for i, a in enumerate(meta["args"])}
    dose: dict[tuple[str, float], list[float]] = defaultdict(list)
    for r in records:
        if "_" not in r["arm"] or r["arm"].startswith("random"):
            continue
        kind, target = r["arm"].split("_", 1)
        if target not in arg_index:
            continue
        dose[(kind, r["alpha"])].append(float(r["dlogit"][arg_index[target]]))
    rows = [
        [
            a,
            f"{med(dose[('local', a)]):+.3f}",
            f"{med(dose[('averaged', a)]):+.3f}",
            len(dose[("local", a)]),
        ]
        for a in alphas
    ]
    t4 = table(
        rows, ["alpha", "local direction", "averaged direction", "trials per cell"]
    )

    # Control: how much of J_bar's poor showing is the position convention?
    # C5 scores it against a last-position perturbation, but J_bar was fitted by
    # averaging over positions -- so it is being asked for something it was not
    # built for. cos(g_x, g_bar) under both conventions separates "the lens is a
    # weak local model" from "we asked it the wrong question". Cached from RQ2;
    # skipped if those vectors are not present locally.
    t5 = _convention_control(args.model, layers)

    report = f"""# C5 - linear response: which Jacobian models the model? ({args.model})

Perturb **one layer at one position** by `delta = alpha * ||h_l[last]|| * unit(d)`
and measure the change in `<u_y, h_final[last]>` for the 16 arguments. Compare
against what each Jacobian predicts. Band {layers[0]}-{layers[-1]}, 64 prompts,
{len(probes) // len(alphas)} measurements per dose.

One layer and one position because that is what a Jacobian is. Our steering
edits every band layer at every position, which no single-layer Jacobian
models; scoring a Jacobian against that operator would measure the operator and
blame the Jacobian. Positions are pinned to the readout token for source and
target alike, which is the convention under which the estimator identity has no
leading `seq_len` factor.

The predicted quantity is the **raw unembedding projection**, not the logit:
`unembedding_rows` deliberately ignores the final norm, so the logit is not
what any of this linearizes. KL is reported to locate the steering regime on
the same axis, not as a prediction target.

## Table C5.1 - prediction quality against dose (random probes)

Cosine is direction only. Scaled error is `sqrt(1 - cos^2)`, the relative error
that survives rescaling the prediction optimally. Gain is
`||predicted|| / ||actual||`. `|actual| / alpha` must be flat: where it is not,
the measurement floor has been reached and that row is not evidence.

{t1}

Crossover (first dose at which the local Jacobian is no longer the better
model): **alpha = {cross}**.

## Table C5.2 - local cosine by layer

{t2}

## Table C5.3 - averaged cosine by layer

{t3}

## Table C5.4 - steering along each direction

Median change in the **injected argument's own logit** when perturbing along
`unit(g)` for that argument. This is a dose-response curve for the write
direction, and it is *not* the swap outcome: success in C1 is graded on the
target *answer* token, which this sweep does not record. Read it as "which
direction moves its own target further at matched dose", not as steering
success.

{t4}

## Table C5.5 - is the averaged cosine a position-convention artifact?

`J_bar` is fitted by averaging over token positions, but C5 scores it against a
perturbation at the readout position alone. This compares `cos(g_x, g_bar)`
under both conventions, on the same band, from the cached RQ2 pullbacks. Note
`g_x` under the "last" convention *is* the true Jacobian C5 measures against,
so the "last" row should reproduce Table C5.1's averaged cosine -- if it does
not, the two pipelines disagree and one of them is wrong.

{t5}
"""
    out = Path(args.out) if args.out else REPO_ROOT / "results" / f"c5_{args.model}.md"
    out.write_text(report)
    print(f"wrote {out}")
    print(t1)
    print(f"\ncrossover: alpha = {cross}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Freeze and audit the expanded write-control evaluation, without loading a model.

The existing GPU runner is deliberately unchanged. A plan captures its exact
expected manifests; result verification rejects stale inputs or changed panels.
This script has no paid launch mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from jsteer.config import load_config
from jsteer.data import probe_swap_items
from jsteer.write_controls import ARMS, WriteControlSpec, file_hash, source_hashes

ROOT = Path(__file__).resolve().parents[1]
DIGEST = "group_mean_digests_qwen3-8b_dom2x32_968c0ba0_pb_t89d64f82_comp_s4.pt"
REFERENCE = "two_hop_qwen3-8b_all_g595bfbbf_twohop89d9c5_shard0of1.json"
PILOT = ROOT / "results/c22/write_controls_7a59c53d116e8b6c.json"
PLAN = ROOT / "data/protocols/write_controls_expansion_plan.json"
CONTRASTS = (
    ("swap_off", "swap_full"),
    ("swap_off_matched", "swap_full"),
    ("frozen_off", "frozen_full"),
    ("add_off", "add_full"),
    ("swap_projected_matched", "swap_random_projection_matched"),
    ("frozen_logit", "frozen_full"),
    ("frozen_random", "frozen_full"),
)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def make_plan():
    directions = ROOT / "results/rq1" / DIGEST
    reference = ROOT / "results/causal" / REFERENCE
    blob = torch.load(directions, map_location="cpu", weights_only=False)
    words = blob["targets"]
    if len(set(words)) != len(words):
        raise ValueError("Duplicate component targets")
    clean = {
        r["name"]
        for r in json.loads(reference.read_text())
        if r["arm"] == "baseline" and r["baseline_ok"]
    }
    items = [
        i
        for i in probe_swap_items()
        if i.name in clean and i.intermediate in words and i.swap_to in words
    ]
    items.sort(key=lambda i: hashlib.sha256(f"2201|{i.name}".encode()).hexdigest())
    if len(items) != 55 or len({i.name for i in items}) != 55:
        raise ValueError(
            "Expected exactly 55 eligible unique items; review panel drift"
        )
    pilot = json.loads(PILOT.read_text())
    pilot_names = sorted(i["name"] for i in pilot["manifest"]["items"])
    if len(pilot_names) != 8 or not set(pilot_names) <= {i.name for i in items}:
        raise ValueError("Pilot overlap changed")
    config = load_config("qwen3-8b")
    manifests = {}
    for corpus in ("gsm8k", "wikitext_a"):
        spec = WriteControlSpec(
            directions_path=f"/results/rq1/{DIGEST}",
            reference_path=f"/results/causal/{REFERENCE}",
            corpus=corpus,
            limit=55,
            expected_sources=source_hashes(),
        )
        spec.out_dir = "/results/c22"
        counts = {}
        for part in ("full", "off"):
            for layer in spec.layers:
                entry = blob["digests"][f"{corpus}_{part}|{layer}"]
                tensor = entry["pullbacks"]
                if tensor.shape != (len(words), config.d_model):
                    raise ValueError("Wrong component dimensions")
                if entry["n"] != 32 or not torch.isfinite(tensor).all():
                    raise ValueError("Invalid fit counts or nonfinite components")
                counts[f"{part}|{layer}"] = entry["n"]
        manifests[corpus] = {
            "spec": asdict(spec),
            "config": asdict(config),
            "arms": ARMS,
            "items": [asdict(i) for i in items],
            "eligible_items": len(items),
            "fit_counts": counts,
            "directions_sha256": file_hash(directions),
            "reference_sha256": file_hash(reference),
            "source_sha256": source_hashes(),
        }
    # Round trip tuples to the runner's exact JSON representation.
    manifests = json.loads(json.dumps(manifests))
    return {
        "schema": 1,
        "status": "prepared_not_launched",
        "evaluation_role": "development expansion, not held-out confirmation",
        "pilot_sha256": file_hash(PILOT),
        "pilot_names": pilot_names,
        "category_counts": dict(sorted(Counter(i.category for i in items).items())),
        "manifests": manifests,
        "expected_outputs": {
            corpus: f"write_controls_{json_hash(m)[:16]}.json"
            for corpus, m in manifests.items()
        },
        "generations_per_corpus": 55 * (1 + len(ARMS) * 2),
        "analysis": {
            "primary_population": "intersection of current clean-correct, counterfactual-incorrect items across both corpora",
            "sensitivity_population": "same intersection excluding the eight pilot items; not a held-out set",
            "primary_contrast": ["swap_off_matched", "swap_full"],
            "contrasts": CONTRASTS,
            "outcomes": ["hit_swap_answer", "emitted_substring"],
            "strengths_separate": [0.5, 1.0],
            "bootstrap_unit": "provided item category; coarse proxy, not a verified template-family mapping",
            "bootstrap_seed": 42101,
            "bootstrap_draws": 10000,
            "intervals": "pointwise exploratory 95% percentile intervals, conditional on fixed lens fits; no multiplicity-adjusted significance claim",
        },
    }


def freeze_plan(path):
    plan = json.loads(json.dumps(make_plan()))
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError(
                "Frozen plan differs; use a new filename and document amendment"
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def category_bootstrap(differences, categories, *, seed=42101, draws=10000):
    """Paired item-weighted mean; resample whole categories with replacement."""
    values = np.asarray(differences, dtype=float)
    if len(values) != len(categories) or not len(values):
        raise ValueError("Nonempty paired differences/categories required")
    groups = sorted(set(categories))
    sums = np.array([values[np.array(categories) == g].sum() for g in groups])
    sizes = np.array([categories.count(g) for g in groups])
    if len(groups) < 2:
        return float(values.mean()), None, None
    sampled = np.random.default_rng(seed).integers(0, len(groups), (draws, len(groups)))
    means = sums[sampled].sum(axis=1) / sizes[sampled].sum(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def verify_result(plan, path):
    # Import from this directory both under CLI execution and tests.
    from scripts.report_write_controls import read_panel

    payload, idx, names = read_panel(path)
    corpus = payload["manifest"]["spec"]["corpus"]
    if corpus not in plan["manifests"]:
        raise ValueError("Unexpected corpus")
    if payload["manifest"] != plan["manifests"][corpus]:
        raise ValueError("Result manifest differs from frozen plan")
    if payload["status"] != "complete":
        raise ValueError("Incomplete run; do not report a partial paired evaluation")
    return corpus, idx, set(names)


def report(plan, paths):
    verified = [verify_result(plan, p) for p in paths]
    if len(verified) != 2 or {v[0] for v in verified} != set(plan["manifests"]):
        raise ValueError("Provide one complete result per frozen corpus")
    common = set.intersection(*(v[2] for v in verified))
    if not common:
        raise ValueError("No common eligible current baselines")
    lines = [
        "# Expanded latent-write controls — Qwen3-8B",
        "",
        f"Development evaluation, conditional on two fixed corpus fits. Common current-clean panel: {len(common)}/55.",
        "",
        "Intervals resample the supplied categories (a coarse grouping), not individual strengths or arm records. They are pointwise exploratory 95% intervals, not adjusted for multiple comparisons or fit uncertainty. Positive answer differences and negative emission differences favor the candidate.",
        "",
        "The original prefix-answer and substring-emission scorers are unchanged. Lower emission alone does not establish downstream computation. Frozen-writer arms match target-term norms, not total edit norms.",
        "",
        "## Absolute outcomes on the common panel",
        "",
        "Each cell is target-answer hits / injected-string emission. These outcomes can overlap; all denominators are the common-panel size above.",
        "",
        "| Arm | GSM8K, strength 0.5 | GSM8K, strength 1 | WikiText, strength 0.5 | WikiText, strength 1 |",
        "|---|---:|---:|---:|---:|",
    ]
    by_corpus = {corpus: idx for corpus, idx, _ in verified}
    for arm in ARMS:
        cells = []
        for corpus in ("gsm8k", "wikitext_a"):
            for strength in plan["analysis"]["strengths_separate"]:
                records = [by_corpus[corpus][n, arm, strength] for n in common]
                cells.append(
                    f"{sum(r['hit_swap_answer'] for r in records)} / {sum(r['emitted_substring'] for r in records)}"
                )
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Paired differences and exploratory uncertainty",
        "",
        "| Population | Corpus | Contrast | Strength | Outcome | Candidate / reference hits | Wins / losses | Difference, pp [95% interval] |",
        "|---|---|---|---:|---|---|---|---|",
    ]
    for population, names in (
        ("all", sorted(common)),
        ("excluding pilot", sorted(common - set(plan["pilot_names"]))),
    ):
        if not names:
            continue
        for corpus, idx, _ in sorted(verified):
            categories = [idx[n, "baseline", 0]["category"] for n in names]
            for candidate, reference in plan["analysis"]["contrasts"]:
                for strength in plan["analysis"]["strengths_separate"]:
                    for outcome in plan["analysis"]["outcomes"]:
                        a = [int(idx[n, candidate, strength][outcome]) for n in names]
                        b = [int(idx[n, reference, strength][outcome]) for n in names]
                        delta = [x - y for x, y in zip(a, b, strict=True)]
                        mean, lo, hi = category_bootstrap(
                            delta,
                            categories,
                            seed=plan["analysis"]["bootstrap_seed"],
                            draws=plan["analysis"]["bootstrap_draws"],
                        )
                        interval = (
                            "not estimable"
                            if lo is None
                            else f"{100 * lo:+.1f}, {100 * hi:+.1f}"
                        )
                        lines.append(
                            f"| {population} n={len(names)} | {corpus} | {candidate} vs {reference} | {strength} | {outcome} | {sum(a)} / {sum(b)} | {delta.count(1)} / {delta.count(-1)} | {100 * mean:+.1f} [{interval}] |"
                        )
    lines += [
        "",
        "## Descriptive category breakdown",
        "",
        "Added as an exhaustive descriptive breakdown after the GSM8K run, not as a new confirmatory test. Each cell shows full to norm-matched-off target-answer hits. Tiny category counts do not support category-specific efficacy claims.",
        "",
        "| Category | n | GSM8K, strength 0.5 | GSM8K, strength 1 | WikiText, strength 0.5 | WikiText, strength 1 |",
        "|---|---:|---|---|---|---|",
    ]
    for category in sorted(
        {by_corpus["gsm8k"][n, "baseline", 0]["category"] for n in common}
    ):
        members = [
            n
            for n in common
            if by_corpus["gsm8k"][n, "baseline", 0]["category"] == category
        ]
        cells = []
        for corpus in ("gsm8k", "wikitext_a"):
            for strength in plan["analysis"]["strengths_separate"]:
                hits = [
                    sum(
                        by_corpus[corpus][n, arm, strength]["hit_swap_answer"]
                        for n in members
                    )
                    for arm in ("swap_full", "swap_off_matched")
                ]
                cells.append(f"{hits[0]} → {hits[1]}")
        lines.append(f"| {category} | {len(members)} | " + " | ".join(cells) + " |")
    lines += ["", "## Execution checks", ""]
    baseline_texts = [
        {n: idx[n, "baseline", 0]["generated"] for n in common}
        for _, idx, _ in verified
    ]
    lines.append(
        f"- Identical clean baseline text across corpus runs: {sum(baseline_texts[0][n] == baseline_texts[1][n] for n in common)}/{len(common)}."
    )
    for corpus, idx, usable in verified:
        selected = [i["name"] for i in plan["manifests"][corpus]["items"]]
        exact = 0
        score_disagreements = []
        errors = []
        for name in selected:
            for strength in plan["analysis"]["strengths_separate"]:
                full = idx[name, "swap_full", strength]
                frozen = idx[name, "frozen_full", strength]
                exact += full["generated"] == frozen["generated"]
                if any(
                    full[key] != frozen[key]
                    for key in ("hit_swap_answer", "emitted_substring")
                ):
                    score_disagreements.append(f"{name} at strength {strength}")
                for arm in (
                    "swap_off_matched",
                    "swap_projected_matched",
                    "swap_random_projection_matched",
                    "add_full",
                    "add_off",
                ):
                    target = idx[name, arm, strength]["edit_row_norms"]
                    if set(target) != set(full["edit_row_norms"]):
                        raise ValueError("Mismatched edit layers")
                    for layer, values in full["edit_row_norms"].items():
                        for a, b in zip(target[layer], values, strict=True):
                            errors.append(abs(a - b) / max(abs(b), 1e-8))
        token_counts = Counter(
            tuple(idx[n, "baseline", 0]["token_counts"]) for n in common
        )
        lines.append(
            f"- {corpus}: {len(idx)} regraded records; {len(usable)}/55 individually clean-eligible; frozen/full identical text {exact}/{len(selected) * 2}; maximum requested fp32 row-norm relative error {max(errors):.3g}; common-panel intermediate token-count pairs `{dict(token_counts)}`."
        )
        lines.append(
            f"- {corpus}: full/frozen-full answer-or-emission score disagreements: {len(score_disagreements)}"
            + (f" ({'; '.join(score_disagreements)})" if score_disagreements else "")
            + "."
        )
    lines += [
        "",
        "Full and frozen-full are algebraically equivalent, but use differently associated fp32 operations before conversion to model dtype. They are not guaranteed to be bitwise identical. Small numerical differences can change a greedy decision; this is a plausible explanation for the discrepancy, not a traced causal diagnosis. Keep the observed frozen-full baseline for frozen-writer comparisons, retain all items, and do not overwrite this result with a post-hoc rerun.",
    ]
    lines += ["", "## Provenance", ""]
    for path in paths:
        lines.append(f"- `{Path(path).name}`: SHA-256 `{file_hash(path)}`.")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("prepare", "report"), default="prepare")
    ap.add_argument("--plan", type=Path, default=PLAN)
    ap.add_argument("--results", type=Path, nargs="+")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.mode == "prepare":
        plan = freeze_plan(args.plan)
        print(
            json.dumps(
                {
                    "plan": str(args.plan),
                    "sha256": file_hash(args.plan),
                    "status": plan["status"],
                    "outputs": plan["expected_outputs"],
                    "total_generations": 2 * plan["generations_per_corpus"],
                },
                indent=2,
            )
        )
    else:
        if not args.results:
            ap.error("--report requires --results")
        output = report(json.loads(args.plan.read_text()), args.results)
        if args.output:
            args.output.write_text(output)
            print(args.output)
        else:
            print(output)


if __name__ == "__main__":
    # The repo root enables importing the reusable offline report module.
    import sys

    sys.path.insert(0, str(ROOT))
    main()

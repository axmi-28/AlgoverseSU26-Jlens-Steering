"""Audit and report the bounded latent-write pilot, distinct from C22 projection.

No model is loaded. Only complete paired item panels are accepted. All
generations remain in the source JSON; table selection uses clean outputs only.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from jsteer.steering import answer_matches
from jsteer.write_controls import file_hash

COMPARISONS = (
    ("swap_off", "swap_full"),
    ("swap_off_matched", "swap_full"),
    ("swap_projected_matched", "swap_full"),
    ("swap_projected_matched", "swap_random_projection_matched"),
    ("frozen_off", "frozen_full"),
    ("frozen_logit", "frozen_full"),
    ("frozen_random", "frozen_full"),
    ("add_off", "add_full"),
)


def read_panel(path):
    payload = json.loads(Path(path).read_text())
    manifest, records = payload["manifest"], payload["records"]
    expected_names = {item["name"] for item in manifest["items"]}
    expected = {("baseline", 0.0)} | {
        (a, float(s)) for a in manifest["arms"] for s in manifest["spec"]["strengths"]
    }
    keys = [(r["name"], r["arm"], r["strength"]) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate item/arm/strength records")
    names = {r["name"] for r in records}
    if not names <= expected_names:
        raise ValueError("Unexpected items outside manifest")
    for name in names:
        found = {(r["arm"], r["strength"]) for r in records if r["name"] == name}
        if found != expected:
            raise ValueError(f"Incomplete paired panel for {name}")
    if payload["status"] == "complete" and names != expected_names:
        raise ValueError("Completion status disagrees with actual item coverage")
    for r in records:
        if r["hit_answer"] != answer_matches(r["generated"], r["answer"]):
            raise ValueError("Stored original-answer score disagrees with text")
        if r["hit_swap_answer"] != answer_matches(r["generated"], r["swap_answer"]):
            raise ValueError("Stored counterfactual-answer score disagrees with text")
        if r["emitted_substring"] != (
            r["swap_to"].casefold() in r["generated"].casefold()
        ):
            raise ValueError("Stored emission score disagrees with text")
    idx = {(r["name"], r["arm"], r["strength"]): r for r in records}
    usable = sorted(
        n
        for n in names
        if idx[n, "baseline", 0]["hit_answer"]
        and not idx[n, "baseline", 0]["hit_swap_answer"]
    )
    return payload, idx, usable


def build_report(path):
    payload, idx, names = read_panel(path)
    manifest = payload["manifest"]
    spec = manifest["spec"]
    rows = payload["records"]
    completed = {r["name"] for r in rows}
    n = len(names)
    w = [
        "# Latent-write control pilot — Qwen3-8B",
        "",
        f"Source: `{Path(path).name}`; SHA-256 `{file_hash(path)}`.",
        f"Status: **{payload['status']}**. {len(completed)}/{len(manifest['items'])} selected items completed; {n} remain clean-correct and counterfactual-incorrect under the current scorer.",
        "",
        "This is the eight-item development pilot prepared before the pause, resumed after reviewing the other sessions' experiments. It is distinct from `c22_projection_control_qwen3-8b.md`, which evaluates native projection swaps on flexible-generalization. This pilot adds per-position norm matching, fixed read/source controls, and target-only additions on latent-intermediate swaps.",
        "",
        f"Corpus: `{spec['corpus']}`; layers: `{spec['layers']}`; strengths: `{spec['strengths']}`; seed: `{spec['seed']}`; greedy continuation: {spec['gen_tokens']} tokens. Items were selected by name hash, not by steered outcomes.",
        "",
        "## Outcomes",
        "",
        "All entries use the same clean-screened items. Answer and emission can overlap. These small counts are exploratory; no population-level success claim or best-per-item selection is made.",
        "",
        "| Arm | Strength | Target answer | Injected-string emission | Mean requested edit norm |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in manifest["arms"]:
        for strength in spec["strengths"]:
            sub = [idx[name, arm, strength] for name in names]
            norms = [
                value
                for r in sub
                for ns in r["edit_row_norms"].values()
                for value in ns
            ]
            w.append(
                f"| {arm} | {strength} | {sum(r['hit_swap_answer'] for r in sub)}/{n} | {sum(r['emitted_substring'] for r in sub)}/{n} | {statistics.mean(norms) if norms else 0:.4f} |"
            )
    w += [
        "",
        "## Paired changes",
        "",
        "Wins and losses count items whose target-answer outcome changes. Each strength is evaluated separately; repeated items across strengths are not independent replicates.",
        "",
        "| Candidate vs reference | Strength | Answer wins / losses | Emission increases / decreases |",
        "|---|---:|---:|---:|",
    ]
    for candidate, reference in COMPARISONS:
        for strength in spec["strengths"]:
            pairs = [
                (idx[name, candidate, strength], idx[name, reference, strength])
                for name in names
            ]
            win = sum(
                a["hit_swap_answer"] and not b["hit_swap_answer"] for a, b in pairs
            )
            loss = sum(
                b["hit_swap_answer"] and not a["hit_swap_answer"] for a, b in pairs
            )
            up = sum(
                a["emitted_substring"] and not b["emitted_substring"] for a, b in pairs
            )
            down = sum(
                b["emitted_substring"] and not a["emitted_substring"] for a, b in pairs
            )
            w.append(
                f"| {candidate} vs {reference} | {strength} | {win} / {loss} | {up} / {down} |"
            )
    errors = []
    identical = 0
    for name in completed:
        for strength in spec["strengths"]:
            full = idx[name, "swap_full", strength]
            frozen = idx[name, "frozen_full", strength]
            identical += full["generated"] == frozen["generated"]
            for arm in (
                "swap_off_matched",
                "swap_projected_matched",
                "swap_random_projection_matched",
                "add_full",
                "add_off",
            ):
                target = idx[name, arm, strength]["edit_row_norms"]
                for layer, norms in full["edit_row_norms"].items():
                    for a, b in zip(target[layer], norms, strict=True):
                        errors.append(abs(a - b) / max(abs(b), 1e-8))
    w += [
        "",
        "## Execution checks",
        "",
        f"- Stored answer and emission scores were independently recomputed from all {len(rows)} continuations; duplicate and incomplete panels are rejected.",
        f"- `frozen_full` and `swap_full` produce identical text in {identical}/{len(completed) * len(spec['strengths'])} item-strength pairs.",
        f"- Maximum relative per-position norm error across matched arms: {max(errors, default=0):.3g}. These are requested fp32 norms before the model's dtype conversion.",
        "- Frozen-writer arms match the target term's magnitude and keep source removal fixed. Their total norms can differ through cancellation; do not describe those arms as globally norm matched.",
        f"- Intermediate token-count pairs among retained items: `{dict(Counter(tuple(idx[name, 'baseline', 0]['token_counts']) for name in names))}`.",
        "- The manifest pins code, component/reference hashes, config, and selected items. Runtime dependency and remote model revisions are not independently pinned by this runner; continuity is checked against the clean baselines and original manifest.",
    ]
    strength = 1.0 if 1.0 in spec["strengths"] else spec["strengths"][0]
    w += [
        "",
        f"## Every retained item at strength {strength}",
        "",
        "Generated-text prefixes are shown below; the JSON retains full six-token continuations. No examples are chosen by success.",
        "",
        "| Item / target answer | Full swap | Off swap | Off matched | Frozen off | Add full | Add off |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in names:
        texts = [
            idx[name, a, strength]["generated"]
            .strip()
            .replace("\n", " / ")
            .replace("|", "\\|")
            for a in (
                "swap_full",
                "swap_off",
                "swap_off_matched",
                "frozen_off",
                "add_full",
                "add_off",
            )
        ]
        w.append(
            f"| {name} / {idx[name, 'baseline', 0]['swap_answer']} | "
            + " | ".join(texts)
            + " |"
        )
    w += [
        "",
        "## Interpretation boundaries",
        "",
        "An improvement after norm matching argues against total edit magnitude alone explaining the native-swap difference on these items. A fixed-writer or additive gain would provide more direct evidence about the target write direction. These are different controls; one cannot substitute for the other. Projection controls here change swap coordinates and then match edit norms, whereas the other session's projection experiment uses its native operator and budget.",
        "",
        "Eight items, one corpus fit, one random seed, and two strengths are insufficient to choose a generally best method. All selected intermediates are single-token: generalization to multi-token intermediates is untested. Prefix answer scoring and substring emission further limit semantic interpretation. The next experiment should follow the observed control pattern on a frozen larger evaluation, not select favorable examples or a new seed after seeing this pilot.",
        "",
        "The raw examples expose a scoring limitation: a numbered continuation such as `1. spring, 2` can contain the requested answer but fail prefix matching. Preserve the original scores here; perform a blinded formatting/alias audit before freezing the next evaluation. Lower intermediate-string emission is not by itself evidence that an edit enters downstream computation.",
        "",
        "See `docs/PROTOCOL.md` for the completed-run interpretation, overlap audit, and prioritized follow-up protocol. This pilot does not authorize another paid run.",
    ]
    return "\n".join(w) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("result", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    report = build_report(args.result)
    if args.output:
        args.output.write_text(report)
        print(args.output)
    else:
        print(report)


if __name__ == "__main__":
    main()

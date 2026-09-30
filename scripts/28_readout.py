"""Does a domain-averaged J_bar READ better, or only write better?

The corpus result so far is entirely about writing. The paper defines J-space
by readout, so "gsm8k gives a better lens" and "gsm8k gives a better actuator"
are different claims that the causal experiments cannot separate. This scores
``unembed(J_bar h)`` on the same prompts and the same concepts the steering
arm writes to.

Why this eval cannot fail the way the causal ones did: the metric is a rank
over the whole vocabulary, so there is no string matching, no surface-form
ambiguity and no answer key. Rank is also invariant to positive rescaling of
J_bar, so corpora whose maps differ in magnitude need no dose matching.

The audit section exists anyway, because "rank 300" is compatible with two very
different situations -- the readout returning a casing or spacing variant of
the right word, or returning noise -- and only the decoded tokens distinguish
them.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="qwen3-8b")
    ap.add_argument("--results", default="results")
    ap.add_argument("--reference", default="published")
    args = ap.parse_args()

    path = Path(args.results) / "readout" / f"readout_{args.config}_all_shard0of1.json"
    if not path.exists():
        cands = sorted((Path(args.results) / "readout").glob("readout_*.json"))
        if not cands:
            raise SystemExit(f"no readout file under {path.parent}")
        path = cands[-1]
    records = json.loads(path.read_text())
    lenses = sorted({r["lens"] for r in records})
    layers = sorted({r["layer"] for r in records})

    out: list[str] = []
    w = out.append
    w(f"# Readout: does the corpus change what the lens can READ? ({args.config})\n")
    w(
        f"`unembed(J_bar h)` on the {len({r['prompt'] for r in records})} eval "
        f"prompts, at the argument and readout positions, layers {layers}. "
        "Scored as the rank of the prompt's own argument token over the whole "
        "vocabulary -- no string matching, no answer key, and invariant to "
        "rescaling of J_bar.\n"
    )

    w("## Overall\n")
    w("| lens | pass@1 | pass@5 | pass@10 | pass@50 | median rank |")
    w("|---|---|---|---|---|---|")
    for lens in lenses:
        sub = [r["rank"] for r in records if r["lens"] == lens]
        cells = [f"{sum(x < k for x in sub) / len(sub):.3f}" for k in (1, 5, 10, 50)]
        w(f"| {lens} | " + " | ".join(cells) + f" | {st.median(sub):.0f} |")
    w("")

    w("## By layer (pass@10)\n")
    w("| lens | " + " | ".join(f"L{ll}" for ll in layers) + " |")
    w("|---" * (len(layers) + 1) + "|")
    for lens in lenses:
        cells = []
        for ll in layers:
            sub = [r["rank"] for r in records if r["lens"] == lens and r["layer"] == ll]
            cells.append(f"{sum(x < 10 for x in sub) / len(sub):.2f}" if sub else "-")
        w(f"| {lens} | " + " | ".join(cells) + " |")
    w("")

    w("## By category (pass@10)\n")
    cats = sorted({r["category"] for r in records})
    w("| lens | " + " | ".join(cats) + " |")
    w("|---" * (len(cats) + 1) + "|")
    for lens in lenses:
        cells = []
        for cat in cats:
            sub = [
                r["rank"] for r in records if r["lens"] == lens and r["category"] == cat
            ]
            cells.append(f"{sum(x < 10 for x in sub) / len(sub):.2f}" if sub else "-")
        w(f"| {lens} | " + " | ".join(cells) + " |")
    w("")

    # --- paired comparison, same (prompt, layer, position) cells -----------
    w(f"## Paired against `{args.reference}`\n")
    w(
        "Same prompt, same layer, same position, so the two lenses are compared "
        "on identical inputs rather than on marginal distributions.\n"
    )
    key = lambda r: (r["prompt"], r["layer"], r["position"])  # noqa: E731
    ref = {key(r): r["rank"] for r in records if r["lens"] == args.reference}
    w("| lens | better | worse | tied | median rank delta |")
    w("|---|---|---|---|---|")
    for lens in lenses:
        if lens == args.reference:
            continue
        pairs = [
            (r["rank"], ref[key(r)])
            for r in records
            if r["lens"] == lens
            if key(r) in ref
        ]
        better = sum(a < b for a, b in pairs)
        worse = sum(a > b for a, b in pairs)
        tied = sum(a == b for a, b in pairs)
        delta = st.median([a - b for a, b in pairs])
        w(f"| {lens} | {better} | {worse} | {tied} | {delta:+.0f} |")
    w("")

    # --- audit -------------------------------------------------------------
    w("## Audit: what the readout actually returns\n")
    w(
        "A high rank is only a genuine miss if the top of the readout is "
        "unrelated. These are the top-5 decoded tokens at the last position of "
        "the middle layer, for the first prompts of each category.\n"
    )
    mid = layers[len(layers) // 2]
    w("| prompt | arg | lens | rank | top-5 readout |")
    w("|---|---|---|---|---|")
    seen: set[tuple[str, str]] = set()
    for r in records:
        if r["layer"] != mid or not r["is_last"]:
            continue
        if (r["category"], r["prompt"]) in seen and r["lens"] == lenses[0]:
            continue
        if (
            sum(1 for c, _ in seen if c == r["category"]) >= 2
            and r["lens"] == lenses[0]
        ):
            continue
        seen.add((r["category"], r["prompt"]))
        tops = " ".join(f"`{t}`" for t in r["top5"])
        w(f"| {r['prompt']} | {r['arg']} | {r['lens']} | {r['rank']} | {tops} |")
    w("")

    # --- the read/write comparison ----------------------------------------
    w("## Read against write\n")
    w(
        "Steering (write) on the same eval: gsm8k 11/39 on the math cell "
        "against the published lens's 1/39, and 35/105 against 10/105 on the "
        "held-out set. Compare the readout deltas above. If gsm8k reads no "
        "better than the published lens while writing far better, the corpus "
        "buys a better *actuator*, not a better map -- and 'towards the "
        "steerable J-space' is the correct framing rather than 'towards the "
        "true J-space'.\n"
    )

    text = "\n".join(out) + "\n"
    dest = Path(args.results) / f"c14_readout_{args.config}.md"
    dest.write_text(text)
    print(text)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()

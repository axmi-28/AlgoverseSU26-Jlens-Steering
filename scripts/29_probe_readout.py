"""Can a lens surface a concept the prompt never states?

Script 28 scored recovery of a concept that is present in the input, which is
the easy case and not what J-space is supposed to be about. This scores
`probe-swap`'s ``intermediate``: for "the language spoken in the country where
the Amazon River ends", the intermediate is *Brazil*, which appears nowhere in
the prompt and exists only as something the model computes en route.

Read the three columns together or not at all:

* ``intermediate`` -- the latent concept. The result of interest.
* ``answer`` -- what the model is about to say. Any competent map should show
  it, so it calibrates whether the lens is working at all on these prompts.
* ``logit_lens`` -- J = I, no transport. The question is never "is the rank
  low" but "is it lower than simply unembedding the residual", because near
  the end of a prompt the raw residual already encodes the upcoming answer.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from math import comb
from pathlib import Path

ORDER = ["logit_lens", "published", "wikitext_a", "gsm8k"]


def mcnemar(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    x = sum(p and not q for p, q in zip(a, b, strict=True))
    y = sum(q and not p for p, q in zip(a, b, strict=True))
    n = x + y
    if n == 0:
        return x, y, 1.0
    tail = sum(comb(n, i) for i in range(min(x, y) + 1))
    return x, y, min(1.0, 2 * tail / 2**n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="qwen3-8b")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    files = sorted(Path(args.results).glob("readout/probe_readout_*.json"))
    if not files:
        raise SystemExit("no probe_readout_*.json under results/readout")
    R = json.loads(files[-1].read_text())
    lenses = [x for x in ORDER if x in {r["lens"] for r in R}]
    layers = sorted({r["layer"] for r in R})

    out: list[str] = []
    w = out.append
    w(f"# Latent-concept readout on probe-swap ({args.config})\n")
    w(
        f"{len({r['name'] for r in R})} items, last 8 positions, layers "
        f"{layers}. Rank is over the full vocabulary, of the word's first "
        "token. **The intermediate never appears in the prompt.**\n"
    )

    for what in ("intermediate", "answer"):
        w(f"## {what}\n")
        w("| lens | top-1 | top-10 | top-100 | median rank |")
        w("|---|---|---|---|---|")
        for lens in lenses:
            sub = [r[f"rank_{what}"] for r in R if r["lens"] == lens]
            cells = [f"{sum(x < k for x in sub) / len(sub):.3f}" for k in (1, 10, 100)]
            w(f"| {lens} | " + " | ".join(cells) + f" | {st.median(sub):.0f} |")
        w("")

    # Best position per lens: a latent concept need not be at the last token.
    w("## Intermediate, by distance from the end (top-10)\n")
    ends = sorted({r["from_end"] for r in R})
    w("| lens | " + " | ".join(f"-{e}" for e in ends) + " |")
    w("|---" * (len(ends) + 1) + "|")
    for lens in lenses:
        cells = []
        for e in ends:
            sub = [
                r["rank_intermediate"]
                for r in R
                if r["lens"] == lens and r["from_end"] == e
            ]
            cells.append(f"{sum(x < 10 for x in sub) / len(sub):.2f}" if sub else "-")
        w(f"| {lens} | " + " | ".join(cells) + " |")
    w("")

    w("## Intermediate, by layer (top-10)\n")
    w("| lens | " + " | ".join(f"L{ll}" for ll in layers) + " |")
    w("|---" * (len(layers) + 1) + "|")
    for lens in lenses:
        cells = []
        for ll in layers:
            sub = [
                r["rank_intermediate"]
                for r in R
                if r["lens"] == lens and r["layer"] == ll
            ]
            cells.append(f"{sum(x < 10 for x in sub) / len(sub):.2f}" if sub else "-")
        w(f"| {lens} | " + " | ".join(cells) + " |")
    w("")

    # Paired on identical (item, layer, position) cells.
    w("## Paired comparisons on the intermediate\n")
    w("Same item, same layer, same position.\n")
    w("| A | B | A only | B only | p (exact McNemar) |")
    w("|---|---|---|---|---|")
    key = lambda r: (r["name"], r["layer"], r["from_end"])  # noqa: E731
    tables = {
        lens: {key(r): r["rank_intermediate"] < 10 for r in R if r["lens"] == lens}
        for lens in lenses
    }
    pairs = [
        ("gsm8k", "published"),
        ("gsm8k", "logit_lens"),
        ("published", "logit_lens"),
    ]
    for a, b in pairs:
        if a not in tables or b not in tables:
            continue
        shared = sorted(set(tables[a]) & set(tables[b]))
        x, y, p = mcnemar(
            [tables[a][k] for k in shared], [tables[b][k] for k in shared]
        )
        w(f"| {a} | {b} | {x} | {y} | {p:.4f} |")
    w("")

    w("## Audit: best intermediate recoveries under each lens\n")
    w(
        "If these are genuine the top-5 should contain the intermediate or its "
        "close variants, not punctuation.\n"
    )
    w("| lens | item | intermediate | rank | top-5 |")
    w("|---|---|---|---|---|")
    for lens in lenses:
        sub = sorted(
            (r for r in R if r["lens"] == lens), key=lambda r: r["rank_intermediate"]
        )[:3]
        for r in sub:
            tops = " ".join(f"`{t}`" for t in r["top5"])
            w(f"| {lens} | {r['name']} | - | {r['rank_intermediate']} | {tops} |")
    w("")

    text = "\n".join(out) + "\n"
    dest = Path(args.results) / f"c15_probe_readout_{args.config}.md"
    dest.write_text(text)
    print(text)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()

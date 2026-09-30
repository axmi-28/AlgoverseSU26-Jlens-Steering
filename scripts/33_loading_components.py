"""C20 - is the component result explained by workspace loading?

The J-Lens paper's own predictor of swap success is **workspace loading**: the
cosine between the clean residual stream and the concept's lens vector, over
the argument and readout positions. An obvious deflationary reading of C18 is
that dropping the diagonal term simply moves the write direction closer to
where the residual already lives, and loading -- not any horizon argument --
is doing the work.

That is testable, because loading is defined per arm: each arm has its own
lens vector for the same source argument, at the same positions and band. So
we can ask three things in order:

1. does loading differ by arm (is the deflationary story even available)?
2. does loading still predict success within an arm (does the paper's
   predictor replicate on these arms at all)?
3. does the per-prompt gain of ``_off`` over ``_full`` track the per-prompt
   change in loading? If the advantage were loading, prompts where ``_off``
   gains the most loading should be the prompts where it gains the most hits.

    python scripts/33_loading_components.py
"""

from __future__ import annotations

import glob
import importlib.util
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("c9", ROOT / "scripts/25_domain_causal.py")
c9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c9)
spec16 = importlib.util.spec_from_file_location("c16", ROOT / "scripts/30_energy_mechanism.py")
c16 = importlib.util.module_from_spec(spec16)
spec16.loader.exec_module(c16)

ARMS = [
    "gsm8k_off", "gsm8k_full", "gsm8k_diag",
    "wikitext_a_off", "wikitext_a_full", "wikitext_a_diag",
]
STRENGTH = 1.0


def main() -> None:
    paths = [p for p in glob.glob(str(ROOT / "results/c2/workspace_loading_qwen3-8b_*.json"))]
    if not paths:
        paths = glob.glob(str(ROOT / "results/**/workspace_loading_qwen3-8b_*.json"), recursive=True)
    loading = json.loads(max((Path(p) for p in paths), key=lambda p: p.stat().st_mtime).read_text())
    has_arms = {k: v for k, v in loading.items() if v.get("by_arm")}
    if not has_arms:
        raise SystemExit("loading file has no per-arm column; rerun `loading` with --arms")

    pattern = str(ROOT / "results/causal/steering_qwen3-8b_swap_comp_shard{shard}of{n}.json")
    probe = json.loads(Path(pattern.format(shard=0, n=2)).read_text())
    arms_all = [a for a in dict.fromkeys(r["arm"] for r in probe) if a != "baseline"]
    strengths = sorted({r["strength"] for r in probe if r["arm"] != "baseline"})
    rows = c9.load(pattern, 2, 1 + len(arms_all) * len(strengths))
    c9.regrade(rows)
    ok = {t.prompt for t, r in rows if r["arm"] == "baseline" and r["hit_re"]}

    # Per prompt-key hit rate for each arm, over that prompt's swap trials.
    hits: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        per: dict[str, list[bool]] = {}
        for t, r in rows:
            if r["arm"] != f"swap_{arm}" or r["strength"] != STRENGTH or t.prompt not in ok:
                continue
            per.setdefault(f"{t.category}/{t.func}/{t.source_arg}", []).append(r["hit_re"])
        hits[arm] = {k: st.mean(v) for k, v in per.items()}

    shared = sorted(set(hits[ARMS[0]]) & set(has_arms))
    lines = [
        "# C20 - workspace loading as a control on the component split (qwen3-8b)",
        "",
        f"{len(shared)} prompts with both a loading value and swap trials at "
        f"alpha={STRENGTH}. Loading is the paper's Sec. 3.4 quantity computed "
        "with **each arm's own** lens vector, same positions and same band.",
        "",
        "| arm | mean loading | hit rate | Spearman(loading, hits) | perm p |",
        "|---|---|---|---|---|",
    ]
    for arm in ARMS:
        load = [has_arms[k]["by_arm"][arm] for k in shared]
        hit = [hits[arm][k] for k in shared]
        lines.append(
            f"| {arm} | {st.mean(load):+.4f} | {st.mean(hit):.3f} | "
            f"{c16.spearman(load, hit):+.2f} | {c16.perm_p(load, hit):.3f} |"
        )
    published = [has_arms[k]["loading"] for k in shared]
    lines.append(
        f"| published (reference) | {st.mean(published):+.4f} | - | - | - |"
    )

    lines += ["", "## Does the gain track the change in loading?", ""]
    lines.append("| corpus | Spearman(d loading, d hits) | perm p | d loading | d hits |")
    lines.append("|---|---|---|---|---|")
    for corpus in ("gsm8k", "wikitext_a"):
        dl = [
            has_arms[k]["by_arm"][f"{corpus}_off"] - has_arms[k]["by_arm"][f"{corpus}_full"]
            for k in shared
        ]
        dh = [hits[f"{corpus}_off"][k] - hits[f"{corpus}_full"][k] for k in shared]
        lines.append(
            f"| {corpus} | {c16.spearman(dl, dh):+.2f} | {c16.perm_p(dl, dh):.3f} | "
            f"{st.mean(dl):+.4f} | {st.mean(dh):+.3f} |"
        )

    lines += ["", "## Matched-loading check", "",
              "Prompts split at the median of the *published* lens's loading -- a "
              "fixed covariate, identical for every arm, so the split cannot be "
              "gamed by an arm's own vector.", "",
              "| stratum | n | " + " | ".join(ARMS) + " |",
              "|---|---|" + "---|" * len(ARMS)]
    median = st.median(published)
    for name, keep in (("low loading", lambda v: v <= median), ("high loading", lambda v: v > median)):
        sub = [k for k in shared if keep(has_arms[k]["loading"])]
        cells = [f"{st.mean([hits[a][k] for k in sub]):.3f}" for a in ARMS]
        lines.append(f"| {name} | {len(sub)} | " + " | ".join(cells) + " |")

    out = ROOT / "results/c20_loading_components_qwen3-8b.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

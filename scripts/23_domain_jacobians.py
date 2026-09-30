"""C8 -- how much of the averaged Jacobian is the corpus it was fitted on?

Reads the pooled domain panel and compares each corpus's ``J_bar_domain``
against the others and against the published lens. Runs on a laptop: the pooled
artifacts are the only input, and no model is loaded.

**Read every number against the null.** ``wikitext_a`` and ``wikitext_b`` are
disjoint halves of the fitting corpus at the same N as every other group, so
whatever they score is what two samples of *the same* distribution produce at
this sample size. A domain pair is evidence only insofar as it exceeds that.
This is not a formality: none of the three measures used here has a chance
floor that can be assumed. Subspace alignment's floor is ``k/d`` (0.016 at
k=64, d=4096); CKA's floor is spectrum-dependent and ranges from 0.03 to 0.50
on synthetic matrices; relative Frobenius has no floor at all but is inflated
by pure gain differences that leave the map's structure intact.

Usage::

    python scripts/23_domain_jacobians.py --config qwen3-8b --n-per 16
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jsteer.analysis import linear_cka, subspace_alignment  # noqa: E402
from jsteer.config import REPO_ROOT  # noqa: E402
from jsteer.metrics import cosine, relative_frobenius  # noqa: E402
from jsteer.sweeps import JacobianSweepSpec  # noqa: E402

#: Any pair of these is two samples of the *same* distribution, so the set of
#: such pairs is the null: it is what this panel's measures return when there
#: is no domain difference at all, at exactly this N.
NULL_PREFIX = "wikitext"


def _is_null(a: str, b: str) -> bool:
    return a.startswith(NULL_PREFIX) and b.startswith(NULL_PREFIX)


def _load(namespace: str, results: Path) -> tuple[dict, dict, list]:
    """Merge one or more pooled panels, addressed by comma-separated namespace.

    Panels run separately are directly comparable when the model, layer grid,
    prompt length, position convention and ``k`` match, because each group mean
    is just an exact average over that corpus under identical settings. Merging
    them here avoids recomputing corpora that are already on disk -- adding two
    math arms to a six-corpus panel would otherwise mean redoing all eight.
    The target-word order is asserted identical across panels, since the
    pullback rows are indexed by it.
    """
    digests: dict = {}
    means: dict = {}
    targets: list | None = None
    for name in [n.strip() for n in namespace.split(",") if n.strip()]:
        digest_path = results / "rq1" / f"group_mean_digests_{name}.pt"
        means_path = results / "rq1" / f"group_means_full_{name}.pt"
        if not digest_path.exists():
            raise SystemExit(
                f"no pooled digests at {digest_path}\nrun pool_domains first"
            )
        blob = torch.load(digest_path, map_location="cpu", weights_only=False)
        if targets is not None and list(blob["targets"]) != targets:
            raise SystemExit(
                f"{name}: target order differs; pullbacks are not comparable"
            )
        targets = list(blob["targets"])
        digests |= blob["digests"]
        if means_path.exists():
            means |= torch.load(means_path, map_location="cpu", weights_only=False)[
                "means"
            ]
        else:
            print(f"! no full means at {means_path}; Frobenius and CKA skipped for it")
    return digests, means, targets or []


def _pair_stats(
    digests: dict, means: dict, a: str, b: str, layer: int
) -> dict[str, float]:
    da, db = digests[f"{a}|{layer}"], digests[f"{b}|{layer}"]
    out = {
        "left": subspace_alignment(da["left"], db["left"]),
        "right": subspace_alignment(da["right"], db["right"]),
        # Per-token write directions: J_bar^T u_y for each of the 16 args. This
        # is the functional measure -- two matrices can agree on their dominant
        # subspace while disagreeing on the specific covector a steer uses.
        "pullback_cos": float(cosine(da["pullbacks"], db["pullbacks"]).mean().item()),
    }
    ka, kb = f"{a}|{layer}", f"{b}|{layer}"
    if ka in means and kb in means:
        out["rel_fro"] = relative_frobenius(means[ka], means[kb])
        out["cka"] = linear_cka(means[ka], means[kb])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="qwen3-8b")
    parser.add_argument("--n-per", type=int, default=16)
    parser.add_argument("--min-tokens", type=int, default=128)
    parser.add_argument(
        "--corpora",
        default="wikitext_a,wikitext_b,wikitext_c,wikitext_d,openwebmath,ultrachat",
    )
    parser.add_argument("--results", default=str(REPO_ROOT / "results"))
    parser.add_argument(
        "--namespace",
        default="",
        help=(
            "Analyse this pooled run directly instead of deriving the "
            "namespace from --config/--corpora/--n-per. Needed only to re-read "
            "a pool whose corpus definitions have since changed -- the derived "
            "namespace is a hash of those definitions, so editing one "
            "correctly makes the old artifacts unaddressable."
        ),
    )
    args = parser.parse_args()

    names = [n.strip() for n in args.corpora.split(",") if n.strip()]
    namespace = (
        args.namespace
        or JacobianSweepSpec(
            config_name=args.config,
            domains=names,
            n_per=args.n_per,
            min_tokens=args.min_tokens,
        ).namespace
    )
    digests, means, targets = _load(namespace, Path(args.results))

    layers = sorted({d["layer"] for d in digests.values()})
    groups = sorted({d["group"] for d in digests.values()})
    counts = {g: {digests[f"{g}|{l}"]["n"] for l in layers} for g in groups}

    lines: list[str] = []
    w = lines.append
    w(f"# C8 - domain-conditional averaged Jacobians ({args.config})\n")
    w(
        f"Panel `{namespace}`: {len(groups)} corpora x {args.n_per} prompts, "
        f"each truncated to {args.min_tokens} tokens and averaged under the "
        f"lens's own position convention. Layers {layers}.\n"
    )
    # The count check the repo insists on: a group averaged over fewer prompts
    # than the panel specifies is a resumed shard whose skipped prompts never
    # entered a saved sum, and it looks exactly like a normal result.
    w("| corpus | prompts averaged (per layer) |")
    w("|---|---|")
    for g in groups:
        flag = "" if counts[g] == {args.n_per} else "  **<- MISMATCH**"
        w(f"| {g} | {sorted(counts[g])}{flag} |")
    w("")
    if any(counts[g] != {args.n_per} for g in groups):
        w(
            "**A corpus above was averaged over the wrong number of prompts.** "
            "Every comparison below is between unequal samples; fix the sweep "
            "before reading any of it.\n"
        )

    pairs = list(itertools.combinations(groups, 2))
    metrics = [
        ("pullback_cos", "pullback cos", "per-token write direction, 16 args"),
        ("left", "left subspace", "output side, top-k"),
        ("right", "right subspace", "input side, top-k"),
        ("cka", "CKA", "scale-invariant"),
        ("rel_fro", "rel Frobenius", "scale-sensitive"),
    ]

    for key, label, note in metrics:
        sample = _pair_stats(digests, means, *pairs[0], layers[0])
        if key not in sample:
            continue
        w(f"## {label} ({note})\n")

        means_by_pair = {}
        for a, b in pairs:
            vals = [_pair_stats(digests, means, a, b, l)[key] for l in layers]
            means_by_pair[(a, b)] = (vals, sum(vals) / len(vals))

        null_means = sorted(
            m for (a, b), (_, m) in means_by_pair.items() if _is_null(a, b)
        )
        lo, hi = null_means[0], null_means[-1]

        w("| pair | " + " | ".join(f"L{l}" for l in layers) + " | mean | vs null |")
        w("|---" * (len(layers) + 3) + "|")
        # Null pairs first, then the comparisons of interest.
        for a, b in sorted(pairs, key=lambda p: (not _is_null(*p), p)):
            vals, mean = means_by_pair[(a, b)]
            cells = " | ".join(f"{v:.3f}" for v in vals)
            if _is_null(a, b):
                verdict = "*null*"
            elif mean < lo:
                verdict = "**below null range**"
            elif mean > hi:
                verdict = "**above null range**"
            else:
                verdict = "inside null range"
            w(f"| {a} vs {b} | {cells} | {mean:.3f} | {verdict} |")
        w("")
        w(
            f"Null range across the {len(null_means)} same-distribution "
            f"wikitext pairs: **{lo:.3f} - {hi:.3f}** "
            f"(median {null_means[len(null_means) // 2]:.3f}). A domain pair is "
            "evidence only if it falls outside that range; a pair inside it is "
            "indistinguishable from two samples of one corpus at this N.\n"
        )

    # ----------------------------------------------------------------
    # cross-application: read one corpus's directions through another's map
    # ----------------------------------------------------------------
    if means:
        w("## Cross-application (one corpus's directions through another's map)\n")
        w(
            "The pairwise measures above are symmetric, but the question "
            '"does the wikitext lens still work on math?" is not. This one '
            "is directional: take the top-k **right** singular vectors of "
            "`J_bar_A` -- the input directions A's own map is most sensitive "
            "to -- push each through both maps, and measure the cosine "
            "between the two images, weighted by A's singular values so that "
            "the directions A actually cares about dominate.\n"
        )
        w(
            "A high value means B's map sends A's preferred directions where "
            "A does, so B's averaged Jacobian is a usable stand-in **for the "
            "directions A cares about** -- which is what a lens transferred "
            "across domains has to do. It is *not* symmetric: B can reproduce "
            "A's action on A's subspace while doing something entirely "
            "different on its own.\n"
        )
        w(
            "| source of directions (A) | applied through (B) | "
            + " | ".join(f"L{l}" for l in layers)
            + " | mean |"
        )
        w("|---" * (len(layers) + 3) + "|")

        rows: list[tuple[str, str, list[float], float]] = []
        for a in groups:
            for b in groups:
                if a == b:
                    continue
                vals = []
                for layer in layers:
                    ka, kb = f"{a}|{layer}", f"{b}|{layer}"
                    if ka not in means or kb not in means:
                        continue
                    basis = digests[ka]["right"].float()  # [k, d]
                    weights = digests[ka]["singular_values"].float()
                    image_a = basis @ means[ka].float().T
                    image_b = basis @ means[kb].float().T
                    cos = cosine(image_a, image_b)
                    vals.append(
                        float(
                            (
                                (cos * weights).sum() / weights.sum().clamp_min(1e-12)
                            ).item()
                        )
                    )
                if vals:
                    rows.append((a, b, vals, sum(vals) / len(vals)))

        null_cross = sorted(m for a, b, _, m in rows if _is_null(a, b))
        for a, b, vals, mean in sorted(
            rows, key=lambda r: (not _is_null(r[0], r[1]), -r[3])
        ):
            cells = " | ".join(f"{v:.3f}" for v in vals)
            tag = "  *(null)*" if _is_null(a, b) else ""
            w(f"| {a}{tag} | {b} | {cells} | {mean:.3f} |")
        w("")
        if null_cross:
            w(
                f"Null range over same-distribution ordered pairs: "
                f"**{null_cross[0]:.3f} - {null_cross[-1]:.3f}**. Read every "
                "row against that, not against 1.0.\n"
            )

    print("\n".join(lines))
    out = Path(args.results) / f"c8_domains_{args.config}.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

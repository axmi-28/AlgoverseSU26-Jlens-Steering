"""Figure for C8: domain dependence of the averaged Jacobian, against the null.

Two panels, one per measure, because the whole finding is that they disagree:
the per-token write directions converge across corpora by the end of the band
while the principal subspaces do not. A single panel would hide that.

Only the pooled *digests* are needed (top-k bases and the 16 arg pullbacks), not
the 2.8 GB of full mean matrices.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jsteer.analysis import subspace_alignment  # noqa: E402
from jsteer.config import REPO_ROOT  # noqa: E402
from jsteer.metrics import cosine  # noqa: E402

# Validated with the dataviz palette checker (light surface, categorical):
# all of lightness band, chroma floor, CVD separation, normal-vision floor and
# contrast pass, worst adjacent pair dE 19.9 under protanopia.
BLUE, CONTEXT = "#1b6ca8", "#98a2aa"
NULL_FILL, INK, MUTED = "#d8d5d1", "#26221f", "#6b645d"
SURFACE = "#fcfcfb"

WIKI = ("wikitext_a", "wikitext_b", "wikitext_c", "wikitext_d")
NS = "qwen3-8b_dom6x32_7e3215f2"


def _measure(digests, a, b, layer, kind):
    da, db = digests[f"{a}|{layer}"], digests[f"{b}|{layer}"]
    if kind == "pullback":
        return float(cosine(da["pullbacks"], db["pullbacks"]).mean().item())
    return subspace_alignment(da["right"], db["right"])


def main() -> None:
    blob = torch.load(
        REPO_ROOT / "results" / "rq1" / f"group_mean_digests_{NS}.pt",
        map_location="cpu",
        weights_only=False,
    )
    digests = blob["digests"]
    layers = sorted({d["layer"] for d in digests.values()})

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.4))
    fig.patch.set_facecolor(SURFACE)

    # Each panel carries its own y-label. A shared "agreement" axis was wrong:
    # the left panel is a plain cosine between two vectors, the right is the
    # mean squared cosine of 64 principal angles between two subspaces. Same
    # 0-1 range, different quantities, and one label cannot mean both.
    panels = [
        (
            "pullback",
            "Write directions",
            "cosine between the write vectors the two lenses\n"
            "give for the same token   ($\\bar{J}^{\\top}u_y$, 16 arg tokens)",
        ),
        (
            "right",
            "Input-side subspace",
            "overlap of the 64 input directions each lens is\n"
            "most sensitive to   (1 = identical, 0.02 = chance)",
        ),
    ]

    for ax, (kind, title, ylabel) in zip(axes, panels, strict=True):
        ax.set_facecolor(SURFACE)

        null = [
            [_measure(digests, a, b, l, kind) for l in layers]
            for i, a in enumerate(WIKI)
            for b in WIKI[i + 1 :]
        ]
        lo = [min(v[j] for v in null) for j in range(len(layers))]
        hi = [max(v[j] for v in null) for j in range(len(layers))]
        ax.fill_between(layers, lo, hi, color=NULL_FILL, zorder=1)

        # Three visual ranks, matching how the comparisons actually rank. The
        # math-vs-chat contrast is the question the panel exists to answer --
        # do two genuinely different domains produce different averaged
        # Jacobians? -- so it is the only saturated line. wikitext is the
        # control at two levels at once: the grey band is wikitext against
        # itself, and the thin dashed lines are each domain against wikitext.
        # Both recede.
        for i, dom in enumerate(("openwebmath", "ultrachat")):
            vals = [
                sum(_measure(digests, dom, w, l, kind) for w in WIKI) / 4
                for l in layers
            ]
            ax.plot(
                layers,
                vals,
                color=CONTEXT,
                lw=1.3,
                ls=(0, (5, 3)),
                zorder=2,
                clip_on=False,
                label="each domain vs wikitext (control)" if i == 0 else None,
            )

        headline = [
            _measure(digests, "openwebmath", "ultrachat", l, kind) for l in layers
        ]
        ax.plot(
            layers,
            headline,
            color=BLUE,
            lw=2.6,
            marker="o",
            ms=6.5,
            label="open-web-math vs UltraChat",
            zorder=4,
            clip_on=False,
        )
        series = [(None, headline, BLUE)]

        # The null band is the reference, not a fourth series, so it is
        # labelled in place. Sat directly under its own left end: a leader line
        # to a band this wide points at nothing in particular.
        ax.text(
            layers[0] - 0.3,
            lo[0] - 0.035,
            "null: wikitext vs wikitext\n(6 same-corpus pairs)",
            fontsize=8.5,
            color=MUTED,
            ha="left",
            va="top",
            linespacing=1.35,
        )

        gap_first = (lo[0] + hi[0]) / 2 - series[0][1][0]
        gap_last = (lo[-1] + hi[-1]) / 2 - series[0][1][-1]
        ax.set_title(title, fontsize=11.5, color=INK, pad=10)
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED, linespacing=1.5)
        ax.text(
            0.975,
            0.05,
            f"gap to null:  L13 {gap_first:.2f}  →  L31 {gap_last:.2f}",
            transform=ax.transAxes,
            ha="right",
            fontsize=9.5,
            color=INK,
            bbox=dict(boxstyle="round,pad=0.42", fc="#ffffff", ec=NULL_FILL, lw=1),
        )

        if kind == "pullback":
            secondary = ax.secondary_yaxis(
                "right",
                # numpy, not math: matplotlib feeds these arrays (sometimes
                # 0-d), and the clip keeps arccos defined at the axis limits.
                functions=(
                    lambda c: np.degrees(np.arccos(np.clip(c, -1.0, 1.0))),
                    lambda d: np.cos(np.radians(d)),
                ),
            )
            secondary.set_ylabel(
                "angle between the two write vectors", fontsize=9, color=MUTED
            )
            secondary.set_yticks([0, 15, 30, 45, 60])
            secondary.set_yticklabels(["0°", "15°", "30°", "45°", "60°"])
            secondary.tick_params(colors=MUTED, labelsize=9)
            secondary.spines["right"].set_color(MUTED)

        ax.set_xlabel("layer (workspace band, Qwen3-8B)", fontsize=9.5, color=MUTED)
        ax.set_xticks(layers)
        ax.set_ylim(0.40, 1.02)
        ax.grid(axis="y", color=NULL_FILL, lw=0.6, alpha=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=9.5,
        labelcolor=INK,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.012),
    )

    fig.suptitle(
        "Math text and chat text give measurably different averaged Jacobians",
        fontsize=12.5,
        color=INK,
        y=0.995,
        x=0.5,
    )
    fig.text(
        0.5,
        0.905,
        "Qwen3-8B, 32 prompts per corpus, 128 tokens each. wikitext is the "
        "control throughout — the corpus the shipped lens was fitted on.",
        ha="center",
        fontsize=9.5,
        color=MUTED,
    )
    # Extra gutter: the left panel carries a right-hand degrees axis, which
    # would otherwise collide with the right panel's y-label.
    fig.tight_layout(rect=(0, 0.075, 1, 0.885), w_pad=4.5)

    out = REPO_ROOT / "results" / "figures" / f"fig8_domains_{NS}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

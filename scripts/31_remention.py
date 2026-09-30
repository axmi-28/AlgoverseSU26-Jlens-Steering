"""C17 - does the corpus teach the lens that context tokens come back?

C16 ruled out the Jacobian energy profile as the mechanism: questions-only
GSM8K has nearly GSM8K's profile and none of its steering. The hypothesis this
tests instead is about *text*, not geometry. J_bar averages how position t
moves the outputs at t' >= t, so a corpus in which a token at t is emitted
again later trains a lens that reads "this token will recur" off h_t -- which
is exactly present-token readout, and exactly the write a swap needs. GSM8K
solutions restate the question's entities and quantities at every step.

Measured on the exact documents the energy run used (``energy_corpora.json``),
tokenized and truncated to 128 as the fit does, over the fitted positions
t >= 16. Content tokens exclude the 200 commonest tokens of the pooled panel
(each corpus weighted equally) and whitespace/punctuation, so a template corpus
cannot score by repeating "plus" and "the".

Prediction registered before the run: content re-mention ranks gsm8k_sol well
above gsm8k_q, and predicts swap success better than diagonal share did.

    python scripts/31_remention.py
"""

from __future__ import annotations

import collections
import importlib.util
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("c16", ROOT / "scripts/30_energy_mechanism.py")
c16 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c16)

MAX_LEN, SKIP_FIRST, TOP_COMMON = 128, 16, 200
#: C10 held-out swaps for the ablation lenses, of 105 (implementation.md).
HELDOUT = {**c16.HELDOUT105, "gsm8k_shuffled": 27, "gsm8k_noent": 34,
           "gsm8k_nonum": 35, "gsm8k_noeq": 33}
#: Read side, from C14/C16 (results/readout/*_g63451680_*.json).
SHORT_HORIZON = {"gsm8k": 0.642, "openwebmath": 0.363, "arith_words": 0.310,
                 "wikitext_a": 0.288, "ultrachat": 0.189}
ICR = {"openwebmath": 7.47, "wikitext_a": 7.04, "ultrachat": 6.98,
       "gsm8k": 5.98, "arith_words": 5.72}


def main() -> None:
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    docs = json.loads((ROOT / "results/energy/energy_corpora.json").read_text())
    ids = {c: [tok.encode(t, add_special_tokens=False)[:MAX_LEN] for t in texts]
           for c, texts in docs.items()}
    short = {c: [len(x) for x in v if len(x) < MAX_LEN] for c, v in ids.items()}
    assert not any(short.values()), f"documents under {MAX_LEN} tokens: {short}"

    # Commonest tokens of the pooled panel, each corpus weighted equally.
    freq: collections.Counter = collections.Counter()
    for v in ids.values():
        counts = collections.Counter(t for d in v for t in d)
        total = sum(counts.values())
        for t, n in counts.items():
            freq[t] += n / total
    common = {t for t, _ in freq.most_common(TOP_COMMON)}

    def is_content(t: int) -> bool:
        s = tok.decode([t]).strip()
        return t not in common and any(ch.isalnum() for ch in s)

    # The GSM8K ablations (C10), scored against the SAME common-token set --
    # added after it is computed so they cannot shift what counts as content.
    # Rebuilt locally with the fit's own ``load_domain_corpora`` call, which is
    # deterministic in the spec, so these are the documents those lenses saw.
    extra_path = ROOT / "results/energy/ablation_corpora.json"
    if extra_path.exists():
        for c, texts in json.loads(extra_path.read_text()).items():
            ids[c] = [tok.encode(t, add_special_tokens=False)[:MAX_LEN] for t in texts]
            assert len(ids[c]) == 32, (c, len(ids[c]))
            assert all(len(x) == MAX_LEN for x in ids[c]), f"{c}: short documents"

    energy = c16.energy_table(c16.latest("energy/energy_qwen3-*_energy_*.json", exclude="0b90e7d2"))
    stats: dict[str, dict] = {}
    for c, v in ids.items():
        rec_all, rec_content, gaps, content_frac = [], [], [], []
        for d in v:
            later: dict[int, list[int]] = collections.defaultdict(list)
            for i, t in enumerate(d):
                later[t].append(i)
            src = range(SKIP_FIRST, len(d))
            nxt = {i: next((j for j in later[d[i]] if j > i), None) for i in src}
            rec_all.append(sum(nxt[i] is not None for i in src) / len(src))
            content = [i for i in src if is_content(d[i])]
            content_frac.append(len(content) / len(src))
            if content:
                rec_content.append(sum(nxt[i] is not None for i in content) / len(content))
                gaps += [nxt[i] - i for i in content if nxt[i] is not None]
        stats[c] = {
            "all": statistics.mean(rec_all),
            # A template corpus can have NO content tokens: its whole
            # vocabulary is among the panel's commonest. Undefined, not zero.
            "content": statistics.mean(rec_content) if rec_content else float("nan"),
            "gap": statistics.median(gaps) if gaps else float("nan"),
            "content_frac": statistics.mean(content_frac),
            "diag": energy[c]["diag"] if c in energy else float("nan"),
        }

    lines = [
        "# C17 - content re-mention vs lens behaviour (qwen3-8b)",
        "",
        "Re-mention = fraction of fitted positions (t >= 16 of 128) whose token "
        "appears again later in the same document. Content excludes the "
        f"{TOP_COMMON} commonest panel tokens and punctuation. Same 32 "
        "documents per corpus as the energy run and the lens fit.",
        "",
        "| corpus | content re-mention | all-token re-mention | median gap | "
        "content share | diag share | math /39 | held-out /105 | present-token | ICR |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in sorted(stats, key=lambda c: -(stats[c]["content"] if stats[c]["content"] == stats[c]["content"] else -1)):
        s = stats[c]
        content = "n/a" if s["content"] != s["content"] else f"{s['content']:.3f}"
        lines.append(
            f"| {c} | {content} | {s['all']:.3f} | {s['gap']:.0f} | "
            f"{s['content_frac']:.3f} | {s['diag']:.3f} | {c16.MATH39.get(c, '')} | "
            f"{HELDOUT.get(c, '')} | {SHORT_HORIZON.get(c, '')} | {ICR.get(c, '')} |"
        )
    lines += ["", "## Predictive power, against C16's diagonal share", "",
              "| predictor | outcome | n | Spearman | perm p |", "|---|---|---|---|---|"]
    for metric in ("content", "all", "diag"):
        for label, table in (("math /39", c16.MATH39), ("held-out /105", HELDOUT),
                             ("present-token", SHORT_HORIZON), ("ICR", ICR)):
            shared = [c for c in stats if c in table and stats[c][metric] == stats[c][metric]]
            xs, ys = [stats[c][metric] for c in shared], [table[c] for c in shared]
            lines.append(f"| {metric} | {label} | {len(shared)} | "
                         f"{c16.spearman(xs, ys):+.2f} | {c16.perm_p(xs, ys):.3f} |")
    s_sol, s_q = stats["gsm8k_sol"]["content"], stats["gsm8k_q"]["content"]
    lines += ["", f"Registered check: gsm8k_sol {s_sol:.3f} vs gsm8k_q {s_q:.3f} -> "
              f"{'HOLDS' if s_sol > s_q else 'FAILS'}."]
    if "gsm8k_noent" in stats:
        s_ne, s_wiki = stats["gsm8k_noent"]["content"], stats["wikitext_a"]["content"]
        lines += [f"Registered check: gsm8k_noent {s_ne:.3f} (steers 34/105) vs wikitext_a "
                  f"{s_wiki:.3f} -> {'CONSISTENT' if s_ne >= 0.25 else 'WOUNDED' if s_ne <= 0.15 else 'AMBIGUOUS'}."]
    out = ROOT / "results/c17_remention_qwen3-8b.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

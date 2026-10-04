"""C40: build, screen, freeze and run the expanded latent-intermediate panel.

    python scripts/57_latent_panel.py generate          # CPU: write candidates
    python scripts/57_latent_panel.py screen            # GPU: clean + one-hop answers
    python scripts/57_latent_panel.py freeze            # CPU: write the frozen panel
    python scripts/57_latent_panel.py run               # GPU: C22 arms, both lenses
    python scripts/57_latent_panel.py report            # CPU: tables

Adding items is a fact-table edit (data/heldout/latent-kb.json) followed by
``generate``; every later stage is keyed by the hash of what it reads.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch

from jsteer.latent_panel import (
    CANDIDATES_PATH,
    KB_PATH,
    PANEL_PATH,
    LatentControlSpec,
    LatentScreenSpec,
    freeze_panel,
    generate_candidates,
)
from jsteer.write_controls import ARMS, file_hash

ROOT = Path(__file__).resolve().parents[1]
DIGEST = "group_mean_digests_qwen3-8b_dom2x32_968c0ba0_pb_t89d64f82_comp_s4.pt"
LOCAL = ROOT / "results" / "latent"
MODAL = shutil.which("modal") or str(ROOT / ".venv/bin/modal")
CORPORA = ("gsm8k", "wikitext_a")


def gpu_fn(timeout):
    import modal

    return modal.Function.from_name("jsteer", "gpu_task").with_options(
        gpu="A100-80GB", max_containers=2, retries=0, timeout=timeout
    )


def download():
    LOCAL.mkdir(parents=True, exist_ok=True)
    subprocess.run([MODAL, "volume", "get", "--force", "jsteer-results", "/latent", str(LOCAL.parent)], check=True)


def generate():
    kb = json.loads(KB_PATH.read_text())
    covered = set(torch.load(ROOT / "results/rq1" / DIGEST, map_location="cpu", weights_only=False)["targets"])
    items = generate_candidates(kb, covered)
    # Independence from the development panels, checked rather than asserted.
    upstream = json.loads((ROOT / "data/jlens-upstream/experiments/probe-swap.json").read_text())["items"]
    concept = json.loads((ROOT / "data/heldout/concept-use.json").read_text())
    concept_clues = {c for t in concept["types"].values() for e in t["entities"].values() for c in e["clues"]}
    reused = [i["name"] for i in items if i["prompt"] in {u["prompt"] for u in upstream} or i["clue"] in concept_clues]
    if reused:
        raise SystemExit(f"Items reuse development prompts or clues: {reused}")
    blob = {"kb_sha256": file_hash(KB_PATH), "items": items}
    blob["sha256"] = hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()
    CANDIDATES_PATH.write_text(json.dumps(blob, indent=1, ensure_ascii=False))
    print(f"{len(items)} candidates -> {CANDIDATES_PATH.relative_to(ROOT)}")
    print(dict(collections.Counter(i["type"] for i in items)))


def screen():
    spec = LatentScreenSpec(candidates_path=f"/root/repo/data/heldout/{CANDIDATES_PATH.name}", out_dir="/results/latent")
    print(json.dumps(gpu_fn(3600).remote("latent_screen", asdict(spec)), indent=2))
    download()


def screen_file():
    cands = json.loads(CANDIDATES_PATH.read_text())
    sha = file_hash(CANDIDATES_PATH)
    matches = [p for p in LOCAL.glob("screen_*.json") if json.loads(p.read_text())["manifest"]["candidates_sha256"] == sha]
    if len(matches) != 1:
        raise SystemExit(f"Expected one screen for candidates {sha[:12]}, found {len(matches)}")
    return cands, matches[0]


def freeze():
    cands, path = screen_file()
    screen_blob = json.loads(path.read_text())
    if len(screen_blob["records"]) != len(cands["items"]):
        raise SystemExit("Screen record count does not match candidates")
    if PANEL_PATH.exists():
        raise SystemExit(f"{PANEL_PATH.name} already exists; a frozen panel is never overwritten")
    panel = freeze_panel(cands, screen_blob, file_hash(path))
    PANEL_PATH.write_text(json.dumps(panel, indent=1, ensure_ascii=False))
    items = panel["items"]
    print(f"froze {len(items)}/{len(cands['items'])} -> {PANEL_PATH.relative_to(ROOT)} sha256={file_hash(PANEL_PATH)}")
    print(dict(collections.Counter(i["type"] for i in items)))
    print(f"one-hop target known: {sum(i['one_hop_target_ok'] for i in items)}/{len(items)}")


def run():
    specs = [
        LatentControlSpec(
            directions_path=f"/results/rq1/{DIGEST}",
            panel_path=f"/root/repo/data/heldout/{PANEL_PATH.name}",
            corpus=c,
            out_dir="/results/latent",
        )
        for c in CORPORA
    ]
    fn = gpu_fn(3600)
    calls = [fn.spawn("latent_controls", asdict(s)) for s in specs]
    for c in calls:
        print(f"call_id={c.object_id}", flush=True)
    for c in calls:
        print(json.dumps(c.get(), indent=2))
    download()


def _load_runs():
    panel_sha = file_hash(PANEL_PATH)
    n = len(json.loads(PANEL_PATH.read_text())["items"])
    runs = {}
    for p in LOCAL.glob("latent_controls_*.json"):
        blob = json.loads(p.read_text())
        m = blob["manifest"]
        if m["panel_sha256"] != panel_sha:
            continue
        corpus = m["spec"]["corpus"]
        if corpus in runs:
            raise SystemExit(f"Two runs for {corpus} on this panel")
        if blob["status"] != "complete" or len(blob["records"]) != n * (1 + 2 * len(ARMS)):
            raise SystemExit(f"{p.name}: status {blob['status']}, {len(blob['records'])} records for {n} items")
        runs[corpus] = blob["records"]
    return runs


def paired(records, arm_a, arm_b, strength, key):
    a = {r["name"]: r[key] for r in records if r["arm"] == arm_a and r["strength"] == strength}
    b = {r["name"]: r[key] for r in records if r["arm"] == arm_b and r["strength"] == strength}
    wins = sum(b[k] and not a[k] for k in a)
    losses = sum(a[k] and not b[k] for k in a)
    return sum(a.values()), sum(b.values()), wins, losses, len(a)


def sign_test(wins, losses):
    from math import comb

    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def cluster_ci(records, strength, key, draws=10000, seed=42101):
    """Bootstrap of the (off_matched - full) difference, resampling intermediates."""
    per = collections.defaultdict(lambda: [0, 0])
    for r in records:
        if r["strength"] == strength and r["arm"] in ("swap_full", "swap_off_matched"):
            per[r["intermediate"]][r["arm"] == "swap_off_matched"] += r[key]
    counts = collections.Counter(r["intermediate"] for r in records if r["arm"] == "swap_full" and r["strength"] == strength)
    ents = sorted(per)
    rng = random.Random(seed)
    diffs = []
    for _ in range(draws):
        s = [rng.choice(ents) for _ in ents]
        n = sum(counts[e] for e in s)
        diffs.append((sum(per[e][1] for e in s) - sum(per[e][0] for e in s)) / n)
    diffs.sort()
    return 100 * diffs[int(0.025 * draws)], 100 * diffs[int(0.975 * draws)]


def report():
    runs = _load_runs()
    items = json.loads(PANEL_PATH.read_text())["items"]
    feasible = {i["name"] for i in items if i["one_hop_target_ok"]}
    lines = [
        "# C40: expanded latent-intermediate panel",
        "",
        f"Frozen panel: {len(items)} items, sha256 `{file_hash(PANEL_PATH)[:16]}`; {len(feasible)} with the replacement's one-hop answer known.",
        "",
        "Primary contrast: full lens vs norm-matched off-diagonal lens (`swap_full` -> `swap_off_matched`).",
        "",
        "| Lens | Strength | Subset | Correct answers | Paired wins/losses | Sign-test p | 95% CI (pp) | Name leakage | Leak wins/losses |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for corpus in CORPORA:
        if corpus not in runs:
            continue
        for s in (1.0, 0.5):
            for label, keep in (("all", None), ("one-hop known", feasible)):
                rs = [r for r in runs[corpus] if keep is None or r["name"] in keep]
                a, b, w, l, n = paired(rs, "swap_full", "swap_off_matched", s, "hit_swap_answer")
                la, lb, lw, ll, _ = paired(rs, "swap_full", "swap_off_matched", s, "emitted_substring")
                lo, hi = cluster_ci(rs, s, "hit_swap_answer")
                lines.append(
                    f"| {corpus} | {s} | {label} | {a}/{n} -> {b}/{n} | {w}/{l} | {sign_test(w, l):.3g} | {lo:+.1f} to {hi:+.1f} | {la}/{n} -> {lb}/{n} | {lw}/{ll} |"
                )
    lines += ["", "## All arms (all items)", "", "| Lens | Arm | Strength | Correct | Leakage |", "|---|---|---|---|---|"]
    for corpus in CORPORA:
        for arm in ("baseline",) + ARMS:
            for s in ([0.0] if arm == "baseline" else [0.5, 1.0]):
                rs = [r for r in runs.get(corpus, []) if r["arm"] == arm and r["strength"] == s]
                if rs:
                    lines.append(f"| {corpus} | {arm} | {s} | {sum(r['hit_swap_answer'] for r in rs)}/{len(rs)} | {sum(r['emitted_substring'] for r in rs)}/{len(rs)} |")
    lines += ["", "## By category (strength 1, correct: full -> off-matched)", "", "| Category | n | GSM8K | WikiText |", "|---|---|---|---|"]
    cats = sorted({i["category"] for i in items})
    for c in cats:
        cells = []
        for corpus in CORPORA:
            rs = [r for r in runs.get(corpus, []) if r["category"] == c]
            a, b, _, _, n = paired(rs, "swap_full", "swap_off_matched", 1.0, "hit_swap_answer")
            cells.append(f"{a} -> {b}")
        lines.append(f"| {c} | {n} | {cells[0]} | {cells[1]} |")
    out = ROOT / "results" / "c40_latent_panel_qwen3-8b.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("generate", "screen", "freeze", "run", "report", "download"))
    args = ap.parse_args()
    {"generate": generate, "screen": screen, "freeze": freeze, "run": run, "report": report, "download": download}[args.stage]()


if __name__ == "__main__":
    main()

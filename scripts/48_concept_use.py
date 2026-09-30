"""C32-C35 launcher: presets pin each stage's protocol; nothing here is analysis.

    python scripts/48_concept_use.py plan  <preset>          # print specs, no GPU
    python scripts/48_concept_use.py launch <preset>         # spawn on the deployed app
    python scripts/48_concept_use.py fetch                   # pull /results/concept
    python scripts/48_concept_use.py freeze                  # clean shards -> eligibility file (+ upload)

Every preset is written here before its run and not edited after it; a changed
protocol is a new preset name. The spec's manifest (all scientific inputs,
source hash, input-file hashes) names the output file, so a changed input can
never be served a stale result.

Presets that depend on a frozen dev selection (``c34test``) read it from
``results/concept/frozen_c34.json``, written by ``scripts/49_concept_report.py
select``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jsteer.concept_use import BAND, ConceptSpec, freeze_eligibility  # noqa: E402

LOCAL = ROOT / "results"
REMOTE = "/results"
COMP = "rq1/group_mean_digests_qwen3-8b_dom2x32_968c0ba0_pb_t89d64f82_comp_s4.pt"
MODAL = shutil.which("modal") or "modal"

LENS = [f"{c}_{p}" for c in ("gsm8k", "wikitext_a") for p in ("full", "off", "proj")]
LAYER_SETS = {f"L{l}": [l] for l in BAND} | {
    "early": [13, 16, 19],
    "mid": [19, 22, 25],
    "late": [25, 28, 31],
    "all": list(BAND),
}


def eligibility_rel() -> str:
    p = LOCAL / "concept" / "eligibility.json"
    if not p.exists():
        raise SystemExit("no frozen eligibility; run `freeze` first")
    return "concept/eligibility.json"


def baselines_rel() -> str:
    hits = sorted((LOCAL / "concept").glob("baselines_fit_*.pt"))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one baselines digest, found {hits}")
    return f"concept/{hits[0].name}"


def preset(name: str) -> tuple[dict, int]:
    """(ConceptSpec kwargs with volume-relative paths, n_shards)."""
    if name == "clean":
        return dict(stage="clean", fillers=[0, 1, 2, 4, 8]), 2
    if name == "c32":
        return dict(
            stage="swap", label="c32", digest_paths=[COMP], panel_path=eligibility_rel(),
            swap_arms=["full", "off_m", "diag_m", "proj_m", "rproj_m", "logit_m", "rand_m"],
            doses=[0.25, 0.5, 1.0], fillers=[0], kinds=["express", "use", "neutral"],
        ), 4
    if name == "c32smoke":
        kw, _ = preset("c32")
        return dict(kw, label="c32smoke", max_units=4, clues=["0", "name"]), 1
    if name == "c32prompt":
        kw, n = preset("c32")
        # Same panel and arms, edits at every prompt position: the convention
        # the earlier two-hop panel used. Reconciliation arm, not the paired design.
        return dict(kw, label="c32prompt", swap_positions="prompt"), 4
    if name == "c32query":
        kw, n = preset("c32")
        # Edits only the question tokens: isolates the answer-position effect.
        return dict(kw, label="c32query", swap_positions="query"), 4
    if name == "c36kappa":
        # Rescaling control. Whole-prompt convention, where the off-vs-full
        # ordering lives. `_cn` arms set kappa = 1 for every family; `_k*`
        # sweep it; `full_koff` / `off_kfull` hand each family the other's
        # per-pair kappa -- if the ordering follows kappa rather than the
        # direction, the filtering effect is a magnitude artifact.
        return dict(
            stage="swap", label="c36kappa", digest_paths=[COMP], panel_path=eligibility_rel(),
            swap_positions="prompt",
            swap_arms=[
                "full", "off_m", "proj_m",
                "full_cn", "off_cn", "proj_cn", "diag_cn", "logit_cn",
                "full_koff", "off_kfull",
                "full_k0.5", "full_k2", "off_k0.5", "off_k2",
            ],
            doses=[0.5, 1.0], fillers=[0], kinds=["express", "use", "neutral"],
            clues=["0", "1"],
        ), 4
    if name.startswith("c37lambda"):
        # J_lambda = J_off + lambda * J_diag, pulled back to v_off + lambda*v_diag
        # (exact: full and diag+off agree to 4e-8 in the digest). lambda=0 is the
        # off-diagonal arm, lambda=1 reproduces the full lens, and >1 overshoots it.
        # `_m` keeps each arm's native columns and matches the delta's row norm;
        # `_cn` additionally unit-normalises both columns, so kappa cannot carry
        # the effect. `full` and `rand_m` anchor the scale at both ends.
        where = "prompt" if name.endswith("prompt") else "prefix"
        return dict(
            stage="swap", label=name, digest_paths=[COMP], panel_path=eligibility_rel(),
            swap_positions=where, swap_arms=["lam0_m", "lam0.125_m", "lam0.25_m", "lam0.5_m", "lam0.75_m", "lam1_m", "lam1.5_m", "lam2_m", "lam0_cn", "lam0.125_cn", "lam0.25_cn", "lam0.5_cn", "lam0.75_cn", "lam1_cn", "lam1.5_cn", "lam2_cn", "full", "rand_m"],
            doses=[0.5, 1.0], fillers=[0], kinds=["express", "use", "neutral"],
            clues=["0", "1"],
        ), 4
    if name.startswith("c39check"):
        # float32 central-difference gate for C39's gradients, on a fixed
        # hash-ordered subsample (the bf16 epsilon column was quantisation).
        where = "prompt" if name.endswith("prompt") else "prefix"
        kw, _ = preset("c38mag" + where)
        kw.update(stage="gradcheck", label=name, max_units=40)
        return kw, 1
    if name.startswith("c39grad"):
        # Mentor spec, Experiment 2: clean-state gradients of the teacher-forced
        # relational (R) and leakage (L) margins at every band hook, dotted with
        # each lambda's Householder axis. Same trials and deltas as C38.
        where = "prompt" if name.endswith("prompt") else "prefix"
        kw, n = preset("c38mag" + where)
        kw.update(stage="gradgeom", label=name)
        return kw, n
    if name.startswith("c38matched"):
        # Spec section 3.5: dynamic Householder of fixed local norm m* = min over
        # lambda of C38's m_actual, on the arriving state. Triggered by C38's
        # pre-registered overlap rule (OVL(0,2) of m_actual_L2 = 0.02-0.04 < 0.2).
        where = "prompt" if name.endswith("prompt") else "prefix"
        kw, n = preset("c38mag" + where)
        kw.update(stage="magmatched", label=name)
        return kw, n
    if name.startswith("c38mag"):
        # Realized edit magnitude on the C37 unit-column sweep (mentor spec,
        # Experiment 1): same eligible units, use queries only, dose 1.0, both
        # position conventions. Deltas are rebuilt exactly as C37 built them.
        where = "prompt" if name.endswith("prompt") else "prefix"
        return dict(
            stage="magnitude", label=name, digest_paths=[COMP], panel_path=eligibility_rel(),
            swap_positions=where,
            swap_arms=["lam0_cn", "lam0.125_cn", "lam0.25_cn", "lam0.5_cn", "lam0.75_cn",
                       "lam1_cn", "lam1.5_cn", "lam2_cn"],
            doses=[1.0], fillers=[0], kinds=["express", "use"], clues=["0", "1"],
        ), 2
    if name == "c33":
        return dict(
            stage="swap", label="c33", digest_paths=[COMP], panel_path=eligibility_rel(),
            swap_arms=["full", "off_m", "proj_m", "logit_m"], doses=[0.5, 1.0],
            fillers=[1, 2, 4, 8], clues=["0", "1"], kinds=["express", "use"],
        ), 4
    if name == "c33b":
        return dict(
            stage="freegen", label="c33b", digest_paths=[COMP], panel_path=eligibility_rel(),
            swap_arms=["full", "off_m", "proj_m", "logit_m"], doses=[0.5, 1.0],
            clues=["0", "1"], gen_tokens=64, max_units=160, chunk=17,
        ), 4
    if name == "fit":
        return dict(stage="fit"), 1
    if name == "c34dev":
        return dict(
            stage="additive", label="c34dev", digest_paths=[COMP, baselines_rel()],
            panel_path=eligibility_rel(),
            methods=LENS + ["logit", "rand", "diffmean", "template"],
            layer_sets=LAYER_SETS, position_sets=["prefix", "last", "tail3", "prompt"],
            alphas=[0.05, 0.1, 0.2, 0.4, 0.8, 1.6], splits=["dev"], clues=["0", "1"],
            kinds=["express", "use", "neutral"], one_use_per_unit=True, max_units=64,
            gen_tokens=0, chunk=384,
        ), 4
    if name == "c34test":
        frozen = json.loads((LOCAL / "concept" / "frozen_c34.json").read_text())
        return dict(
            stage="additive", label="c34test", digest_paths=[COMP, baselines_rel()],
            panel_path=eligibility_rel(), layer_sets=LAYER_SETS,
            condition_list=frozen["condition_list"], splits=["test"], clues=["0", "1"],
            kinds=["express", "use", "neutral"], gen_tokens=8, chunk=128,
        ), 4
    raise SystemExit(f"unknown preset {name}")


def remote_spec(kwargs: dict, shard: int, n: int) -> dict:
    kw = dict(kwargs)
    kw["digest_paths"] = [f"{REMOTE}/{p}" for p in kw.get("digest_paths", [])]
    if kw.get("panel_path"):
        kw["panel_path"] = f"{REMOTE}/{kw['panel_path']}"
    kw.update(shard=shard, n_shards=n, out_dir=REMOTE)
    return kw


def local_spec(kwargs: dict) -> ConceptSpec:
    kw = dict(kwargs)
    kw["digest_paths"] = [str(LOCAL / p) for p in kw.get("digest_paths", [])]
    if kw.get("panel_path"):
        kw["panel_path"] = str(LOCAL / kw["panel_path"])
    kw["out_dir"] = str(LOCAL)
    return ConceptSpec(**kw)


def cmd_plan(name: str) -> None:
    kwargs, n = preset(name)
    spec = local_spec(kwargs)
    print(json.dumps({"n_shards": n, "tag": spec.tag, "spec": asdict(spec)}, indent=1)[:4000])


def cmd_launch(name: str, max_containers: int) -> None:
    import modal

    kwargs, n = preset(name)
    tag = local_spec(kwargs).tag
    print(f"preset {name}: tag {tag}, {n} shards, <= {max_containers} containers", flush=True)
    fn = modal.Function.from_name("jsteer", "gpu_task").with_options(
        max_containers=max_containers, retries=0
    )
    calls = [fn.spawn("concept", remote_spec(kwargs, i, n)) for i in range(n)]
    for c in calls:
        print("call", c.object_id, flush=True)
    for c in calls:
        print(json.dumps(c.get(), indent=1), flush=True)


def cmd_fetch() -> None:
    dest = LOCAL / "concept"
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run([MODAL, "volume", "get", "--force", "jsteer-results", "/concept", str(LOCAL)], check=True)
    for p in sorted(dest.iterdir()):
        print(p.name, p.stat().st_size)


def cmd_freeze() -> None:
    kwargs, n = preset("clean")
    tag = local_spec(kwargs).tag
    shards = [LOCAL / "concept" / f"{tag}_shard{i}of{n}.json" for i in range(n)]
    missing = [s for s in shards if not s.exists()]
    if missing:
        raise SystemExit(f"missing clean shards {missing}")
    rows = [r for s in shards for r in json.loads(s.read_text())["rows"]]
    from jsteer.concept_use import build_units

    units = build_units(clues=("0", "1", "name"))
    expected = sum(len(u.queries) for u in units) * len(kwargs["fillers"])
    if len(rows) != expected:
        raise SystemExit(f"{len(rows)} clean rows, expected {expected}")
    elig = freeze_eligibility(rows)
    out = LOCAL / "concept" / "eligibility.json"
    body = {
        "clean_tag": tag,
        "clean_sha": {s.name: hashlib.sha256(s.read_bytes()).hexdigest() for s in shards},
        "eligible": elig,
    }
    out.write_text(json.dumps(body, indent=1, sort_keys=True))
    subprocess.run(
        [MODAL, "volume", "put", "--force", "jsteer-results", str(out), "/concept/eligibility.json"],
        check=True,
    )
    print(f"{len(elig)} eligible units of {len(units)}; wrote and uploaded {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("plan", "launch", "fetch", "freeze"))
    ap.add_argument("preset", nargs="?")
    ap.add_argument("--max-containers", type=int, default=4)
    a = ap.parse_args()
    if a.command == "plan":
        cmd_plan(a.preset)
    elif a.command == "launch":
        cmd_launch(a.preset, a.max_containers)
    elif a.command == "fetch":
        cmd_fetch()
    else:
        cmd_freeze()


if __name__ == "__main__":
    main()

"""Print reproducible local experiment plans; --execute runs the requested stage.

GPU stages write to an isolated directory. Reference data are never overwritten.
No Modal account is needed. The reference component digest/eligibility can be
replaced with newly fitted/generated files via --directions and --eligibility.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jsteer.concept_use import ConceptSpec, run_concept  # noqa: E402
from jsteer.config import load_config  # noqa: E402
from jsteer.sweeps import ComponentSpec, run_component_pullbacks  # noqa: E402
from jsteer.write_controls import WriteControlSpec, run_write_controls  # noqa: E402

COMP = (
    ROOT
    / "results/rq1/group_mean_digests_qwen3-8b_dom2x32_968c0ba0_pb_t89d64f82_comp_s4.pt"
)
BAND = [13, 16, 19, 22, 25, 28, 31]
PRESETS = [
    "clean",
    "c36kappa",
    "c37lambdaprompt",
    "c37lambdaprefix",
    "c38magprompt",
    "c38magprefix",
    "c38matchedprompt",
    "c38matchedprefix",
    "c39gradprompt",
    "c39gradprefix",
    "c39checkprompt",
    "c39checkprefix",
]


def build(args):
    output = args.output_dir.resolve()
    reference = (ROOT / "results").resolve()
    if output == reference or not output.is_relative_to(reference):
        raise ValueError("--output-dir must be a new subdirectory of results/")
    common = dict(config_name="qwen3-8b", layers=BAND, out_dir=str(output))
    if args.stage == "components":
        settings = json.loads((ROOT / "configs/paper_components.json").read_text())
        spec = ComponentSpec(
            **common,
            **settings,
            corpus_cache=str(
                ROOT / "results/c22/component_fit_corpus_snapshot_20260914.json"
            ),
        )
        return spec, run_component_pullbacks
    if args.stage == "latent":
        return WriteControlSpec(
            config_name="qwen3-8b",
            directions_path=str(args.directions.resolve()),
            reference_path=str(
                ROOT
                / "results/causal/two_hop_qwen3-8b_all_g595bfbbf_twohop89d9c5_shard0of1.json"
            ),
            corpus=args.corpus,
            layers=tuple(BAND),
            limit=55,
            max_seconds=7200,
            out_dir=str(output / "c22"),
        ), run_write_controls
    spec = importlib.util.spec_from_file_location(
        "presets", ROOT / "scripts/48_concept_use.py"
    )
    presets = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(presets)
    kwargs, n = presets.preset(args.stage)
    kwargs.update(common, shard=args.shard, n_shards=n)
    if args.stage != "clean":
        kwargs["digest_paths"] = [str(args.directions.resolve())]
        kwargs["panel_path"] = str(args.eligibility.resolve())
    if not 0 <= args.shard < n:
        raise ValueError(f"--shard must be between 0 and {n - 1}")
    return ConceptSpec(**kwargs), run_concept


def freeze(args):
    """Freeze complete local clean shards without uploading to a service."""
    from jsteer.concept_use import build_units, freeze_eligibility

    paths = sorted((args.output_dir / "concept").glob("clean_*_shard*of2.json"))
    if len(paths) != 2 or not all(
        f"shard{i}of2" in p.name for i, p in enumerate(paths)
    ):
        raise ValueError(
            "Expected exactly two completed clean shards in output-dir/concept"
        )
    blobs = [json.loads(p.read_text()) for p in paths]
    if blobs[0]["manifest"] != blobs[1]["manifest"]:
        raise ValueError("Clean shards disagree on manifest")
    rows = [r for b in blobs for r in b["rows"]]
    expected = {
        (u.key, q.relation, f)
        for u in build_units(clues=("0", "1", "name"))
        for q in u.queries
        for f in (0, 1, 2, 4, 8)
    }
    actual = {(r["key"], r["relation"], r["filler"]) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError(
            f"Incomplete or duplicate clean population: {len(rows)} rows, expected {len(expected)}"
        )
    body = {
        "clean_sha": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "eligible": freeze_eligibility(rows),
    }
    out = args.output_dir / "concept/eligibility.json"
    if out.exists() and json.loads(out.read_text()) != body:
        raise ValueError("Refusing to overwrite different eligibility")
    if args.execute:
        out.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
        print(f"Frozen {len(body['eligible'])} eligible units: {out}")
    else:
        print(
            f"Would freeze {len(body['eligible'])} eligible units to {out}; add --execute"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["components", "latent", "freeze"] + PRESETS)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "results/rerun")
    ap.add_argument("--directions", type=Path, default=COMP)
    ap.add_argument(
        "--eligibility", type=Path, default=ROOT / "results/concept/eligibility.json"
    )
    ap.add_argument("--corpus", choices=["gsm8k", "wikitext_a"], default="gsm8k")
    ap.add_argument("--shard", type=int, default=0)
    args = ap.parse_args()
    if args.stage == "freeze":
        freeze(args)
        return
    spec, runner = build(args)
    print(json.dumps(asdict(spec), indent=2))
    if not args.execute:
        print("Plan only. Add --execute to load models and run this stage.")
        return
    # New run directories protect historical outputs and make changed code visible.
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    run_path = out / f"run_{args.stage}_{args.corpus}_{args.shard}.json"
    manifest = dict(
        stage=args.stage,
        spec=asdict(spec),
        model=asdict(load_config("qwen3-8b")),
        packages={
            n: version(n)
            for n in ["torch", "transformers", "jlens", "numpy", "datasets"]
        },
        source_sha256={
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "src/jsteer").glob("*.py"))
        },
    )
    for key in ["directions_path", "reference_path", "corpus_cache", "panel_path"]:
        p = getattr(spec, key, None)
        if p:
            manifest.setdefault("input_sha256", {})[key] = hashlib.sha256(
                Path(p).read_bytes()
            ).hexdigest()
    for p in getattr(spec, "digest_paths", []):
        manifest.setdefault("input_sha256", {})[Path(p).name] = hashlib.sha256(
            Path(p).read_bytes()
        ).hexdigest()
    if run_path.exists():
        old = json.loads(run_path.read_text())
        if old["inputs"] != manifest:
            raise SystemExit("Run inputs changed. Choose a new --output-dir.")
        if old["status"] == "complete":
            raise SystemExit(
                "Stage already completed. Choose a new --output-dir to repeat."
            )
    record = dict(inputs=manifest, status="running")
    run_path.write_text(json.dumps(record, indent=2))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        result = runner(spec)
    except Exception:
        record["status"] = "failed"
        run_path.write_text(json.dumps(record, indent=2))
        raise
    record.update(status=result.get("status", "complete"), result=result)
    run_path.write_text(json.dumps(record, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

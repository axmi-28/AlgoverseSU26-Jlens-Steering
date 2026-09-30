"""Rebuild paper results on CPU from the verified reference artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.release import reference_items  # noqa: E402


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_inputs():
    from scripts.release import sha

    missing = []
    for item, _ in reference_items():
        p = ROOT / item["path"]
        if not p.exists() or sha(p.read_bytes()) != item["sha256"]:
            missing.append(item["path"])
    if missing:
        raise SystemExit(
            "Restore the reference inputs with scripts/release.py restore. "
            f"Missing or changed inputs: {missing}"
        )


def tables():
    b = module("paper_figure_data", "50_paper_figures.py")
    c = module("paper_corpus_data", "53_corpus_figures.py")
    output = {
        "weighting": "Concept rates average queries within units, then average units.",
        "latent": {},
        "corpus_replicates": {},
        "interpolation": {},
        "direction_scale": {},
    }
    for corpus, suffix, expected in [
        ("gsm8k", "dac7603b1af57dbd", [15, 21, 25, 5]),
        ("wikitext_a", "6fff00659415ddf5", [13, 21, 23, 9]),
    ]:
        blob = b.read(f"results/c22/write_controls_{suffix}.json")
        assert blob["status"] == "complete"
        rows = blob["records"]
        index = {(r["name"], r["arm"], r["strength"]): r for r in rows}
        assert len(index) == len(rows)
        names = sorted({r["name"] for r in rows})
        assert len(names) == 55
        output["latent"][corpus] = {}
        for dose in (0.5, 1.0):
            vals = [
                sum(index[n, arm, dose][field] for n in names)
                for arm in ("swap_full", "swap_off_matched")
                for field in ("hit_swap_answer", "emitted_substring")
            ]
            output["latent"][corpus][str(dose)] = dict(
                n=55,
                full_answer=vals[0],
                full_leakage=vals[1],
                off_answer=vals[2],
                off_leakage=vals[3],
            )
            if dose == 1:
                assert vals == expected, (corpus, vals)
    for corpus, label, prefix in [
        ("gsm8k", "tier0", "gsm8k"),
        ("wikitext_a", "tier0", "wikirep"),
        ("aqua_rat", "aqua", "aqua_rat"),
    ]:
        best = c.corpus_rows(label)
        output["corpus_replicates"][corpus] = {
            "numerical_of_39": [best(f"swap_{prefix}_r{i}", True) for i in range(5)],
            "other_of_105": [best(f"swap_{prefix}_r{i}", False) for i in range(5)],
            "cells": "[hits, selected_strength]; strength selected on evaluated subset",
        }
    for where, stem in [
        ("prompt", "swap_c37lambdaprompt_qwen3-8b_034da26ce9"),
        ("prefix", "swap_c37lambdaprefix_qwen3-8b_75f74629d0"),
    ]:
        rows, conditions = b.load_concept(stem)
        a = b.unit_arrays(rows, conditions)
        assert len(a["keys"]) == 661 and a["n_queries"] == 1339
        output["interpolation"][where] = dict(n_units=661, n_queries=1339, corpora={})
        for corpus in ("gsm8k", "wikitext_a"):
            cells = []
            for lam in (0, 0.125, 0.25, 0.5, 0.75, 1, 1.5, 2):
                j = b.col(a, corpus, f"lam{lam:g}_cn", 1.0)
                cells.append(
                    {
                        "lambda": lam,
                        **{
                            m: float(a[m][:, j].mean())
                            for m in ("express", "use", "leak", "kl")
                        },
                    }
                )
            output["interpolation"][where]["corpora"][corpus] = cells
    rows, conditions = b.load_concept("swap_c36kappa_qwen3-8b_3b28f16205")
    a = b.unit_arrays(rows, conditions)
    output["direction_scale"] = dict(
        n_units=len(a["keys"]),
        n_queries=a["n_queries"],
        note="Complete stored run; the manuscript's older 649-unit table is not this population.",
        corpora={},
    )
    for corpus in ("gsm8k", "wikitext_a"):
        output["direction_scale"]["corpora"][corpus] = {
            arm: {
                m: float(a[m][:, b.col(a, corpus, arm, 1.0)].mean())
                for m in ("use", "leak")
            }
            for arm in ("full_cn", "full_koff", "off_cn", "off_kfull")
        }
    output["sources"] = b.SOURCES | c.b.SOURCES
    out = ROOT / "output/reproduced"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables.json").write_text(json.dumps(output, indent=2) + "\n")
    lines = [
        "# Recomputed reference tables",
        "",
        "See tables.json for every cell and source hash.",
        "",
        "| Corpus | Full answer | Off answer | Full leakage | Off leakage |",
        "|---|---:|---:|---:|---:|",
    ]
    for corpus, doses in output["latent"].items():
        v = doses["1.0"]
        lines.append(
            f"| {corpus} | {v['full_answer']}/55 | {v['off_answer']}/55 | "
            f"{v['full_leakage']}/55 | {v['off_leakage']}/55 |"
        )
    lines += [
        "",
        "Direction-scale population: " + str(len(a["keys"])) + " units. "
        "Do not substitute the historical 649-unit table; see docs/PAPER_AUDIT.md.",
    ]
    (out / "tables.md").write_text("\n".join(lines) + "\n")
    print(f"Recomputed latent, corpus, interpolation and direction-scale tables: {out}")


def run(filename, *args):
    env = __import__("os").environ.copy()
    env["PYTHONPATH"] = str(ROOT) + __import__("os").pathsep + str(ROOT / "src")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / filename), *args],
        cwd=ROOT,
        env=env,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("tables", "figures", "magnitude", "gradients", "latent", "all"),
    )
    args = parser.parse_args()
    check_inputs()
    if args.command in ("tables", "all"):
        tables()
    if args.command in ("figures", "all"):
        run("56_final_paper_figures.py")
    if args.command in ("magnitude", "all"):
        run("51_magnitude_report.py")
    if args.command in ("gradients", "all"):
        run("55_gradgeom_report.py")
    if args.command in ("latent", "all"):
        m = module("latent_report", "42_write_controls_followup.py")
        report = m.report(
            json.loads(m.PLAN.read_text()),
            [
                ROOT / "results/c22/write_controls_dac7603b1af57dbd.json",
                ROOT / "results/c22/write_controls_6fff00659415ddf5.json",
            ],
        )
        out = ROOT / "output/reproduced/latent.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)
        print(out)


if __name__ == "__main__":
    main()

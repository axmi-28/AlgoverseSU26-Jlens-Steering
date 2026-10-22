"""Build the complete filtering comparison from saved artifacts; CPU only."""

from __future__ import annotations

import hashlib
import html
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/core-results"
SOURCES = {}


def read(path):
    p = ROOT / path
    raw = p.read_bytes()
    SOURCES[str(p.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rep = module("core_paired_report", "60_cross_model_report.py")


def pair(rows, a, b, concept=False):
    return {
        metric: rep.paired(rows, a, b, metric, concept=concept)
        for metric in ("success", "incorrect_name_emission")
    }


def summarize(latent, concept, a, b, sensitivity=None):
    latent = rep.emission_metrics(latent)
    concept = rep.emission_metrics(concept)
    metrics = {
        f"latent / {s:g}": pair(
            [r for r in latent if r["strength"] == s], "full", "off_m"
        )
        for s in (0.5, 1.0)
    }
    metrics["concept use / 1"] = pair(
        [r for r in concept if r["kind"] == "use"], a, b, True
    )
    naming = rep.paired(
        [r for r in concept if r["kind"] == "express"], a, b, "success", concept=True
    )
    checks = {}
    if sensitivity:
        la, ca = sensitivity
        checks = {
            f"latent / {s:g}": rep.paired(
                [r for r in latent if r["strength"] == s], "full", la, "success"
            )
            for s in (0.5, 1.0)
        }
        checks["concept use / 1"] = rep.paired(
            [r for r in concept if r["kind"] == "use"], a, ca, "success", concept=True
        )
    return metrics, naming, checks


def campaign_rows(folder, study, prefix, published=False):
    summary = read(f"results/{folder}/summary.json")["models"]
    out = []
    for name, info in sorted(summary.items()):
        path = Path(info["path"])
        if path.is_absolute():
            path = Path("results") / path.relative_to("/results")
        manifest = read(path / "manifest.json")
        assert read(path / "status.json")["stage"] == "complete"
        latent, concept = (
            read(path / "latent_results.json"),
            read(path / "concept_results.json"),
        )
        a, b = ("full_cn", "lam0_cn") if published else ("lam1_cn", "lam0_cn")
        metrics, naming, sensitivity = summarize(
            latent, concept, a, b, ("off32_m", "off32_cn") if published else None
        )
        assert metrics == info["contrasts"], (
            name,
            "saved summary differs from raw paired results",
        )
        label = name.removesuffix("-np")
        full_n, diag_n = 32, 32
        meta = None
        if published:
            meta = read(path / "published_metadata.json")
            assert meta["full_lens_refitted"] is False
            full_n, diag_n = meta["n_prompts"], 64
        out.append(
            {
                "id": f"{prefix}-{len(out) + 1:02d}",
                "model": label,
                "study": study,
                "lens": "Neuronpedia WikiText + estimated D64"
                if published
                else "Self-fit GSM8K / exact components",
                "method": "published full − estimated diagonal"
                if published
                else "jointly fitted full / off-diagonal",
                "corpus": "WikiText" if published else "GSM8K",
                "full_fit_prompts": full_n,
                "diagonal_fit_prompts": diag_n,
                "model_revision": manifest["config"].get("model_revision"),
                "layers": manifest["layers"],
                "source_path": str(path),
                "published_metadata": meta,
                "contrasts": metrics,
                "naming": naming,
                "sensitivity32": sensitivity,
            }
        )
    return out


def original_qwen():
    paths = {
        "gsm8k": [
            "results/c22/write_controls_dac7603b1af57dbd.json",
            "results/latent/latent_controls_gsm8k_80a6934224f0ac55.json",
        ],
        "wikitext_a": [
            "results/c22/write_controls_6fff00659415ddf5.json",
            "results/latent/latent_controls_wikitext_a_393f229cc9c869be.json",
        ],
    }
    stem = "results/concept/swap_c37lambdaprompt_qwen3-8b_034da26ce9"
    blobs = [read(f"{stem}_shard{i}of4.json") for i in range(4)]
    conditions = blobs[0]["conditions"]
    assert all(b["conditions"] == conditions for b in blobs)
    records = [
        r
        for b in blobs
        for r in b["records"]
        if r["clue"] in ("0", "1")
        and r.get("filler", 0) == 0
        and r["kind"] in ("use", "express")
    ]
    assert len({(r["key"], r["relation"]) for r in records}) == len(records)
    allowed = {r["key"] for r in records if r["kind"] == "use"} & {
        r["key"] for r in records if r["kind"] == "express"
    }
    records = [r for r in records if r["key"] in allowed]
    out = []
    for corp, latent_paths in paths.items():
        latent = []
        for p in latent_paths:
            blob = read(p)
            for r in blob["records"]:
                if r["arm"] not in ("swap_full", "swap_off_matched") or r[
                    "strength"
                ] not in (0.5, 1.0):
                    continue
                latent.append(
                    {
                        "name": r["name"],
                        "source": r["intermediate"],
                        "arm": "full" if r["arm"] == "swap_full" else "off_m",
                        "strength": r["strength"],
                        "success": bool(r["hit_swap_answer"]),
                        "leakage": bool(r["emitted_substring"]),
                    }
                )
        concept = []
        for arm in ("lam1_cn", "lam0_cn", "rand_m", "lam0.5_cn", "lam2_cn"):
            k = conditions.index([corp, arm, 1.0])
            for r in records:
                concept.append(
                    {
                        "key": r["key"],
                        "relation": r["relation"],
                        "source": r["type"] + "/" + r["source"],
                        "arm": arm,
                        "kind": r["kind"],
                        "success": bool(r["hit_t"][k]),
                        "leakage": bool(r["name_t"][k]),
                    }
                )
        metrics, naming, _ = summarize(latent, concept, "lam1_cn", "lam0_cn")
        assert metrics["latent / 1"]["success"]["n"] == 194
        assert metrics["concept use / 1"]["success"]["n"] == 661
        assert naming["n"] == 661
        expected = (62, 92) if corp == "gsm8k" else (68, 96)
        d = metrics["latent / 1"]["success"]
        assert (
            tuple(round(d[key] * 194 / 100) for key in ("baseline", "filtered"))
            == expected
        )
        out.append(
            {
                "id": "P-GSM" if corp == "gsm8k" else "P-WIKI",
                "model": "qwen3-8b",
                "study": "Original Qwen paper panel",
                "lens": f"Self-fit {'GSM8K' if corp == 'gsm8k' else 'WikiText'} / exact components",
                "method": "jointly fitted full / off-diagonal",
                "corpus": corp,
                "full_fit_prompts": 32,
                "diagonal_fit_prompts": 32,
                "source_path": "; ".join(latent_paths) + f"; {stem}_shard[0–3]of4.json",
                "layers": blobs[0]["manifest"]["layers"],
                "contrasts": metrics,
                "naming": naming,
                "sensitivity32": {},
            }
        )
    return out


def pressure_rows():
    folders = sorted((ROOT / "results/gemma_pressure").glob("*/status.json"))
    assert len(folders) == 1
    folder = folders[0].parent.relative_to(ROOT)
    assert read(folder / "status.json")["stage"] == "complete"
    rows = read(folder / "validation_results.json")
    selection = read(folder / "selection.json")
    manifest = read(folder / "manifest.json")
    released = read(folder / "released_metadata.json")
    selected = {r["id"] for r in selection["selected"]}
    output = []
    for setting in sorted({r["setting"] for r in rows}):
        corpus, layers, strength, columns = setting.split("/")
        metrics = {}
        for task in ("latent", "concept"):
            rs = [
                dict(r, key=r["unit"], name=r["query"], relation=r["query"])
                for r in rows
                if r["setting"] == setting and r["task"] == task
            ]
            arm = "projection_proxy" if corpus == "released" else "filtered"
            metrics[task] = pair(rs, "full", arm, task == "concept")
        output.append(
            {
                "setting": setting,
                "corpus": corpus,
                "layers": layers,
                "strength": float(strength),
                "columns": columns,
                "selected_on_dev": setting in selected,
                "contrasts": metrics,
                "method": f"Neuronpedia WikiText ({released['n_prompts']} texts) / unembedding-projection proxy"
                if corpus == "released"
                else f"Self-fit {'GSM8K' if corpus == 'gsm8k' else 'WikiText'} (32 texts) / exact components",
                "full_fit_prompts": released["n_prompts"]
                if corpus == "released"
                else 32,
                "model_revision": manifest["model"]["model_revision"],
                "published_metadata": released if corpus == "released" else None,
                "source_path": str(folder / "validation_results.json"),
            }
        )
    return output


def prefix_rows():
    stem = "results/concept/swap_c37lambdaprefix_qwen3-8b_75f74629d0"
    blobs = [read(f"{stem}_shard{i}of4.json") for i in range(4)]
    conditions = blobs[0]["conditions"]
    assert all(b["conditions"] == conditions for b in blobs)
    records = [
        r
        for b in blobs
        for r in b["records"]
        if r["clue"] in ("0", "1") and r.get("filler", 0) == 0 and r["kind"] == "use"
    ]
    out = []
    for corp in ("gsm8k", "wikitext_a"):
        rows = []
        for arm in ("lam1_cn", "lam0_cn"):
            k = conditions.index([corp, arm, 1.0])
            rows += [
                dict(
                    key=r["key"],
                    relation=r["relation"],
                    source=r["type"] + "/" + r["source"],
                    arm=arm,
                    success=bool(r["hit_t"][k]),
                    leakage=bool(r["name_t"][k]),
                )
                for r in records
            ]
        out.append(
            {
                "corpus": corp,
                "contrasts": pair(
                    rep.emission_metrics(rows), "lam1_cn", "lam0_cn", True
                ),
                "source_path": stem,
            }
        )
    return out


def projection_rows():
    path = ROOT / "results/c31_second_model_qwen3-4b.md"
    raw = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    cells = {}
    for line in raw.decode().splitlines():
        fields = [f.strip() for f in line.split("|")[1:-1]]
        if fields and fields[0] in ("b0", "b1"):
            cells[fields[0]] = fields
    return [
        {
            "model": model,
            "n": n,
            "baseline_count": int(cells["b0"][k]),
            "filtered_count": int(cells["b1"][k]),
            "baseline": 100 * int(cells["b0"][k]) / n,
            "filtered": 100 * int(cells["b1"][k]) / n,
            "delta": 100 * (int(cells["b1"][k]) - int(cells["b0"][k])) / n,
            "method": "published WikiText / unembedding-projection proxy (beta=1)",
            "source_path": str(path.relative_to(ROOT)),
        }
        for model, n, k in [("qwen3-4b", 99, 1), ("qwen3-8b", 105, 3)]
    ]


def delta(d, ci=True):
    text = f"{d['delta']:+.1f}"
    if ci:
        text += f" [{d['ci'][0]:+.1f}, {d['ci'][1]:+.1f}]"
    return text


def rates(d):
    return f"{d['baseline']:.1f} → {d['filtered']:.1f}"


def table(headers, rows):
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ] + [
        "| " + " | ".join(str(c).replace("|", "/") for c in row) + " |" for row in rows
    ]


def cell(d):
    return f"{rates(d)}; {delta(d)}"


def build_tables(core, pressure, prefix, projection):
    main = []
    naming = []
    detailed = []
    sensitivity = []
    provenance = []
    for row in core:
        l = row["contrasts"]["latent / 1"]["success"]
        u = row["contrasts"]["concept use / 1"]["success"]
        name = row["naming"]
        em = row["contrasts"]["concept use / 1"]["incorrect_name_emission"]
        main.append(
            [
                row["id"],
                row["model"],
                row["lens"],
                f"{row['full_fit_prompts']} / {row['diagonal_fit_prompts']}",
                l["n"],
                cell(l),
                u["n"],
                cell(u),
            ]
        )
        assert name["n"] == u["n"]
        naming.append(
            [
                row["id"],
                row["model"],
                u["n"],
                rates(name),
                delta(name),
                rates(u),
                delta(u),
                rates(em),
            ]
        )
        for task, m in row["contrasts"].items():
            d = m["success"]
            e = m["incorrect_name_emission"]
            detailed.append(
                [
                    row["id"],
                    row["model"],
                    task,
                    d["n"],
                    d["clusters"],
                    rates(d),
                    delta(d),
                    rates(e),
                    delta(e),
                ]
            )
        for task, d in row["sensitivity32"].items():
            primary = row["contrasts"][task]["success"]
            sensitivity.append(
                [
                    row["id"],
                    row["model"],
                    task,
                    d["n"],
                    f"{d['baseline']:.1f}",
                    f"{d['filtered']:.1f}",
                    f"{primary['filtered']:.1f}",
                    delta(d),
                    delta(primary),
                ]
            )
        provenance.append(
            [
                row["id"],
                row.get("model_revision") or "See source manifests",
                ", ".join(map(str, row["layers"])),
                row["source_path"],
            ]
        )
    ptable = []
    for row in pressure:
        l = row["contrasts"]["latent"]["success"]
        u = row["contrasts"]["concept"]["success"]
        ptable.append(
            [
                "gemma-3-12b-it",
                row["setting"],
                "Dev-selected" if row["selected_on_dev"] else "Original fixed control",
                row["method"],
                l["n"],
                cell(l),
                u["n"],
                cell(u),
            ]
        )
    return [
        (
            "Main comparison · strength 1",
            [
                "ID",
                "Model",
                "J-lens / filtering",
                "Fit texts: full / D",
                "Latent N",
                "Latent full → filtered (%); Δ pp [95% CI]",
                "Concept N",
                "Use full → filtered (%); Δ pp [95% CI]",
            ],
            main,
        ),
        (
            "Naming versus downstream use · same concept units, strength 1",
            [
                "ID",
                "Model",
                "Concept N",
                "Naming full → filtered (%)",
                "Naming Δ pp [95% CI]",
                "Use full → filtered (%)",
                "Use Δ pp [95% CI]",
                "Incorrect name on use full → filtered (%)",
            ],
            naming,
        ),
        (
            "Both latent strengths and complete paired metrics",
            [
                "ID",
                "Model",
                "Task / strength",
                "N",
                "Source clusters",
                "Accuracy full → filtered (%)",
                "Accuracy Δ pp [95% CI]",
                "Incorrect name full → filtered (%)",
                "Incorrect-name Δ pp [95% CI]",
            ],
            detailed,
        ),
        (
            "Published-lens sensitivity · diagonal estimated on 32 vs 64 texts",
            [
                "ID",
                "Model",
                "Task / strength",
                "N",
                "Published full (%)",
                "Filtered D32 (%)",
                "Filtered D64 (%)",
                "D32 Δ pp [95% CI]",
                "D64 Δ pp [95% CI]",
            ],
            sensitivity,
        ),
        (
            "Gemma 3 12B IT follow-up · validation only",
            [
                "Model",
                "Setting: corpus / layers / strength / columns",
                "Selection",
                "Method",
                "Latent N",
                "Latent full → filtered (%); Δ pp [95% CI]",
                "Concept N",
                "Concept full → filtered (%); Δ pp [95% CI]",
            ],
            ptable,
        ),
        (
            "Published-lens projection controls · explicit non-numerical swaps",
            [
                "Model",
                "Method",
                "N",
                "Full correct",
                "Projected correct",
                "Full → projected (%)",
                "Δ pp",
            ],
            [
                [
                    r["model"],
                    r["method"],
                    r["n"],
                    r["baseline_count"],
                    r["filtered_count"],
                    rates(r),
                    delta(r, False),
                ]
                for r in projection
            ],
        ),
        (
            "Original Qwen 3 8B prefix-only placement control · strength 1",
            [
                "Model",
                "Self-fit lens",
                "Concept N",
                "Use full → filtered (%)",
                "Use Δ pp [95% CI]",
                "Incorrect name full → filtered (%)",
            ],
            [
                [
                    "qwen3-8b",
                    r["corpus"],
                    r["contrasts"]["success"]["n"],
                    rates(r["contrasts"]["success"]),
                    delta(r["contrasts"]["success"]),
                    rates(r["contrasts"]["incorrect_name_emission"]),
                ]
                for r in prefix
            ],
        ),
        (
            "Run provenance",
            ["ID", "Model revision", "Edited layers", "Source artifacts"],
            provenance,
        ),
    ]


INTRO = [
    "Completed position-pair filtering evaluations: 13 model checkpoints, 16 main model/lens evaluations. Rates and paired intervals are derived from saved outputs; no new model calls or GPU work.",
    "IDs: P = original Qwen paper reference; H = initial Gemma/Llama replication; F = broader-model balanced replication; NP = published Neuronpedia-lens replication. Full per-run provenance is listed below.",
    "Full → filtered reports target-answer accuracy in percent. Δ is filtered − full in percentage points. All displayed paired intervals are recomputed with the same source-entity cluster bootstrap (2000 draws, seed 42102), conditional on one fitted lens/diagonal estimate; pointwise and unadjusted for multiple comparisons. The 14 replication rows are checked against their existing summaries. Original Qwen reference intervals can differ slightly from the paper’s 10,000-draw intervals.",
    "Lens provenance: self-fit GSM8K/WikiText rows use jointly fitted full/diagonal/off components (32 fitting texts). Neuronpedia rows hold the published full lens fixed and subtract a separately estimated 64-text WikiText diagonal: this is approximate filtering. Full-fit prompt counts refer to the released lens; D32 is a sensitivity with the same full baseline. Projection proxies remove the unembedding-aligned component and are reported separately.",
    "The main table uses strength 1 and whole-prompt edits. Latent swaps retain native direction-column scales and match filtered edit norms to full. Concept-use/naming comparisons unit-normalize both columns, with per-position/per-layer norm matching. Concept accuracy averages eligible relations within units, then units equally. N is latent items or concept units, not generated tokens or relation queries.",
    "Populations, layer bands, lens-fitting corpus and sampling differ across campaigns; compare full vs filtered within each row. The original Qwen panel uses 194 latent items and 661 country-heavy concept units; later replications select up to 40 concept units per type. Original Qwen paper-reference and later Qwen control rows are distinct evaluations and are not pooled. Naming accuracy and incorrect-name emission are separate endpoints; emission reduction alone does not establish improved use.",
    "Gemma pressure-test validation is exploratory: source groups are separated from development selection, but earlier aggregate/some latent outcomes were already observed. All six evaluated validation settings are retained, including unsuccessful settings. Its projection row is a proxy, and its native-column concept controls differ from the main unit-column comparison.",
    "Coverage: Qwen3-1.7B and Qwen3.6-27B have other domain/geometry or local-vs-averaged experiments, but no completed paired latent-plus-concept position-filtering replication was located for this table. They are untested for these endpoints, not negative results. The configured Gemma 2 2B base checkpoint likewise has no completed core pair here.",
]


def render_html(tables):
    parts = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>J-lens · all core results</title>',
        "<style>body{font:15px/1.5 system-ui,sans-serif;color:#162b3c;background:#f5f7fa;margin:0}main{max-width:1800px;margin:auto;padding:30px}h1{font-size:32px;line-height:1.2;margin:0 0 12px}h2{font-size:21px;margin:34px 0 12px}p{max-width:1200px}.note{color:#41586b}input{font:inherit;padding:10px 14px;border:1px solid #b7c5d1;border-radius:6px;width:min(650px,90%)}.wrap{overflow:auto;border:1px solid #d7e0e7;border-radius:8px;background:white;max-height:760px}table{border-collapse:separate;border-spacing:0;font-size:13px;width:100%}th,td{padding:10px 12px;border-bottom:1px solid #e1e7ec;text-align:left;vertical-align:top;white-space:nowrap}th{position:sticky;top:0;background:#eaf0f5;color:#213a50;z-index:1}tbody tr:nth-child(even){background:#f8fafc}tbody tr:hover{background:#e9f4f6}details p{font-size:14px}.count{font-size:13px;color:#526879}a{color:#086b83}nav a{margin-right:18px}footer{margin:30px 0;color:#526879}</style><main>",
        "<h1>J-lens filtering · all core results</h1><p>13 model checkpoints · 16 main model/lens evaluations · completed saved runs only</p>",
        '<p class="note">Full → filtered accuracy; changes in percentage points. Source, method and evaluation populations remain explicit.</p>',
        '<nav><a href="full_table.md">Markdown table</a><a href="full_table.json">Machine-readable data</a><a href="provenance.json">Source checksums</a></nav>',
        '<p><input id="search" type="search" aria-label="Filter table rows" placeholder="Filter rows: model, lens, study or task…"><span class="count" id="count"></span></p>',
        "<details><summary>Methods, comparability and coverage</summary>"
        + "".join("<p>" + html.escape(x) + "</p>" for x in INTRO)
        + "</details>",
    ]
    for title, headers, rows in tables:
        parts += [
            "<h2>"
            + html.escape(title)
            + '</h2><div class="wrap"><table><thead><tr>'
            + "".join('<th scope="col">' + html.escape(x) + "</th>" for x in headers)
            + "</tr></thead><tbody>"
        ]
        for row in rows:
            parts.append(
                "<tr>"
                + "".join("<td>" + html.escape(str(x)) + "</td>" for x in row)
                + "</tr>"
            )
        parts.append("</tbody></table></div>")
    parts.append(
        '<footer>Built from saved artifacts by scripts/75_all_core_results.py. Original experiment outputs are unchanged.</footer></main><script>const rows=[...document.querySelectorAll("tbody tr")],field=document.querySelector("#search");function filter(){const q=field.value.toLowerCase().trim();let n=0;for(const r of rows){r.hidden=!r.textContent.toLowerCase().includes(q);if(!r.hidden)n++}document.querySelector("#count").textContent=`  ${n} / ${rows.length} rows shown`;}field.addEventListener("input",filter);filter();</script></html>'
    )
    return "\n".join(parts)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    core = (
        original_qwen()
        + campaign_rows("cross_model", "Initial Gemma/Llama replication", "H")
        + campaign_rows(
            "filtering_replication", "Broader-model balanced replication", "F"
        )
        + campaign_rows(
            "neuronpedia_replication", "Published-lens replication", "NP", True
        )
    )
    pressure = pressure_rows()
    prefix = prefix_rows()
    projection = projection_rows()
    assert len(core) == 16 and len({r["model"] for r in core}) == 13
    tables = build_tables(core, pressure, prefix, projection)
    lines = ["# All core J-lens filtering results", ""]
    for paragraph in INTRO:
        lines += [paragraph, ""]
    for title, headers, rows in tables:
        lines += ["## " + title, ""] + table(headers, rows) + [""]
    lines += [
        "## Rebuild",
        "",
        "`.venv/bin/python scripts/75_all_core_results.py`",
        "",
        "Sources include the original Qwen C22/C40/C37 artifacts, completed cross_model, filtering_replication and neuronpedia_replication runs, Gemma pressure validation, and the C31 published-lens projection report. The JSON provenance file hashes every artifact read.",
        "",
    ]
    (OUT / "full_table.md").write_text("\n".join(lines))
    payload = {
        "scope": "completed core position-pair filtering evaluations",
        "bootstrap": {
            "draws": 2000,
            "seed": 42102,
            "cluster": "source entity",
            "pointwise": True,
        },
        "notes": INTRO,
        "core": core,
        "gemma_pressure_validation": pressure,
        "published_projection_controls": projection,
        "qwen_prefix_controls": prefix,
    }
    (OUT / "full_table.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "index.html").write_text(render_html(tables))
    for path in (Path(__file__), ROOT / "scripts/60_cross_model_report.py"):
        SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    (OUT / "provenance.json").write_text(
        json.dumps(
            {
                "sources_sha256": SOURCES,
                "primary_evaluations": len(core),
                "model_checkpoints": len({r["model"] for r in core}),
                "tables": {t: len(r) for t, h, r in tables},
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(OUT),
                "primary_evaluations": len(core),
                "model_checkpoints": len({r["model"] for r in core}),
                "source_artifacts": len(SOURCES),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

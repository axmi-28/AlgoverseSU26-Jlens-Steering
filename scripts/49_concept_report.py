"""C32-C35 reports. Reads only local copies of the volume files; writes markdown.

    python scripts/49_concept_report.py c32
    python scripts/49_concept_report.py c33
    python scripts/49_concept_report.py c33b
    python scripts/49_concept_report.py select      # C34 dev grid -> frozen_c34.json
    python scripts/49_concept_report.py c34         # C34/C35 test

Statistics follow docs/PROTOCOL.md: cluster bootstrap over
source entities (10,000 resamples, seed 3201), exact sign tests on unit-level
paired differences, no multiplicity adjustment.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "results"
CON = RES / "concept"

from jsteer.concept_use import ConceptSpec  # noqa: E402
from jsteer.concept_use import _eligible_units as cu_eligible  # noqa: E402

_spec = importlib.util.spec_from_file_location("launcher", ROOT / "scripts/48_concept_use.py")
launcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launcher)

N_BOOT = 10_000
#: preset -> caveat text, populated by ``find_shards`` and printed into reports.
LOAD_CAVEATS: dict[str, str] = {}
SEED = 3201


def find_shards(preset: str) -> tuple[dict, int, list[Path]]:
    """Locate a preset's outputs by the manifest *inside* each file.

    Filenames carry a hash of the spec, so adding any new spec field -- even
    one whose default reproduces the old behaviour -- renames every finished
    output. Matching on the stored manifest instead makes the lookup survive
    that: a field the stored manifest does not have is accepted when the
    current value is its default, and any other difference is a genuine
    mismatch.
    """
    kwargs, n = launcher.preset(preset)
    want = launcher.local_spec(kwargs).manifest()
    defaults = ConceptSpec(stage=kwargs["stage"]).manifest()
    hits = []
    for path in sorted(CON.glob("*.json")):
        try:
            got = json.loads(path.read_text()).get("manifest")
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(got, dict) or got.get("stage") != want["stage"] or got.get("label") != want.get("label"):
            continue
        ok = all(
            got[k] == v if k in got else v == defaults.get(k)
            for k, v in want.items()
            if k != "source_sha"
        )
        if ok:
            hits.append(path)
    if len(hits) != n:
        raise SystemExit(
            f"{preset}: found {len(hits)} matching shards, expected {n}"
            f"{' -- fetch, or the run is unfinished' if len(hits) < n else ''}: {[h.name for h in hits]}"
        )
    return kwargs, n, hits


def load(preset: str) -> tuple[dict, list[dict], list]:
    kwargs, n, shards = find_shards(preset)
    payloads = [json.loads(s.read_text()) for s in shards]
    conds = payloads[0]["conditions"]
    if any(p["conditions"] != conds for p in payloads):
        raise SystemExit("shards disagree on conditions")
    # A shard is checkpointed as it runs, so a file existing does not mean the
    # shard finished: c32query lost two shards to a crash and still produced
    # four readable files, 537 units instead of 661. Check the unit count
    # against what this shard was asked to do.
    if payloads[0]["manifest"].get("stage") in ("swap", "additive"):
        spec = launcher.local_spec(kwargs)
        want_units = [u.key for u, _, _ in cu_eligible(spec)]
        conds_n = len(payloads[0]["conditions"])
        missing_all = []
        for i, (path, p) in enumerate(zip(shards, payloads)):
            expected = set(want_units[i :: len(shards)])
            got = {r["key"] for r in p["records"]}
            missing_all += sorted(expected - got)
            # A unit present with short condition lists is the dangerous case:
            # arms would be compared on different populations.
            ragged = [r["key"] for r in p["records"] if len(r["kl"]) != conds_n]
            if ragged:
                raise SystemExit(
                    f"{preset}: {path.name} has {len(ragged)} records with fewer than {conds_n} "
                    f"conditions (e.g. {ragged[:3]}). Rerun it."
                )
        if missing_all:
            # Uniformly absent units shrink the panel but cannot bias a paired
            # between-arm comparison, since every arm loses the same units.
            LOAD_CAVEATS[preset] = (
                f"{len(missing_all)} of {len(want_units)} eligible units are absent from **every** arm "
                f"of this run (e.g. {missing_all[:3]}), so the panel is smaller than the frozen eligibility "
                "list. Paired arm comparisons are unaffected; absolute rates are over the reduced panel."
            )
            print(f"NOTE: {preset}: {LOAD_CAVEATS[preset]}", file=sys.stderr)
    srcs = {p["manifest"]["source_sha"] for p in payloads}
    if len(srcs) != 1:
        raise SystemExit(f"shards were produced by different source versions: {srcs}")
    local = launcher.local_spec(kwargs).manifest()["source_sha"]
    if srcs != {local}:
        print(f"WARNING: {preset} was produced by concept_use.py {srcs}, local is {local}", file=sys.stderr)
    records = [r for p in payloads for r in p["records"]]
    return kwargs, records, [tuple(c) for c in conds]


# --------------------------------------------------------------------------
# statistics


def cluster_boot(values: dict, stat=statistics.fmean, n=N_BOOT, seed=SEED):
    """values: cluster -> list of unit-level numbers. Percentile 95% CI of the
    mean over units, resampling clusters."""
    rng = random.Random(seed)
    keys = sorted(values)
    flat = [v for k in keys for v in values[k]]
    point = stat(flat)
    draws = []
    for _ in range(n):
        sample = [v for k in (rng.choice(keys) for _ in keys) for v in values[k]]
        draws.append(stat(sample))
    draws.sort()
    return point, draws[int(0.025 * n)], draws[int(0.975 * n) - 1]


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def fmt_ci(t, pct=True):
    p, lo, hi = t
    if pct:
        return f"{100*p:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]"
    return f"{p:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def pct(x):
    return f"{100*x:.1f}%"


# --------------------------------------------------------------------------
# C32


def unit_table(records, conds, *, clues, filler=0):
    """key -> {'express': rec, 'use': [rec...], 'neutral': [rec...], meta}."""
    units = defaultdict(lambda: {"use": [], "neutral": []})
    for r in records:
        if r["clue"] not in clues or r["filler"] != filler:
            continue
        u = units[r["key"]]
        u["meta"] = r
        if r["kind"] == "express":
            u["express"] = r
        else:
            u[r["kind"]].append(r)
    return {k: v for k, v in units.items() if "express" in v and v["use"]}


def dlo(r, k):
    c = r["clean"]
    return (r["t"][k] - r["s"][k]) - (c["t"] - c["s"])


def per_unit(u, k, metric):
    e = u["express"]
    if metric == "hit":
        ex = float(e["hit_t"][k])
        us = statistics.fmean(float(r["hit_t"][k]) for r in u["use"])
    elif metric == "dlo":
        ex = dlo(e, k)
        us = statistics.fmean(dlo(r, k) for r in u["use"])
    elif metric == "leak":
        ex = float("nan")
        us = statistics.fmean(float(r["name_t"][k]) for r in u["use"])
    elif metric == "src":
        ex = float(e["hit_s"][k])
        us = statistics.fmean(float(r["hit_s"][k]) for r in u["use"])
    else:
        raise ValueError(metric)
    return ex, us


def neutral_stats(u, k):
    kl = statistics.fmean(r["kl"][k] for r in u["neutral"])
    ok = statistics.fmean(float(r["hit_t"][k]) for r in u["neutral"])
    return kl, ok


def c32_section(records, conds, clues, title):
    units = unit_table(records, conds, clues=clues)
    idx = {c: i for i, c in enumerate(conds)}
    corpora = sorted({c for c, _, _ in conds})
    arms = list(dict.fromkeys(a for _, a, _ in conds))
    doses = sorted({d for _, _, d in conds})
    L = [f"## {title}", ""]
    n_units = len(units)
    n_use = sum(len(u["use"]) for u in units.values())
    by_type = defaultdict(int)
    for u in units.values():
        by_type[u["meta"]["type"]] += 1
    L.append(
        f"{n_units} units with an eligible express query and at least one eligible use query "
        f"({n_use} use queries; by type {dict(by_type)}). A unit's use score is the mean over its eligible use queries; "
        "every cell weights units equally."
    )
    L.append("")
    L.append("### Absolute outcomes")
    L.append("")
    L.append("Express and use are target-answer hit rates. Leak is how often the target *name* appears in a use continuation. Neutral KL and neutral accuracy are measured on the same edited prefix.")
    L.append("")
    L.append("| corpus | dose | arm | express | use | use − express | leak (use) | express Δlog-odds | use Δlog-odds | neutral KL | neutral acc |")
    L.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in corpora:
        for d in doses:
            for a in arms:
                k = idx[(c, a, d)]
                ex = [per_unit(u, k, "hit") for u in units.values()]
                lo = [per_unit(u, k, "dlo") for u in units.values()]
                lk = [per_unit(u, k, "leak")[1] for u in units.values()]
                nk = [neutral_stats(u, k) for u in units.values()]
                e = statistics.fmean(x for x, _ in ex)
                s = statistics.fmean(y for _, y in ex)
                L.append(
                    f"| {c} | {d} | {a} | {pct(e)} | {pct(s)} | {100*(s-e):+.1f} | {pct(statistics.fmean(lk))} | "
                    f"{statistics.fmean(x for x, _ in lo):+.2f} | {statistics.fmean(y for _, y in lo):+.2f} | "
                    f"{statistics.fmean(x for x, _ in nk):.3f} | {pct(statistics.fmean(y for _, y in nk))} |"
                )
    L.append("")
    L.append("### Primary contrast: gain over `full`, on expression vs on use")
    L.append("")
    L.append(
        "G = arm − full for the same unit and dose. DiD = G_use − G_express, so a positive DiD means the arm helps use more than expression. "
        "The intervals are cluster-bootstrap 95% intervals over source entities. W/L counts units where the arm's use hit is higher/lower than full's."
    )
    L.append("")
    L.append("| corpus | dose | arm | G_express (pp) | G_use (pp) | DiD (pp) | use W/L, sign p | DiD Δlog-odds |")
    L.append("|---|---:|---|---|---|---|---|---|")
    for c in corpora:
        for d in doses:
            kf = idx[(c, "full", d)]
            for a in arms:
                if a == "full":
                    continue
                k = idx[(c, a, d)]
                ge, gu, dd, dl = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
                w = l_ = 0
                for u in units.values():
                    cl = (u["meta"]["type"], u["meta"]["source"])
                    e1, u1 = per_unit(u, k, "hit")
                    e0, u0 = per_unit(u, kf, "hit")
                    ge[cl].append(e1 - e0)
                    gu[cl].append(u1 - u0)
                    dd[cl].append((u1 - u0) - (e1 - e0))
                    x1, y1 = per_unit(u, k, "dlo")
                    x0, y0 = per_unit(u, kf, "dlo")
                    dl[cl].append((y1 - y0) - (x1 - x0))
                    w += u1 > u0
                    l_ += u1 < u0
                L.append(
                    f"| {c} | {d} | {a} | {fmt_ci(cluster_boot(ge))} | {fmt_ci(cluster_boot(gu))} | "
                    f"**{fmt_ci(cluster_boot(dd))}** | {w}/{l_}, p={sign_test(w, l_):.3g} | {fmt_ci(cluster_boot(dl), pct=False)} |"
                )
    L.append("")
    L.append("### Conditional on the write landing: P(use hit | express hit)")
    L.append("")
    L.append(
        "The percentage-point DiD above is confounded by where each arm sits on its own dose curve: a *weaker* arm loses more "
        "on express (the higher baseline) than on use, so it scores a positive DiD for no good reason. `rand_m` -- a direction that "
        "does nothing -- makes that visible, and is the reason this section, not that table, carries the claim. "
        "Here the population is restricted to units where the concept was demonstrably written (the express query returned the target), "
        "and the question is whether the same edit is then usable."
    )
    L.append("")
    L.append("| corpus | dose | arm | n express hits | P(use \\| express) | P(use hit) overall |")
    L.append("|---|---:|---|---:|---|---:|")
    for c in corpora:
        for d in doses:
            for a in arms:
                k = idx[(c, a, d)]
                cond = defaultdict(list)
                n = 0
                for u in units.values():
                    e, us = per_unit(u, k, "hit")
                    if e > 0.5:
                        cond[(u["meta"]["type"], u["meta"]["source"])].append(us)
                        n += 1
                overall = statistics.fmean(per_unit(u, k, "hit")[1] for u in units.values())
                cell = fmt_ci(cluster_boot(cond)).split(" [")[0].lstrip("+") if n >= 10 else "-"
                ci = fmt_ci(cluster_boot(cond)) if n >= 10 else ""
                L.append(f"| {c} | {d} | {a} | {n} | {ci if n >= 10 else 'too few express hits'} | {pct(overall)} |")
    L.append("")
    L.append("### Matched expression effect")
    L.append("")
    L.append(
        "Each arm's (express hit, use hit) pairs across the three doses, so arms can be compared where they write equally often "
        "rather than at a shared norm. Read down a column: at a given express rate, which arm converts it into use?"
    )
    L.append("")
    L.append("| corpus | arm | (express, use) at dose 0.25 | 0.5 | 1.0 |")
    L.append("|---|---|---|---|---|")
    for c in corpora:
        for a in arms:
            cells = []
            for d in doses:
                k = idx[(c, a, d)]
                e = statistics.fmean(per_unit(u, k, "hit")[0] for u in units.values())
                us = statistics.fmean(per_unit(u, k, "hit")[1] for u in units.values())
                cells.append(f"({pct(e)}, {pct(us)})")
            L.append(f"| {c} | {a} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("### Is the projection gain specific to the target's own unembedding row?")
    L.append("")
    L.append("`proj_m` − `rproj_m`: both remove the same norm from each full column, one along the entity's own u and the other along a random orientation.")
    L.append("")
    L.append("| corpus | dose | express (pp) | use (pp) | leak (pp) |")
    L.append("|---|---:|---|---|---|")
    for c in corpora:
        for d in doses:
            k1, k0 = idx[(c, "proj_m", d)], idx[(c, "rproj_m", d)]
            ge, gu, gl = defaultdict(list), defaultdict(list), defaultdict(list)
            for u in units.values():
                cl = (u["meta"]["type"], u["meta"]["source"])
                e1, u1 = per_unit(u, k1, "hit")
                e0, u0 = per_unit(u, k0, "hit")
                ge[cl].append(e1 - e0)
                gu[cl].append(u1 - u0)
                gl[cl].append(per_unit(u, k1, "leak")[1] - per_unit(u, k0, "leak")[1])
            L.append(f"| {c} | {d} | {fmt_ci(cluster_boot(ge))} | {fmt_ci(cluster_boot(gu))} | {fmt_ci(cluster_boot(gl))} |")
    L.append("")
    L.append("### By type and relation (dose 1.0, `off_m` vs `full`)")
    L.append("")
    L.append("| corpus | type | relation | n queries | full hit | off_m hit | proj_m hit |")
    L.append("|---|---|---|---:|---:|---:|---:|")
    for c in corpora:
        cells = defaultdict(lambda: [0, 0, 0, 0])
        for u in units.values():
            for r in [u["express"]] + u["use"]:
                cell = cells[(r["type"], r["relation"])]
                cell[0] += 1
                for j, a in enumerate(("full", "off_m", "proj_m")):
                    cell[j + 1] += r["hit_t"][idx[(c, a, 1.0)]]
        for (ty, rel), (n, f, o, p) in sorted(cells.items()):
            L.append(f"| {c} | {ty} | {rel} | {n} | {f} | {o} | {p} |")
    L.append("")
    return L


def norms_table(preset_name):
    kwargs, n, shards = find_shards(preset_name)
    per = defaultdict(list)
    for path in shards:
        payload = json.loads(path.read_text())
        for unit, byc in payload.get("norms", {}).items():
            for corpus, byarm in byc.items():
                for arm, bylayer in byarm.items():
                    per[(corpus, arm)].append(statistics.fmean(bylayer.values()))
    L = ["### Mean edit row norm at dose 1.0", "",
         "The swap clamp's magnitude is set by the geometry of the two lens directions, so a dose is **not** "
         "comparable across corpora: the same dose buys a different perturbation. Matched arms equal `full` by construction.",
         "", "| corpus | arm | mean row norm |", "|---|---|---:|"]
    for (c, a), v in sorted(per.items()):
        L.append(f"| {c} | {a} | {statistics.fmean(v):.3f} |")
    L.append("")
    return L


def cmd_c32():
    kwargs, records, conds = load("c32")
    L = [
        "# C32 - one edit, two readouts: does filtering help *using* a concept more than *saying* it? (qwen3-8b)",
        "",
        "The same rank-2 swap (source concept → target concept) is applied to the prefix positions of a shared prefix, and only the unedited question after it varies: *name the concept* (express) or *compute with it* (use: capital, language, continent, first name, birth country, chemical symbol, state). The model is causal, so the prefix edit is literally the same tensor under both questions. Every `_m` arm is rescaled to `full`'s per-layer, per-position edit norm. Protocol: `docs/PROTOCOL.md`. Panel: `data/heldout/concept-use.json`.",
        "",
        f"Conditions: {len(conds)} (corpora × arms × doses). Records: {len(records)}.",
        "",
    ]
    L += norms_table("c32")
    L += c32_section(records, conds, {"0", "1"}, "Latent clues (primary)")
    L += c32_section(records, conds, {"name"}, "Explicit name in the prefix (secondary)")
    out = RES / "c32_express_vs_use_qwen3-8b.md"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")


def cmd_c32prompt():
    """Same panel and arms, edits at every prompt position (the earlier convention)."""
    _, rec_p, conds = load("c32prompt")
    _, rec_x, conds_x = load("c32")
    if conds != conds_x:
        raise SystemExit("the two runs disagree on conditions")
    L = [
        "# C32-prompt - the same swaps applied at every prompt position (qwen3-8b)",
        "",
        "Identical panel, arms, doses and norm matching as C32; the only change is *where* the edit goes. "
        "C32 edits the shared prefix, so one intervention serves both questions. This run edits every prompt position, "
        "including the question itself -- the convention the J-lens paper uses (\"at all token positions\") and the one the "
        "earlier two-hop panel used, where norm-matched `off` beat `full`. The paired reading does not hold here: "
        "the edit differs between the express and use prompts because those prompts differ.",
        "",
    ]
    L += c32_section(rec_p, conds, {"0", "1"}, "Latent clues, edits at every prompt position")
    # side-by-side on the common units
    up = unit_table(rec_p, conds, clues={"0", "1"})
    ux = unit_table(rec_x, conds, clues={"0", "1"})
    keys = sorted(set(up) & set(ux))
    idx = {c: i for i, c in enumerate(conds)}
    L += ["", "## Prefix-only versus whole-prompt edits, on the same units", "",
          f"{len(keys)} units in both runs. Each cell is the use hit rate.", "",
          "| corpus | dose | arm | prefix only | whole prompt | difference |",
          "|---|---:|---|---:|---:|---:|"]
    for c in sorted({c for c, _, _ in conds}):
        for d in sorted({d for _, _, d in conds}):
            for a in list(dict.fromkeys(a for _, a, _ in conds)):
                k = idx[(c, a, d)]
                x = statistics.fmean(per_unit(ux[u], k, "hit")[1] for u in keys)
                y = statistics.fmean(per_unit(up[u], k, "hit")[1] for u in keys)
                L.append(f"| {c} | {d} | {a} | {pct(x)} | {pct(y)} | {100*(y-x):+.1f} |")
    out = RES / "c32prompt_position_convention_qwen3-8b.md"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")


def cmd_c36():
    """Rescaling control: does the off-vs-full ordering follow kappa or the direction?"""
    _, records, conds = load("c36kappa")
    units = unit_table(records, conds, clues={"0", "1"})
    idx = {c: i for i, c in enumerate(conds)}
    corpora = sorted({c for c, _, _ in conds})
    arms = list(dict.fromkeys(a for _, a, _ in conds))
    doses = sorted({d for _, _, d in conds})
    L = [
        "# C36 - is the filtering effect geometry or magnitude? (qwen3-8b)",
        "",
        "The rank-2 clamp is invariant to *uniform* rescaling of `V = [v_s, v_t]` but not to scaling one column "
        "against the other, so `kappa = ||v_t||/||v_s||` is a free parameter of every arm, and per-position norm "
        "matching cannot undo it. Off-diagonal directions are about 30% shorter than full ones and their kappa "
        "spreads differently (10-90%: 0.82-1.21 vs 0.86-1.17), so the whole off-versus-full result could in "
        "principle be a kappa effect.",
        "",
        "`_cn` arms unit-normalise both columns (kappa = 1 for every family, so arms differ only in direction). "
        "`_k<v>` sweeps kappa explicitly. **`full_koff` gives `full`'s directions `off`'s per-pair kappa, and "
        "`off_kfull` the reverse: if the ordering follows kappa, these swap places.** "
        "All arms except `full` are norm-matched to `full` as before, and the convention is whole-prompt, "
        "where the off-versus-full gap lives.",
        "",
        f"{len(units)} units, {len(conds)} conditions.",
        "",
    ] + ([f"> **Panel note.** {LOAD_CAVEATS['c36kappa']}", ""] if "c36kappa" in LOAD_CAVEATS else []) + [
        "| corpus | dose | arm | express | use | leak | P(use\\|express) | neutral KL |",
        "|---|---:|---|---:|---:|---:|---|---:|",
    ]
    for c in corpora:
        for d in doses:
            for a in arms:
                k = idx[(c, a, d)]
                e = statistics.fmean(per_unit(u, k, "hit")[0] for u in units.values())
                us = statistics.fmean(per_unit(u, k, "hit")[1] for u in units.values())
                lk = statistics.fmean(per_unit(u, k, "leak")[1] for u in units.values())
                nk = statistics.fmean(neutral_stats(u, k)[0] for u in units.values())
                cond = defaultdict(list)
                n = 0
                for u in units.values():
                    ex, uu = per_unit(u, k, "hit")
                    if ex > 0.5:
                        cond[(u["meta"]["type"], u["meta"]["source"])].append(uu)
                        n += 1
                cv = fmt_ci(cluster_boot(cond)) if n >= 10 else "-"
                L.append(f"| {c} | {d} | {a} | {pct(e)} | {pct(us)} | {pct(lk)} | {cv} | {nk:.3f} |")
    L += ["", "## The decisive comparison", "",
          "If filtering is a magnitude artifact, `full_koff` should behave like `off`, and `off_kfull` like `full`. "
          "If it is geometric, each arm should track its own family whatever kappa it is given.", "",
          "| corpus | dose | quantity | full_cn | full_koff | off_cn | off_kfull |",
          "|---|---:|---|---:|---:|---:|---:|"]
    for c in corpora:
        for d in doses:
            for label, metric in (("use hit", "hit"), ("leak", "leak")):
                cells = []
                for a in ("full_cn", "full_koff", "off_cn", "off_kfull"):
                    k = idx[(c, a, d)]
                    v = statistics.fmean(per_unit(u, k, metric)[1] for u in units.values())
                    cells.append(pct(v))
                L.append(f"| {c} | {d} | {label} | " + " | ".join(cells) + " |")
    out = RES / "c36_rescaling_control_qwen3-8b.md"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")


def cmd_c37():
    """Lambda interpolation: J_lambda = J_off + lambda*J_diag."""
    out_lines = [
        "# C37 - the diagonal as a continuum, not a switch (qwen3-8b)",
        "",
        "`J_lambda = J_off + lambda * J_diag`. The pullback is linear, so the write direction is exactly "
        "`v_lambda = v_off + lambda * v_diag`, taken from the existing component fit (`full` and `diag + off` agree "
        "to 4e-8 relative). lambda = 0 is the off-diagonal arm, lambda = 1 reproduces the full lens, and lambda > 1 "
        "overshoots it. Every arm is norm-matched per layer and position to `full`; `_cn` arms additionally "
        "unit-normalise both swap columns, so the relative column scale (kappa) cannot carry the effect.",
        "",
        "The prediction under test: adding the diagonal should move the intervention smoothly from a "
        "future-use write toward an immediate lexical emission -- leakage up, downstream use down, alignment with "
        "`u_y` up -- rather than the off-diagonal result being an artifact of deleting a term.",
        "",
    ]
    for preset_name, title in (("c37lambdaprompt", "Edits at every prompt position"),
                               ("c37lambdaprefix", "Edits on the prefix only (control)")):
        try:
            _, records, conds = load(preset_name)
        except SystemExit as e:
            out_lines += [f"## {title}", "", f"_not available: {e}_", ""]
            continue
        units = unit_table(records, conds, clues={"0", "1"})
        idx = {c: i for i, c in enumerate(conds)}
        geom = {}
        kwargs, n, shards = find_shards(preset_name)
        geom = json.loads(shards[0].read_text()).get("geometry", {})
        out_lines += [f"## {title}", "", f"{len(units)} units.", ""]
        for variant, label in (("_m", "native columns, delta norm-matched"),
                               ("_cn", "columns unit-normalised (kappa = 1)")):
            out_lines += [
                f"### {label}", "",
                "| corpus | dose | lambda | use hit | express | leak | P(use\\|express) | neutral KL | cos(v,off) | cos(v,u_y) | \\|v\\| |",
                "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
            ]
            for c in sorted({x for x, _, _ in conds}):
                for d in sorted({x for _, _, x in conds}):
                    arms_v = list(dict.fromkeys(x for _, x, _ in conds if x.endswith(variant)))
                    if variant == "_m":
                        arms_v = [x for x in dict.fromkeys(y for _, y, _ in conds) if x == "full"] + arms_v
                    for a in arms_v:
                        if (c, a, d) not in idx:
                            continue
                        k = idx[(c, a, d)]
                        e = statistics.fmean(per_unit(u, k, "hit")[0] for u in units.values())
                        us = statistics.fmean(per_unit(u, k, "hit")[1] for u in units.values())
                        lk = statistics.fmean(per_unit(u, k, "leak")[1] for u in units.values())
                        nk = statistics.fmean(neutral_stats(u, k)[0] for u in units.values())
                        cond = defaultdict(list)
                        cnt = 0
                        for u in units.values():
                            ex, uu = per_unit(u, k, "hit")
                            if ex > 0.5:
                                cond[(u["meta"]["type"], u["meta"]["source"])].append(uu)
                                cnt += 1
                        cv = fmt_ci(cluster_boot(cond)) if cnt >= 10 else "-"
                        g = [geom.get(f"{c}/{a}/L{l}", {}) for l in (13, 16, 19, 22, 25, 28, 31)]
                        g = [x for x in g if x]
                        co = f"{statistics.fmean(x['cos_off'] for x in g):.3f}" if g else "-"
                        cu_ = f"{statistics.fmean(x['cos_uy'] for x in g):.3f}" if g else "-"
                        nm = f"{statistics.fmean(x['norm'] for x in g):.2f}" if g else "-"
                        lam = a.replace(variant, "").replace("lam", "")
                        out_lines.append(
                            f"| {c} | {d} | {lam} | {pct(us)} | {pct(e)} | {pct(lk)} | {cv} | {nk:.3f} | {co} | {cu_} | {nm} |"
                        )
            out_lines.append("")
    path = RES / "c37_lambda_interpolation_qwen3-8b.md"
    path.write_text("\n".join(out_lines) + "\n")
    print(f"wrote {path}")


# --------------------------------------------------------------------------
# C33 - distance


def cmd_c33():
    _, rec0, cond0 = load("c32")
    _, recd, condd = load("c33")
    arms = [a for _, a, _ in condd]
    arms = list(dict.fromkeys(arms))
    doses = sorted({d for _, _, d in condd})
    corpora = sorted({c for c, _, _ in condd})
    i0 = {c: i for i, c in enumerate(cond0)}
    id_ = {c: i for i, c in enumerate(condd)}
    fillers = sorted({r["filler"] for r in recd})
    # units eligible at EVERY filler, latent clues only
    tabs = {0: unit_table(rec0, cond0, clues={"0", "1"}, filler=0)}
    for f in fillers:
        tabs[f] = unit_table(recd, condd, clues={"0", "1"}, filler=f)
    keys = set.intersection(*[set(t) for t in tabs.values()])
    L = [
        "# C33 - how far does a prompt-only concept write reach? (qwen3-8b)",
        "",
        "One rank-2 swap on the prefix, then 0/1/2/4/8 unrelated filler sentences, then the same express or use question. "
        "The filler is fixed text (teacher-forced), never edited, and nothing is re-injected during generation. "
        "The population is the units eligible at *every* distance, so the curves are paired.",
        "",
        f"{len(keys)} units eligible at all {len(tabs)} distances. Filler 0 comes from the C32 run; the rest from the C33 run.",
        "",
        "## Target-answer hit rate and Δlog-odds by distance",
        "",
        "| corpus | dose | arm | measure | " + " | ".join(f"f={f}" for f in sorted(tabs)) + " |",
        "|---|---:|---|---|" + "---:|" * len(tabs),
    ]
    for c in corpora:
        for d in doses:
            for a in arms:
                for kind, metric in (("express", "hit"), ("use", "hit"), ("use", "dlo")):
                    cells = []
                    for f in sorted(tabs):
                        k = (i0 if f == 0 else id_)[(c, a, d)]
                        vals = [per_unit(tabs[f][u], k, metric) for u in sorted(keys)]
                        v = statistics.fmean(x if kind == "express" else y for x, y in vals)
                        cells.append(pct(v) if metric == "hit" else f"{v:+.2f}")
                    lab = f"{kind} {'hit' if metric == 'hit' else 'Δlog-odds'}"
                    L.append(f"| {c} | {d} | {a} | {lab} | " + " | ".join(cells) + " |")
    L += ["", "## Does filtering's advantage change with distance?", "",
          "`off_m` − `full` on use, at each distance, with cluster-bootstrap intervals over source entities.", "",
          "| corpus | dose | " + " | ".join(f"f={f}" for f in sorted(tabs)) + " |",
          "|---|---:|" + "---|" * len(tabs)]
    for c in corpora:
        for d in doses:
            cells = []
            for f in sorted(tabs):
                idx = i0 if f == 0 else id_
                g = defaultdict(list)
                for u in sorted(keys):
                    rec = tabs[f][u]
                    cl = (rec["meta"]["type"], rec["meta"]["source"])
                    g[cl].append(per_unit(rec, idx[(c, "off_m", d)], "hit")[1]
                                 - per_unit(rec, idx[(c, "full", d)], "hit")[1])
                cells.append(fmt_ci(cluster_boot(g)))
            L.append(f"| {c} | {d} | " + " | ".join(cells) + " |")
    out = RES / "c33_distance_qwen3-8b.md"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# C33b - free generation


def cmd_c33b():
    _, records, conds = load("c33b")
    by = defaultdict(list)
    for r in records:
        by[tuple(r["cond"])].append(r)
    order = [("clean", "clean", 0.0)] + [tuple(c) for c in conds]
    L = [
        "# C33b - free generation after a prompt-only concept write (qwen3-8b)",
        "",
        "The prefix is edited once, then 64 tokens are generated with nothing re-injected. "
        "A mention counts if the entity's name, capital or language appears in that token window. "
        "The probe appends *In one word, the capital city of this country is* after the model's own text, under the same prefix edit. "
        "Disruption is the clean model's mean NLL of the generated text (lower is more ordinary text).",
        "",
        f"{len({r['key'] for r in records})} country units.",
        "",
        "| corpus | arm | dose | target 0-16 | 16-32 | 32-64 | source 0-16 | 16-32 | 32-64 | probe target | probe source | clean NLL |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cond in order:
        rs = by.get(cond, [])
        if not rs:
            continue
        t = [statistics.fmean(float(r["win_t"][i]) for r in rs) for i in range(3)]
        s = [statistics.fmean(float(r["win_s"][i]) for r in rs) for i in range(3)]
        L.append(
            f"| {cond[0]} | {cond[1]} | {cond[2]} | " + " | ".join(pct(x) for x in t) + " | "
            + " | ".join(pct(x) for x in s) + " | "
            + pct(statistics.fmean(float(r["probe_t"]) for r in rs)) + " | "
            + pct(statistics.fmean(float(r["probe_s"]) for r in rs)) + " | "
            + f"{statistics.fmean(r['nll'] for r in rs):.2f} |"
        )
    out = RES / "c33b_freegen_qwen3-8b.md"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# C34 - dev grid selection


KL_GATE = 0.5
ACC_GATE = 0.9


def dev_stats(records, conds):
    """condition index -> dict of dev means."""
    use, exp, nkl, nok = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    clean_ok = []
    for r in records:
        for k in range(len(conds)):
            if r["kind"] == "use":
                use[k].append(float(r["top1"][k]))
            elif r["kind"] == "express":
                exp[k].append(float(r["top1"][k]))
            else:
                nkl[k].append(r["kl"][k])
                nok[k].append(float(r["top1"][k]))
        if r["kind"] == "neutral":
            clean_ok.append(float(r["clean"]["top1"]))
    base = statistics.fmean(clean_ok) if clean_ok else 1.0
    return {
        k: {
            "use": statistics.fmean(use[k]), "express": statistics.fmean(exp[k]),
            "kl": statistics.fmean(nkl[k]), "neutral_ok": statistics.fmean(nok[k]),
        }
        for k in range(len(conds))
    }, base


def cmd_select():
    kwargs, records, conds = load("c34dev")
    stats, base = dev_stats(records, conds)
    methods = list(dict.fromkeys(m for m, *_ in conds))
    layer_sets = list(dict.fromkeys(ls for _, ls, _, _ in conds))
    pos_sets = list(dict.fromkeys(ps for _, _, ps, _ in conds))
    alphas = sorted({a for *_, a in conds})
    frozen, rows = {}, []
    for m in methods:
        cand = [
            (k, c) for k, c in enumerate(conds)
            if c[0] == m and stats[k]["neutral_ok"] >= ACC_GATE * base and stats[k]["kl"] <= KL_GATE
        ]
        if not cand:
            rows.append((m, None, None, None, None, None))
            continue
        k, c = max(cand, key=lambda kc: (stats[kc[0]]["use"], -kc[1][3], -len(kwargs["layer_sets"][kc[1][1]])))
        frozen[m] = {"layer_set": c[1], "position_set": c[2], "alpha": c[3],
                     "dev_use": stats[k]["use"], "dev_express": stats[k]["express"],
                     "dev_kl": stats[k]["kl"], "dev_neutral_ok": stats[k]["neutral_ok"]}
        rows.append((m, c[1], c[2], c[3], stats[k]["use"], stats[k]["express"]))
    cond_list = []
    for m, f in frozen.items():
        for a in alphas:
            cond_list.append([m, f["layer_set"], f["position_set"], a])
            if (f["layer_set"], f["position_set"]) != ("all", "prefix"):
                cond_list.append([m, "all", "prefix", a])
    out = CON / "frozen_c34.json"
    out.write_text(json.dumps({"gates": {"kl": KL_GATE, "neutral_acc_ratio": ACC_GATE, "clean_neutral": base},
                               "frozen": frozen, "condition_list": cond_list}, indent=1))
    L = [
        "# C34 dev - where and how hard to write (development grid, dev entities only)",
        "",
        "Exploratory. Selection rule, fixed before the run: among conditions with neutral accuracy ≥ "
        f"{ACC_GATE}× clean ({base:.2f}) and mean neutral KL ≤ {KL_GATE} nats, maximise dev use success "
        "(the answer's first token is top-1), breaking ties by smaller alpha, then fewer layers. "
        "The chosen layer set and position set are frozen per method; the test run sweeps the whole alpha grid at that configuration "
        "and also at a common `all`/`prefix` reference.",
        "",
        "## Selected configuration per method",
        "",
        "| method | layers | positions | alpha | dev use | dev express |",
        "|---|---|---|---:|---:|---:|",
    ]
    for m, ls, ps, a, u, e in rows:
        L.append(f"| {m} | {ls or '-'} | {ps or '-'} | {a if a is not None else '-'} | "
                 f"{pct(u) if u is not None else 'no config passes the gate'} | {pct(e) if e is not None else '-'} |")
    L += ["", "## Best gated use success by layer set x position set", "",
          "Each cell is the best gated success over the alpha grid.", ""]
    for m in methods:
        L += [f"### {m}", "", "| layers | " + " | ".join(pos_sets) + " |", "|---|" + "---:|" * len(pos_sets)]
        for ls in layer_sets:
            cells = []
            for ps in pos_sets:
                ok = [stats[k]["use"] for k, c in enumerate(conds)
                      if c[0] == m and c[1] == ls and c[2] == ps
                      and stats[k]["neutral_ok"] >= ACC_GATE * base and stats[k]["kl"] <= KL_GATE]
                cells.append(pct(max(ok)) if ok else "-")
            L.append(f"| {ls} | " + " | ".join(cells) + " |")
        L.append("")
    path = RES / "c34_dev_grid_qwen3-8b.md"
    path.write_text("\n".join(L) + "\n")
    print(f"wrote {path} and {out}")


# --------------------------------------------------------------------------
# C34/C35 test


def cmd_c34():
    kwargs, records, conds = load("c34test")
    frozen = json.loads((CON / "frozen_c34.json").read_text())["frozen"]
    alphas = sorted({a for *_, a in conds})
    idx = {tuple(c): i for i, c in enumerate(conds)}
    units = defaultdict(lambda: {"use": [], "neutral": [], "express": []})
    for r in records:
        units[r["key"]][r["kind"]].append(r)
    keys = [k for k, v in units.items() if v["use"] and v["neutral"]]

    def curve(method, cfg):
        rows = []
        for a in alphas:
            k = idx.get((method, cfg[0], cfg[1], a))
            if k is None:
                continue
            use = statistics.fmean(float(r["hit_t"][k]) for u in keys for r in units[u]["use"])
            exp = statistics.fmean(float(r["hit_t"][k]) for u in keys for r in units[u]["express"])
            leak = statistics.fmean(float(r["name_t"][k]) for u in keys for r in units[u]["use"])
            kl = statistics.fmean(r["kl"][k] for u in keys for r in units[u]["neutral"])
            ok = statistics.fmean(float(r["top1"][k]) for u in keys for r in units[u]["neutral"])
            rows.append((a, use, exp, leak, kl, ok))
        return rows

    def crossing(method, cfg, tau):
        """Per use query: the alpha at which p(target) first reaches tau, by
        log-linear interpolation between grid points; and the neutral KL there."""
        reach, kls = [], []
        for u in keys:
            ks = [(a, idx.get((method, cfg[0], cfg[1], a))) for a in alphas]
            ks = [(a, k) for a, k in ks if k is not None]
            nkl = [statistics.fmean(r["kl"][k] for r in units[u]["neutral"]) for _, k in ks]
            for r in units[u]["use"]:
                p = [math.exp(r["t"][k]) for _, k in ks]
                hit = next((i for i, v in enumerate(p) if v >= tau), None)
                if hit is None:
                    reach.append(None)
                    continue
                if hit == 0:
                    a_star, kl_star = ks[0][0], nkl[0]
                else:
                    x0, x1 = math.log(ks[hit - 1][0]), math.log(ks[hit][0])
                    y0, y1 = p[hit - 1], p[hit]
                    w = 0 if y1 == y0 else (tau - y0) / (y1 - y0)
                    a_star = math.exp(x0 + w * (x1 - x0))
                    kl_star = nkl[hit - 1] + w * (nkl[hit] - nkl[hit - 1])
                reach.append(a_star)
                kls.append(kl_star)
        cov = sum(x is not None for x in reach) / len(reach)
        med_a = statistics.median([x for x in reach if x is not None]) if kls else float("nan")
        med_kl = statistics.median(kls) if kls else float("nan")
        return cov, med_a, med_kl

    L = [
        "# C34/C35 - targeted additive steering on held-out entities (qwen3-8b)",
        "",
        "Target-only additions (no source subtraction) at `alpha × (layer mean residual norm) × unit(v)`, "
        "so at a given alpha every method spends the same per-position edit norm. "
        "Each method uses the layer set and position set frozen on the dev entities, and a common `all`/`prefix` reference. "
        "Test entities never appear in the dev grid, as source or as target.",
        "",
        f"{len(keys)} test units. Success-disruption pairs are (neutral KL, use hit rate) along the alpha grid.",
        "",
        "## Frozen configuration and the full dose curve",
        "",
        "| method | layers | positions | alpha | use hit | express hit | leak | neutral KL | neutral acc |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m, f in frozen.items():
        for a, use, exp, leak, kl, ok in curve(m, (f["layer_set"], f["position_set"])):
            star = " **" if a == f["alpha"] else " "
            L.append(f"| {m} | {f['layer_set']} | {f['position_set']} |{star}{a}{star.strip()} | {pct(use)} | {pct(exp)} | {pct(leak)} | {kl:.3f} | {pct(ok)} |")
    L += ["", "The starred alpha is the frozen operating point. ", "",
          "## Common reference configuration (all band layers, prefix positions)", "",
          "| method | alpha | use hit | express hit | leak | neutral KL |", "|---|---:|---:|---:|---:|---:|"]
    for m in frozen:
        for a, use, exp, leak, kl, ok in curve(m, ("all", "prefix")):
            L.append(f"| {m} | {a} | {pct(use)} | {pct(exp)} | {pct(leak)} | {kl:.3f} |")
    L += ["", "## Equal success: dose needed to reach a target probability, and the disruption there", "",
          "Interpolated between grid points (log-linear), not exact bisection. Coverage is the share of use queries reaching the level within the grid; "
          "the medians are over those that do.", "",
          "| method | tau | coverage | median alpha | median neutral KL |", "|---|---:|---:|---:|---:|"]
    for m, f in frozen.items():
        for tau in (0.3, 0.5):
            cov, a, kl = crossing(m, (f["layer_set"], f["position_set"]), tau)
            L.append(f"| {m} | {tau} | {pct(cov)} | {a:.3f} | {kl:.3f} |")
    L += ["", "## Supervision and fitting cost", "",
          "| method | what it needs |", "|---|---|",
          "| `*_full`, `*_off` | 32 unlabelled corpus documents; backward passes at stride-4 positions |",
          "| `*_proj` | the same, plus the target's unembedding row (free) |",
          "| `logit` | nothing beyond the model |",
          "| `diffmean` | 12 labelled sentences per concept, plus the type's other concepts as negatives |",
          "| `template` | 12 labelled continuation contexts per concept (J-lens App. A.9.1 recipe) |",
          "| `rand` | nothing; the floor |", ""]
    out = RES / "c34_c35_additive_test_qwen3-8b.md"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("c32", "c32prompt", "c33", "c33b", "select", "c34", "c36", "c37"))
    a = ap.parse_args()
    {"c32": cmd_c32, "c32prompt": cmd_c32prompt, "c36": cmd_c36, "c37": cmd_c37, "c33": cmd_c33, "c33b": cmd_c33b, "select": cmd_select, "c34": cmd_c34}[a.command]()


if __name__ == "__main__":
    main()

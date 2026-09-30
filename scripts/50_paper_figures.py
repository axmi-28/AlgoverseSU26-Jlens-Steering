"""Audited, publication-size figures from pinned result artifacts. No model/GPU calls.

Run: .venv/bin/python scripts/50_paper_figures.py
Outputs: output/figures/ (SVG/PNG, source audit and captions); output/pdf/ (vector PDFs).
Intervals: source-cluster percentile bootstrap, 10,000 draws, fixed seed. They
condition on fitted directions and do not estimate fitting/model uncertainty.
"""
from __future__ import annotations

import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

import pymupdf as fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/figures"
PDF = ROOT / "output/pdf"
OUT.mkdir(parents=True, exist_ok=True)
PDF.mkdir(parents=True, exist_ok=True)
SEED, NBOOT = 160926, 10000
C = {"full": "#28536B", "off": "#00856A", "proj": "#A45486", "diag": "#929AA3",
     "logit": "#BD892C", "diffmean": "#C35437", "template": "#689DB6", "rand": "#B8BDC3"}
LABEL = {"full": "Full J-lens", "off": "Off-diagonal", "proj": "Unembedding projection",
         "diag": "Diagonal only", "logit": "Unembedding", "diffmean": "DiffMean (supervised)",
         "template": "Template lens", "rand": "Random"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
 "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "axes.titleweight": "normal",
 "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
 "axes.linewidth": .65, "lines.linewidth": 1.35, "lines.markersize": 4.5,
 "xtick.major.width": .65, "ytick.major.width": .65,
 "xtick.major.size": 3, "ytick.major.size": 3, "axes.spines.top": False,
 "axes.spines.right": False, "axes.grid": False, "pdf.fonttype": 42,
 "ps.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "jlens-20260916",
 "savefig.facecolor": "white", "figure.facecolor": "white", "text.color": "#20252A",
 "axes.labelcolor": "#20252A", "xtick.color": "#41484F", "ytick.color": "#41484F"})
SOURCES, STATS, FIGS, AUDIT = {}, {}, [], {}


def read(rel):
    p = ROOT / rel
    raw = p.read_bytes()
    SOURCES[rel] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return json.loads(raw)


def raw_shards(stem, n):
    return [r for i in range(n) for r in read(f"results/causal/{stem}_shard{i}of{n}.json")]


def load_concept(stem):
    ps = [read(f"results/concept/{stem}_shard{i}of4.json") for i in range(4)]
    conditions = [tuple(x) for x in ps[0]["conditions"]]
    assert len(set(conditions)) == len(conditions)
    assert all(p["conditions"] == ps[0]["conditions"] for p in ps)
    assert len({p["manifest"]["source_sha"] for p in ps}) == 1
    rows = [r for p in ps for r in p["records"]]
    ids = [(r["key"], r["relation"], r.get("filler", 0)) for r in rows]
    assert len(ids) == len(set(ids)), f"duplicate query: {stem}"
    for r in rows:
        for field in ("hit_t", "kl", "name_t"):
            assert len(r[field]) == len(conditions), (stem, field)
    AUDIT[stem] = {"query_records": len(rows), "units_all_clues": len({r["key"] for r in rows}),
                  "conditions": len(conditions), "dead_rows": sum(sum(p.get("dead_rows", {}).values()) for p in ps),
                  "manifest": ps[0]["manifest"]}
    return rows, conditions


def unit_arrays(rows, conditions, filler=0, keys=None):
    units = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["clue"] not in ("0", "1") or r.get("filler", 0) != filler:
            continue
        units[r["key"]][r["kind"]].append(r)
    allowed = {k for k, u in units.items() if u["express"] and u["use"]}
    if keys is not None:
        assert set(keys) <= allowed
        allowed = set(keys)
    keys = sorted(allowed)
    data = {name: [] for name in ("use", "express", "leak", "kl", "dlo")}
    groups, query_counts = [], []
    for k in keys:
        u = units[k]
        assert len(u["express"]) == 1
        e = u["express"][0]
        groups.append(e["type"] + "/" + e["source"])
        query_counts.append(len(u["use"]))
        data["express"].append(e["hit_t"])
        data["use"].append(np.mean([r["hit_t"] for r in u["use"]], axis=0))
        data["leak"].append(np.mean([r["name_t"] for r in u["use"]], axis=0))
        data["kl"].append(np.mean([r["kl"] for r in u["neutral"]], axis=0) if u["neutral"] else np.full(len(conditions), np.nan))
        data["dlo"].append(np.mean([np.asarray(r["t"])-r["s"]-(r["clean"]["t"]-r["clean"]["s"]) for r in u["use"]], axis=0))
    return {**{m: np.asarray(v, float) for m, v in data.items()}, "keys": keys,
            "groups": groups, "n_queries": sum(query_counts), "conditions": conditions}


def stats(values, groups, tag):
    v = np.asarray(values, float)
    if v.ndim == 1:
        v = v[:, None]
    assert len(v) == len(groups) and np.isfinite(v).all(), tag
    gs = sorted(set(groups)); ix = {g: i for i, g in enumerate(gs)}
    totals = np.zeros((len(gs), v.shape[1])); counts = np.zeros(len(gs))
    for row, g in zip(v, groups):
        totals[ix[g]] += row; counts[ix[g]] += 1
    rng = np.random.default_rng(SEED)
    weights = rng.multinomial(len(gs), np.ones(len(gs))/len(gs), size=NBOOT)
    draws = (weights @ totals) / (weights @ counts)[:, None]
    mean = v.mean(axis=0); lo, hi = np.quantile(draws, [.025, .975], axis=0)
    STATS[tag] = {"mean": mean.tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
                  "n_observations": len(v), "n_clusters": len(gs)}
    return mean, lo, hi


def panel(ax, letter, title):
    ax.set_title(title, loc="left", pad=12)
    ax.text(-.14, 1.10, letter, transform=ax.transAxes, fontweight="bold", fontsize=10,
            va="bottom", ha="left")


def foot(fig, text):
    fig.text(.085, .022, text, fontsize=6.7, color="#65717B", va="bottom")


def legend_methods(fig, methods, y=.98, ncol=None):
    handles = [Line2D([], [], color=C[m], marker="o", label=LABEL[m]) for m in methods]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, y),
               ncol=ncol or len(methods), frameon=False, columnspacing=1.3, handlelength=1.7)


def save(fig, name, title, caption, note):
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.png", dpi=450)
    buf = io.BytesIO(); fig.savefig(buf, format="pdf"); buf.seek(0)
    if name == "fig01_core_swap_gain":
        (PDF / "jlens_core_swap_gain.pdf").write_bytes(buf.getvalue())
    FIGS.append({"name": name, "title": title, "caption": caption, "note": note,
                 "pdf": buf.getvalue(), "size": fig.get_size_inches().tolist()})
    plt.close(fig)
    print(f"Created {name}", flush=True)


def explicit_data(rows, arms, n):
    clean = {r["prompt_key"] for r in rows if r["arm"] == "baseline" and r["hit_generated"] and r["category"] != "numbers"}
    banks = {}
    for arm in arms:
        rr = [r for r in rows if r["arm"] == arm and r["strength"] == 1 and r["prompt_key"] in clean]
        bank = {(r["prompt_key"], r["target_arg"]): int(r["hit_generated"]) for r in rr}
        assert len(bank) == len(rr) == n, (arm, len(bank))
        banks[arm] = bank
    keys = sorted(banks[arms[0]])
    assert all(set(b) == set(keys) for b in banks.values())
    return np.array([[banks[a][k] for a in arms] for k in keys]), [k[0] for k in keys]


def figure1():
    rr = raw_shards("steering_qwen3-8b_swap_comp", 2)
    arms = [f"swap_{c}_{a}" for c in ("gsm8k", "wikitext_a") for a in ("full", "off")]
    vals, groups = explicit_data(rr, arms, 105)
    assert vals.sum(axis=0).tolist() == [35, 60, 20, 50]
    left = stats(vals, groups, "fig1_explicit_rates")
    stats(vals[:, [1, 3]]-vals[:, [0, 2]], groups, "fig1_explicit_paired_gain")
    banks = []
    for corp, leaf in [("gsm8k", "dac7603b1af57dbd"), ("wikitext_a", "6fff00659415ddf5")]:
        a = read(f"results/c22/write_controls_{leaf}.json")
        r = a["records"]; names = sorted({x["name"] for x in r})
        assert len(names) == 55 and len(r) == 1265
        lookup = {(x["name"], x["arm"], x["strength"]): x for x in r}
        v = np.array([[lookup[(n, ar, 1.)]["hit_swap_answer"] for ar in ("swap_full", "swap_off_matched")] for n in names], float)
        gs = [lookup[(n, "swap_full", 1.)]["category"] for n in names]
        banks.append((v, gs, stats(v, gs, "fig1_latent_"+corp)))
        stats(v[:, 1]-v[:, 0], gs, "fig1_latent_gain_"+corp)
    assert [b[0].sum(axis=0).tolist() for b in banks] == [[15, 25], [13, 23]]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    fig.subplots_adjust(left=.12, right=.98, bottom=.23, top=.73, wspace=.42)
    for j, ax in enumerate(axs):
        panel(ax, "ab"[j], ["Explicit-argument swaps", "Latent-intermediate swaps"][j])
        for i, corp in enumerate(["GSM8K", "WikiText"]):
            m, lo, hi = (tuple(z[2*i:2*i+2] for z in left) if j == 0 else banks[i][2])
            counts = vals[:, 2*i:2*i+2].sum(axis=0) if j == 0 else banks[i][0].sum(axis=0)
            y = 1-i
            ax.plot(m*100, [y+.10, y-.10], color="#B9C3C8", lw=1, zorder=1)
            for k, method in enumerate(["full", "off"]):
                yy = y+.10-k*.20
                ax.errorbar(m[k]*100, yy, xerr=[[100*(m[k]-lo[k])], [100*(hi[k]-m[k])]], fmt="o", color=C[method], capsize=2, lw=1, zorder=3)
                ax.text(m[k]*100, yy+(.18 if k == 0 else -.20), f"{int(counts[k])}/{105 if j==0 else 55}", color=C[method], ha="center", va="center", fontsize=7.5)
            ax.text(80, y, f"+{100*(m[1]-m[0]):.1f} pp", ha="right", va="center", fontsize=8, color=C["off"])
        ax.set(xlim=(0, 82), ylim=(-.55, 1.55), yticks=[1, 0], yticklabels=["GSM8K", "WikiText"], xlabel="Target-answer success (%)", xticks=[0, 20, 40, 60, 80])
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0, pad=7)
    legend_methods(fig, ["full", "off"], y=1.0)
    foot(fig, "Qwen3-8B  |  strength 1  |  whole-prompt edits  |  development panels; 95% cluster-bootstrap intervals")
    save(fig, "fig01_core_swap_gain", "Diagonal removal improves whole-prompt swapping",
         "Removing same-position Jacobian terms improves counterfactual-answer success in both fitting corpora. The latent-intermediate gain survives matching each layer/position edit norm to the full-lens swap, so larger requested edits do not explain that contrast.",
         "Panel a: 105 non-numerical swaps from 35 clean-correct source prompts, with native swap magnitudes (C18). Panel b: 55 reused latent-intermediate items, with per-position norm matching (expanded C22). Both use strength 1 and seven layers. Intervals resample source prompts in a and supplied item categories in b, conditional on one fit per corpus. Counts and gains are paired within each panel; the panels are not independent replications or held-out tests.")


def load_positions():
    spec = {"prefix": "swap_c32_qwen3-8b_44ba057d98", "prompt": "swap_c32prompt_qwen3-8b_98dd7890f4", "query": "swap_c32query_qwen3-8b_4279d3b14d"}
    result = {}
    for pos, stem in spec.items():
        rows, conds = load_concept(stem)
        a = unit_arrays(rows, conds)
        assert len(a["keys"]) == 661 and a["n_queries"] == 1339, (pos, len(a["keys"]), a["n_queries"])
        result[pos] = a
    assert all(a["keys"] == result["prefix"]["keys"] for a in result.values())
    AUDIT["c32_common"] = {"units": 661, "queries": 1339, "source_clusters": len(set(result["prefix"]["groups"]))}
    return result


def col(a, corpus, method, dose=1):
    return a["conditions"].index((corpus, method, dose))


def figure2(P):
    fig = plt.figure(figsize=(7.2, 3.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[.86, 1, 1], left=.065, right=.98, bottom=.21, top=.76, wspace=.48)
    ax = fig.add_subplot(gs[0]); ax.axis("off")
    panel(ax, "a", "Where the edit is applied")
    for i, (lab, mask) in enumerate([("Prefix only", (1, 0)), ("Whole prompt", (1, 1)), ("Question only", (0, 1))]):
        y = .79-i*.29
        ax.text(0, y+.10, lab, fontsize=8)
        for k, word in enumerate(["Prefix", "Question"]):
            ax.add_patch(Rectangle((k*.50, y-.03), .46, .09, facecolor="#D9EAE5" if mask[k] else "white", edgecolor="#88979C", linewidth=.65))
            ax.text(k*.50+.23, y+.015, word, ha="center", va="center", fontsize=6.7)
    ax.text(0, -.015, "Shaded tokens receive edits.\nNew output tokens are not edited.", fontsize=7, color="#65717B")
    for j, corpus in enumerate(["gsm8k", "wikitext_a"]):
        ax = fig.add_subplot(gs[j+1]); panel(ax, "bc"[j], ["GSM8K fit", "WikiText fit"][j])
        arm_stats = {}
        for method in ["full", "off_m"]:
            v = np.column_stack([P[p]["use"][:, col(P[p], corpus, method)] for p in ("prefix", "prompt", "query")])
            m, lo, hi = stats(v, P["prefix"]["groups"], f"fig2_{corpus}_{method}")
            arm_stats[method] = (m, lo, hi)
            ax.errorbar(np.arange(3), 100*m, yerr=100*np.array([m-lo, hi-m]), color=C["off" if method=="off_m" else "full"], marker="o", capsize=2, lw=1.3)
        for method, (m, lo, hi) in arm_stats.items():
            other = arm_stats["full" if method == "off_m" else "off_m"][0]
            for i in range(3):
                above = m[i] > other[i] or (abs(m[i]-other[i]) < .0005 and method == "off_m")
                yy = 100*hi[i]+1.8 if above else 100*lo[i]-2.1
                ax.text(i, yy, f"{100*m[i]:.1f}", ha="center", va="bottom" if above else "top", fontsize=7, color=C["off" if method=="off_m" else "full"])
        ax.set(xticks=[0, 1, 2], xticklabels=["Prefix", "Whole\nprompt", "Question"], xlim=(-.2, 2.2), ylim=(30, 90), yticks=[30, 50, 70, 90])
        ax.set_ylabel("Downstream-answer success (%)" if j==0 else "")
        a, b = P["prefix"], P["prompt"]
        dif = np.column_stack([P[p]["use"][:,col(P[p],corpus,"off_m")]-P[p]["use"][:,col(P[p],corpus,"full")] for p in ("prefix","prompt","query")])
        stats(dif,a["groups"],f"fig2_paired_gains_{corpus}")
    legend_methods(fig, ["full", "off"], y=.98)
    foot(fig, "661 common units  |  1,339 use queries  |  strength 1  |  matched norms within each placement; 95% intervals")
    save(fig, "fig02_position_dependence", "The filtering advantage depends on intervention placement",
         "The same fitted directions give different full-versus-filtered rankings when the edit moves from the prefix to the whole prompt. On GSM8K, off-diagonal filtering changes from an 8.2-point disadvantage to an 8.5-point advantage; WikiText also favors filtering most clearly for whole-prompt edits.",
         "C32, C32-prompt and C32-query, latent clues; 661 units from countries, people and elements. Each unit is weighted equally after averaging its eligible use queries. Error bars resample source entities. Norms are matched across methods within each placement, not across placements with different numbers of edited tokens. Prefix-only edits are identical across expression/use questions; question-inclusive edits are query-dependent. The expanded placement comparison is exploratory, and does not isolate the final token alone.")


def figure3(P):
    a=P["prompt"]; methods=["logit_m","diag_m","full","off_m","proj_m"]
    names=["Unembedding","Diagonal only","Full J-lens","Off-diagonal","Projection"]
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.2),gridspec_kw={"height_ratios":[1.65,1]})
    fig.subplots_adjust(left=.17,right=.97,top=.85,bottom=.14,hspace=.80,wspace=.45)
    for j,corp in enumerate(["gsm8k","wikitext_a"]):
        ids=[col(a,corp,m) for m in methods]
        e=stats(a["express"][:,ids],a["groups"],f"fig3_express_{corp}")
        u=stats(a["use"][:,ids],a["groups"],f"fig3_use_{corp}")
        l=stats(a["leak"][:,ids],a["groups"],f"fig3_leak_{corp}")
        ax=axs[0,j]; panel(ax,"ab"[j],["GSM8K fit","WikiText fit"][j])
        for i,m in enumerate(methods):
            color=C[m.replace("_m","")]
            ax.plot([u[0][i]*100,e[0][i]*100],[i,i],color="#ADB9C0",lw=1)
            for st,fill in [(e,"white"),(u,color)]:
                ax.errorbar(st[0][i]*100,i,xerr=[[100*(st[0][i]-st[1][i])],[100*(st[2][i]-st[0][i])]],fmt="o",color=color,mfc=fill,ms=4.5,capsize=1.5,lw=.8)
        ax.set(xlim=(0,105),xticks=[0,25,50,75,100],yticks=range(5),yticklabels=names if j==0 else [],ylim=(4.6,-.6),xlabel="Target-answer success (%)")
        ax.tick_params(axis="y",length=0);ax.spines["left"].set_visible(False)
        ax=axs[1,j];panel(ax,"cd"[j],"Literal target-name emission")
        for i,m in enumerate(methods):
            st=l;color=C[m.replace("_m","")]
            ax.hlines(i,0,st[0][i]*100,color=color,lw=2.2,alpha=.65)
            ax.errorbar(st[0][i]*100,i,xerr=[[100*(st[0][i]-st[1][i])],[100*(st[2][i]-st[0][i])]],fmt="o",color=color,capsize=1.5,lw=.8)
            ax.text(min(st[2][i]*100+2,55),i,f"{100*st[0][i]:.1f}",va="center",fontsize=7)
        ax.set(xlim=(0,60),xticks=[0,20,40,60],yticks=range(5),yticklabels=names if j==0 else [],ylim=(4.6,-.6),xlabel="Use continuations containing target name (%)")
        ax.tick_params(axis="y",length=0);ax.spines["left"].set_visible(False)
    fig.legend(handles=[Line2D([],[],color="#455A64",marker="o",mfc="white",ls="none",label="Name the concept"),Line2D([],[],color="#455A64",marker="o",ls="none",label="Use the concept")],loc="upper center",bbox_to_anchor=(.56,.98),ncol=2,frameon=False)
    foot(fig,"Whole-prompt swaps  |  strength 1  |  661 units  |  outcomes may overlap; 95% source-entity bootstrap intervals")
    save(fig,"fig03_expression_use_emission","Naming, downstream use and literal emission are different outcomes",
         "Whole-prompt unembedding and diagonal-only swaps make the model name the replacement much more often than answer a question about it. Off-diagonal directions improve downstream answering while reducing literal target-name emission; projection suppresses emission further but is not uniformly the most successful writer.",
         "Same C32-prompt units as Figure 2; no conditioning on successful expression. Top: open markers are expression success and filled markers are downstream-use success. Bottom: the target entity name appears somewhere in the eight-token use continuation. Naming and use are separate prompts, and name emission can coexist with a correct answer. These are behavioral endpoints, not unique identification of a lexical versus semantic causal pathway.")


def figure4(P):
    rows,conds=load_concept("swap_c33_qwen3-8b_52074de562")
    dist={0:P["prefix"]}
    for f in [1,2,4,8]:dist[f]=unit_arrays(rows,conds,filler=f)
    common=sorted(set.intersection(*[set(a["keys"]) for a in dist.values()]));assert len(common)==489
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.2));fig.subplots_adjust(left=.10,right=.97,bottom=.23,top=.76,wspace=.30)
    for j,corp in enumerate(["gsm8k","wikitext_a"]):
        ax=axs[j];panel(ax,"ab"[j],["GSM8K fit","WikiText fit"][j])
        for method in ["full","off_m","proj_m"]:
            cols=[]
            for f,a in dist.items():
                ix=[a["keys"].index(k) for k in common]
                cols.append(a["use"][ix,col(a,corp,method)])
            groups=[P["prefix"]["groups"][P["prefix"]["keys"].index(k)] for k in common]
            m,lo,hi=stats(np.column_stack(cols),groups,f"fig4_{corp}_{method}")
            x=[0,1,2,4,8];color=C[method.replace("_m","")]
            ax.fill_between(x,lo*100,hi*100,color=color,alpha=.09,linewidth=0)
            ax.plot(x,m*100,color=color,marker="o",ms=4)
        ax.set(xlim=(-.2,8.4),ylim=(25,100),xticks=[0,1,2,4,8],yticks=[25,50,75,100],xlabel="Intervening filler sentences")
        if j==0:ax.set_ylabel("Downstream-answer success (%)")
        ax.text(.97,.03,"8 sentences ≈ 120 tokens",transform=ax.transAxes,ha="right",fontsize=7,color="#65717B")
    legend_methods(fig,["full","off","proj"],y=.99)
    foot(fig,"489 common units  |  prefix-only swaps, strength 1  |  fixed filler text; no new-token injections  |  95% intervals")
    save(fig,"fig04_distractor_distance","Prompt-local edits remain accessible after intervening text",
         "Counterfactual answering remains substantial after eight unrelated filler sentences, for both full and filtered directions. The off-diagonal advantage does not systematically grow with distance; these measurements do not support a uniquely longer-lived filtered write.",
         "C33 uses fixed teacher-forced filler before the question and retains the 489 units eligible at every distance. Each unit is weighted equally; eligible relations can vary across distance, so this is a common-unit rather than strictly common-query panel. Shading is a pointwise source-entity bootstrap interval. The edited prefix remains available through attention/KV state: this measures accessibility after distractors, not autonomous hidden-state persistence or free-generation success. See Supplement S2 for free generation.")


def figure5():
    rows,conds=load_concept("additive_c34test_qwen3-8b_513696465f")
    frozen=read("results/concept/frozen_c34.json")["frozen"]
    eligible_keys=({r["key"] for r in rows if r["kind"]=="use"} & {r["key"] for r in rows if r["kind"]=="neutral"})
    assert len(eligible_keys)==174
    use=[r for r in rows if r["kind"]=="use" and r["key"] in eligible_keys]
    neutral=[r for r in rows if r["kind"]=="neutral" and r["key"] in eligible_keys]
    groups=[r["type"]+"/"+r["source"] for r in use]
    order=["diffmean","template","logit","gsm8k_full","wikitext_a_full","gsm8k_off","wikitext_a_off","wikitext_a_proj","gsm8k_proj","rand"]
    labels=["DiffMean*","Template lens*","Unembedding","GSM8K: full","WikiText: full","GSM8K: off","WikiText: off","WikiText: projection","GSM8K: projection","Random"]
    def color(m):return C["proj" if m.endswith("proj") else m.split("_")[-1] if m.startswith(("gsm8k","wikitext")) else m]
    def ix(m,config,alpha):return conds.index((m,*config,alpha))
    ids=[ix(m,(frozen[m]["layer_set"],frozen[m]["position_set"]),frozen[m]["alpha"]) for m in order]
    vals=np.array([[r["hit_t"][k] for k in ids] for r in use],float)
    mean,lo,hi=stats(vals,groups,"fig5_frozen_test")
    AUDIT["additive_population"]={"units":174,"use_queries":len(use),"source_clusters":len(set(groups)),"weighting":"eligible use queries, as in C34 report"}
    fig,axs=plt.subplots(1,2,figsize=(7.2,4.0),gridspec_kw={"width_ratios":[1,1.1]})
    fig.subplots_adjust(left=.19,right=.97,bottom=.25,top=.84,wspace=.47)
    ax=axs[0];panel(ax,"a","Frozen development-selected settings")
    for i,m in enumerate(order):
        ax.errorbar(mean[i]*100,i,xerr=[[100*(mean[i]-lo[i])],[100*(hi[i]-mean[i])]],fmt="o",color=color(m),capsize=2)
        ax.text(min(hi[i]*100+2,76),i,f"{mean[i]*100:.1f}",va="center",fontsize=7)
    ax.set(yticks=range(10),yticklabels=labels,ylim=(9.6,-.6),xlim=(-1,85),xticks=[0,20,40,60,80],xlabel="Test answer success (%)")
    ax.spines["left"].set_visible(False);ax.tick_params(axis="y",length=0)
    ax=axs[1];panel(ax,"b","Shared placement, full dose curves")
    methods=["diffmean","gsm8k_full","gsm8k_off","wikitext_a_off","logit","template"]
    for m in methods:
        xx=[];yy=[]
        for alpha in [.05,.1,.2,.4,.8,1.6]:
            k=ix(m,("all","prefix"),alpha)
            xx.append(np.mean([r["kl"][k] for r in neutral]));yy.append(100*np.mean([r["hit_t"][k] for r in use]))
        c=color(m);style="--" if m=="wikitext_a_off" else "-"
        ax.plot(xx,yy,color=c,ls=style,marker="o",ms=3.5,label={"gsm8k_full":"GSM8K full","gsm8k_off":"GSM8K off","wikitext_a_off":"WikiText off","diffmean":"DiffMean*","logit":"Unembedding","template":"Template*"}[m])
        STATS[f"fig5_curve_{m}"]={"alpha":[.05,.1,.2,.4,.8,1.6],"neutral_kl":xx,"use_percent":yy}
    ax.set(xscale="log",xlim=(.004,.6),ylim=(0,85),xlabel="Neutral-query KL (nats; log scale)",ylabel="Test answer success (%)",yticks=[0,20,40,60,80])
    ax.legend(loc="lower center",bbox_to_anchor=(.45,-.38),ncol=2,frameon=False,fontsize=6.7,columnspacing=1,handlelength=1.7)
    foot(fig,"174 test units  |  *Labelled fitting contexts  |  a: frozen layer/position/dose + 95% CI  |  b: all seven layers, prefix")
    save(fig,"fig05_additive_baselines","Swap improvements do not imply additive-steering superiority",
         "At development-selected operating points, the supervised DiffMean baseline is substantially more effective than lens-derived target-only additions. The shared-placement dose curves also show why conclusions should consider both answer success and disruption, rather than each method's best test score alone.",
         "C34/C35: 174 test units, with source and target entities excluded from the development grid. Panel a freezes layer, position and dose per method before test; intervals resample source entities over query-weighted success. DiffMean and the template baseline use 12 labelled contexts per entity, including test-entity fitting contexts; this is held out from configuration selection, not zero-shot concept learning. Panel b is descriptive: all seven layers, prefix positions, all six doses in increasing-dose order, without constructing an oracle envelope. KL measures two neutral questions, not general capability preservation. Absolute budgets differ between methods in a; b shares the placement convention.")


def figure6():
    datasets=[]
    for model,tag,n in [("8b","beta",105),("4b","beta4b",99)]:
        r=raw_shards(f"steering_qwen3-{model}_swap_{tag}",4)
        betas=[0,.5,1,2,3] if model=="8b" else [0,.5,1,2]
        arms=[f"swap_published_b{b:g}" for b in betas]
        v,g=explicit_data(r,arms,n);datasets.append((model,betas,stats(v,g,f"fig6_beta_{model}"),n))
    r=raw_shards("steering_qwen3-8b_swap_beta",4)
    arms=[f"swap_{co}_{arm}" for co in ["gsm8k","wikitext_a"] for arm in ["b0","b1","galpha","off"]]
    v,g=explicit_data(r,arms,105);s=stats(v,g,"fig6_scalar")
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.2));fig.subplots_adjust(left=.1,right=.97,bottom=.24,top=.81,wspace=.30)
    ax=axs[0];panel(ax,"a","Correction on downloaded lenses")
    for model,bs,(m,lo,hi),n in datasets:
        co=C["proj"] if model=="8b" else C["full"]
        ax.errorbar(bs,m*100,yerr=100*np.array([m-lo,hi-m]),marker="o" if model=="8b" else "s",ls="-" if model=="8b" else "--",color=co,capsize=2,label=f"Qwen3-{model.upper()} (n={n})")
    ax.axvline(1,ls=":",color="#8B9398",lw=.8,zorder=0)
    ax.set(xlim=(-.1,3.1),ylim=(0,75),xticks=[0,.5,1,2,3],xlabel="Projection coefficient beta",ylabel="Target-answer success (%)",yticks=[0,25,50,75])
    ax.legend(frameon=False,loc="upper right",fontsize=7)
    ax=axs[1];panel(ax,"b","Cheap correction versus positional refit")
    for j,co in enumerate(["gsm8k","wikitext_a"]):
        m,lo,hi=(z[4*j:4*j+4] for z in s);x=np.arange(4)+(-.055 if j==0 else .055)
        ax.errorbar(x,m*100,yerr=100*np.array([m-lo,hi-m]),marker="o" if j==0 else "s",ls="-" if j==0 else "--",color=C["off"] if j==0 else C["full"],capsize=2,label=["GSM8K","WikiText"][j])
    ax.set(xticks=range(4),xticklabels=["Full","Projection","Shared\nscalar","Off-diagonal"],xlim=(-.25,3.25),ylim=(0,75),yticks=[0,25,50,75],xlabel="Direction construction")
    ax.legend(frameon=False,loc="upper left",fontsize=7)
    foot(fig,"Explicit non-numerical swaps  |  whole-prompt edits, strength 1  |  development evidence; 95% source-prompt intervals")
    save(fig,"fig06_cheap_correction","A cheap correction recovers much of the original swap gain",
         "Removing the token-aligned component improves the downloaded lenses at both Qwen model sizes, with the best tested coefficient at beta=1. A shared scalar per layer captures much of the positional-filtering benefit in the original explicit-swap regime, without requiring a component refit.",
         "C25/C31, native swaps at strength 1. The 8B and 4B clean-correct populations differ (105 versus 99 swaps), so compare changes within each model. Bootstrap clusters are source prompts. The shared scalar is calibrated on the 16 evaluation argument tokens without answer labels. This is a development result at a specific intervention placement, not evidence that the correction is universally optimal or that it helps the prefix-only C32 setting.")


def supplement1(P):
    fig,axs=plt.subplots(2,3,figsize=(7.2,4.6),sharex=True,sharey=True)
    fig.subplots_adjust(left=.1,right=.97,bottom=.16,top=.84,hspace=.4,wspace=.25)
    for j,corp in enumerate(["gsm8k","wikitext_a"]):
        for i,pos in enumerate(["prefix","prompt","query"]):
            ax=axs[j,i];panel(ax,"abcdef"[j*3+i],["Prefix only","Whole prompt","Question only"][i]+(" / GSM8K" if j==0 else " / WikiText"))
            a=P[pos]
            for method in ["full","off_m","proj_m"]:
                v=a["use"][:,[col(a,corp,method,d) for d in [.25,.5,1]]]
                m,lo,hi=stats(v,a["groups"],f"s1_{corp}_{pos}_{method}");color=C[method.replace("_m","")]
                ax.errorbar([.25,.5,1],m*100,yerr=100*np.array([m-lo,hi-m]),color=color,marker="o",capsize=1.5,ms=3)
            ax.set(ylim=(0,100),yticks=[0,50,100],xlim=(.2,1.05),xticks=[.25,.5,1])
            if i==0:ax.set_ylabel("Use success (%)")
            if j==1:ax.set_xlabel("Swap strength")
    legend_methods(fig,["full","off","proj"],y=.99)
    foot(fig,"661 common units  |  all three tested doses  |  95% source-entity bootstrap intervals")
    save(fig,"figS01_position_dose","Position dependence across the full tested dose grid",
         "The full three-dose comparison retains the unsuccessful regimes as well as the positive whole-prompt result. Filtering and projection should be described as configuration-dependent, not as an improvement at every strength or location.",
         "C32/C32-prompt/C32-query; same 661 latent-clue units and unit weighting as Figures 2 and 3. All three doses are shown without per-item or per-panel strength optimization. Pointwise intervals condition on fixed fitted directions and do not adjust for multiple comparisons.")


def supplement2():
    ps=[read(f"results/concept/freegen_c33b_qwen3-8b_25f68bcb51_shard{i}of4.json") for i in range(4)]
    rows=[r for p in ps for r in p["records"]]
    # The free-generation format stores one record per unit and condition.
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.2));fig.subplots_adjust(left=.11,right=.97,bottom=.25,top=.76,wspace=.4)
    for j,corp in enumerate(["gsm8k","wikitext_a"]):
        ax=axs[j];panel(ax,"ab"[j],["GSM8K fit","WikiText fit"][j])
        for i,method in enumerate(["full","off_m","proj_m","logit_m"]):
            rr=[r for r in rows if tuple(r["cond"])==(corp,method,1.)]
            assert len(rr)==125,(corp,method,len(rr))
            groups=[r["key"].split("/")[1] for r in rr]
            v=np.array([[r["probe_t"],r["win_t"][0]] for r in rr],float)
            m,lo,hi=stats(v,groups,f"s2_{corp}_{method}")
            color=C[method.replace("_m","")]
            for k in range(2):
                yy=i+(-.10 if k==0 else .10)
                ax.errorbar(m[k]*100,yy,xerr=[[100*(m[k]-lo[k])],[100*(hi[k]-m[k])]],fmt="o",mfc=color if k==0 else "white",color=color,capsize=1.5,ms=4)
        ax.set(yticks=range(4),yticklabels=["Full","Off-diagonal","Projection","Unembedding"] if j==0 else [],xlim=(0,40),ylim=(3.5,-.5),xticks=[0,10,20,30,40],xlabel="Target outcome (%)")
        ax.tick_params(axis="y",length=0);ax.spines["left"].set_visible(False)
    fig.legend(handles=[Line2D([],[],color="#455A64",marker="o",ls="none",label="Capital probe after 64 tokens"),Line2D([],[],color="#455A64",marker="o",mfc="white",ls="none",label="Target mention in first 16 tokens")],loc="upper center",bbox_to_anchor=(.55,.99),frameon=False,ncol=2)
    foot(fig,"125 country units  |  prefix-only swaps, strength 1  |  clean probe: 0%; clean early target mention: 0.8%")
    save(fig,"figS02_free_generation","Forced-context accessibility does not guarantee free-generation control",
         "After 64 freely generated tokens, the target-capital probe succeeds only infrequently. This weaker result limits interpreting the distractor experiment as reliable long-form concept installation.",
         "C33b, 125 country units, prefix-only edits and no injection at new token positions. Filled markers show target-capital probe success after the model's own text; open markers show target-name/capital/language mentions in the first 16 tokens. Both outcomes are coded from stored text. Source-entity bootstrap intervals reflect repeated source/target combinations; this is a different population and task from Figure 4.")


def assemble():
    book=fitz.open()
    for n,entry in enumerate(FIGS):
        page=book.new_page(width=612,height=792)
        page.insert_text((42,40),entry["name"].replace("_"," ").upper(),fontsize=8,color=(.35,.4,.43))
        page.insert_textbox(fitz.Rect(42,52,570,90),entry["title"],fontsize=14,fontname="hebo")
        src=fitz.open(stream=entry["pdf"],filetype="pdf")
        h=528*entry["size"][1]/entry["size"][0]
        page.show_pdf_page(fitz.Rect(42,96,570,96+h),src,0)
        y=112+h
        text=entry["caption"]+"\n\n"+entry["note"]
        remainder=page.insert_textbox(fitz.Rect(48,y,564,744),text,fontsize=9,fontname="helv",lineheight=1.35,color=(.12,.15,.17))
        assert remainder>=0,(entry["name"],remainder)
        page.insert_text((48,766),"J-lens steering | figure review set | 16 September 2026",fontsize=7,color=(.4,.45,.48))
        page.insert_text((550,766),str(n+1),fontsize=7)
    book.save(PDF/"jlens_figure_atlas.pdf",garbage=4,deflate=True)
    audit={"seed":SEED,"bootstrap_draws":NBOOT,"interval":"95% pointwise percentile, cluster resampling", "sources":SOURCES,"panels":STATS,"audit":AUDIT}
    (OUT/"figure_data_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    lines=["# Paper figure set - 16 September 2026","",
           "Six main figures and two supplementary figures, generated directly from pinned JSON results. Vector SVGs have editable text; PNGs are 450 dpi. The core finding also has a standalone vector PDF, and the full vector PDF atlas includes captions.","",
           "**Narrative update:** the new data support a placement-dependent swap result, not an unconditional superiority claim. Figure 1 is the historical core finding; Figure 2 is its essential qualification. The additive and free-generation limitations must remain visible.","",
           "**Rebuild:** run `.venv/bin/python scripts/50_paper_figures.py`. Dependencies: NumPy, Matplotlib, PyMuPDF. No model calls or GPU access. SHA-256 inputs, plotted means, bootstrap intervals and population counts are in `figure_data_audit.json`.","",
           "**Statistics:** intervals resample source prompts/entities (or supplied categories in the 55-item control panel), retain paired observations, and condition on the fitted lenses. They do not establish fit replication or adjust for model selection/multiple comparisons. C32 uses unit-weighted use scores; C34 uses query-weighted scores, matching the reports.","",
           "**Design:** 180-mm-class figure width, Arial typography, restrained colorblind-friendly colors, direct labels, no decorative shading/3-D, embedded PDF text and editable SVG labels. Followed [Nature's artwork specifications](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/) for vector output and clear axes; final venue-specific font/width requirements should be applied at submission.","",
           "**Submission caveats:** whole-prompt and question-only conditions use query-dependent edits, so they are not literally the same intervention tensor across naming/use questions. Matching row norms within a placement does not equalize total intervention budget across placements. Conditional P(use|expression) is not plotted as a causal conversion measure. Neutral KL concerns two queries, not general collateral damage. Saved source hashes and fit uncertainty remain necessary in manuscript methods.",""]
    for entry in FIGS:
        name=entry["name"]
        lines += ["## "+entry["title"],"",f"![{entry['title']}]({name}.png)","",entry["caption"],"",entry["note"],"",f"[Editable SVG]({name}.svg) | [450-dpi PNG]({name}.png)",""]
    (OUT/"README.md").write_text("\n".join(lines)+"\n")
    # Render the actual final PDF pages, rather than only inspecting PNG exports.
    qa=ROOT/"tmp/pdfs/jlens_figures";qa.mkdir(parents=True,exist_ok=True)
    for i,page in enumerate(book):
        page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).save(qa/f"atlas_{i+1:02d}.png")
    for entry in FIGS:
        assert b"<text" in (OUT/f"{entry['name']}.svg").read_bytes()
    print("PDF atlas pages:",len(book),"; sources:",len(SOURCES),flush=True)


if __name__=="__main__":
    figure1()
    P=load_positions()
    figure2(P);figure3(P);figure4(P);figure5();figure6();supplement1(P);supplement2()
    assemble()

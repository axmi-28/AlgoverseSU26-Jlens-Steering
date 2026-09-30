"""Manuscript-width, claim-led figures. CPU-only rebuild from pinned raw results.

Run: .venv/bin/python scripts/56_final_paper_figures.py
Does not edit the manuscript or earlier figures. PDF/SVG are vector originals.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("corpus", ROOT / "scripts/53_corpus_figures.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
b = c.b
OUT = ROOT / "output/figures/final"
PDF = ROOT / "output/pdf"
QA = ROOT / "tmp/pdfs/redesign"
for d in (OUT, PDF, QA):
    d.mkdir(parents=True, exist_ok=True)

# LNCS text width: 122 mm. All lettering is sized for this width, not for a slide.
MM = 1 / 25.4
W = 122 * MM
INK = "#242F37"
GRAY = "#75828B"
LIGHT = "#D9DFE2"
PALE = "#F3F5F6"
OFF = "#007F73"
DIAG = "#B85D29"
FULL = "#43556A"
CORPORA = ["gsm8k", "wikitext_a"]
CN = {"gsm8k": "GSM8K", "wikitext_a": "WikiText", "aqua_rat": "AQuA-RAT"}
CC = {"gsm8k": "#245D85", "wikitext_a": "#8466A2", "aqua_rat": "#408371"}
CM = {"gsm8k": "o", "wikitext_a": "s", "aqua_rat": "^"}
LAM = [0, .125, .25, .5, .75, 1, 1.5, 2]
ENTRIES = []
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5,
    "mathtext.fontset": "dejavusans", "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": GRAY, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": .6, "lines.linewidth": 1.35, "lines.markersize": 3.3,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "jlens-final-20260925",
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def title(ax, letter, text):
    ax.set_title(text, loc="left", pad=12, fontsize=8.4)
    ax.text(-.16, 1.08, letter, transform=ax.transAxes, fontsize=9,
            weight="bold", va="bottom")


def text(ax, x, y, s, size=8, color=INK, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, va="center", **kw)


def box(ax, x, y, w, h, color="white", edge=LIGHT, radius=.8, lw=.65):
    p = FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=color, edgecolor=edge, linewidth=lw)
    ax.add_patch(p)
    return p


def arrow(ax, xy0, xy1, color=GRAY, lw=.8, style="-|>", rad=0, **kw):
    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle=style, mutation_scale=6,
        color=color, linewidth=lw, connectionstyle=f"arc3,rad={rad}", **kw))


def save(fig, name, heading, caption, where, audit_note=""):
    p = PDF / f"jlens_final_{name}.pdf"
    fig.savefig(p)
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.png", dpi=400)
    # Overleaf gets standalone PDF assets as well as editable SVGs and previews.
    shutil.copyfile(p, OUT / f"{name}.pdf")
    plt.close(fig)
    doc = fitz.open(p)
    assert len(doc) == 1 and not doc[0].get_images(), name
    assert abs(doc[0].rect.width - 122 * 72 / 25.4) < .05, name
    page = doc[0]
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                x0, y0, x1, y1 = span["bbox"]
                assert x0 >= -.5 and y0 >= -.5 and x1 <= page.rect.width + .5 and y1 <= page.rect.height + .5, (name, span["text"], span["bbox"])
    page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5)).save(QA / f"{name}.png")
    ENTRIES.append(dict(name=name, heading=heading, caption=caption, where=where,
                        audit_note=audit_note, path=str(p), height=page.rect.height))
    print(f"Built {name}", flush=True)


def load_latent():
    banks = {}
    for corp, leaf in [("gsm8k", "dac7603b1af57dbd"), ("wikitext_a", "6fff00659415ddf5")]:
        blob = b.read(f"results/c22/write_controls_{leaf}.json")
        rows = blob["records"]
        lookup = {(r["name"], r["arm"], r["strength"]): r for r in rows}
        names = sorted({r["name"] for r in rows})
        assert len(names) == 55
        v = np.array([[lookup[n, arm, 1.][field]
                       for arm in ("swap_full", "swap_off_matched")
                       for field in ("hit_swap_answer", "emitted_substring")] for n in names], float)
        groups = [lookup[n, "swap_full", 1.]["category"] for n in names]
        banks[corp] = dict(values=v, groups=groups, lookup=lookup)
    assert banks["gsm8k"]["values"].sum(0).tolist() == [15, 21, 25, 5]
    assert banks["wikitext_a"]["values"].sum(0).tolist() == [13, 21, 23, 9]
    return banks


def overview(latent):
    fig = plt.figure(figsize=(W, 117 * MM))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 122), ylim=(0, 117))
    ax.axis("off")
    text(ax, 2, 112, "a", 9, weight="bold")
    text(ax, 8, 112, "Construct a direction from downstream effects", 8.6, weight="bold")
    text(ax, 8, 106.5, "Average over a fitting corpus: 32 documents, 128 tokens each", 7.4, GRAY)
    text(ax, 9.5, 101.5, "Token-wise residual streams", 7.3)

    # Token-wise residual streams. Colored arrows summarize derivatives, not
    # identified attention heads or a discovered semantic circuit.
    xs = [15, 25, 35, 45]
    for y in [93, 85, 77]:
        box(ax, 9.5, y - 2.2, 41, 4.4, PALE, edge=LIGHT, radius=.4, lw=.4)
    for x in xs:
        arrow(ax, (x, 98), (x, 70), LIGHT, lw=1.15)
        for y in [96.5, 69.5]:
            ax.plot(x, y, "o", color="white", mec=GRAY, mew=.5, ms=3)
    text(ax, 3, 96.5, r"$h_{\ell}$", 8, ha="left")
    text(ax, 1.5, 69.5, r"$h_{\mathrm{final}}$", 7.5)
    text(ax, 52, 85, "remaining\nlayers", 7.1, GRAY, ha="left", linespacing=1.15)
    arrow(ax, (59, 96), (69, 96), GRAY, lw=.8)
    arrow(ax, (15, 96.2), (15, 70.2), DIAG, lw=1.35)
    for x in xs[1:]:
        arrow(ax, (15.8, 95.6), (x, 70.3), OFF, lw=1, rad=.04)
    for x, s in zip(xs, [r"$t$", r"$t+1$", r"$t+2$", r"$t+3$"]):
        text(ax, x, 64.8, s, 7.1, ha="center")
    text(ax, 31, 60.3, "token position", 7.2, GRAY, ha="center")

    # One cell is a d_model x d_model Jacobian, not one hidden dimension.
    x0, y0, cell = 76.5, 77, 5.2
    text(ax, x0 + 2 * cell, 103, r"Final position $t'$", 7.3, ha="center")
    text(ax, x0 - 5, y0 + 10.4, r"Source $t$", 7.3, rotation=90, ha="center")
    for i in range(4):
        for j in range(4):
            fc = "#E0EFEB" if j > i else "#F3DCCC" if j == i else PALE
            ax.add_patch(Rectangle((x0 + j * cell, y0 + (3-i)*cell), cell-.3, cell-.3,
                                   facecolor=fc, edgecolor="white", linewidth=.3))
            text(ax, x0+(j+.5)*cell-.15, y0+(3-i+.5)*cell-.15,
                 "O" if j > i else "D" if j == i else "", 7.5,
                 OFF if j > i else DIAG, ha="center")
    text(ax, 88, 71.5, r"$J_\lambda = J_O + \lambda J_D$", 9, ha="center")
    text(ax, 75, 66.3, "D", 7.5, DIAG, weight="bold")
    text(ax, 80, 66.3, "same position", 7.3)
    text(ax, 75, 61.5, "O", 7.5, OFF, weight="bold")
    text(ax, 80, 61.5, "later positions", 7.3)

    ax.plot([2, 120], [55.9, 55.9], color=LIGHT, lw=.65)
    text(ax, 2, 51.5, "b", 9, weight="bold")
    text(ax, 8, 51.5, "Change the construction, keep the swap task", 8.6, weight="bold")
    # Recorded generation, not an invented illustrative result.
    r = latent["gsm8k"]["lookup"]
    name = "ex-city-capital-Toronto-Lyon"
    assert r[name, "swap_full", 1.]["generated"].startswith(" France.")
    assert r[name, "swap_off_matched", 1.]["generated"].startswith(" Paris.")
    text(ax, 8, 45.5, '"The capital of the country where Toronto is located is ..."', 7.6)
    text(ax, 8, 39, "Swap Canada", 8)
    arrow(ax, (35.5, 39), (43.5, 39), INK, lw=1)
    text(ax, 45.5, 39, "France", 8, weight="bold")
    text(ax, 78, 39, r"$v_y = J_\lambda^{\mathsf{T}}u_y$", 8.5)
    # Full / filtered outputs are parallel branches of the same intervention.
    box(ax, 8, 23, 50, 10, "#F2F4F6", edge=FULL)
    box(ax, 65, 23, 50, 10, "#EDF6F3", edge=OFF)
    text(ax, 11, 29.7, r"Full lens  ($\lambda=1$)", 7.3)
    text(ax, 68, 29.7, r"Filtered  ($\lambda=0$)", 7.3)
    text(ax, 11, 25.6, '"France."', 8, DIAG, weight="bold")
    text(ax, 32, 25.6, "names the target", 7)
    text(ax, 68, 25.6, '"Paris."', 8, OFF, weight="bold")
    text(ax, 85, 25.6, "answers the relation", 7)
    arrow(ax, (37, 36), (33, 33.5), FULL)
    arrow(ax, (63, 36), (89, 33.5), OFF)

    text(ax, 8, 17.7, "Across 55 latent-intermediate swaps (GSM8K fit)", 7.4, weight="bold")
    text(ax, 8, 11.8, "Correct answers", 7.5)
    text(ax, 41, 11.8, "15/55", 8, FULL)
    arrow(ax, (54, 11.8), (62, 11.8), GRAY)
    text(ax, 64, 11.8, "25/55", 8, OFF, weight="bold")
    text(ax, 85, 11.8, "+18.2 points", 7.5, OFF)
    text(ax, 8, 6.2, "Target-name leakage", 7.5)
    text(ax, 41, 6.2, "21/55", 8, FULL)
    arrow(ax, (54, 6.2), (62, 6.2), GRAY)
    text(ax, 64, 6.2, "5/55", 8, OFF, weight="bold")
    text(ax, 85, 6.2, "less name emission", 7.3)
    save(fig, "01_overview", "Lens construction changes what a swap demonstrates",
         "(a) A J-lens averages local derivatives across a fitting corpus. Same-position contributions "
         "(D, orange) and later-position contributions (O, green) are indexed by token-position pairs, "
         "not hidden-state coordinates. The diagram is schematic, not a measured attention map. "
         "(b) We vary J_lambda = J_O + lambda J_D before forming token directions. A saved GSM8K-lens "
         "trial swaps Canada for France: the full lens emits 'France.', whereas the filtered lens "
         "answers 'Paris.' (first answer spans shown). Across the 55-item latent-intermediate panel, "
         "filtering improves answer success and reduces name leakage at strength 1 with matched edit norms. "
         "This is evidence about the intervention construction, not identification of a unique semantic circuit.",
         "After the introduction, before Related Work.",
         "The example is ex-city-capital-Toronto-Lyon in C22; its full prompt starts with 'Fact:'. "
         "Seven layers and all original prompt positions are edited. Matrix shading denotes membership, not energy.")


def corpus_main():
    rep, aq = c.corpus_rows("tier0"), c.corpus_rows("aqua")
    specs = [("wikitext_a", "wikirep", rep), ("gsm8k", "gsm8k", rep), ("aqua_rat", "aqua_rat", aq)]
    expected = [[[1,0,1,0,2],[39,41,41,41,42]], [[10,8,8,8,9],[44,42,41,39,43]], [[14,12,12,15,10],[41,42,38,42,41]]]
    fig, axs = plt.subplots(1, 2, figsize=(W, 65*MM), sharey=True)
    fig.subplots_adjust(left=.105, right=.985, bottom=.24, top=.79, wspace=.16)
    for i, (corp, prefix, fn) in enumerate(specs):
        counts = [[fn(f"swap_{prefix}_r{k}", numeric)[0] for k in range(5)] for numeric in (True, False)]
        assert counts == expected[i]
        b.AUDIT[f"final_replicates_{corp}"] = counts
        for j, denom in enumerate((39,105)):
            ax = axs[j]
            vals = np.array(counts[j])*100/denom
            for count in sorted(set(counts[j])):
                n = counts[j].count(count)
                xx = i + np.linspace(-.14,.14,n) if n > 1 else [i]
                ax.scatter(xx, [100*count/denom]*n, s=15, marker=CM[corp], color=CC[corp], zorder=4)
            ax.plot([i-.19,i+.19], [np.mean(vals)]*2, color=CC[corp], lw=.85, zorder=3)
            ax.text(i, max(vals)+4, f"{min(counts[j])}-{max(counts[j])}", ha="center", fontsize=7.5, color=CC[corp])
    for j, ax in enumerate(axs):
        title(ax, "ab"[j], ["Numerical arguments", "Other original arguments"][j])
        ax.set(xlim=(-.45,2.45), ylim=(0,52), yticks=[0,10,20,30,40,50], xticks=[0,1,2],
               xticklabels=["WikiText", "GSM8K", "AQuA-RAT"])
        ax.tick_params(axis="x", length=0, labelsize=7)
        ax.text(.96,.06,["39 trials", "105 trials"][j], transform=ax.transAxes, fontsize=7.2, color=GRAY, va="bottom",ha="right")
        if j == 0: ax.set_ylabel("Swap success (%)")
    fig.text(.105,.065,"Each point: one 32-document fit. Labels: successful-trial range.", fontsize=7, color=GRAY)
    save(fig, "02_corpus", "Corpus effects replicate, but are task dependent",
         "Five disjoint 32-document fits per corpus on the original flexible-generalization evaluation. "
         "(a) Every GSM8K and AQuA-RAT fit exceeds every WikiText fit on the 39 numerical-argument trials. "
         "(b) Results overlap on the 105 non-numerical trials. Points are individual fits, short horizontal "
         "ticks mark their means, and labels give count ranges, not confidence intervals. Ties are offset "
         "horizontally. Strength is selected separately for each fit and subset from {0.5, 1, 2} on this "
         "same evaluation; this is a development comparison. The two panels share a percentage scale.",
         "Fitting Corpus: after the paragraph introducing five independent fits.",
         "The numerical subset includes non-arithmetic templates and the inherited square-task answer-key limitation. "
         "The separate new-argument panel also has 105 eligible trials; it is not panel b.")


def core(latent):
    rows = b.raw_shards("steering_qwen3-8b_swap_comp", 2)
    arms = [f"swap_{corp}_{arm}" for corp in CORPORA for arm in ("full","off")]
    v, groups = b.explicit_data(rows, arms, 105)
    assert v.sum(0).tolist() == [35,60,20,50]
    fig, axs = plt.subplots(1,2,figsize=(W, 74*MM))
    fig.subplots_adjust(left=.105,right=.965,bottom=.23,top=.76,wspace=.42)
    ax = axs[0]; title(ax,"a","Explicit arguments")
    for j, corp in enumerate(CORPORA):
        m,lo,hi = b.stats(v[:,j*2:j*2+2], groups, f"final_explicit_{corp}")
        b.stats(v[:,j*2+1]-v[:,j*2],groups,f"final_explicit_gain_{corp}")
        xs=np.array([0,1]) + (j-.5)*.055
        ax.errorbar(xs,100*m,yerr=100*np.array([m-lo,hi-m]), color=CC[corp], fmt=CM[corp]+"-", lw=1.2, capsize=2, ms=3.6)
        for k in range(2):
            ax.text(xs[k]+(-.09 if k==0 else .09),m[k]*100+(-5 if j else 5),f"{int(v[:,j*2+k].sum())}/105",ha="right" if k==0 else "left",fontsize=7,color=CC[corp])
    ax.set(ylim=(0,85),yticks=[0,20,40,60,80],xlim=(-.55,1.6),xticks=[0,1],xticklabels=["Full","Filtered"],ylabel="Correct answers (%)")
    ax.tick_params(axis="x",length=0)
    ax=axs[1];title(ax,"b","Latent intermediates")
    for j,corp in enumerate(CORPORA):
        vv=latent[corp]["values"]; m=vv.mean(0)*100
        b.stats(vv,latent[corp]["groups"],f"final_latent_{corp}")
        b.stats(vv[:,2:]-vv[:,:2],latent[corp]["groups"],f"final_latent_gain_{corp}")
        arrow(ax,(m[1],m[0]),(m[3],m[2]),CC[corp],lw=1.4,shrinkA=4,shrinkB=4)
        ax.plot(m[1],m[0],marker=CM[corp],mfc="white",mec=CC[corp],ms=4.5,ls="none")
        ax.plot(m[3],m[2],marker=CM[corp],color=CC[corp],ms=4.5,ls="none")
    ax.set(xlim=(0,52),ylim=(0,62),xticks=[0,20,40],yticks=[0,20,40,60],xlabel="Name leakage (%)",ylabel="Correct answers (%)")
    ax.text(6,57,"fewer names,\nmore answers",fontsize=7.2,color=OFF,va="top")
    ax.text(40,29.5,"full",ha="left",fontsize=7,color=GRAY)
    ax.text(7,37.5,"filtered",fontsize=7,color=GRAY)
    fig.legend(handles=[Line2D([],[],color=CC[x],marker=CM[x],label=CN[x],ms=3.5) for x in CORPORA],
               ncol=2,loc="upper center",bbox_to_anchor=(.55,.985),frameon=False)
    fig.text(.105,.055,"Strength 1; arrows in b connect full to filtered on the same 55 items.",fontsize=7,color=GRAY)
    save(fig,"03_filtering","Filtering improves answers while reducing name leakage",
         "(a) Removing same-position contributions improves explicit-argument swapping on 105 clean-correct "
         "non-numerical trials: 35 to 60 successes for GSM8K and 20 to 50 for WikiText. Error bars are 95% "
         "source-prompt cluster-bootstrap intervals. (b) Each arrow connects full (open) to filtered (filled) "
         "directions on the same 55 latent-intermediate items. Correct answers rise from 15 to 25 (GSM8K) "
         "and 13 to 23 (WikiText), while name leakage falls from 21 to 5 and 21 to 9. Both panels use "
         "strength 1 and whole-prompt edits; panel b additionally matches edit norms at each layer and position. "
         "Leakage and success need not be mutually exclusive.",
         "Positional Filtering: after the latent-intermediate results.",
         "C18 and expanded C22 are separate reused development panels, not independent held-out replications. "
         "The bivariate plot shows raw rates, not confidence regions; C22 category-bootstrap statistics are in the audit.")


def load_interp():
    out={}
    for pos,stem in [("prompt","swap_c37lambdaprompt_qwen3-8b_034da26ce9"),
                     ("prefix","swap_c37lambdaprefix_qwen3-8b_75f74629d0")]:
        rows,conditions=b.load_concept(stem)
        a=b.unit_arrays(rows,conditions)
        assert len(a["keys"])==661 and a["n_queries"]==1339
        out[pos]=a
    assert out["prefix"]["keys"]==out["prompt"]["keys"]
    return out


MET=[("express","Name requested",GRAY,"s"),("use","Relation answer",OFF,"o"),("leak","Name leakage",DIAG,"^")]


def interp_axis(ax,a,corp,dose,tag,labels=False):
    ax.axvline(1,color=LIGHT,lw=.9,ls=(0,(3,3)),zorder=0)
    ids=[b.col(a,corp,f"lam{x:g}_cn",dose) for x in LAM]
    for metric,label,color,marker in MET:
        vals=a[metric][:,ids]
        m,lo,hi=b.stats(vals,a["groups"],f"final_{tag}_{corp}_{metric}")
        ax.fill_between(LAM,100*lo,100*hi,color=color,alpha=.08,lw=0)
        ax.plot(LAM,100*m,color=color,marker=marker,ms=2.8,lw=1.15)
        b.stats(vals[:,0]-vals[:,5],a["groups"],f"final_{tag}_{corp}_{metric}_off_minus_full")
        if labels:
            ax.text(2.06,m[-1]*100,f"{m[-1]*100:.1f}",color=color,va="center",fontsize=7)
            for k in (0,5):
                if metric in ("use","leak"):
                    ax.annotate(f"{m[k]*100:.1f}",(LAM[k],100*m[k]),xytext=(0,9 if metric=="use" else -12),textcoords="offset points",ha="center",fontsize=6.8,color=color)
    ax.set(xlim=(-.05,2.45 if labels else 2.06),ylim=(0,104),xticks=[0,.5,1,1.5,2],yticks=[0,25,50,75,100])
    ax.set_xlabel(r"Same-position weight $\lambda$")


def interpolation(P):
    fig,axs=plt.subplots(1,2,figsize=(W,75*MM),sharey=True)
    fig.subplots_adjust(left=.105,right=.97,bottom=.23,top=.77,wspace=.20)
    for j,corp in enumerate(CORPORA):
        title(axs[j],"ab"[j],CN[corp]+"-fitted lens")
        interp_axis(axs[j],P["prompt"],corp,1,"interp_prompt1",True)
        if j==0:axs[j].set_ylabel("Outcome rate (%)")
    fig.legend(handles=[Line2D([],[],color=col,marker=marker,label=lab,ms=3) for _,lab,col,marker in MET],
               loc="upper center",bbox_to_anchor=(.53,1.005),ncol=3,frameon=False,columnspacing=.9,handlelength=1.1)
    fig.text(.105,.062,r"$\lambda=0$: filtered     $\lambda=1$: full lens     $\lambda=2$: double the diagonal",fontsize=7.2)
    save(fig,"04_interpolation","Restoring same-position terms shifts the output toward names",
         "Eight measured mixtures J_lambda = J_O + lambda J_D on the expanded concept panel. At strength 1, "
         "whole-prompt swaps increasingly name the replacement, while correct relational answers decline and "
         "name leakage increases. Both source and target columns are unit-normalized; applied edit norms are "
         "matched to the full-lens reference at each edited site. Each of 661 eligible units is equally weighted "
         "after averaging its use queries (1,339 queries total). Shading is a pointwise 95% source-entity "
         "cluster-bootstrap interval, conditional on the fitted lenses. Lines join tested weights; they are "
         "not fitted curves. Naming is measured on a separate query; name leakage is measured in use answers.",
         "Controlled Characterization: after the interpolation and norm-matching setup.",
         "Reused concept panel; not a newly held-out confirmation. Question-inclusive edits are recomputed for "
         "each query, not one shared edit tensor. The advantage is placement- and dose-dependent (S2).")


def load_grad():
    blobs=[b.read(f"results/concept/gradgeom_c39gradprompt_qwen3-8b_ec4eca685b_shard{i}of2.json") for i in range(2)]
    assert blobs[0]["conditions"]==blobs[1]["conditions"]
    rows=sorted([r for bb in blobs for r in bb["records"]],key=lambda r:r["trial"])
    assert len(rows)==len({r["trial"] for r in rows})==1339
    cond=[tuple(x) for x in blobs[0]["conditions"]]
    groups=[r["type"]+"/"+r["source"] for r in rows]
    b.AUDIT["final_gradient_population"]={"queries":len(rows),"source_clusters":len(set(groups)),"dead_rows":sum(sum(bb["dead_rows"].values()) for bb in blobs)}
    return rows,cond,groups


def corr_boot(x,y,groups):
    # Exact cluster-weighted Pearson correlation using sufficient statistics.
    gs=sorted(set(groups)); ix={g:i for i,g in enumerate(gs)}
    sums=np.zeros((len(gs),6))
    for a,bb,g in zip(x,y,groups):
        sums[ix[g]] += [1,a,bb,a*a,bb*bb,a*bb]
    weights=np.random.default_rng(b.SEED).multinomial(len(gs),np.ones(len(gs))/len(gs),size=b.NBOOT)
    n,sx,sy,sxx,syy,sxy=(weights@sums).T
    rr=(sxy-sx*sy/n)/np.sqrt((sxx-sx*sx/n)*(syy-sy*sy/n))
    return float(np.corrcoef(x,y)[0,1]),np.quantile(rr,[.025,.975]).tolist()


def competition():
    rows,cond,groups=load_grad()
    fig,axs=plt.subplots(2,2,figsize=(W,124*MM))
    fig.subplots_adjust(left=.13,right=.97,bottom=.17,top=.885,hspace=.81,wspace=.42)
    arrays={}
    for j,F in enumerate("RL"):
        ax=axs[0,j]
        title(ax,"ab"[j],["Answer vs. source answer", "Name vs. target answer"][j])
        for corp in CORPORA:
            ids=[cond.index((corp,f"lam{lam:g}_cn")) for lam in LAM]
            a=np.array([[r[f"d{F}"][i] for i in ids] for r in rows])
            pred=np.array([[r[f"hat_applied_{F}"][i] for i in ids] for r in rows])
            arrays[corp,F]=(a,pred)
            m,lo,hi=b.stats(a,groups,f"final_gradient_mean_{corp}_{F}")
            ax.fill_between(LAM,lo,hi,color=CC[corp],alpha=.07,lw=0)
            ax.plot(LAM,m,color=CC[corp],marker=CM[corp],ms=2.7,lw=1.1,ls="-" if corp=="gsm8k" else "--")
        ax.axvline(1,color=LIGHT,ls=(0,(3,3)),lw=.7,zorder=0)
        ax.set(xlim=(-.05,2.05),xticks=[0,1,2],xlabel=r"Same-position weight $\lambda$",ylabel=f"Change in {F} from clean (nats)")
        ax.set_ylim((20,40) if F=="R" else (0,11))
        ax.set_yticks([20,25,30,35,40] if F=="R" else [0,5,10])
    cmap=LinearSegmentedColormap.from_list("density",["#E5EFEE","#78AFA6",OFF,"#124A46"])
    for j,corp in enumerate(CORPORA):
        ax=axs[1,j];title(ax,"cd"[j],CN[corp]+": leakage prediction")
        obs,pred=arrays[corp,"L"]
        x=(pred[:,1:]-pred[:,[0]]).ravel();y=(obs[:,1:]-obs[:,[0]]).ravel()
        corr,ci=corr_boot(x,y,np.repeat(groups,7))
        big=np.abs(y)>.1; sign=float(np.mean(np.sign(x[big])==np.sign(y[big])))
        b.STATS[f"final_leak_prediction_{corp}"]={"r":corr,"ci":ci,"pairs":len(x),"queries":1339,"sign_agreement":sign}
        # Show every pair; common full extents, no outlier trimming or smoothing.
        h=ax.hexbin(x,y,gridsize=35,extent=(-15,30,-15,30),mincnt=1,cmap=cmap,norm=LogNorm(1,1000),linewidths=0)
        assert x.min()>-15 and y.min()>-15 and x.max()<30 and y.max()<30,(x.min(),x.max(),y.min(),y.max())
        assert h.get_array().sum()==len(x) and h.get_array().max()<=1000
        b.AUDIT[f"final_hexbin_{corp}"]={"points_shown":int(h.get_array().sum()),"max_bin_count":int(h.get_array().max()),"axis_limits":[-15,30],"color_limits":[1,1000]}
        ax.plot([-15,30],[-15,30],color=GRAY,ls=(0,(3,3)),lw=.7)
        ax.axhline(0,color=LIGHT,lw=.5,zorder=0);ax.axvline(0,color=LIGHT,lw=.5,zorder=0)
        ax.set(xlim=(-15,30),ylim=(-15,30),xticks=[-10,0,10,20,30],yticks=[-10,0,10,20,30],
               xlabel="First-order prediction (nats)",ylabel="Observed change (nats)")
        ax.set_aspect("equal",adjustable="box")
        ax.text(.04,.96,f"r = {corr:.2f}\n{100*sign:.0f}% sign agreement",transform=ax.transAxes,va="top",fontsize=7.1,
                bbox=dict(facecolor="white",edgecolor="none",pad=1.3))
    fig.legend(handles=[Line2D([],[],color=CC[corp],marker=CM[corp],ls="-" if corp=="gsm8k" else "--",label=CN[corp],ms=3) for corp in CORPORA],
               loc="upper center",bbox_to_anchor=(.55,.995),ncol=2,frameon=False)
    # A shared, explicitly numerical density key.
    cax=fig.add_axes([.43,.07,.33,.017])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=LogNorm(1,1000),cmap=cmap),cax=cax,orientation="horizontal",ticks=[1,10,100,1000])
    cb.solids.set_rasterized(False)
    cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=6.7,length=2);cb.outline.set_linewidth(.5)
    fig.text(.13,.079,"Pairs per hexagon",fontsize=7,va="center")
    save(fig,"05_competition","The name competes with an answer that remains favored over the source",
         "(a,b) Teacher-forced margin changes relative to the unedited model: R = log p(target answer) - "
         "log p(source answer), and L = log p(target name) - log p(target answer). Increasing the diagonal "
         "weight improves the target answer relative to the source answer, but also improves the target name "
         "relative to that answer. These are changes in margins, not their absolute values. Means weight "
         "the 1,339 use queries equally; bands are 95% source-entity bootstrap intervals. (c,d) The clean-pass "
         "first-order prediction sum_site grad(L)^T delta_h predicts observed within-swap changes relative "
         "to lambda=0. Hexagons show all 9,373 query-by-nonzero-weight pairs per corpus; dashed lines indicate "
         "perfect calibration. Pearson correlations are 0.68 and 0.67, not a claim of exact calibration. "
         "Sign agreement excludes observed changes of at most 0.1 nat. Whole-prompt edits; strength 1.",
         "Results: after the paragraph interpreting competition and first-order predictions.",
         "C39 uses the applied edit and gradients on the clean graph; C38 independently checks matched "
         "norms and reproduces margin scores. One dead edit row is logged. Nonlinear effects remain. "
         "The upper plots are query-weighted, unlike unit-weighted generation outcomes in Figure 4.")


def corpus_details():
    best=c.corpus_rows("expanded")
    arms=["wikitext_a","gsm8k","aqua_rat","svamp","math_algebra","openwebmath","arith_words","ordered_scale"]
    names=["WikiText","GSM8K","AQuA-RAT","SVAMP","Competition algebra","OpenWebMath","Arithmetic in words","Ordered-scale text"]
    cells=[best("swap_"+arm) for arm in arms]
    assert [v[0] for v in cells]==[0,11,14,6,8,4,0,1]
    # A compact graphical table makes corpus, count, and chosen dose explicit.
    fig=plt.figure(figsize=(W,118*MM))
    ax=fig.add_axes([.03,.59,.94,.37],xlim=(0,116),ylim=(-.7,9.2));ax.axis("off")
    text(ax,0,8.9,"a",9,weight="bold");text(ax,6,8.9,"Exploratory corpus screen",8.5,weight="bold")
    for x,label,ha in [(0,"Fitting corpus","left"),(72,"Success / 39","right"),(105,"Selected strength","right")]:
        text(ax,x,7.9,label,7.2,GRAY,ha=ha)
    ax.plot([0,115],[7.45,7.45],color=LIGHT,lw=.65)
    for i,(name,(hits,dose)) in enumerate(zip(names,cells)):
        y=6.85-i
        text(ax,0,y,name,7.5)
        text(ax,72,y,str(hits),7.7,weight="bold" if name in ("GSM8K","AQuA-RAT") else "normal",ha="right")
        text(ax,102,y,f"{dose:g}",7.5,ha="right")

    raw=b.read("results/causal/steering_qwen3-8b_swap_oscale_abl_shard0of1.json")
    trials={(t.category,t.func,t.source_arg,t.target_arg):t for t in c.heldout_trials("ordered-scale")}
    for r in raw:
        t=trials[r["category"],r["func"],r["source_arg"],r["target_arg"]]
        want=t.source_answer if r["arm"]=="baseline" else t.target_answer
        r["scored"]=bool(r.get("generated")) and c.answer_matches(r["generated"],want,digit=c._digit_form(want))
    clean={r["prompt_key"] for r in raw if r["arm"]=="baseline" and r["scored"]}
    arm2=["gsm8k","gsm8k_q","gsm8k_sol","gsm8k_shuffled","gsm8k_noent","gsm8k_nonum","wikitext_a"]
    lab2=["Original GSM8K","Questions only","Solutions only","Sentences shuffled","Entity names altered","Digits removed","WikiText"]
    banks=[{(r["prompt_key"],r["target_arg"]):int(r["scored"]) for r in raw if r["arm"]=="swap_"+a and r["strength"]==1 and r["prompt_key"] in clean} for a in arm2]
    keys=sorted(banks[0]);assert len(keys)==105 and all(set(x)==set(keys) for x in banks)
    vals=np.array([[d[k] for d in banks] for k in keys]);assert vals.sum(0).tolist()==[30,6,26,27,34,35,9]
    m,lo,hi=b.stats(vals,[k[0] for k in keys],"final_corpus_ablation")
    ax=fig.add_axes([.36,.14,.46,.34])
    fig.text(.03,.523,"b",fontsize=9,weight="bold")
    fig.text(.085,.523,"Corpus ablation on new arguments",fontsize=8.4)
    ax.axvline(m[0]*100,color=LIGHT,ls=(0,(3,3)),lw=.9)
    for i in range(7):
        col=CC["gsm8k"] if i<6 else CC["wikitext_a"]
        ax.errorbar(m[i]*100,i,xerr=[[100*(m[i]-lo[i])],[100*(hi[i]-m[i])]],fmt="o",color=col,ms=3.5,capsize=2,lw=.8)
        ax.text(1.07,i,f"{int(vals[:,i].sum())}/105",transform=ax.get_yaxis_transform(),ha="left",va="center",fontsize=7)
    ax.axhline(5.5,color=LIGHT,lw=.7)
    ax.set(xlim=(0,50),ylim=(6.6,-.5),xticks=[0,20,40],yticks=range(7),yticklabels=lab2,xlabel="Swap success (%)")
    ax.tick_params(axis="y",length=0);ax.spines["left"].set_visible(False)
    fig.text(.03,.043,"Panel b: a separate 105-trial evaluation; all fits use strength 1.",fontsize=7.2,color=GRAY)
    b.AUDIT["final_corpus_screen"]=dict(arms=arms,cells=cells)
    save(fig,"S1_corpus_details","Corpus changes are not explained by topic matching alone",
         "(a) Exploratory comparison on 39 numerical-argument trials, with 32 documents per fit and strength "
         "selected on the evaluated subset. (b) Corpus ablations at a common strength of 1 on a separate "
         "new-argument/template panel (weekdays, ordinals, sports, instruments; 105 of 120 trials pass the "
         "clean-answer screen). Solutions-only and other modified texts retain more of the original GSM8K "
         "result than questions alone. The dashed reference marks original GSM8K. Error bars are 95% "
         "source-prompt bootstrap intervals conditional on each fit. These controls do not identify a "
         "unique useful property of solution text, and this 105-trial panel is not the original "
         "non-numerical subset of Figure 2.",
         "Appendix, with references from the Fitting Corpus section.",
         "The exploratory numerical counts retain the inherited grading limitation. The downloaded lens is "
         "omitted here because it uses a different fitting budget. Matched 32-document fits are compared.")


def boundary(P):
    fig,axs=plt.subplots(2,2,figsize=(W,112*MM),sharex=True,sharey=True)
    fig.subplots_adjust(left=.105,right=.97,bottom=.18,top=.84,wspace=.21,hspace=.63)
    for i,(pos,dose) in enumerate([("prefix",1),("prompt",.5)]):
        for j,corp in enumerate(CORPORA):
            ax=axs[i,j]
            title(ax,"abcd"[2*i+j],CN[corp]+(" / prefix only" if i==0 else " / lower strength"))
            interp_axis(ax,P[pos],corp,dose,f"boundary_{pos}{dose}")
            if i==0: ax.set_xlabel("")
            if j==0:ax.set_ylabel("Outcome rate (%)")
    fig.legend(handles=[Line2D([],[],color=col,marker=marker,label=lab,ms=3) for _,lab,col,marker in MET],loc="upper center",bbox_to_anchor=(.53,.98),ncol=3,frameon=False,columnspacing=.9,handlelength=1.1)
    fig.text(.105,.065,"Top: prefix-only edits, strength 1. Bottom: whole prompt, strength 0.5.",fontsize=7.1)
    fig.text(.105,.026,"Same units, column normalization, and within-condition edit-norm matching.",fontsize=7,color=GRAY)
    save(fig,"S2_scope","The benefit is not independent of placement or strength",
         "Controls on the same 661-unit panel as Figure 4. (a,b) With prefix-only edits at strength 1, "
         "GSM8K no longer favors the filtered endpoint; WikiText is similar at the two endpoints. "
         "(c,d) With whole-prompt edits at strength 0.5, GSM8K also favors retaining the diagonal. "
         "Both columns are unit-normalized and edit norms are matched within each placement/strength, "
         "not across different placements or strengths. Lines join measured weights; shading shows "
         "pointwise 95% source-entity bootstrap intervals. These controls limit any claim of a universal "
         "advantage from same-position filtering.",
         "Appendix; cite explicitly in Limitations and alongside Figure 4.",
         "The manuscript currently underemphasizes these boundaries. Prefix-only edit tensors are shared "
         "across query types; whole-prompt tensors are query-dependent.")


def latex_caption(entry):
    # ASCII captions are also readable without math rendering.
    s=entry["caption"].replace("%",r"\%").replace("{",r"\{").replace("}",r"\}")
    s=s.replace("J_lambda = J_O + lambda J_D",r"$J_\lambda=J_O+\lambda J_D$")
    s=s.replace("lambda=0",r"$\lambda=0$")
    s=s.replace("sum_site grad(L)^T delta_h",r"$\sum_{\mathrm{site}}\nabla L^\top\delta h$")
    s=s.replace("Figure 2",r"Figure~\ref{fig:02-corpus}").replace("Figure 4",r"Figure~\ref{fig:04-interpolation}")
    return r"\textbf{"+entry["heading"]+".} "+s


def assemble():
    atlas=fitz.open()
    readme=["# Final manuscript figures: claim-led redesign", "",
        "Rebuilt from the checksummed reference data. See docs/PAPER_AUDIT.md for manuscript consistency notes.","",
        "## Start here", "",
        "Use Figures 1-5 in the main paper; S1-S2 are supporting figures. The overview is an original vector schematic with a verified saved-generation example, not an invented illustration. All figures are **122 mm wide**, sized for the LNCS text block. The review PDF displays that actual size, not a page-filling enlargement.","",
        "- `PDF`: insert directly with `\\includegraphics[width=\\linewidth]{figures/01_overview.pdf}`.",
        "- `SVG`: editable original for Illustrator, Inkscape, Figma, or another vector editor.",
        "- `PNG`: 400-dpi preview, not needed for Overleaf.",
        "- `figure_blocks.tex`: standalone, captioned figure environments with suggested insertion locations. Copy individual blocks, not the entire file at one location.",
        "- `data_audit.json`: pinned raw-source hashes, populations, counts, and uncertainty estimates.","",
        "Rebuild: `.venv/bin/python scripts/56_final_paper_figures.py`. No GPU calls. Statistical intervals use 10,000 source-cluster resamples, conditional on fitted directions. Replicate dots represent fits, not bootstrap uncertainty. Every PDF was rendered and inspected at insertion size; all fonts are embedded and no figure contains raster artwork. No TeX engine is installed locally, so the complete manuscript has not been compiled here.","",
        "## Design references", "",
        "The design follows [Nature's figure specifications](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/) and [Nature Reviews' figure-design guide](https://www.nature.com/documents/natrev-artworkguide_PS.pdf): logical reading order, meaningful and consistent color, vector text, and minimal decoration. The conceptual use of a residual-stream schematic is informed by [Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html); the paper's [J-lens precursor](https://transformer-circuits.pub/2026/workspace/) and [Short Horizons](https://arxiv.org/html/2608.25347v1) were reviewed for context. No artwork is copied or traced.","",
        "## Before submission: text/figure consistency", "",
        "1. The +26.2 to +31.3 and +2.7 to +7.6 values in Results are **changes from the clean margins**, not absolute R or L. Figure 5 labels this explicitly.",
        "2. Whole-prompt edits are recomputed for each question. The Evaluation paragraph's statement that each prefix edit is read out by all queries describes prefix-only runs, not the principal whole-prompt curves.",
        "3. S2 shows real placement/strength reversals. 'Broadly useful' should not be read as dose- or position-independent. The main claim should specify whole-prompt swaps at strength 1.",
        "4. The separate 649-unit norm-ratio sentence in the manuscript comes from an older report; current complete C36 shards contain 661 common units. Do not silently mix those runs or populations.",
        "5. The 105 original non-numerical trials in Figure 2 and 105 new-argument trials in S1 are distinct evaluations. Best-on-evaluation strengths in Figure 2 differ from S1's fixed strength.",
        "6. The term 'off-diagonal' here means all later-position pairs, not exclusively the sparse-concept positions of Yan et al. Likewise, filtering t'=t does not remove all nearby short-horizon effects.",
        "7. These concept-panel follow-ups reuse the same evaluation items. 'Held-out' needs an explicit held-out-from-what definition; these are not fresh confirmatory replications.","",
        "## Figures", ""]
    tex=["% Standalone figure blocks; copy to the suggested positions in main.tex.",
         "% graphicx is already loaded. Main manuscript has NOT been edited.", ""]
    for i,e in enumerate(ENTRIES):
        title_str=(f"Figure {i+1}" if i<5 else f"Supplement {i-4}")+" | "+e["heading"]
        page=atlas.new_page(width=595.276,height=841.89)
        page.insert_text((54,45),title_str,fontsize=11,fontname="hebo",color=(.14,.18,.21))
        page.insert_text((54,63),"122 mm manuscript-width proof | vector originals | 25 September 2026",fontsize=8,color=(.4,.45,.48))
        src=fitz.open(e["path"])
        x=(page.rect.width-src[0].rect.width)/2
        bottom=85+src[0].rect.height
        page.show_pdf_page(fitz.Rect(x,85,x+src[0].rect.width,bottom),src,0)
        ret=page.insert_textbox(fitz.Rect(66,bottom+20,529,775),e["caption"]+"\n\nPlacement: "+e["where"],fontsize=9,lineheight=1.2,color=(.14,.18,.21))
        assert ret>=0,(e["name"],ret)
        page.insert_text((66,814),str(i+1),fontsize=8,color=(.4,.45,.48))
        readme += ["### "+title_str,"",f"![{e['heading']}]({e['name']}.png)","",e["caption"],"", "**Place:** "+e["where"],"", "**Audit:** "+e["audit_note"],"",f"[PDF]({e['name']}.pdf) | [SVG]({e['name']}.svg)",""]
        tex += ["% "+e["where"], r"\begin{figure}[t]",r"  \centering",rf"  \includegraphics[width=\linewidth]{{figures/{e['name']}.pdf}}",r"  \caption{"+latex_caption(e)+"}",rf"  \label{{fig:{e['name'].replace('_','-')}}}",r"\end{figure}",""]
    p=PDF/"jlens_final_visual_review.pdf"
    atlas.save(p,garbage=4,deflate=True)
    doc=fitz.open(p)
    for i,page in enumerate(doc):
        assert not page.get_images()
        page.get_pixmap(matrix=fitz.Matrix(1.6,1.6)).save(QA/f"review_{i+1}.png")
    (OUT/"README.md").write_text("\n".join(readme))
    (OUT/"figure_blocks.tex").write_text("\n".join(tex))
    (OUT/"data_audit.json").write_text(json.dumps(dict(seed=b.SEED,n_boot=b.NBOOT,sources=b.SOURCES,statistics=b.STATS,audit=b.AUDIT),indent=2))
    print(f"Checked {len(ENTRIES)} vector figures; {len(b.SOURCES)} raw input artifacts. Review: {p}")


if __name__=="__main__":
    latent=load_latent()
    overview(latent)
    corpus_main()
    core(latent)
    P=load_interp()
    interpolation(P)
    competition()
    corpus_details()
    boundary(P)
    assemble()

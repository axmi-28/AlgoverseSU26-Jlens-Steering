"""Curated manuscript figures, including C36/C37. CPU-only; raw JSON inputs.

Run .venv/bin/python scripts/52_submission_figures.py.
Earlier figures and manuscripts are preserved. Outputs in output/figures/supporting/.
"""
from pathlib import Path
import importlib.util
import io
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("previous", ROOT / "scripts/50_paper_figures.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
OUT = ROOT / "output/figures/supporting"
OUT.mkdir(parents=True, exist_ok=True)
FIGS = []
NAMES = {"fig01_core_swap_gain": "02_core_swap_gain", "fig05_additive_baselines": "06_additive_baselines"}


def save(fig, name, title, caption, note):
    name = NAMES.get(name, name)
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.png", dpi=450)
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    FIGS.append(dict(name=name, title=title, caption=caption, note=note,
                     pdf=buf.getvalue(), size=fig.get_size_inches()))
    plt.close(fig)
    print(name, flush=True)


b.save = save


def schematic():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.5), gridspec_kw={"width_ratios": [1, 1.45]})
    fig.subplots_adjust(left=.09, right=.97, bottom=.20, top=.82, wspace=.38)
    a, z = axs
    b.panel(a, "a", "Which position pairs enter the lens?")
    n = 6
    for source in range(n):
        for target in range(n):
            color = "#ECEEF0" if target < source else b.C["diag"] if target == source else "#C8E4DC"
            a.add_patch(Rectangle((target, source), 1, 1, fc=color, ec="white", lw=1))
    a.set(xlim=(0,n), ylim=(n,0), aspect="equal", xticks=[], yticks=[],
          xlabel="Later-layer target position t'", ylabel="Source position t")
    for s in a.spines.values(): s.set_visible(False)
    a.text(3, 1.25, "future-position\ninfluence", ha="center", fontsize=7.5)
    a.text(1.55, 4.5, "causally\nexcluded", ha="center", fontsize=7.5, color="#74808A")
    a.text(3, 6.9, r"$J_\lambda = J_{\mathrm{off}} + \lambda J_{\mathrm{diag}}$", ha="center", fontsize=9)
    b.panel(z, "b", "What should the intervention change?")
    z.axis("off")
    z.text(0, .94, "Illustrative task (not a measured trial)", color="#65717B", fontsize=7)
    z.text(0, .79, "Clue: the country containing Lyon", fontsize=9)
    z.text(0, .64, "Intervene: France  →  Italy", fontsize=9, color=b.C["off"])
    for y, q, answer in [(.43, "Naming question", "Italy"), (.20, "Capital question", "Rome")]:
        z.text(.03,y,q,fontsize=8)
        z.annotate("", xy=(.68,y+.015),xytext=(.49,y+.015),arrowprops=dict(arrowstyle="->",lw=.8,color="#65717B"))
        z.text(.72,y,answer,fontsize=9,color=b.C["off"])
    z.text(.03,.015,"Name emission in the capital answer: “Italy …”",fontsize=7.5,color="#A25531")
    b.foot(fig,"Schematic only  |  diagonal means t' = t, not diagonal entries of a hidden-state matrix")
    save(fig,"01_design","Separate direction construction from behavioral evaluation",
         "The lens averages same-position and future-position influence. We vary their mixture and measure naming, relation answering, and literal name emission separately.",
         "The matrix shows admissible position pairs, not measured Jacobian energy. The example illustrates the scoring distinction and is not a claim about a particular saved generation. Naming and relation prompts receive identical edit tensors only in the prefix-only condition. Name emission and correct relation answering are not mutually exclusive.")


def load(stem, expected):
    rows, cond = b.load_concept(stem)
    a = b.unit_arrays(rows, cond)
    assert len(a["keys"]) == expected, (stem,len(a["keys"]))
    return a


LAM = [0,.125,.25,.5,.75,1,1.5,2]
MET = [("express", "Naming", "#65737E", "s"),
       ("use", "Relation answer", b.C["off"], "o"),
       ("leak", "Name emission", "#B76B36", "^")]


def interpolation(P, dose):
    fig, axs = plt.subplots(2,2,figsize=(7.2,5.05),sharex=True,sharey=True)
    fig.subplots_adjust(left=.09,right=.97,bottom=.16,top=.83,hspace=.48,wspace=.20)
    for r,pos in enumerate(["prompt","prefix"]):
        a=P[pos]
        for j,corp in enumerate(["gsm8k","wikitext_a"]):
            ax=axs[r,j]
            b.panel(ax,"abcd"[r*2+j],f"{'Whole prompt' if r==0 else 'Prefix only'} · {'GSM8K' if j==0 else 'WikiText'}")
            ax.axvline(1,color="#B8BFC5",lw=.75,ls=(0,(3,3)),zorder=0)
            inds=[b.col(a,corp,f"lam{x:g}_cn",dose) for x in LAM]
            for metric,label,color,marker in MET:
                vals=a[metric][:,inds]
                m,lo,hi=b.stats(vals,a["groups"],f"c37_{pos}_{corp}_{dose}_{metric}")
                ax.fill_between(LAM,lo*100,hi*100,color=color,alpha=.09,lw=0)
                ax.plot(LAM,m*100,color=color,marker=marker,ms=3.2,lw=1.25)
                b.stats(vals[:,0]-vals[:,5],a["groups"],f"c37_endpoint_{pos}_{corp}_{dose}_{metric}")
            ax.set(xlim=(-.04,2.04),ylim=(0,103),yticks=[0,25,50,75,100],xticks=[0,.5,1,1.5,2])
            if j==0: ax.set_ylabel("Outcome rate (%)")
            if r==1: ax.set_xlabel(r"Diagonal weight $\lambda$ (0 = off; 1 = full)")
    fig.legend(handles=[Line2D([],[],color=c,marker=m,label=l,ms=4) for _,l,c,m in MET],
               loc="upper center",bbox_to_anchor=(.53,.985),ncol=3,frameon=False)
    b.foot(fig,f"661 units  |  strength {dose:g}  |  unit-normalized columns; matched edit norms  |  pointwise 95% intervals")
    save(fig,"03_diagonal_interpolation" if dose==1 else "S01_lower_strength",
         "Adding the diagonal changes the naming-answering trade-off" if dose==1 else "The trade-off depends on intervention strength",
         "At unit strength, adding diagonal influence increases naming and name emission while reducing whole-prompt relation-answer success. Prefix-only edits do not show the same trade-off." if dose==1 else
         "At strength 0.5, adding the diagonal improves GSM8K whole-prompt answering. This lower-strength control rules out a dose-independent advantage for diagonal removal.",
         "C37, Qwen3-8B, two fixed 32-document component fits. All eight measured lambda values are shown; connecting lines are guides, not fitted models. Both swap columns are unit-normalized and each edit row is matched to its corpus-specific full-lens reference. Queries are averaged within 661 latent-clue units. Shading is a 10,000-draw source-entity cluster bootstrap; intervals are pointwise and conditional on fitted directions. Whole-prompt edits depend on the question, unlike prefix-only edits. These are reused-panel follow-up analyses, not independent fit replications.")


def placement(P):
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.25),sharex=True)
    fig.subplots_adjust(left=.19,right=.97,bottom=.24,top=.79,wspace=.40)
    for j,corp in enumerate(["gsm8k","wikitext_a"]):
        ax=axs[j]; b.panel(ax,"ab"[j],["GSM8K fit","WikiText fit"][j])
        ax.axvline(0,color="#929AA3",ls=(0,(3,3)),lw=.8)
        vals=np.column_stack([P[p]["use"][:,b.col(P[p],corp,"off_m")]-P[p]["use"][:,b.col(P[p],corp,"full")] for p in ["prefix","prompt","query"]])
        m,lo,hi=b.stats(vals,P["prefix"]["groups"],f"placement_difference_{corp}")
        ax.errorbar(m*100,[2,1,0],xerr=100*np.array([m-lo,hi-m]),fmt="o",color=b.C["off"],capsize=2)
        for i,y in enumerate([2,1,0]):
            label = "0.0 pp" if abs(m[i]*100)<.05 else f"{m[i]*100:+.1f} pp"
            ax.text(m[i]*100,y+.20,label,ha="center",fontsize=8,color=b.C["off"])
        ax.set(xlim=(-20,25),ylim=(-.45,2.6),yticks=[2,1,0],yticklabels=["Prefix only","Whole prompt","Question only"] if j==0 else [],xticks=[-20,-10,0,10,20],xlabel="Off-diagonal minus full (pp)")
        ax.spines["left"].set_visible(False);ax.tick_params(axis="y",length=0)
    b.foot(fig,"661 common units  |  strength 1  |  paired differences with 95% source-entity bootstrap intervals")
    save(fig,"04_placement","The benefit changes with intervention placement",
         "Filtering helps most clearly when edits cover the whole prompt. Prefix-only edits reverse the GSM8K advantage and eliminate the WikiText advantage.",
         "C32 placement runs; these paired differences use the same 661 units and 1,339 eligible relation questions. Positive values favor filtering. Norms are matched within each placement, not across placements. This comparison changes several edited positions and does not isolate the final prompt position. The interpolation runs in Figure 3 additionally normalize the swap columns and are a separate execution.")


def scaling(a):
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.5),sharey=True)
    fig.subplots_adjust(left=.085,right=.97,bottom=.29,top=.76,wspace=.26)
    for j,corp in enumerate(["gsm8k","wikitext_a"]):
        ax=axs[j];b.panel(ax,"ab"[j],["GSM8K fit","WikiText fit"][j])
        for fam,arms in [("full",["full","full_cn","full_koff"]),("off",["off_m","off_cn","off_kfull"])]:
            inds=[b.col(a,corp,x) for x in arms]
            v=a["use"][:,inds];m,lo,hi=b.stats(v,a["groups"],f"c36_{corp}_{fam}")
            x=np.arange(3)+(-.08 if fam=="full" else .08)
            ax.errorbar(x,m*100,yerr=100*np.array([m-lo,hi-m]),color=b.C[fam],fmt="o",ls="-",capsize=2)
            for i in range(3):ax.text(x[i],100*(hi[i] if fam=="off" else lo[i])+(2 if fam=="off" else -3),f"{m[i]*100:.1f}",ha="center",va="bottom" if fam=="off" else "top",fontsize=7.5,color=b.C[fam])
        full=[b.col(a,corp,x) for x in ["full","full_cn","full_koff"]]
        off=[b.col(a,corp,x) for x in ["off_m","off_cn","off_kfull"]]
        b.stats(a["use"][:,off]-a["use"][:,full],a["groups"],f"c36_paired_{corp}")
        ax.set(xticks=range(3),xticklabels=["Native\nratio","Equal\ncolumn norms","Other family's\nratio"],ylim=(40,100),yticks=[40,60,80,100],xlim=(-.3,2.3))
        if j==0:ax.set_ylabel("Relation-answer success (%)")
    b.legend_methods(fig,["full","off"],y=.99)
    b.foot(fig,"661 available units  |  whole-prompt edits, strength 1  |  matched edit norms; 95% cluster-bootstrap intervals")
    save(fig,"05_scaling_control","Relative column scales do not explain the filtering gain",
         "The off-diagonal advantage survives equalizing both direction-column norms and exchanging native source-target norm ratios between the two families.",
         "C36. The current raw shards contain 661 common eligible latent-clue units, superseding the older 649-unit Markdown report. The third condition retains each family's directions but adopts the other family's per-pair target/source norm ratio. Output edits are row-norm matched in all three comparisons. Connected categories indicate the same direction family, not a continuous fitted curve. Intervals resample source entities and do not estimate fitting uncertainty.")


def assemble():
    book=fitz.open()
    lines=["# Curated manuscript figures - 21 September 2026", "",
           "Six main figures and one supplementary control. Earlier figures and manuscripts are unchanged. Use 02 for the core finding; 03 is the new centerpiece. 01 is optional if space is tight. The older distance, projection, and free-generation figures are better suited to the appendix.", "",
           "**Formats:** 450-dpi PNG for Word/Docs; editable-text SVG for vector insertion and editing. The captioned vector PDF review atlas is `output/pdf/jlens_submission_figures.pdf`. Each atlas page contains one vector figure, not a screenshot.", "",
           "**Rebuild:** `.venv/bin/python scripts/52_submission_figures.py`. Data are recomputed from raw saved JSON, not transcribed report tables. `data_audit.json` records source SHA-256 hashes, means, cluster intervals, and paired contrasts. No GPU or model calls.", "",
           "**Statistics:** 10,000 cluster resamples, conditional on fixed fitted lenses. Bands are pointwise, not simultaneous; no multiplicity correction. C37 and C36 are follow-ups on a reused evaluation panel. Read captions before merging numbers across runs or normalization conventions.", ""]
    for i,e in enumerate(FIGS):
        page=book.new_page(width=612,height=792)
        page.insert_text((42,35),f"{'SUPPLEMENT' if e['name'].startswith('S') else 'FIGURE'} {e['name'].split('_')[0]}",fontsize=8,color=(.35,.4,.43))
        assert page.insert_textbox(fitz.Rect(42,49,570,91),e["title"],fontsize=13,fontname="hebo")>=0
        src=fitz.open(stream=e["pdf"],filetype="pdf")
        h=528*e["size"][1]/e["size"][0]
        page.show_pdf_page(fitz.Rect(42,100,570,100+h),src,0)
        assert page.insert_textbox(fitz.Rect(48,116+h,564,749),e["caption"]+"\n\n"+e["note"],fontsize=9,lineheight=1.3)>=0,e["name"]
        page.insert_text((48,769),"J-lens steering | manuscript figure review | 21 September 2026",fontsize=7,color=(.4,.45,.48))
        page.insert_text((550,769),str(i+1),fontsize=7)
        lines += ["## "+e["name"]+": "+e["title"],"",f"![{e['title']}]({e['name']}.png)","",e["caption"],"",e["note"],"",f"[PNG]({e['name']}.png) | [SVG]({e['name']}.svg)",""]
    path=ROOT/"output/pdf/jlens_submission_figures.pdf"
    book.save(path,garbage=4,deflate=True)
    (OUT/"README.md").write_text("\n".join(lines))
    (OUT/"data_audit.json").write_text(json.dumps(dict(seed=b.SEED,n_boot=b.NBOOT,sources=b.SOURCES,statistics=b.STATS,populations=b.AUDIT),indent=2))
    qa=ROOT/"tmp/pdfs/submission";qa.mkdir(parents=True,exist_ok=True)
    final=fitz.open(path)
    for i,p in enumerate(final):
        assert not p.get_images(), "Unexpected raster figure"
        p.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(qa/f"page_{i+1}.png")
    print(f"Verified {len(final)} vector PDF pages; {len(b.SOURCES)} raw inputs")


if __name__ == "__main__":
    schematic()
    b.figure1()
    P={"prompt":load("swap_c37lambdaprompt_qwen3-8b_034da26ce9",661),
       "prefix":load("swap_c37lambdaprefix_qwen3-8b_75f74629d0",661)}
    assert P["prompt"]["keys"]==P["prefix"]["keys"]
    interpolation(P,1)
    placement(b.load_positions())
    scaling(load("swap_c36kappa_qwen3-8b_3b28f16205",661))
    b.figure5()
    interpolation(P,.5)
    assemble()

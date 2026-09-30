"""Two corpus figures, regraded from raw generations. CPU only."""
import importlib.util
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pymupdf as fitz

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
def module(name,filename):
    s=importlib.util.spec_from_file_location(name,ROOT/'scripts'/filename)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
b=module('style','50_paper_figures.py')
c9=module('grader','25_domain_causal.py')
from jsteer.data import heldout_trials
from jsteer.steering import answer_matches
from jsteer.sweeps import _digit_form
OUT=ROOT/'output/figures/supporting'
ENTRIES=[]

def corpus_rows(label):
    rel=f'results/causal/steering_qwen3-8b_swap_{label}_shard0of1.json'
    raw=b.read(rel)
    arms=set(r['arm'] for r in raw)-{'baseline'}
    doses=sorted({r['strength'] for r in raw if r['arm']!='baseline'})
    rows=c9.load(str(ROOT/rel),1,1+len(arms)*len(doses));c9.regrade(rows)
    clean={t.prompt for t,r in rows if r['arm']=='baseline' and r['hit_re']}
    def best(arm,numeric=True):
        cells=[]
        for dose in doses:
            selected=[r for t,r in rows if t.prompt in clean and (t.category=='numbers')==numeric and r['arm']==arm and r['strength']==dose]
            assert len(selected)==(39 if numeric else 105)
            cells.append((sum(r['hit_re'] for r in selected),dose))
        return max(cells,key=lambda p:p[0])
    return best

def save(fig,name,title,caption):
    fig.savefig(OUT/f'{name}.png',dpi=450)
    fig.savefig(OUT/f'{name}.svg')
    path=ROOT/'output/pdf'/f'jlens_{name}.pdf'
    fig.savefig(path)
    plt.close(fig)
    doc=fitz.open(path);assert not doc[0].get_images()
    qa=ROOT/'tmp/pdfs/corpus';qa.mkdir(parents=True,exist_ok=True)
    doc[0].get_pixmap(matrix=fitz.Matrix(2,2)).save(qa/f'{name}.png')
    ENTRIES.append((name,title,caption))

def comparison():
    best=corpus_rows('expanded');rep=corpus_rows('tier0');aqua=corpus_rows('aqua')
    names=['WikiText','GSM8K','AQuA-RAT','SVAMP','Competition algebra','OpenWebMath','Arithmetic in words','Ordered-scale text']
    arms=['wikitext_a','gsm8k','aqua_rat','svamp','math_algebra','openwebmath','arith_words','ordered_scale']
    cells=[best('swap_'+a) for a in arms]
    assert [x[0] for x in cells]==[0,11,14,6,8,4,0,1]
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.65),gridspec_kw={'width_ratios':[1.18,1]})
    fig.subplots_adjust(left=.235,right=.965,bottom=.23,top=.80,wspace=.72)
    ax=axs[0];b.panel(ax,'a','Different fitting distributions')
    for i,(hits,dose) in enumerate(cells):
        color=b.C['off'] if i in (1,2) else b.C['full']
        ax.hlines(i,0,100*hits/39,color='#D5DCDF',lw=1)
        ax.plot(100*hits/39,i,'o',color=color,ms=4.5)
        ax.text(100*hits/39+1.8,i,f'{hits}/39',va='center',fontsize=7.5,color=color)
    ax.set(yticks=range(8),yticklabels=names,ylim=(7.65,-.65),xlim=(-1,46),xticks=[0,10,20,30,40],xlabel='Swap success (%)')
    ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    ax=axs[1];b.panel(ax,'b','Five document draws per corpus')
    values=[]
    for i,(name,prefix,fn) in enumerate([('WikiText','wikirep',rep),('GSM8K','gsm8k',rep),('AQuA-RAT','aqua_rat',aqua)]):
        counts=[fn(f'swap_{prefix}_r{k}')[0] for k in range(5)]
        non=[fn(f'swap_{prefix}_r{k}',False)[0] for k in range(5)]
        values.append(dict(corpus=name,numerical=counts,other=non))
        # Vertical offsets display coincident fits without changing their x values.
        for count in sorted(set(counts)):
            n=counts.count(count);ys=i+np.linspace(-.12,.12,n) if n>1 else [i]
            ax.scatter([100*count/39]*n,ys,s=23,color=b.C['full'] if i==0 else b.C['off'],zorder=3)
        ax.text(100*max(counts)/39+2,i,f'{min(counts)}-{max(counts)}/39',va='center',fontsize=7.5)
    ax.set(yticks=range(3),yticklabels=['WikiText','GSM8K','AQuA-RAT'],ylim=(2.65,-.65),xlim=(-1,53),xticks=[0,10,20,30,40,50],xlabel='Swap success (%)')
    ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    b.foot(fig,'Qwen3-8B | 32 documents per fit | 39 numerical-argument trials | best tested strength per fit/subset')
    b.AUDIT['corpus_comparison']=dict(arms=arms,cells=cells,replicates=values)
    save(fig,'corpus_comparison','The fitting distribution changes steering efficacy',
         'Problem-and-solution corpora outperform WikiText on numerical-argument swaps, while mathematical topic alone is insufficient. Each dot in panel b represents a separate 32-document fit; every GSM8K and AQuA-RAT draw exceeds every WikiText draw. Panel a uses one fit per corpus. Strength is selected separately on this development subset from 0.5, 1, and 2; these are not common-dose or held-out-selected comparisons. The 39 trials include non-arithmetic operations and a known square-task answer-key limitation. Non-numerical outcomes overlap across fits (GSM8K 39–44/105, WikiText 39–42/105, AQuA-RAT 38–42/105), so the result is not universal superiority. Vertical offsets separate overlapping dots and are not uncertainty intervals.')

def ablations(only_six=False):
    raw=b.read('results/causal/steering_qwen3-8b_swap_oscale_abl_shard0of1.json')
    trials={(t.category,t.func,t.source_arg,t.target_arg):t for t in heldout_trials('ordered-scale')}
    for r in raw:
        t=trials[(r['category'],r['func'],r['source_arg'],r['target_arg'])]
        want=t.source_answer if r['arm']=='baseline' else t.target_answer
        r['scored']=bool(r.get('generated')) and answer_matches(r['generated'],want,digit=_digit_form(want))
    clean={r['prompt_key'] for r in raw if r['arm']=='baseline' and r['scored']}
    arms=['gsm8k','gsm8k_q','gsm8k_sol','gsm8k_shuffled','gsm8k_noent','gsm8k_nonum','wikitext_a','averaged']
    labels=['Original GSM8K','Questions only','Solutions only','Sentences shuffled','Entity names altered','Digits removed','WikiText fit','Downloaded reference']
    if only_six:
        arms,labels=arms[:6],labels[:6]
    banks=[]
    for arm in arms:
        rows=[r for r in raw if r['arm']=='swap_'+arm and r['strength']==1 and r['prompt_key'] in clean]
        bank={(r['prompt_key'],r['target_arg']):int(r['scored']) for r in rows}
        assert len(rows)==len(bank)==105
        banks.append(bank)
    keys=sorted(banks[0]);assert all(set(x)==set(keys) for x in banks)
    vals=np.array([[x[k] for x in banks] for k in keys]);assert vals.sum(0).tolist()==[30,6,26,27,34,35,9,7][:len(arms)]
    m,lo,hi=b.stats(vals,[k[0] for k in keys],'corpus_ablation_rates')
    fig,ax=plt.subplots(figsize=(7.2,3.8));fig.subplots_adjust(left=.29,right=.93,bottom=.22,top=.84)
    ax.axvline(100*m[0],color='#A7B4BB',ls=(0,(3,3)),lw=.8)
    if not only_six: ax.axhline(5.5,color='#DFE4E6',lw=.7)
    for i in range(len(arms)):
        color=b.C['off'] if i<6 else b.C['full']
        ax.errorbar(m[i]*100,i,xerr=[[100*(m[i]-lo[i])],[100*(hi[i]-m[i])]],fmt='o',color=color,capsize=2,ms=4.5)
        ax.text(52,i,f'{int(vals[:,i].sum())}/105',ha='right',va='center',fontsize=8,color=color)
    ax.set(yticks=range(len(arms)),yticklabels=labels,ylim=(len(arms)-.35,-.65),xlim=(0,54),xticks=[0,10,20,30,40,50],xlabel='Swap success (%)')
    ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    ax.set_title('Changing the fitting text at a common intervention strength',loc='left',pad=14,fontsize=10)
    b.foot(fig,'Qwen3-8B | new-argument panel, 105 trials | strength 1 | 95% source-prompt bootstrap intervals')
    save(fig,'corpus_ablations','Solution-bearing text matters beyond mathematical symbols',
         'On new arguments and templates, a questions-only fit loses much of the GSM8K advantage. Retaining solutions alone, shuffling sentences, altering entity names, or deleting digits retains substantially more success. All methods use strength 1; the dashed line marks original GSM8K. The evaluation covers weekdays, ordinals, sports, and instruments, not the 39-trial numerical subset in the companion figure. Counts are screened on clean correctness (105 of 120 trials). Points use the same paired trials; intervals are 10,000-draw source-prompt bootstrap intervals conditional on each fitted lens, not independent fitting uncertainty. The downloaded reference differs in fitting budget and is contextual, while the new fits use 32 documents. This evidence does not isolate a unique causal property of solution text.')

if __name__=='__main__':
    comparison();ablations()
    lines=['# Corpus construction figures','', 'Use the comparison figure before the diagonal-filtering results. The ablation figure can follow it or move to the appendix. Both have standalone vector PDFs in `output/pdf/`, editable SVGs, and 450-dpi PNGs. Existing submission figures are unchanged.','', 'Rebuild: `.venv/bin/python scripts/53_corpus_figures.py`. Saved generations are regraded using the original evaluation definitions; source hashes and plotted data are in `corpus_data_audit.json`.','']
    for name,title,caption in ENTRIES:
        lines += ['## '+title,'',f'![{title}]({name}.png)','',caption,'',f'[PNG]({name}.png) | [SVG]({name}.svg)','']
    (OUT/'CORPUS.md').write_text('\n'.join(lines))
    (OUT/'corpus_data_audit.json').write_text(json.dumps(dict(sources=b.SOURCES,statistics=b.STATS,audit=b.AUDIT),indent=2))

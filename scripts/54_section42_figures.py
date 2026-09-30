"""Three figures corresponding one-to-one to Section 4.2 tables; raw-data rebuild."""
import importlib.util
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('corpus',ROOT/'scripts/53_corpus_figures.py')
c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
b=c.b

def comparison():
    best=c.corpus_rows('expanded')
    labels=['WikiText','GSM8K','AQuA-RAT','SVAMP','Competition algebra','OpenWebMath','Synthetic arithmetic in words','Synthetic ordered-scale text']
    arms=['wikitext_a','gsm8k','aqua_rat','svamp','math_algebra','openwebmath','arith_words','ordered_scale']
    cells=[best('swap_'+a) for a in arms]
    assert [x[0] for x in cells]==[0,11,14,6,8,4,0,1]
    fig,ax=plt.subplots(figsize=(7.2,3.8));fig.subplots_adjust(left=.33,right=.93,bottom=.22,top=.83)
    for i,(hits,dose) in enumerate(cells):
        color=b.C['off'] if i in (1,2) else b.C['full']
        ax.hlines(i,0,100*hits/39,color='#DAE0E3',lw=1.3)
        ax.plot(100*hits/39,i,'o',ms=5,color=color)
        ax.text(43,i,f'{hits}/39',ha='right',va='center',color=color,fontsize=8)
    ax.set(yticks=range(8),yticklabels=labels,ylim=(7.65,-.65),xlim=(-.8,44),xticks=[0,10,20,30,40],xlabel='Numerical-argument swap success (%)')
    ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    ax.set_title('Steering efficacy varies with the fitting corpus',loc='left',pad=15,fontsize=10)
    b.foot(fig,'Qwen3-8B | 32 documents per fit | 39 trials | strength selected separately on this evaluation subset')
    b.AUDIT['section42_corpora']=dict(arms=arms,hits_and_selected_strength=cells)
    c.save(fig,'section42_1_corpora','Fitting distribution changes numerical-argument steering',
        'Problem-and-solution corpora yield higher swap success than the tested WikiText fit, whereas synthetic arithmetic text does not. Each point is one 32-document fit on Qwen3-8B, evaluated on 39 clean-correct numerical-argument trials. The best strength from 0.5, 1, and 2 is selected separately on this same subset, so this is an exploratory comparison rather than held-out tuning. These trials include non-arithmetic operations and a known square-task answer-key limitation; this is not a pure arithmetic benchmark. No fitting uncertainty is estimated from these single fits.')

def replicates():
    rep=c.corpus_rows('tier0');aqua=c.corpus_rows('aqua')
    specs=[('GSM8K','gsm8k',rep),('WikiText','wikirep',rep),('AQuA-RAT','aqua_rat',aqua)]
    expected=[[[10,8,8,8,9],[44,42,41,39,43]],[[1,0,1,0,2],[39,41,41,41,42]],[[14,12,12,15,10],[41,42,38,42,41]]]
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.25),sharey=True)
    fig.subplots_adjust(left=.14,right=.97,bottom=.23,top=.77,wspace=.25)
    values=[]
    for i,(name,prefix,fn) in enumerate(specs):
        counts=[[fn(f'swap_{prefix}_r{k}',numeric)[0] for k in range(5)] for numeric in [True,False]]
        assert counts==expected[i],(name,counts)
        values.append(dict(corpus=name,numerical=counts[0],non_numerical=counts[1]))
        for j,denom in enumerate([39,105]):
            ax=axs[j];color=b.C['full'] if name=='WikiText' else b.C['off']
            for count in sorted(set(counts[j])):
                n=counts[j].count(count)
                ys=i+np.arange(n)*.09-(n-1)*.045
                ax.scatter([100*count/denom]*n,ys,color=color,s=16,zorder=3)
            ax.text(50,i-.30,f'{min(counts[j])}-{max(counts[j])}/{denom}',ha='right',va='center',fontsize=7.5,color=color)
    for j,ax in enumerate(axs):
        b.panel(ax,'ab'[j],['Numerical arguments (39 trials)','Non-numerical arguments (105 trials)'][j])
        ax.set(xlim=(-1,52),xticks=[0,10,20,30,40,50],yticks=range(3),yticklabels=['GSM8K','WikiText','AQuA-RAT'],ylim=(2.6,-.6),xlabel='Swap success (%)')
        ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    b.foot(fig,'Five fits per corpus | 32 documents per fit | each dot is a fit; vertical offsets separate ties')
    b.AUDIT['section42_replicates']=values
    c.save(fig,'section42_2_replicates','The numerical-argument gain repeats across fitting draws',
        'Every GSM8K and AQuA-RAT fit exceeds every WikiText fit on numerical arguments, while the non-numerical outcomes overlap. Dots represent five separately sampled 32-document fits per corpus; labels give observed count ranges, not confidence intervals. Identical scores are separated vertically without changing their x values. Both panels use the same percentage scale, but different trial denominators. Strength is selected separately for each fit and evaluation subset from 0.5, 1, and 2. This is the original flexible-generalization evaluation, not the new-argument panel used in Figure 3.')

if __name__=='__main__':
    comparison();replicates()
    original=c.save
    def save_ablation(fig,name,title,caption):
        caption='Using a questions-only fitting corpus reduces success markedly; solution-only text and the other modifications retain more of the GSM8K result. All six fits are compared at strength 1 on the same 105 clean-correct trials from a 120-trial new-argument evaluation covering weekdays, ordinals, sports, and instruments. This is not the non-numerical remainder of flexible-generalization. The dashed line marks original GSM8K. Error bars are 95% source-prompt cluster-bootstrap intervals (10,000 draws), conditional on the fitted directions, not uncertainty across independent fits. These controls do not identify a unique causal property of solution text.'
        original(fig,'section42_3_ablations','Solution-bearing text retains more of the steering benefit',caption)
    c.save=save_ablation;c.ablations(only_six=True)
    lines=['# Section 4.2: three corpus figures','',
        'These figures correspond one-to-one to Tables 1, 2, and 3. Use in the order below. All have vector PDF exports in `output/pdf/`, editable-text SVGs, and 450-dpi PNGs. Earlier two-figure exports are retained but superseded for this section.','',
        '**Text correction:** 30/105 (GSM8K) versus 9/105 (WikiText), at strength 1, is from the new-argument panel of weekdays, ordinals, sports, and instruments. It is not the remainder of the original flexible-generalization set. Also describe 0/39 as the tested 32-document WikiText fit, not a universal failure of the original published J-lens.','',
        'Rebuild: `.venv/bin/python scripts/54_section42_figures.py`. Counts are verified by regrading raw generations, not manually entered plot values.','']
    for name,title,caption in c.ENTRIES:
        lines += ['## '+title,'',f'![{title}]({name}.png)','',caption,'',f'[PNG]({name}.png) | [SVG]({name}.svg)','']
    (c.OUT/'CORPUS.md').write_text('\n'.join(lines))
    (c.OUT/'section42_data_audit.json').write_text(json.dumps(dict(sources=b.SOURCES,statistics=b.STATS,audit=b.AUDIT),indent=2))

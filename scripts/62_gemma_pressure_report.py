"""CPU report of frozen Gemma development search and validation outcomes."""
from __future__ import annotations
import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cross_report',ROOT/'scripts/60_cross_model_report.py')
rep=importlib.util.module_from_spec(spec);spec.loader.exec_module(rep)


def convert(rows):
    return [dict(r, key=r['unit'], name=r['query'], relation=r['query']) for r in rows]


def paired(rows,metric):
    other='projection_proxy' if any(r['arm']=='projection_proxy' for r in rows) else 'filtered'
    return rep.paired(convert(rows),'full',other,metric,concept=rows[0]['task']=='concept')


def mean(rows,metric):
    units=defaultdict(list)
    for r in rows:units[r['unit']].append(float(r[metric]))
    return statistics.fmean(statistics.fmean(v) for v in units.values()) if units else float('nan')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path,nargs='?',default=ROOT/'results/gemma_pressure')
    args=ap.parse_args()
    runs=sorted(args.root.glob('*/status.json'))
    if len(runs)!=1:raise SystemExit(f'Expected one run, found {len(runs)}')
    root=runs[0].parent;status=json.loads(runs[0].read_text())
    sensitivity=args.root/'question_sensitivity.json'
    clear_queries=set(json.loads(sensitivity.read_text())['queries']) if sensitivity.exists() else set()
    lines=['# Gemma adversarial replication diagnostics','',f"Status: **{status['stage']}**; recorded GPU function time {status['seconds']/60:.1f} minutes.",'',
           'This is an exploratory follow-up after observing the initial Gemma result. Source entities are disjoint between the development search and validation; validation concept units were not in the initial intervention sample. Prior aggregate outcomes and some latent outcomes were already observed, so this is not a pristine independent replication. No validation outcomes enter configuration selection.','',
           'All comparisons are paired on identical prompts, generation limits and answer aliases. Correct answers remain successes even if they contain the target name. Incorrect-name emission means an incorrect target answer plus name substring presence; answer-initial-name error is a narrower diagnostic.','',
           'The released WikiText lens is the full 844-prompt lens. Its unembedding projection is a **proxy control**, not an exact position-pair decomposition. Exact full/off comparisons use jointly fitted 32-prompt GSM8K or WikiText components.']
    check=root/'qwen_positive_control.json'
    if check.exists():
        checks=json.loads(check.read_text());n=sum(len(r['expected']) for r in checks);bad=sum(a!=b for r in checks for a,b in zip(r['expected'],r['observed']))
        lines += ['',f'Qwen replay: {n-bad}/{n} historical accuracy labels matched.']
    if (root/'question_audit.json').exists():
        cases=json.loads((root/'question_audit.json').read_text())
        lines += ['',f"Question audit: {sum(c['clean_ok'] for c in cases)}/{len(cases)} clean source answers replayed. Target one-hop knowledge is a separately reported sensitivity; unknown or unavailable one-hop checks do not silently change the main population."]
    if (root/'fit_stability.json').exists():
        stable=json.loads((root/'fit_stability.json').read_text());lines += ['','## Fit stability','','Cosines compare directions fitted on the first and second 16 fitting texts. These are estimates of stability, not a proof of convergence.','','| Corpus | Component | Mean cosine across layers/tokens |','|---|---|---:|']
        for corpus in ['gsm8k','wikitext_a']:
            for part in ['total','diag','off']:
                lines.append(f"| {corpus} | {part} | {statistics.fmean(r['split_half_cos_mean'] for r in stable if r['corpus']==corpus and r['part']==part):.3f} |")
    selection=root/'selection.json'
    if selection.exists():
        sel=json.loads(selection.read_text());lines+=['','## Frozen selection','',f"Original GSM8K full-lens development accuracy, equal weight to the two tasks: {100*sel['baseline_dev_accuracy']:.1f}%.",'', '| Setting | Dev full | Dev filtered | Change (pp) | Passed improvement criterion |','|---|---:|---:|---:|---|']
        for r in sel['selected']:
            lines.append(f"| {r['id']} | {100*r['full']:.1f}% | {100*r['filtered']:.1f}% | {100*r['gain']:+.1f} | {r['qualifies']} |")
    for split in ['dev','validation']:
        path=root/f'{split}_results.json'
        if not path.exists():continue
        rows=json.loads(path.read_text())
        settings=sorted({r['setting'] for r in rows})
        if split=='dev' and selection.exists():
            keep={r['id'] for r in sel['selected']}|{'gsm8k/original/1/native','gsm8k/original/1/unit','wikitext_a/original/1/native','released/original/1/native'}
            settings=[s for s in settings if s in keep]
        lines += ['',f'## {split.capitalize()} results','', '| Setting | Task | Subset | Units | Metric | Full | Filtered / proxy | Change (pp; 95% CI) |','|---|---|---|---:|---|---:|---:|---|']
        for setting in settings:
            for task in ['latent','concept']:
                for subset in (['all','target-known','strict-question','clear-relation'] if split=='validation' else ['all']):
                    rs=[r for r in rows if r['setting']==setting and r['task']==task and
                        (subset=='all' or (subset=='target-known' and r['target_known'] is True) or
                         (subset=='strict-question' and r['strict_question']) or
                         (subset=='clear-relation' and r['query'] in clear_queries))]
                    if not rs:continue
                    for metric in (['success','incorrect_name_emission','answer_initial_name_error'] if subset=='all' else ['success']):
                        d=paired(rs,metric)
                        lines.append(f"| {setting} | {task} | {subset} | {d['n']} | {metric} | {d['baseline']:.1f}% | {d['filtered']:.1f}% | {d['delta']:+.1f} [{d['ci'][0]:+.1f}, {d['ci'][1]:+.1f}] |")
    if selection.exists():
        lines += ['','## Full development search','','All configurations are retained, including unsuccessful ones. These development comparisons are selected and have no confirmatory significance claim.','','| Setting | Full | Filtered | Change (pp) |','|---|---:|---:|---:|']
        for r in sorted(sel['all_settings'],key=lambda r:-r['gain']):
            lines.append(f"| {r['id']} | {100*r['full']:.1f}% | {100*r['filtered']:.1f}% | {100*r['gain']:+.1f} |")
    out=args.root/'report.md';out.write_text('\n'.join(lines)+'\n');print(out)


if __name__=='__main__':main()

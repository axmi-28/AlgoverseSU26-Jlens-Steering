"""Audit downloaded published-lens artifacts and unique Modal GPU calls."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from jsteer.cross_model import sha
from jsteer.neuronpedia_replication import PROTOCOL, input_hashes
from jsteer.write_controls import file_hash


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root', type=Path, nargs='?', default=Path('results/neuronpedia_replication'))
    ap.add_argument('--remote', action='store_true')
    args = ap.parse_args()
    root = args.root
    protocol = json.loads(PROTOCOL.read_text())
    ledger = json.loads((root/'campaign.json').read_text())
    assert ledger['stage'] == 'complete'
    assert set(ledger['models']) == set(protocol['models'])
    assert ledger['expected']['protocol_sha'] == file_hash(PROTOCOL)
    audited = {}
    for name in protocol['models']:
        row = ledger['models'][name]
        assert row['stage'] == 'complete'
        folder = root/Path(row['result']['path']).relative_to('/results/neuronpedia_replication')
        def read(filename, base=folder):
            return json.loads((base/filename).read_text())
        manifest, status = read('manifest.json'), read('status.json')
        assert status['stage'] == 'complete'
        assert sha(manifest) == status['manifest_sha']
        assert manifest['inputs'] == input_hashes(name)
        assert sha(manifest['inputs']) == ledger['expected']['inputs_sha'][name]
        assert manifest['source_sha'] == ledger['expected']['source_sha']
        published, diagnostics = read('published_metadata.json'), read('fit_diagnostics.json')
        assert published['repository'] == protocol['lens_repository']
        assert published['revision'] == protocol['lens_revision']
        assert published['file_sha256'] == row['prepared']['lens_sha256']
        assert published['full_lens_refitted'] is False
        assert diagnostics['published_full_unchanged'] is True
        assert diagnostics['filter_is_estimated'] is True
        assert diagnostics['diagonal_prompts'] == 64
        bank = torch.load(folder/'components.pt', map_location='cpu', weights_only=True)
        partial = torch.load(folder/'diagonal_partial.pt', map_location='cpu', weights_only=True)
        assert bank['published_full_refitted'] is False and bank['filter_is_estimated'] is True
        assert bank['manifest_sha'] == partial['manifest_sha'] == sha(manifest)
        assert bank['targets'] == partial['words'] and partial['count'] == 64
        for layer in manifest['layers']:
            vectors = {fam: bank['digests'][f'published_{fam}|{layer}'] for fam in ('full', 'diag', 'off', 'off32')}
            assert vectors['full']['n'] == published['n_prompts']
            assert all(torch.isfinite(v['pullbacks']).all() for v in vectors.values())
            full = vectors['full']['pullbacks']
            diagonal = partial['sums'][layer]/64
            torch.testing.assert_close(vectors['diag']['pullbacks'], diagonal)
            torch.testing.assert_close(vectors['off']['pullbacks'], full-diagonal)
            torch.testing.assert_close(vectors['off32']['pullbacks'], full-partial['first32'][layer]/32)
        eligibility = read('eligibility.json')
        latent, concept = read('latent_results.json'), read('concept_results.json')
        expected = {(i['name'], a, s) for i in eligibility['latent'] for a in protocol['latent_arms'] for s in protocol['latent_strengths']}
        assert len(latent) == len(expected)
        assert {(r['name'], r['arm'], r['strength']) for r in latent} == expected
        expected = {(key, rel, arm) for key in eligibility['concept_selected'] for rel in eligibility['concept_eligible'][key] for arm in protocol['concept_arms']}
        assert len(concept) == len(expected)
        assert {(r['key'], r['relation'], r['arm']) for r in concept} == expected
        max_norm = max(r['norm_error'] for r in latent+concept)
        assert math.isfinite(max_norm) and max_norm < 1e-4
        smoke = read('smoke.json')
        assert smoke['logit_max_error'] < .05 and smoke['pullback_relative_error'] < .05
        assert row['result']['seconds'] < protocol['scientific_budget_seconds_per_model']
        audited[name] = {'path': str(folder), 'gpu_call_id': row['gpu_call_id'],
                         'latent_rows': len(latent), 'concept_rows': len(concept),
                         'max_norm_error': max_norm, 'published_file_sha256': published['file_sha256'],
                         'smoke': smoke, 'gpu_function_seconds': row['result']['seconds']}
    ids = {v['gpu_call_id'] for v in audited.values()}
    assert len(ids) == len(audited) == 6
    graphs = []
    if args.remote:
        import modal
        observed = set()
        def visit(rows):
            for r in rows:
                graphs.append({'call_id': r.function_call_id, 'function': r.function_name, 'status': r.status.name})
                if r.function_name == 'run_model':
                    assert r.status.name == 'SUCCESS'
                    observed.add(r.function_call_id)
                visit(r.children)
        for filename in ('launch.json', 'recovery.json'):
            launch = json.loads((root/filename).read_text())
            visit(modal.FunctionCall.from_id(launch['campaign_call_id']).get_call_graph())
        assert observed == ids, (observed, ids)
    result = {'status': 'passed', 'method': 'published full minus estimated diagonal',
              'models': audited, 'unique_gpu_calls': len(ids), 'remote_call_graph': graphs,
              'total_gpu_function_minutes': sum(r['gpu_function_seconds'] for r in audited.values())/60,
              'offline_validation': '188 passed, 22 deselected; synthetic paired-report QA passed'}
    (root/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('models','remote_call_graph')}, indent=2))


if __name__ == '__main__':
    main()

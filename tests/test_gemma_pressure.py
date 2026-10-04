import pytest
from jsteer.gemma_pressure import choose_settings, task_mean


def test_selection_uses_dev_only_and_does_not_reward_breaking_full():
    settings = []
    rows = []
    for corpus in ['gsm8k', 'wikitext_a']:
        for grid, full, off in [('original', .5, .4), ('good', .6, .8), ('bad', 0., .4)]:
            sid = f'{corpus}/{grid}/1/native'
            settings.append({'id': sid, 'corpus': corpus})
            for arm, score in [('full', full), ('filtered', off)]:
                for task in ['concept', 'latent']:
                    rows.append({'setting': sid, 'arm': arm, 'success': score,
                                 'task': task, 'unit': 'one', 'split': 'dev'})
    result = choose_settings(rows, settings)
    assert all('/good/' in s['id'] for s in result['selected'])
    with pytest.raises(ValueError):
        choose_settings([dict(r, split='validation') for r in rows], settings)


def test_task_mean_does_not_overweight_more_queries():
    rows = [{'task': 'concept', 'unit': 'one', 'success': 1}] * 10
    rows += [{'task': 'latent', 'unit': 'two', 'success': 0}]
    assert task_mean(rows, 'success') == .5

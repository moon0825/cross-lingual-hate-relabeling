#!/usr/bin/env python3
"""Portable verification of released post-hoc statistics; no private inputs."""
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
TARGETS = ('gender', 'age', 'race_or_region_of_origin', 'religion',
           'politics', 'occupation', 'disability')
MODELS = ('kcelectra', 'xlmr')
SEEDS = tuple(range(20260901, 20260924))


def rows(name):
    with (DATA / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def close(a, b, tolerance=1e-12):
    if not np.isfinite(float(a)) or not np.isfinite(float(b)) or abs(float(a)-float(b)) > tolerance:
        raise AssertionError('Published numeric value differs')


def human():
    old = json.loads((DATA / 'annotation_audit_statistics.json').read_text())
    new = json.loads((DATA / 'human_reinforcement_summary.json').read_text())
    humans = [r for r in old['raters'] if r['family'] == 'Human']
    ids = {r['id'] for r in humans}
    pairs = [r for r in old['pairwise'] if r['rater_a'] in ids and r['rater_b'] in ids]
    assert len(humans) == 4 and len(pairs) == 6 and new['n_items'] == 300
    counts = np.array([sum(r['label_counts'].get(c, 0) for r in humans)
                       for c in ('non-hate', 'hate', 'unclear')], dtype=float)
    n = float(counts.sum())
    observed = 1 - sum(r['agreement_count'] for r in pairs) / (6 * 300)
    expected = (n*n - counts @ counts) / (n*(n-1))
    alpha = 1-observed/expected if expected else None
    close(alpha, new['human_panel_alpha']['alpha'])
    close(observed, new['human_panel_alpha']['observed_disagreement'])
    close(expected, new['human_panel_alpha']['expected_disagreement'])
    panel = {label: sum(r['agreement_counts'][label] for r in humans)/12
             for label in ('source', 'exaone', 'gemma4')}
    for label, value in panel.items():
        close(value, new['panel_agreement_percent'][label])
    assert len(new['conditional_concordance']) == 36
    by_id = {r['id']: r for r in humans}
    conditional = {(r['human_id'], r['human_response'], r['label_source']): r
                   for r in new['conditional_concordance']}
    for identity, r in conditional.items():
        human_id, response, label = identity
        assert r['n_items'] == by_id[human_id]['label_counts'].get(response, 0)
        assert r['non_hate_labels'] + r['hate_labels'] == r['n_items']
        agreed = r['hate_labels'] if response == 'hate' else (
                 r['non_hate_labels'] if response == 'non-hate' else 0)
        assert r['agreement_count'] == agreed
        close(r['agreement_percent'], 100*agreed/r['n_items'])
        source = conditional[human_id, response, 'source']['agreement_count']
        assert r['newly_agreeing_vs_source'] - r['lost_agreement_vs_source'] == agreed-source
        close(r['gain_pp'], 100*(agreed-source)/r['n_items'])
    for r in humans:
        for label in ('source', 'exaone', 'gemma4'):
            cells = [conditional[r['id'], response, label]
                     for response in ('non-hate', 'hate', 'unclear')]
            assert sum(c['agreement_count'] for c in cells) == r['agreement_counts'][label]
            if label != 'source':
                assert sum(c['newly_agreeing_vs_source'] for c in cells) == r['contrasts'][label]['agreement_gains']
                assert sum(c['lost_agreement_vs_source'] for c in cells) == r['contrasts'][label]['agreement_losses']
    boot = np.genfromtxt(DATA / 'human_bootstrap_estimates.csv', delimiter=',', names=True)
    assert len(boot) == 10000 and np.isfinite(boot['nominal_alpha']).all()
    for actual, reference in zip(np.quantile(boot['nominal_alpha'], [.025, .975]),
                                 new['human_panel_alpha']['ci95_pointwise']):
        close(actual, reference, 1e-9)
    for r in new['panel_contrasts']:
        label = r['label_source']
        close(r['gain_pp'], panel[label]-panel['source'])
        close(r['gain_pp'], 100*(r['newly_agreeing_judgments']-r['lost_agreement_judgments'])/1200)
        assert r['newly_agreeing_judgments'] == sum(h['contrasts'][label]['agreement_gains'] for h in humans)
        assert r['lost_agreement_judgments'] == sum(h['contrasts'][label]['agreement_losses'] for h in humans)
        col = 'exaone_gain_pp' if label == 'exaone' else 'gemma_gain_pp'
        for actual, reference in zip(np.quantile(boot[col], [.025, .975]), r['ci95_pointwise_pp']):
            close(actual, reference, 1e-9)
    return dict(status='PASS', nominal_alpha=alpha, panel_agreement_percent=panel,
                ci_scope='Quantiles of saved 10,000 author bootstrap estimates; private row resampling is not rerun',
                point_tolerance=1e-12, saved_bootstrap_quantile_tolerance=1e-9,
                conditional_aggregate_cells=36)


def targets():
    source = rows('target_recall_reproduction.csv')
    scores = {(r['model'], int(r['repeat_seed']), r['arm'], r['target']): float(r['recall'])
              for r in source}
    assert len(source) == len(scores) == 1288
    expected = set(itertools.product(MODELS, SEEDS, ('S','L','W','P'), TARGETS))
    assert set(scores) == expected
    old = {(r['model'], 20260900+int(r['repeat']), r['arm'], r['target']): float(r['recall'])
           for r in rows('control_target_recall.csv')}
    assert set(old) == expected
    for identity, score in scores.items():
        close(score, old[identity])
    differences = rows('target_lw_repeat_differences.csv')
    assert len(differences) == 322
    for r in differences:
        model, seed, target = r['model'], int(r['repeat_seed']), r['target']
        close(r['L_recall'], scores[model, seed, 'L', target])
        close(r['W_recall'], scores[model, seed, 'W', target])
        close(r['L_minus_W'], scores[model, seed, 'L', target]-scores[model, seed, 'W', target])
    refs = rows('target_lw_contrasts.csv')
    assert len(refs) == 14
    computed = []
    for r in refs:
        values = np.array([scores[r['model'], s, 'L', r['target']]-scores[r['model'], s, 'W', r['target']] for s in SEEDS])
        mean, sd = float(values.mean()), float(values.std(ddof=1))
        if np.ptp(values) == 0:
            sd = 0.
        se = sd/np.sqrt(23)
        half = float(stats.t.ppf(.975, 22))*se
        p = float(2*stats.t.sf(abs(mean/se),22)) if se else (1. if mean == 0 else None)
        for col, value in [('mean',mean),('sd',sd),('ci95_lower',mean-half),('ci95_upper',mean+half)]:
            close(r[col], value)
        assert (r['t_test_defined'] == 'True') == bool(se)
        if p is None:
            assert r['p_raw'] == ''
        else:
            close(r['p_raw'], p)
        computed.append(dict(model=r['model'],target=r['target'],mean=mean,p_raw=p,
                             ci95_lower=mean-half,ci95_upper=mean+half))
    p = np.array([1 if r['p_raw'] is None else r['p_raw'] for r in computed])
    order = np.argsort(p, kind='stable')
    adj = np.empty(14)
    adj[order] = np.minimum(1,np.maximum.accumulate(p[order]*np.arange(14,0,-1)))
    for r, c, a, slot in zip(refs,computed,adj,p):
        close(r['p_holm_14'],a)
        close(r['holm_slot_p'],slot)
        c['p_holm_14'] = float(a)
        pos = c['p_raw'] is not None and a < .05 and c['ci95_lower'] > 0
        neg = c['p_raw'] is not None and a < .05 and c['ci95_upper'] < 0
        assert (r['positive_after_adjustment'] == 'True') == pos
        assert (r['negative_after_adjustment'] == 'True') == neg
    return dict(status='PASS', repeats=23, family_size=14,
                confidence_intervals='Pointwise paired t22', adjustment='Holm14',
                posthoc=True, fixed_gold_and_input_pools=True, contrasts=computed)


def turnover():
    repeated = rows('target_lw_turnover_by_repeat.csv')
    summaries = rows('target_lw_turnover_summary.csv')
    hist = rows('target_lw_item_turnover_histograms.csv')
    fields = ('both_correct','recovered_by_L','new_errors_under_L','both_wrong')
    assert len(repeated) == 414 and len(summaries) == 18
    for r in repeated:
        n = int(r['n_distinct_items']); bc, rec, new, wrong = [int(r[f]) for f in fields]
        assert bc+rec+new+wrong == n
        close(r['W_correct_rate'], (bc+new)/n)
        close(r['L_correct_rate'], (bc+rec)/n)
        close(r['L_minus_W_correct_rate'], (rec-new)/n)
        close(r['recovery_minus_new_error_over_n'], (rec-new)/n)
    for ref in summaries:
        key = ref['model'],ref['stratum']
        cells = [r for r in repeated if (r['model'],r['stratum']) == key]
        bins = [r for r in hist if (r['model'],r['stratum']) == key]
        n = int(ref['n_distinct_items'])
        assert len(cells) == 23 and sum(int(r['frequency']) for r in bins) == n
        assert int(ref['n']) == n and int(ref['paired_repeats']) == 23
        assert int(ref['item_repeat_observations_not_independent_n']) == n*23
        for f in fields:
            total = sum(int(r[f]) for r in cells)
            assert int(ref[f+'_item_repeat_observations']) == total
            assert sum(int(r[f])*int(r['frequency']) for r in bins) == total
            close(ref[f+'_mean_per_repeat'],total/23)
        aliases = {'both_correct_mean':'both_correct','recovered_mean':'recovered_by_L',
                   'new_error_mean':'new_errors_under_L','both_wrong_mean':'both_wrong'}
        for col, f in aliases.items():
            close(ref[col],sum(int(r[f]) for r in cells)/23)
        checks = {
            'unique_items_ever_recovered':lambda r: int(r['recovered_by_L']) > 0,
            'unique_items_ever_new_error':lambda r: int(r['new_errors_under_L']) > 0,
            'unique_items_ever_both_recovered_and_new_error':lambda r: int(r['recovered_by_L']) > 0 and int(r['new_errors_under_L']) > 0,
            'unique_items_ever_discordant_union':lambda r: int(r['recovered_by_L']) > 0 or int(r['new_errors_under_L']) > 0}
        for col, predicate in checks.items():
            assert int(ref[col]) == sum(int(r['frequency']) for r in bins if predicate(r))
        close(ref['L_minus_W_correct_rate_mean'],np.mean([float(r['L_minus_W_correct_rate']) for r in cells]))
    return dict(status='PASS', repeat_tables=414, stratum_summaries=18,
                aggregate_histogram_bins=len(hist), item_identifiers_distributed=False,
                scope='Descriptive paired error turnover; overlapping target-positive masks, no pooled item-repeat test')


def run(output):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    report = dict(status='PASS',human=human(),target_family=targets(),turnover=turnover())
    (output/'reinforcement_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs')
    print(json.dumps(run(parser.parse_args().output),indent=2))

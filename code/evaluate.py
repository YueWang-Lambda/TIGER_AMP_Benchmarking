"""Reconstruct V17 benchmark results from archived predictions, without weights.

Run: python code/evaluate.py --check-reference
All file paths are resolved relative to this repository, independently of cwd.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
KEYS = ['family', 'anchor_sequence', 'query_sequence']
METRICS = ['PCC', 'KCC', 'log10MAE', 'RSE']
SEEDS = set(range(120, 130))
CONDITIONS = {'with_negatives', 'without_negatives'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return pd.read_csv(path, float_precision='round_trip')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aligned_prediction(frame, pairs):
    """Validate identities and labels, then align by identity (never row number)."""
    require(set(KEYS+['y_true', 'y_pred']).issubset(frame), 'Missing pair columns')
    require(not frame.duplicated(KEYS).any(), 'Duplicate pair identities')
    require(set(map(tuple, frame[KEYS].to_numpy())) == set(map(tuple, pairs[KEYS].to_numpy())), 'Incomplete or unexpected endpoint')
    aligned = pairs[KEYS].merge(frame, on=KEYS, validate='one_to_one', sort=False)
    require(np.allclose(aligned.y_true, pairs.y_true, rtol=0, atol=1e-10), 'Pair labels differ from canonical labels')
    pred = aligned.y_pred.to_numpy(float)
    require(np.isfinite(pred).all(), 'Nonfinite predictions')
    return pred


def family_metrics(truth, pred):
    """Primary policy: undefined correlations propagate; never nanmean."""
    truth, pred = np.asarray(truth, float), np.asarray(pred, float)
    require(truth.shape == pred.shape and np.isfinite(truth).all() and np.isfinite(pred).all(), 'Invalid metric inputs')
    sst = np.square(truth-truth.mean()).sum()
    require(sst > 0, 'Constant family target: RSE undefined')
    constant = np.ptp(pred) == 0
    error = pred-truth
    return {
        'PCC': np.nan if constant else float(stats.pearsonr(truth, pred).statistic),
        'KCC': np.nan if constant else float(stats.kendalltau(truth, pred, variant='b').statistic),
        'log10MAE': float(np.abs(error).mean()*np.log10(2)),
        'RSE': float(np.square(error).sum()/sst),
    }


def summarize(frame):
    rows = []
    for (method, condition), group in frame.groupby(['method', 'condition'], sort=True):
        require(len(group) == 10 and set(group.seed) == SEEDS, f'Incomplete seeds: {method}/{condition}')
        for metric in METRICS:
            v = group[metric].to_numpy(float)
            mean, sd = v.mean(), v.std(ddof=1)  # Propagate any undefined seed.
            half = stats.t.ppf(.975, 9)*sd/np.sqrt(10)
            rows.append(dict(method=method, condition=condition, metric=metric, n=10,
                             n_defined=int(np.isfinite(v).sum()), mean=mean, sd=sd,
                             ci_low=mean-half, ci_high=mean+half))
    return pd.DataFrame(rows)


def holm(p):
    p = np.asarray(p, float)
    result = np.full(len(p), np.nan)
    ix = np.flatnonzero(np.isfinite(p))
    order = ix[np.argsort(p[ix], kind='stable')]
    result[order] = np.minimum(1, np.maximum.accumulate(p[order]*(len(p)-np.arange(len(order)))))
    return result


def paired_tests(frame):
    rows = []
    for method, group in frame.groupby('method'):
        a = group[group.condition == 'with_negatives'].set_index('seed').sort_index()
        b = group[group.condition == 'without_negatives'].set_index('seed').sort_index()
        require(set(a.index) == set(b.index) == SEEDS, f'Unpaired seeds: {method}')
        for metric in METRICS:
            delta = (a[metric]-b[metric]).to_numpy()
            valid = np.isfinite(delta).all()
            mean = delta.mean()
            half = stats.t.ppf(.975, 9)*stats.sem(delta)
            p = (1.0 if np.all(delta == 0) else stats.wilcoxon(delta, method='auto').pvalue) if valid else np.nan
            rows.append(dict(method=method, contrast='with_minus_without', metric=metric,
                             n=10, delta_a_minus_b=mean, ci_low=mean-half, ci_high=mean+half, p_raw=p))
    out = pd.DataFrame(rows)
    out['p_holm'] = np.nan
    for _, group in out.groupby('metric'):
        out.loc[group.index, 'p_holm'] = holm(group.p_raw)
    return out


def pareto(frame, metric, time_col='time_s_with_struct'):
    """Return nondominated points; ties retained if neither strictly dominates."""
    good = frame[np.isfinite(frame[metric]) & np.isfinite(frame[time_col])].copy()
    y = good[metric].to_numpy() * (1 if metric == 'PCC' else -1)
    x = good[time_col].to_numpy()
    keep = []
    for i in range(len(good)):
        dominated = (x <= x[i]) & (y >= y[i]) & ((x < x[i]) | (y > y[i]))
        keep.append(not dominated.any())
    return good.loc[keep].sort_values(time_col)


def compare(actual, reference, keys, columns, atol=1e-9):
    require(not actual.duplicated(keys).any() and not reference.duplicated(keys).any(), f'Duplicate comparison keys: {keys}')
    a, b = actual.set_index(keys).sort_index(), reference.set_index(keys).sort_index()
    require(a.index.equals(b.index), f'Reference identities differ: {keys}')
    for col in columns:
        require(np.allclose(a[col], b[col], atol=atol, rtol=0, equal_nan=True), f'Reference mismatch: {col}')
    return len(a)*len(columns)


def peptide_to_pair(path, unit, pairs, peptides, masses):
    p = read(path)
    seqcol = 'sequence' if 'sequence' in p else p.columns[0]
    cols = [c for c in p if 'log10mic' in c.lower()]
    require(len(cols) == 1, f'Expected one potency column: {path}')
    require(p[seqcol].is_unique and set(p[seqcol]) == set(peptides.sequence), f'Incomplete peptide identities: {path}')
    values = p.set_index(seqcol)[cols[0]].astype(float)
    if unit == 'log10_uM':
        mw = np.array([sum(masses['residue_mass_Da'][aa] for aa in seq)+masses['water_mass_Da'] for seq in values.index])
        values = values+np.log10(mw/1000)
    else:
        require(unit == 'log10_ug_mL', f'Unknown unit {unit}')
    return ((pairs.anchor_sequence.map(values)-pairs.query_sequence.map(values))/np.log10(2)).to_numpy()


def run(output, check_reference=True):
    output = Path(output).resolve()
    require(not output.is_relative_to(ROOT/'results/reference'), 'Reference results are read-only')
    output.mkdir(parents=True, exist_ok=True)
    index = read(ROOT/'provenance/prediction_index.csv')
    registry = read(ROOT/'provenance/method_registry.csv')
    require(len(index) == 460 and not index.duplicated(['method','condition','seed']).any(), 'Expected 460 unique predictors')
    require(set(index.method) == set(registry.method) and len(registry) == 23, 'Unexpected method set')
    expected = {(m,c,s) for m in registry.method for c in CONDITIONS for s in SEEDS}
    require(set(map(tuple,index[['method','condition','seed']].to_numpy())) == expected, 'Missing condition/seed')
    pairs = read(ROOT/'data/canonical_pairs.csv')
    peptides = read(ROOT/'data/external_peptides.csv')
    require(len(pairs) == 2211 and pairs.family.nunique() == 11 and not pairs.duplicated(KEYS).any(), 'Invalid canonical endpoint')
    require(len(peptides) == 208 and peptides.sequence.is_unique, 'Invalid external peptides')
    require(set(pairs.anchor_sequence)|set(pairs.query_sequence) == set(peptides.sequence), 'Endpoint peptide coverage differs')
    mic = peptides.set_index('sequence').MIC_E_coli_ug_mL
    truth = np.log2(pairs.anchor_sequence.map(mic)/pairs.query_sequence.map(mic))
    require(np.allclose(truth, pairs.y_true, atol=1e-12, rtol=0), 'Concentration-to-delta labels differ')
    groups = [(family,g.index.to_numpy()) for family,g in pairs.groupby('family')]
    y = pairs.y_true.to_numpy()
    masses = json.loads((ROOT/'data/evo_residue_masses.json').read_text())
    family_rows, seed_rows, cache = [], [], {}
    peptide_checks = 0
    def score(method, condition, seed, pred):
        identity = dict(method=method, condition=condition, seed=seed)
        fam = []
        for family, ix in groups:
            vals = family_metrics(y[ix], pred[ix]); fam.append(vals)
            family_rows.append(dict(**identity, family=family, n_pairs=len(ix), **vals))
        seed_rows.append(dict(**identity, **{m:np.mean([v[m] for v in fam]) for m in METRICS}))
    for row in index.itertuples():
        path = ROOT/row.prediction_path
        require(sha256(path) == row.sha256, f'Prediction checksum mismatch: {path.name}')
        pred = aligned_prediction(read(path), pairs)
        if isinstance(row.peptide_prediction_path, str):
            rawpath = ROOT/row.peptide_prediction_path
            require(sha256(rawpath) == row.original_sha256, f'Peptide checksum mismatch: {rawpath}')
            independent = peptide_to_pair(rawpath, row.peptide_unit, pairs, peptides, masses)
            require(np.allclose(independent, pred, atol=1e-10, rtol=0), f'Peptide-to-pair mismatch: {row.method}/{row.seed}')
            peptide_checks += 1
        score(row.method, row.condition, row.seed, pred)
        if row.method in ['TIGER-pair','TIGER-single'] and row.condition == 'with_negatives':
            cache[(row.method,row.seed)] = pred
    family = pd.DataFrame(family_rows); seed = pd.DataFrame(seed_rows)
    summary = summarize(seed); tests = paired_tests(seed)
    counts = 0
    if check_reference:
        reference = ROOT/'results/reference'
        for frame, name, keys, cols in [
            (family, 'benchmark_family_per_seed', ['method','condition','seed','family'], METRICS),
            (seed, 'benchmark_per_seed', ['method','condition','seed'], METRICS),
            (summary, 'benchmark_summary', ['method','condition','metric'], ['n','n_defined','mean','sd','ci_low','ci_high']),
            (tests, 'benchmark_paired_tests', ['method','metric'], ['delta_a_minus_b','ci_low','ci_high','p_raw','p_holm']),
        ]:
            counts += compare(frame, read(reference/(name+'.csv')), keys, cols)
    # Independent ensemble reconstruction, using matched calibrated predictions.
    family_rows.clear(); seed_rows.clear()
    for s in sorted(SEEDS):
        pred=(cache[('TIGER-pair',s)]+cache[('TIGER-single',s)])/2
        score('TIGER-ensemble','with_negatives',s,pred)
    ensemble_seed=pd.DataFrame(seed_rows); ensemble_summary=summarize(ensemble_seed)
    if check_reference:
        counts += compare(ensemble_summary,read(ROOT/'results/reference/ensemble/ensemble_summary.csv'),['method','condition','metric'],['mean','sd','ci_low','ci_high'])
    selected = summary[summary.metric=='PCC'].sort_values(['method','mean'],ascending=[True,False],na_position='last').drop_duplicates('method')[['method','condition']]
    table = selected.merge(summary.pivot(index=['method','condition'],columns='metric',values='mean').reset_index(),on=['method','condition'],validate='one_to_one')
    timing = read(ROOT/'results/reference/cost_timing_v17.csv')
    require(set(timing.method)==set(registry.method) and len(timing)==23, 'Timing coverage differs')
    for r in timing.itertuples():
        t=json.loads((ROOT/f'results/timing/{r.method}.json').read_text())
        times=np.asarray(t['repeats_s'],float)
        require(len(times)==20 and np.isfinite(times).all() and (times>0).all(),f'Invalid timing samples: {r.method}')
        require(np.isclose(np.median(times),r.time_s_no_struct,atol=1e-12,rtol=0),f'Timing median mismatch: {r.method}')
        require(np.isclose(r.time_s_no_struct+r.structure_increment_s,r.time_s_with_struct,atol=1e-10,rtol=0),f'Structure increment mismatch: {r.method}')
    table=table.merge(timing,on='method',validate='one_to_one').sort_values('PCC',ascending=False)
    if check_reference:
        ref=read(ROOT/'results/reference/cost_best_PCC.csv')
        require(table.set_index('method').condition.sort_index().equals(ref.set_index('method').condition.sort_index()),'PCC-selected conditions differ')
    both=summary.pivot(index=['method','condition'],columns='metric',values='mean').reset_index().merge(timing,on='method',validate='many_to_one')
    for name, frame in [('benchmark_family_per_seed',family),('benchmark_per_seed',seed),('benchmark_summary',summary),('benchmark_paired_tests',tests),('ensemble_per_seed',ensemble_seed),('ensemble_summary',ensemble_summary),('pcc_selected',table),('all_conditions',both)]:
        frame.to_csv(output/(name+'.csv'),index=False)
    for metric in ['PCC','RSE']:
        for label,frame in [('selected',table),('both_conditions',both)]:
            pareto(frame,metric).to_csv(output/f'pareto_{label}_{metric}.csv',index=False)
    report=dict(status='PASS',methods=23,conditions=2,seeds=sorted(SEEDS),peptides=208,families=11,pairs=2211,
                prediction_sets=460,peptide_to_pair_checks=peptide_checks,family_metric_rows=len(family),
                numeric_reference_checks=counts,ensemble_seeds=10,timing_methods_verified=23,
                undefined_correlations=family.loc[family.PCC.isna()|family.KCC.isna(),['method','condition','seed','family']].to_dict('records'))
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'reproduced')
    parser.add_argument('--check-reference',action='store_true',help='Explicitly request comparison (enabled by default)')
    parser.add_argument('--no-reference-check',action='store_true',help='For new predictions only; does not reproduce the frozen benchmark')
    args=parser.parse_args()
    run(args.output,not args.no_reference_check)

"""Recompute 10 matched-seed ensembles, preserve the original 23 predictors."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[4]
V15 = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1]
KEYS = ['family','anchor_sequence','query_sequence']
METRICS = ['PCC','KCC','log10MAE','RSE']
NAME = 'TIGER-ensemble'

def read(path):
    return pd.read_csv(path, float_precision='round_trip')

def scores(y, p):
    e = p-y
    return dict(PCC=stats.pearsonr(y,p).statistic, KCC=stats.kendalltau(y,p,variant='b').statistic,
                log10MAE=np.abs(e).mean()*np.log10(2), RSE=np.sum(e**2)/np.sum((y-y.mean())**2))

def frontier(data, time_col, metric):
    ascending = metric == 'RSE'
    ordered = data[np.isfinite(data['mean'])].sort_values([time_col,'mean'], ascending=[True,ascending])
    keep=[]; best=np.inf if ascending else -np.inf
    for i,r in ordered.iterrows():
        if (r['mean'] < best) if ascending else (r['mean'] > best):
            keep.append(i); best=r['mean']
    return data.loc[keep].copy()

def prepare():
    tab=OUT/'tables';tab.mkdir(exist_ok=True)
    predout=OUT/'predictions';predout.mkdir(exist_ok=True)
    canon=read(V15/'source_snapshot/canonical_pairs.csv').set_index(KEYS)
    assert len(canon)==2211 and canon.index.is_unique
    families=sorted(set(canon.index.get_level_values('family')))
    assert len(families)==11
    archive=read(V15/'benchmark_per_seed.csv')
    rows=[]; famrows=[]; decomp=[]; source_hashes={}
    for seed in range(120,130):
        preds={}
        for arm,method in [('P11','TIGER-pair'),('S11','TIGER-single')]:
            path=V15/f'source_snapshot/predictions/tiger/factor_{arm}_seed{seed}.csv'
            source_hashes[str(path.relative_to(V15))]=hashlib.sha256(path.read_bytes()).hexdigest()
            x=read(path).set_index(KEYS)
            assert x.index.is_unique and set(x.index)==set(canon.index)
            x=x.reindex(canon.index)
            assert np.allclose(x.y_true,canon.y_true,atol=1e-12,rtol=0)
            preds[method]=x.y_pred.to_numpy()
        preds[NAME]=(preds['TIGER-pair']+preds['TIGER-single'])/2
        for method,p in preds.items():
            per=[]
            for f in families:
                ix=canon.index.get_level_values('family')==f
                y=canon.y_true.to_numpy()[ix];v=p[ix]
                m=scores(y,v);per.append(m)
                famrows.append(dict(method=method,condition='with_negatives',seed=seed,family=f,n_pairs=int(ix.sum()),**m))
                r=m['PCC']; c=np.std(v)/np.std(y); b=(v.mean()-y.mean())/np.std(y)
                terms=dict(correlation_term=1-r*r,scale_term=(c-r)**2,bias_term=b*b,sd_ratio=c)
                assert abs(terms['correlation_term']+terms['scale_term']+terms['bias_term']-m['RSE'])<1e-12
                decomp.append(dict(method=method,seed=seed,family=f,**terms))
            assert np.isfinite(pd.DataFrame(per).to_numpy()).all()
            macro=pd.DataFrame(per).mean().to_dict()
            rows.append(dict(method=method,condition='with_negatives',seed=seed,**macro))
            if method!=NAME:
                old=archive[(archive.method==method)&(archive.condition=='with_negatives')&(archive.seed==seed)].iloc[0]
                assert all(np.isclose(old[m],macro[m],atol=1e-10,rtol=0) for m in METRICS),(method,seed)
        x=canon.copy()
        for method,values in preds.items():x[method]=values
        x.to_csv(predout/f'TIGER_ensemble_seed{seed}.csv')
    three=pd.DataFrame(rows);fam=pd.DataFrame(famrows)
    benchmark=pd.concat([archive,three[three.method==NAME]],ignore_index=True)
    old_fam=read(V15/'benchmark_family_per_seed.csv')
    benchmark_fam=pd.concat([old_fam,fam[fam.method==NAME]],ignore_index=True)
    summary=[]
    for metric in METRICS:
        v=three.loc[three.method==NAME,metric].to_numpy();sd=v.std(ddof=1);h=stats.t.ppf(.975,9)*sd/np.sqrt(10)
        summary.append(dict(method=NAME,condition='with_negatives',metric=metric,n=10,n_defined=10,mean=v.mean(),sd=sd,ci_low=v.mean()-h,ci_high=v.mean()+h))
    summary=pd.DataFrame(summary)
    extended_summary=pd.concat([read(V15/'benchmark_summary.csv'),summary],ignore_index=True)
    assert len(benchmark.query('condition=="with_negatives"'))==240
    assert benchmark.query('condition=="with_negatives"').method.nunique()==24
    paired=[]
    ens=three[three.method==NAME].set_index('seed').sort_index()
    for baseline in ['TIGER-pair','TIGER-single']:
        base=three[three.method==baseline].set_index('seed').sort_index()
        for m in METRICS:
            delta=(ens[m]-base[m]).to_numpy()
            paired.append(dict(baseline=baseline,metric=m,mean_delta=delta.mean(),seeds_improved=int(((delta>0) if m in ['PCC','KCC'] else (delta<0)).sum()),p_raw=stats.wilcoxon(delta).pvalue))
    paired=pd.DataFrame(paired);ix=paired.p_raw.sort_values().index
    paired.loc[ix,'p_holm_8']=np.minimum(1,np.maximum.accumulate(paired.loc[ix,'p_raw'].to_numpy()*np.arange(8,0,-1)))
    frames={'benchmark_per_seed':benchmark,'benchmark_family_per_seed':benchmark_fam,'benchmark_summary':extended_summary,
            'ensemble_summary':summary,'tiger_three_per_seed':three,'tiger_three_family_per_seed':fam,
            'paired_comparisons':paired,'rse_decomposition':pd.DataFrame(decomp)}
    timing=json.loads((OUT/'timing/result.json').read_text())
    assert timing['status']=='PASS' and len(timing['repeats_s'])==20 and len(timing['checks'])==21
    for c in timing['checks']:
        for m in ['P11','S11',NAME]:
            assert c[m]['n_pair']==2211 and c[m]['all_finite'] and c[m]['max_abs_pair_diff']<=1e-5
    for metric in ['PCC','RSE']:
        r=summary[summary.metric==metric].iloc[0].to_dict()
        r.update(time_s_no_struct=timing['median_s'],time_s_with_struct=timing['median_s']+328.12,
                 time_min=timing['min_s'],time_max=timing['max_s'],device=timing['device'],seed=123,n_repeats=20,
                 max_pair_diff=max(c[NAME]['max_abs_pair_diff'] for c in timing['checks']),structure_increment_s=328.12,
                 n_defined_conditions=1,selection='Fixed matched-seed P11 + S11, equal weights; no selection or recalibration')
        best=pd.concat([read(V15/f'cost_best_{metric}.csv'),pd.DataFrame([r])],ignore_index=True)
        assert len(best)==24 and best.method.nunique()==24
        frames[f'cost_best_{metric}']=best
        for tag,col in [('new_structures','time_s_with_struct'),('structures_ready','time_s_no_struct')]:
            frames[f'pareto_{metric}_{tag}']=frontier(best,col,metric)
    pcc_sel=frames['cost_best_PCC'].set_index('method').condition
    rse_sel=frames['cost_best_RSE'].set_index('method').condition
    assert pcc_sel.equals(rse_sel), 'Main RSE and PCC must use the same condition per method'
    for name,frame in frames.items():frame.to_csv(tab/f'{name}.csv',index=False)
    # The new S11 must be the ablation single head, seed by seed. The old
    # V15 ensemble is historical evidence and is not a numeric target.
    for seed in range(120,130):
        a=read(V15/f'source_snapshot/predictions/tiger/factor_S11_seed{seed}.csv').set_index(KEYS).reindex(canon.index)
        b=read(V15/f'source_snapshot/predictions/tiger/s4_single_n1n2_seed{seed}.csv').set_index(KEYS).reindex(canon.index)
        assert np.array_equal(a.y_pred.to_numpy(),b.y_pred.to_numpy())
    audit={'status':'PASS','seeds':list(range(120,130)),'n_pairs':2211,'n_families':11,'n_predictors':24,
           'prediction_rule':'(calibrated P11_s + calibrated S11_s) / 2, separately for each seed s',
           'weights_fitted_on_test':False,'recalibrated':False,'20_component_metrics_reproduced':True,
           'V16_single_alias_verified':True,'original_method_rows_unchanged':True,'source_sha256':source_hashes,
           'timing_median_s':timing['median_s'],'structure_increment_s':328.12}
    (OUT/'analysis_validation.json').write_text(json.dumps(audit,indent=2))
    print(summary.to_string(index=False))
    return frames

if __name__=='__main__':prepare()

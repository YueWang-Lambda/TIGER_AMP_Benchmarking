"""Independent, fixed-endpoint V15 reconstruction; no training or model selection.

The primary TIGER negative-control comparison is the registered 00 versus 11
factorial contrast. The historical submitted P10 reference remains an ablation
reference, and is not relabelled as wholly negative-free.
"""
import ast
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'reports/S4_study_nature_v15'
OLD = ROOT/'reports/S4_study_nature_v14'
WORK = ROOT/'artifacts/repair_v15_v14input'
METRICS = ['PCC', 'KCC', 'log10MAE', 'RSE']
KEYS = ['family', 'anchor_sequence', 'query_sequence']
SEEDS = set(range(120, 130))

def load(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, x):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False))
def literal(p, name):
    return next(ast.literal_eval(n.value) for n in ast.parse(Path(p).read_text()).body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))

NAME = {'Das2021-regr':'Das', 'HydrAMP-arch-regr':'HydrAMP', 'EvoGrad-strict':'EvoGradient',
        'APEX-restr(lrfix)+neg':'APEX', 'DeepAMP-CNN':'DeepAMP', 'AMPLiT-regr vB':'AMPLiT',
        'AMPlify-regr (ours)':'AMPlify', 'BERT-AmPEP+neg':'BERT-AmPEP60',
        'SenseXAMP-Ecoli+neg':'SenseXAMP', 'tsAMP-CS+neg':'tsAMP'}
def canonical(x): return NAME.get(x, x.replace('+neg',''))

def summarize(d, keys):
    rows=[]
    for key,g in d.groupby(keys, sort=False):
        if not isinstance(key,tuple):key=(key,)
        assert len(g)==10 and set(g.seed)==SEEDS
        for m in METRICS:
            a=g[m].to_numpy(float); valid=np.isfinite(a).all()
            mean=float(a.mean()) if valid else np.nan
            sd=float(a.std(ddof=1)) if valid else np.nan
            half=stats.t.ppf(.975,9)*sd/np.sqrt(10)
            rows.append({**dict(zip(keys,key)), 'metric':m,'n':10,'n_defined':int(np.isfinite(a).sum()),
                         'mean':mean,'sd':sd,'ci_low':mean-half,'ci_high':mean+half})
    return pd.DataFrame(rows)

def contrast(a,b,identifier):
    a=a.set_index('seed').reindex(sorted(SEEDS)); b=b.set_index('seed').reindex(sorted(SEEDS)); rows=[]
    for m in METRICS:
        delta=(a[m]-b[m]).to_numpy(float); ok=np.isfinite(delta).all()
        mean=float(delta.mean()) if ok else np.nan
        half=float(stats.t.ppf(.975,9)*stats.sem(delta)) if ok else np.nan
        p=(1. if np.all(delta==0) else float(stats.wilcoxon(delta,method='auto').pvalue)) if ok else np.nan
        rows.append({**identifier,'metric':m,'n':10,'delta_a_minus_b':mean,'ci_low':mean-half,'ci_high':mean+half,'p_raw':p})
    return rows

def holm(frame, groups):
    frame=frame.copy();frame['p_holm']=np.nan
    for _,g in frame.groupby(groups,dropna=False):
        valid=g[g.p_raw.notna()].sort_values('p_raw'); n=len(g)
        # Undefined tests retain a place in the registered multiplicity family
        # (equivalent to a non-rejectable p=1), rather than shrinking its size.
        frame.loc[valid.index,'p_holm']=np.minimum(1,np.maximum.accumulate(valid.p_raw.to_numpy()*np.arange(n,n-len(valid),-1)))
    return frame

def main():
    OUT.mkdir(exist_ok=True); snap=OUT/'source_snapshot';snap.mkdir(exist_ok=True)
    assert load(WORK/'training_and_inference_complete.json')['status']=='PASS'
    # The training freeze is verified without importing a possibly modified package.
    for p,h in load(WORK/'freeze_manifest.json')['file_hashes'].items():assert sha(p)==h,p
    pep=pd.read_csv(WORK/'source_snapshot/2_external_testset/data/final11_peptides.csv')
    edges=pd.read_csv(WORK/'source_snapshot/4_S4_benchmark/results/pair_keys_primary_2211.csv')
    pep.family=pep.family.str.rstrip('/'); edges.family=edges.family.str.rstrip('/')
    assert len(pep)==208 and pep.sequence.is_unique and len(edges)==2211 and edges.family.nunique()==11
    assert set(edges.anchor_sequence)|set(edges.query_sequence)==set(pep.sequence)
    assert not edges.duplicated(KEYS).any()
    mic=pep.set_index('sequence').MIC_E_coli_ug_mL
    edges['y_true']=np.log2(edges.anchor_sequence.map(mic)/edges.query_sequence.map(mic))
    edges.to_csv(snap/'canonical_pairs.csv',index=False);pep.to_csv(snap/'external_peptides.csv',index=False)
    groups=[(f,g.index.to_numpy()) for f,g in edges.groupby('family')]
    family_rows=[];seed_rows=[];provenance=[];undefined=[]
    def score(path, identity, level='pair', unit='log10_ug_mL', expected_hash=None):
        if expected_hash:assert sha(path)==expected_hash,path
        q=pd.read_csv(path)
        if level=='pair':
            q.family=q.family.str.rstrip('/')
            assert not q.duplicated(KEYS).any()
            assert set(map(tuple,q[KEYS].to_numpy()))==set(map(tuple,edges[KEYS].to_numpy())),path
            joined=edges.merge(q[KEYS+['y_pred']+(['y_true'] if 'y_true' in q else [])],on=KEYS,validate='one_to_one',suffixes=('_canonical','_provided'))
            if 'y_true' in q:assert np.allclose(joined.y_true_canonical,joined.y_true_provided,rtol=0,atol=1e-7),path
            pred=joined.y_pred.to_numpy(float)
        else:
            seq='sequence' if 'sequence' in q else q.columns[0]
            cols=[c for c in q if 'log10mic' in c.lower()]
            assert len(cols)==1,(path,q.columns)
            values=q.set_index(seq)[cols[0]]
            assert values.index.is_unique and set(values.index)==set(pep.sequence),path
            if unit=='log10_uM':
                assert cols[0]=='log10mic_um'
                masses=literal(OLD/'source_snapshot/methods_package_S4_v1/4_S4_benchmark/code/sota_bench_retrain/retrain_evo_strict.py','RES_MASS')
                values=values+np.log10(np.array([sum(masses[a] for a in s)+18.0152 for s in values.index])/1000.)
            else:assert unit=='log10_ug_mL'
            pred=((edges.anchor_sequence.map(values)-edges.query_sequence.map(values))/np.log10(2)).to_numpy(float)
        assert np.isfinite(pred).all() and pred.shape==(2211,)
        target=snap/'predictions'/identity['scope']/f"{identity['key']}_seed{identity['seed']}.csv"
        target.parent.mkdir(parents=True,exist_ok=True)
        frame=edges.copy();frame['y_pred']=pred;frame.to_csv(target,index=False)
        fam=[]
        for family,ix in groups:
            y=edges.y_true.to_numpy()[ix];p=pred[ix]; e=y-p;sst=np.square(y-y.mean()).sum()
            assert sst>0
            constant=np.ptp(p)==0
            values={'PCC':np.nan if constant else float(stats.pearsonr(y,p).statistic),
                    'KCC':np.nan if constant else float(stats.kendalltau(y,p,variant='b').statistic),
                    'log10MAE':float(np.abs(e).mean()*np.log10(2)), 'RSE':float(np.square(e).sum()/sst)}
            row={**identity,'family':family,'n_pairs':len(ix),**values};family_rows.append(row);fam.append(row)
            if not np.isfinite([values['PCC'],values['KCC']]).all():undefined.append(row)
        macro={m:float(np.mean([x[m] for x in fam])) for m in METRICS} # np.mean deliberately propagates NaN
        seed_rows.append({**identity,**macro,**{m+'_zero_sensitivity':float(np.nan_to_num([x[m] for x in fam],nan=0).mean()) for m in ['PCC','KCC']}})
        provenance.append({**identity,'original':str(Path(path).relative_to(ROOT)),'original_sha256':sha(path),
                           'snapshot':str(target.relative_to(OUT)),'sha256':sha(target),'source_level':level,'source_unit':unit if level!='pair' else 'log2_difference'})
        return macro
    reg=load(WORK/'run_registry.json'); assert len(reg)==490
    lookup={r['task_id']:r for r in reg if 'task_id' in r}
    for r in reg:
        tid=r.get('canonical_task',r.get('task_id')); p=WORK/'tiger_external'/tid
        accepted=load(p/'EVALUATED.json');assert accepted['status']=='PASS' and accepted['n_pairs']==2211
        met=score(p/'pair_predictions.csv',{'scope':'tiger','key':r['label'],'seed':r['seed'],'action':r['action']},expected_hash=accepted['prediction_sha256'])
        archived=load(p/'metrics.json')['macro']
        for m in METRICS:
            if archived[m] is None:assert not np.isfinite(met[m])
            else:assert abs(met[m]-archived[m])<1e-8,(tid,m,met[m],archived[m])
    print('TIGER: 490 result slots independently reconstructed',flush=True)
    for item in load(WORK/'retained_external_jobs.json'):
        path=Path(item['completion']);assert sha(path)==item['sha256'];d=load(path);assert d['status']=='PASS'
        if 'AMPLiT' in path.parts:
            method='AMPLiT';condition=path.parent.parent.name;seed=int(path.parent.name[4:])
        else:method=path.parent.parent.name;condition='with_negatives';seed=int(path.parent.name[4:])
        score(path.parent/'predictions.csv',{'scope':'external','key':method+'__'+condition,'method':method,'condition':condition,'seed':seed,'action':'retained_retrained'},level='peptide')
    sources=pd.read_csv(OLD/'noneg_prediction_sources.csv')
    for r in sources.itertuples():
        method=canonical(r.method)
        if method in ['AMPLiT','TIGER-single']:continue
        assert r.prediction_level=='peptide'
        score(OLD/r.path,{'scope':'external','key':method+'__without_negatives','method':method,'condition':'without_negatives','seed':r.seed,'action':'V14_prediction_reuse'},level='peptide',unit='log10_uM' if method=='EvoGradient' else 'log10_ug_mL',expected_hash=r.sha256)
    seed=pd.DataFrame(seed_rows);family=pd.DataFrame(family_rows)
    tiger=seed[seed.scope=='tiger'].rename(columns={'key':'arm'}).drop(columns=['method','condition'],errors='ignore')
    tfam=family[family.scope=='tiger'].rename(columns={'key':'arm'}).drop(columns=['method','condition'],errors='ignore')
    bench=seed[seed.scope=='external'].copy();bfam=family[family.scope=='external'].copy()
    # Registered 00/11 comparison, with the two-stage 10/01 cells kept separately.
    for model in ['P','S']:
        for bits,condition in [('00','without_negatives'),('11','with_negatives')]:
            arm=f'factor_{model}{bits}';method='TIGER-pair' if model=='P' else 'TIGER-single'
            for source,target in [(tiger,bench),(tfam,bfam)]:
                q=source[source.arm==arm].copy();q['method']=method;q['condition']=condition;q['key']=method+'__'+condition;q['scope']='benchmark'
                if source is tiger:bench=pd.concat([bench,q],ignore_index=True)
                else:bfam=pd.concat([bfam,q],ignore_index=True)
    assert len(tiger)==490 and len(tfam)==5390
    assert len(bench)==460 and len(bfam)==5060 and bench.method.nunique()==23
    assert not bench.duplicated(['method','condition','seed']).any()
    ts=summarize(tiger,['arm']);bs=summarize(bench,['method','condition'])
    tests=[]
    for method,g in bench.groupby('method'):
        tests+=contrast(g[g.condition=='with_negatives'],g[g.condition=='without_negatives'],{'method':method,'contrast':'with_minus_without'})
    tests=holm(pd.DataFrame(tests),['metric'])
    factorial=[]
    for model in ['P','S']:
        for a,b,tag in [('10','00','pretraining_without_regression_neg'),('11','01','pretraining_with_regression_neg'),('01','00','regression_without_pretraining'),('11','10','regression_with_pretraining')]:
            factorial+=contrast(tiger[tiger.arm==f'factor_{model}{a}'],tiger[tiger.arm==f'factor_{model}{b}'],{'model':model,'contrast':tag,'a':f'factor_{model}{a}','b':f'factor_{model}{b}'})
    for bits in ['00','10','01','11']:
        factorial+=contrast(tiger[tiger.arm=='factor_P'+bits],tiger[tiger.arm=='factor_S'+bits],{'model':'pair_vs_single','contrast':bits,'a':'factor_P'+bits,'b':'factor_S'+bits})
    factorial=holm(pd.DataFrame(factorial),['metric'])
    abtests=[]
    for arm in tiger.arm.unique():
        if arm=='S4' or arm.startswith('factor_'):continue
        abtests+=contrast(tiger[tiger.arm==arm],tiger[tiger.arm=='S4'],{'arm':arm,'contrast':'arm_minus_reference'})
    abtests=holm(pd.DataFrame(abtests),['metric'])
    for fn,d in [('tiger_per_seed',tiger),('tiger_family_per_seed',tfam),('tiger_summary',ts),('benchmark_per_seed',bench),('benchmark_family_per_seed',bfam),('benchmark_summary',bs),('benchmark_paired_tests',tests),('factorial_contrasts',factorial),('ablation_contrasts',abtests),('prediction_provenance',pd.DataFrame(provenance)),('undefined_correlations',pd.DataFrame(undefined,columns=family.columns))]:d.to_csv(OUT/(fn+'.csv'),index=False)
    labels=literal(OLD/'ablation_plotting_source.py','LABELS');groups0=literal(OLD/'ablation_plotting_source.py','GROUPS')
    labels.update({'S4_sim00':'Similarity ≥ 0.0','s4_gate_off':'No training gate','s4_mod_s':'S only','S4_negpairs':'Negative pairs (top-k = 1)'})
    for r in reg:
        if r['label'].startswith('factor_'):labels[r['label']]=r['label'].replace('factor_','')
    for arm in tiger.arm.unique():labels.setdefault(arm,arm.replace('s4_bal_signed_bal','Signed quota ').replace('s4_bal_unsigned_bal','Unsigned quota '))
    dump(OUT/'labels.json',labels);dump(OUT/'ablation_groups.json',groups0)
    # Carry unchanged dataset evidence forward, but do not execute old metric or timing code.
    for fn in ['external_edge_source_data.csv','external_family_source_data.csv','external_peptide_source_data.csv','training_cleaned_replay.csv','training_delta_distributions.csv','training_pathogen_counts.csv','structure_timing_paired_source_data.csv']:
        shutil.copy2(OLD/fn,OUT/fn)
    shutil.copy2(OLD/'S4_complete_analysis_and_figures_Fig5_style.ipynb',snap/'V14_original_notebook.ipynb')
    shutil.copy2(WORK/'run_registry.json',snap/'run_registry.json')
    for fn in ['input_and_replay_acceptance.json','reuse_provenance_acceptance.json','freeze_manifest.json','training_and_inference_complete.json','retained_external_jobs.json']:
        shutil.copy2(WORK/fn,snap/fn)
    dump(OUT/'reconstruction_acceptance.json',{'status':'PASS','tiger_result_slots':490,'unique_tiger_predictors':470,'tiger_aliases':20,'external_prediction_sets':420,'benchmark_method_condition_seed_sets':460,'families':11,'pairs':2211,'peptides':208,'undefined_family_rows':len(undefined),'primary_TIGER_conditions':'factorial 00/11; historical P10 reference analyzed separately','macro_policy':'all 11 families; undefined correlation propagates; no dropped seeds or families','training_input_policy':'V14 labels, terminal features, external uniform CFU; positive precedence on external negative pool retained','all_predictions_independently_reconstructed':True})
    print(bs[(bs.metric=='PCC')&(bs.condition=='with_negatives')][['method','mean','sd']].sort_values('mean',ascending=False).to_string(index=False),flush=True)
    print('Undefined family rows:',len(undefined),flush=True)

if __name__=='__main__':main()

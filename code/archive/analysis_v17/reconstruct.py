"""Independent V17 reconstruction; formal delivery requires 400 accepted jobs."""
from pathlib import Path
import os,json,hashlib,shutil,importlib.util
import numpy as np
import pandas as pd
from scipy import stats
ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'artifacts/ablation_v17_u20000'
OLD=ROOT/'reports/S4_study_nature_v16'
OUT=Path(os.environ.get('V17_DELIVERY_DIR',str(ROOT/'reports/S4_study_nature_v17')))
FIXTURE=os.environ.get('V17_PIPELINE_FIXTURE')=='1'
METRICS=['PCC','KCC','log10MAE','RSE'];KEYS=['family','anchor_sequence','query_sequence']
def load(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def read(p):return pd.read_csv(p,float_precision='round_trip')
spec=importlib.util.spec_from_file_location('v15_stats',ROOT/'scripts/analysis_v15/reconstruct.py')
st=importlib.util.module_from_spec(spec);spec.loader.exec_module(st)

def initialize():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'source_snapshot/canonical_pairs.csv').exists():return
 # Copy supporting tables and drawing sources, never historical success flags or results prose.
 for p in OLD.iterdir():
  if p.is_file() and p.suffix in ['.py','.csv','.json','.txt'] and not any(k in p.stem for k in ['acceptance','validation','check','manifest','complete','review']):shutil.copy2(p,OUT/p.name)
 shutil.copytree(OLD/'source_snapshot',OUT/'source_snapshot',ignore=shutil.ignore_patterns('predictions','configs','build_history','continuation_before_completion','*.ipynb'))
 for directory in ['predictions/tiger','predictions/external','configs']:(OUT/'source_snapshot'/directory).mkdir(parents=True,exist_ok=True)
 shutil.copytree(OLD/'source_snapshot/predictions/external',OUT/'source_snapshot/predictions/external',dirs_exist_ok=True)
 addon=OUT/'TIGER_ensemble_addendum'
 shutil.copytree(OLD/'TIGER_ensemble_addendum/code',addon/'code',ignore=shutil.ignore_patterns('__pycache__'))
 (addon/'timing').mkdir(exist_ok=True)
 (OUT/'panels').mkdir(exist_ok=True)
 for name in ['fig3d_manuscript_update']:
  shutil.copytree(OLD/name,OUT/name,ignore=shutil.ignore_patterns('*.png','*.pdf','*.svg','*.tiff','*.docx','__pycache__'))
 shutil.copy2(OLD/'S4_complete_analysis_and_figures_Fig5_style.ipynb',OUT/'source_snapshot/V16_original_notebook.ipynb')
 for name in ['factorial_contrasts.csv','single_unification_acceptance.json']:(OUT/name).unlink(missing_ok=True)

def main():
 if FIXTURE:assert OUT.is_relative_to(WORK),'Fixture output must remain in artifacts, never formal reports'
 else:assert load(WORK/'training_and_inference_complete.json')['status']=='PASS'
 for path,h in load(WORK/'freeze_manifest.json')['file_hashes'].items():assert sha(path)==h,path
 structures=load(WORK/'structure_sources_manifest.json');assert structures['status']=='PASS'
 for path,r in structures['files'].items():assert sha(path)==r['sha256'],path
 initialize();snap=OUT/'source_snapshot'
 edges=read(snap/'canonical_pairs.csv');assert len(edges)==2211 and edges.family.nunique()==11 and not edges.duplicated(KEYS).any()
 idx=[(f,g.index.to_numpy()) for f,g in edges.groupby('family')];y=edges.y_true.to_numpy()
 reg=load(WORK/'run_registry.json');refs={r['seed']:r for r in reg if r['label']=='factor_P11'}
 seedrows=[];familyrows=[];provenance=[];runcheck=[]
 def score(path,arm,seed,action,config):
  p=read(path);p.family=p.family.str.rstrip('/')
  assert not p.duplicated(KEYS).any() and set(map(tuple,p[KEYS].to_numpy()))==set(map(tuple,edges[KEYS].to_numpy()))
  p=edges[KEYS].merge(p,on=KEYS,validate='one_to_one')
  assert np.allclose(p.y_true,y,atol=1e-10,rtol=0) and np.isfinite(p.y_pred).all()
  target=snap/f'predictions/tiger/{arm}_seed{seed}.csv'
  q=edges.copy();q['y_pred']=p.y_pred.to_numpy();q.to_csv(target,index=False)
  ident=dict(scope='tiger',arm=arm,seed=seed,action=action,reference='factor_P11');fam=[]
  for f,i in idx:
   truth=y[i];pred=p.y_pred.to_numpy()[i];error=truth-pred
   values=dict(PCC=np.nan if np.ptp(pred)==0 else stats.pearsonr(truth,pred).statistic,KCC=np.nan if np.ptp(pred)==0 else stats.kendalltau(truth,pred,variant='b').statistic,log10MAE=abs(error).mean()*np.log10(2),RSE=np.square(error).sum()/np.square(truth-truth.mean()).sum())
   fam.append(values);familyrows.append(dict(**ident,family=f,n_pairs=len(i),**values))
  vals={m:np.mean([v[m] for v in fam]) for m in METRICS}
  seedrows.append(dict(**ident,**vals,**{m+'_zero_sensitivity':np.nan_to_num([v[m] for v in fam]).mean() for m in ['PCC','KCC']}))
  provenance.append(dict(scope='tiger',key=arm,seed=seed,action=action,original=str(path),original_sha256=sha(path),snapshot=str(target.relative_to(OUT)),sha256=sha(target),source_level='pair',source_unit='log2_difference'))
  shutil.copy2(config,snap/f'configs/{arm}_seed{seed}.json')
  return vals
 for r in reg:
  arm,seed=r['label'],r['seed']
  if r['action']=='train':
   if FIXTURE:
    prior_arm='factor_P11' if arm=='s4_bal_signed_bal4000' else arm
    path=OLD/f'source_snapshot/predictions/tiger/{prior_arm}_seed{seed}.csv'
   else:
    run=Path(r['run_dir']);ep=WORK/'tiger_external'/r['task_id'];a=load(ep/'EVALUATED.json')
    assert a['status']=='PASS' and a['calibration_value_check'] and a['n_pairs']==2211
    path=ep/'pair_predictions.csv';assert sha(path)==a['prediction_sha256'] and sha(run/'config.json')==a['config_sha256']
    done=load(run/'DONE.json');assert done['status']=='PASS'
    for f in range(1,6):assert load(run/f'intermediate/fold{f}_indices.json')==load(Path(refs[seed]['run_dir'])/f'intermediate/fold{f}_indices.json')
    cal=load(run/'results/calibrator.json');p=read(path)
    assert np.allclose(p.y_pred,cal['slope']*p.raw_fold_mean+cal['intercept'],rtol=0,atol=1e-10)
    runcheck.append(dict(task_id=r['task_id'],folds=5,fold_indices_match_reference=True,calibration_value_match=True))
   vals=score(path,arm,seed,'PIPELINE_FIXTURE_ONLY' if FIXTURE else 'U20000_reference_retrained',r['config_path'])
   if not FIXTURE:
    archived=load(ep/'metrics.json')['macro']
    for m in METRICS:assert (archived[m] is None and not np.isfinite(vals[m])) or np.isclose(vals[m],archived[m],atol=1e-8,rtol=0),(arm,seed,m)
  else:
   path=Path(r['prediction_path']);assert sha(path)==r['prediction_sha256']
   score(path,arm,seed,'V16_entire_ten_seed_batch_reuse',r['config_path'])
 for seed in range(120,130):
  score(OLD/f'source_snapshot/predictions/tiger/factor_S00_seed{seed}.csv','factor_S00',seed,'V16_archive_reuse',OLD/f'source_snapshot/configs/factor_S00_seed{seed}.json')
  r=next(x for x in reg if x['label']=='s4_single_n1n2' and x['seed']==seed)
  score(Path(r['prediction_path']),'factor_S11',seed,'Alias_of_reused_single_head',r['config_path'])
 tiger=pd.DataFrame(seedrows);family=pd.DataFrame(familyrows);assert len(tiger)==440 and tiger.arm.nunique()==44
 groups=load(WORK/'landscape_groups.json');arms=[a for _,aa in groups for a in aa];assert len(arms)==42 and len(set(arms))==42
 tests=[]
 for arm in arms:
  if arm!='factor_P11':tests+=st.contrast(tiger[tiger.arm==arm],tiger[tiger.arm=='factor_P11'],dict(arm=arm,reference='factor_P11',contrast='arm_minus_P11',multiplicity='all_ablation_metrics'))
 tests=pd.DataFrame(tests);assert len(tests)==164;tests=st.holm(tests,['multiplicity']);tests['multiplicity_family_size']=164
 for name,frame in [('tiger_per_seed',tiger),('tiger_family_per_seed',family),('tiger_summary',st.summarize(tiger,['arm'])),('ablation_contrasts',tests)]:frame.to_csv(OUT/f'{name}.csv',index=False)
 external=read(OLD/'prediction_provenance.csv').query('scope=="external"');assert len(external)==420
 provenance=pd.concat([pd.DataFrame(provenance),external],ignore_index=True);assert len(provenance)==860
 provenance.to_csv(OUT/'prediction_provenance.csv',index=False)
 for name,source in [('benchmark_per_seed',tiger),('benchmark_family_per_seed',family)]:
  old=read(OLD/(name+'.csv'));b=old[~old.method.isin(['TIGER-pair','TIGER-single'])].copy()
  for head,model in [('P','TIGER-pair'),('S','TIGER-single')]:
   for cond,bits in [('without_negatives','00'),('with_negatives','11')]:
    q=source[source.arm=='factor_'+head+bits].copy();q['method']=model;q['condition']=cond;q['key']=model+'__'+cond;q['scope']='benchmark'
    b=pd.concat([b,q.reindex(columns=old.columns)],ignore_index=True)
  assert len(b)==(460 if name=='benchmark_per_seed' else 5060)
  keys=['method','condition','seed']+(['family'] if 'family' in b else [])
  assert not b.duplicated(keys).any();b.to_csv(OUT/(name+'.csv'),index=False)
  # Every non-pair method, including both single conditions, remains identical.
  x=b[b.method!='TIGER-pair'].set_index(keys).sort_index();z=old[old.method!='TIGER-pair'].set_index(keys).sort_index()
  assert np.allclose(x[METRICS],z[METRICS],rtol=0,atol=1e-10,equal_nan=True)
 combined=pd.concat([family,read(OUT/'benchmark_family_per_seed.csv')],ignore_index=True)
 combined[~np.isfinite(combined.PCC)|~np.isfinite(combined.KCC)].to_csv(OUT/'undefined_correlations.csv',index=False)
 b=read(OUT/'benchmark_per_seed.csv');st.summarize(b,['method','condition']).to_csv(OUT/'benchmark_summary.csv',index=False)
 bt=[]
 for method,g in b.groupby('method'):bt+=st.contrast(g[g.condition=='with_negatives'],g[g.condition=='without_negatives'],dict(method=method,contrast='with_minus_without'))
 st.holm(pd.DataFrame(bt),['metric']).to_csv(OUT/'benchmark_paired_tests.csv',index=False)
 tests[tests.arm=='factor_P00'].to_csv(OUT/'negative_control_contrasts.csv',index=False)
 labels=load(OLD/'labels.json');labels.update(factor_P11='TIGER-pair',s4_bal_signed_bal4000='Signed quota 4000',factor_P00='P00: neither stage')
 for label in ['factor_P10','factor_P01','factor_S10','factor_S01','s4_bal_unsigned_bal20000']:labels.pop(label,None)
 dump(OUT/'labels.json',labels);dump(OUT/'ablation_groups.json',groups)
 tiger.to_csv(snap/'ablation_per_seed_flat.csv',index=False)
 records={a:dict(n=10,per_seed={str(r.seed):{m:float(getattr(r,m)) for m in METRICS} for r in g.itertuples()},mean={m:float(np.mean(g[m].to_numpy())) for m in METRICS}) for a,g in tiger.groupby('arm')}
 dump(snap/'ablation_metrics.json',json.loads(json.dumps(records).replace('NaN','null')))
 for name in ['run_registry.json','interventions.json']:shutil.copy2(WORK/name,snap/name)
 dump(OUT/'training_run_checks.json',dict(status='PASS',validation_only=FIXTURE,jobs=runcheck))
 dump(OUT/'reconstruction_acceptance.json',dict(status='PASS',validation_only=FIXTURE,training_jobs=0 if FIXTURE else 400,landscape_configurations=42,reused_landscape_slots=20,tiger_result_slots=440,external_prediction_sets=420,benchmark_method_condition_seed_sets=460,reference='factor_P11',reference_protocol='unsigned quota 20000; neighbour validation',families=11,pairs=2211,peptides=208,registered_ablation_tests=164,all_predictions_independently_reconstructed=True))
 print('Reconstructed 42-row landscape; 440 TIGER + 420 external snapshots',flush=True)
if __name__=='__main__':main()

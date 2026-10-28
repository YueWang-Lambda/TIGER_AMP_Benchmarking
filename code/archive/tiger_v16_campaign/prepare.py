"""Register the approved P11 ablations; no smoke runs and no training here."""
from pathlib import Path
import json,hashlib,shutil,sys,os,ast
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'artifacts/repair_v15_v14input'
WORK=ROOT/'artifacts/ablation_v16_p11'
RUNS=ROOT/'runs_ablation_v16_p11'
SCRIPTS=ROOT/'scripts/ablation_v16'
V15=ROOT/'reports/S4_study_nature_v15'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))

def main():
 assert not (WORK/'freeze_manifest.json').exists(), 'Already registered; no replacement'
 oldfreeze=json.loads((OLD/'freeze_manifest.json').read_text())
 for p,h in oldfreeze['file_hashes'].items():assert sha(p)==h,('Changed frozen V15 source',p)
 shutil.copytree(OLD/'runtime',WORK/'runtime',ignore=shutil.ignore_patterns('__pycache__','*.pyc'),dirs_exist_ok=True)
 for rel in ['training.csv','negative.csv','2_external_testset/data/final11_peptides.csv','4_S4_benchmark/results/pair_keys_primary_2211.csv']:
  dst=WORK/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OLD/'source_snapshot'/rel,dst)
 sys.path.insert(0,str(WORK/'runtime'))
 from code.utils.config import load_config
 oldreg=json.loads((OLD/'run_registry.json').read_text());lookup={(r['label'],r['seed']):r for r in oldreg}
 oldbase=asdict(load_config(lookup['S4',123]['config_path']))
 groups=json.loads((V15/'ablation_groups.json').read_text())
 for group,arms in groups:
  if group=='Similarity gate':arms.insert(0,'s4_gate_off')
 arms=[(g,a) for g,aa in groups if g!='Reference' for a in aa];assert len(arms)==40
 excluded={'name','out_dir','shared_cache_dir','train_csv','neg_csv','device','seed'}
 interventions={};registry=[]
 for group,label in arms:
  old=asdict(load_config(lookup[label,123]['config_path']))
  delta={k:old[k] for k in old if old[k]!=oldbase[k] and k not in excluded}
  delta.pop('neg_train_inject',None)  # Already true in P11; do not inherit the P10 regime.
  if label=='S4_negpairs':delta={'neg_pair_mode':'topk'}
  delta.pop('validation_similarity_threshold',None)  # P11 already fixes it to 0.7.
  assert delta,(label,'empty intervention')
  interventions[label]={'group':group,'changed_fields':delta}
  for seed in range(120,130):
   base=asdict(load_config(lookup['factor_P11',seed]['config_path']))
   cfg=dict(base);cfg.update(delta)
   task=f'v16_p11_{label}_seed{seed}'
   cfg.update(name=task,out_dir=str(RUNS/label/task),train_csv=str(WORK/'source_snapshot/training.csv'),
       neg_csv=str(WORK/'source_snapshot/negative.csv'),shared_cache_dir=str(WORK/'graph_cache'),device='cuda:0')
   actual={k:cfg[k] for k in cfg if cfg[k]!=base[k] and k not in excluded}
   assert actual==delta,(label,actual,delta)
   assert cfg['neg_train_inject'] and cfg['cls_pretrain_epochs']==15
   assert cfg['validation_similarity_threshold']==.7 and cfg['folds']==5
   path=WORK/'configs'/f'{task}.json';dump(path,cfg)
   registry.append(dict(label=label,seed=seed,group=group,task_id=task,action='train',reference='factor_P11',
       config_path=str(path),config_hash=sha(path),run_dir=cfg['out_dir'],intervention=delta))
 for label in ['factor_P11','factor_P00','factor_P10','factor_P01']:
  for seed in range(120,130):
   r=lookup[label,seed]
   if r['action']=='alias':r=next(x for x in oldreg if x.get('task_id')==r['canonical_task'])
   registry.append(dict(label=label,seed=seed,group='Reference' if label=='factor_P11' else 'Negative-sample exposure',
     task_id=f'v16_reuse_{label}_seed{seed}',action='reuse_archive',reference='factor_P11',run_dir=r['run_dir'],
     config_path=r['config_path'],config_hash=sha(r['config_path']),
     prediction_path=str(V15/f'source_snapshot/predictions/tiger/{label}_seed{seed}.csv'),
     prediction_sha256=sha(V15/f'source_snapshot/predictions/tiger/{label}_seed{seed}.csv')))
 dump(WORK/'run_registry.json',registry);dump(WORK/'interventions.json',interventions)
 groups[0]=['Reference',['factor_P11']]
 groups.append(['Negative-sample exposure',['factor_P10','factor_P01','factor_P00']])
 dump(WORK/'landscape_groups.json',groups)
 assert len(registry)==440 and sum(r['action']=='train' for r in registry)==400
 assert sum(len(aa) for _,aa in groups)==44
 # Hash the exact scientific implementation, inputs and configurations. The
 # queue and postprocessing live outside this lock so they can be maintained.
 files=list((WORK/'runtime/code').rglob('*.py'))+list((WORK/'configs').glob('*.json'))+list((WORK/'source_snapshot').rglob('*.csv'))
 files += [WORK/'run_registry.json',WORK/'interventions.json',WORK/'landscape_groups.json']
 files += [SCRIPTS/n for n in ['common.py','train_one.py','execute_job.py','tiger_external.py','external_metrics.py']]
 for p in files:
  if p.suffix=='.py':ast.parse(p.read_text())
 dump(WORK/'freeze_manifest.json',dict(status='FROZEN',reference='V15 TIGER-pair P11',
     input_policy='V14 actual terminal/CFU inputs retained; positive-record priority; V15 accepted scientific runtime',
     short_preflight='Explicitly skipped by user; no smoke training',file_hashes={str(p):sha(p) for p in files}))
 dump(WORK/'full_training_release.json',dict(status='PASS',authorization='User approved formal queue directly, two GPUs, five processes per GPU; no short preflight',
     config_interventions_checked=400,training_jobs=400,seeds=list(range(120,130)),folds=5,reused_result_slots=40))
 print('REGISTERED: 400 formal five-fold tasks + 40 archived slots; 44 landscape rows; no smoke runs',flush=True)

if __name__=='__main__':main()

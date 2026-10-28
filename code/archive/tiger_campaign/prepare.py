"""Register 400 formal tasks, verify frozen inputs, and reuse 20 archived slots."""
import ast
import copy
import importlib.metadata
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from common import *

SOURCE=ROOT/'artifacts/ablation_v16_p11'
PLAN=ROOT/'artifacts/v17_unsigned20000_plan_20261015'


def main():
    assert not (WORK/'freeze_manifest.json').exists(), 'Already registered'
    WORK.mkdir(parents=True,exist_ok=True)
    proposal=load(PLAN/'proposed_interventions.json')
    assert len(proposal['interventions'])==40 and 's4_single_n1n2' not in proposal['interventions']
    for path,digest in load(SOURCE/'freeze_manifest.json')['file_hashes'].items():
        assert sha(path)==digest,('Frozen source modified',path)
    for name in ['reference_source_audit.json','single_reuse_audit.json']:
        for path,digest in load(PLAN/name)['source_sha256'].items():
            assert sha(path)==digest,('Reused model/prediction modified',path)
    shutil.copytree(SOURCE/'runtime',RUNTIME,ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'),dirs_exist_ok=True)
    for rel in ['training.csv','negative.csv','2_external_testset/data/final11_peptides.csv',
                '4_S4_benchmark/results/pair_keys_primary_2211.csv']:
        dst=PKG/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SOURCE/'source_snapshot'/rel,dst)
    for ext in ['md','docx']:
        p=ROOT/f'reports/V17_unsigned20000_ablation_work_plan_20261015.{ext}'
        dest=WORK/'approved_plan'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for name in ['proposed_interventions.json','reference_source_audit.json','single_reuse_audit.json']:
        shutil.copy2(PLAN/name,WORK/'approved_plan'/name)
    def verify(item):
        path,info=item;assert sha(path)==info['sha256'],('Structure/features changed',path)
        return path,dict(info)
    with ThreadPoolExecutor(max_workers=8) as pool:
        structures=dict(pool.map(verify,load(SOURCE/'structure_sources_manifest.json')['files'].items()))
    dump(WORK/'structure_sources_manifest.json',{'status':'PASS','verified':time.time(),'files':structures})
    print('Structure/feature SHA-256 PASS',len(structures),flush=True)
    registry=[]
    excluded={'name','out_dir','train_csv','neg_csv','shared_cache_dir','device'}
    for label,intervention in proposal['interventions'].items():
        for seed in range(120,130):
            source=ROOT/f'runs_ablation_v16_p11/s4_bal_unsigned_bal20000/v16_p11_s4_bal_unsigned_bal20000_seed{seed}'
            base=load(source/'config.json');cfg=dict(base);cfg.update(intervention['changed_fields'])
            task=f'v17_u20000_{label}_seed{seed}'
            cfg.update(name=task,out_dir=str(RUNS/label/task),train_csv=str(PKG/'training.csv'),
                       neg_csv=str(PKG/'negative.csv'),shared_cache_dir=str(WORK/'graph_cache'),device='cuda:0')
            actual={k:v for k,v in cfg.items() if k not in excluded and base.get(k)!=v}
            expected={k:v for k,v in intervention['changed_fields'].items() if base.get(k)!=v}
            assert actual==expected and actual,(label,actual,expected)
            assert cfg['model_kind']=='pair' and cfg['seed']==seed and cfg['folds']==5
            assert cfg['validation_similarity_threshold']==.7
            assert cfg['neg_train_inject']==(label!='factor_P00')
            assert cfg['cls_pretrain_epochs']==(0 if label=='factor_P00' else 15)
            path=WORK/'configs'/f'{task}.json';dump(path,cfg)
            registry.append({'label':label,'seed':seed,'group':intervention['group'],'task_id':task,
                             'action':'train','reference':'factor_P11','sampling_reference_run':str(source),
                             'run_dir':cfg['out_dir'],'config_path':str(path),'config_hash':sha(path),
                             'intervention':intervention['changed_fields'],'effective_changes':actual})
    for label,source_arm,group in [('factor_P11','s4_bal_unsigned_bal20000','Reference'),
                                   ('s4_single_n1n2','s4_single_n1n2','Prediction design')]:
        for seed in range(120,130):
            task=f'v17_reuse_{label}_seed{seed}'
            run=ROOT/f'runs_ablation_v16_p11/{source_arm}/v16_p11_{source_arm}_seed{seed}'
            src=SOURCE/f'tiger_external/v16_p11_{source_arm}_seed{seed}'
            out=WORK/'tiger_external'/task;out.mkdir(parents=True,exist_ok=True)
            for name in ['metrics.json','family_metrics.csv','pair_predictions.csv']:
                shutil.copy2(src/name,out/name)
            dump(out/'EVALUATED.json',{'status':'PASS','action':'reuse_archive','source':str(src),
                                      'n_pairs':2211,'n_families':11,'prediction_sha256':sha(out/'pair_predictions.csv')})
            registry.append({'label':label,'seed':seed,'group':group,'task_id':task,'action':'reuse_archive',
                             'reference':'factor_P11','run_dir':str(run),'config_path':str(run/'config.json'),
                             'config_hash':sha(run/'config.json'),'source_arm':source_arm,
                             'prediction_path':str(out/'pair_predictions.csv'),
                             'prediction_sha256':sha(out/'pair_predictions.csv')})
    assert len(registry)==420 and sum(r['action']=='train' for r in registry)==400
    groups=[['Reference',['factor_P11']],*proposal['comparison_groups']]
    assert sum(len(aa) for _,aa in groups)==42
    dump(WORK/'run_registry.json',registry);dump(WORK/'landscape_groups.json',groups)
    dump(WORK/'interventions.json',proposal['all_comparisons'])
    dump(WORK/'training_interventions.json',proposal['interventions'])
    # The previously independently enumerated strict tables remain exactly valid.
    src=ROOT/'artifacts/v16_pair_unsigned20000_strict/expected_validation'
    shutil.copytree(src,WORK/'expected_validation',dirs_exist_ok=True)
    assert len(list((WORK/'expected_validation').glob('*.csv')))==50
    import torch,pandas as pd
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    from code.utils.config import load_config
    from code.utils.graph_cache import cache_identity
    import code
    assert Path(code.__file__).resolve()==RUNTIME/'code/__init__.py'
    # Copy graph tensors without recomputation; only verified CSV locations differ.
    cache_checks=[]
    for old_path in sorted((SOURCE/'graph_cache').glob('*.pt')):
        payload=torch.load(old_path,map_location='cpu',weights_only=False)
        old_identity=payload['cache_identity'];negative=old_identity['negative']
        cfg=load_config(registry[0]['config_path'])
        for key,value in old_identity['options'].items():setattr(cfg,key,value)
        new_path,new_identity=cache_identity(cfg,ROOT,negative)
        normalized=copy.deepcopy(new_identity)
        for a,b in zip(normalized['source'],old_identity['source']):a['csv']=b['csv']
        assert normalized==old_identity,('Cache content/options mismatch',old_path)
        payload['cache_identity']=new_identity
        new_path.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,new_path)
        dump(new_path.with_suffix('.json'),new_identity)
        again=torch.load(new_path,map_location='cpu',weights_only=False)
        key='neg_graphs' if negative else 'graphs';rowkey='neg_rows' if negative else 'rows'
        pd.testing.assert_frame_equal(payload[rowkey],again[rowkey],check_exact=True)
        tensors=0
        for a,b in zip(payload[key],again[key]):
            assert set(a.keys())==set(b.keys())
            for field in a.keys():
                if torch.is_tensor(a[field]):assert torch.equal(a[field],b[field]);tensors+=1
                else:assert a[field]==b[field]
        cache_checks.append({'source':str(old_path),'target':str(new_path),'graphs':len(payload[key]),
                             'tensor_fields_checked':tensors,'rows_and_tensors_exact':True})
        print('Cache tensors preserved',negative,len(payload[key]),new_path.name,flush=True)
        del payload,again
    assert len(cache_checks)==6  # Three graph settings, each with training and negative caches.
    dump(WORK/'input_acceptance.json',{'status':'PASS','structure_files':len(structures),
         'cache_checks':cache_checks,'formal_configurations_checked':400,'reused_seed_results':20})
    # Snapshot file hashes only: never allow the V17 pipeline to mutate V16.
    old_report=ROOT/'reports/S4_study_nature_v16'
    preserve={str(p):sha(p) for p in old_report.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    dump(WORK/'existing_v16_manifest.json',{'file_hashes':preserve})
    dump(WORK/'environment.json',{'python':sys.version,'packages':{x:importlib.metadata.version(x) for x in
         ['torch','torch-geometric','numpy','pandas','scipy','scikit-learn']},
         'threads':2,'interop':1,'max_jobs_per_gpu':6,'seeds':list(range(120,130))})
    files=list((RUNTIME/'code').rglob('*.py'))+list(PKG.rglob('*.csv'))+list((WORK/'configs').glob('*.json'))
    files+=list((WORK/'expected_validation').glob('*.csv'))
    files+=[WORK/n for n in ['run_registry.json','landscape_groups.json','interventions.json','training_interventions.json','input_acceptance.json']]
    files+=[SCRIPTS/n for n in ['common.py','train_one.py','execute_job.py','tiger_external.py','external_metrics.py','validation.py','strict_audit.py']]
    for p in files:
        if p.suffix=='.py':ast.parse(p.read_text())
    dump(WORK/'freeze_manifest.json',{'status':'FROZEN','input_policy':'V14 actual inputs; V16 frozen science',
                                    'file_hashes':{str(p):sha(p) for p in files}})
    check_freeze()
    dump(WORK/'full_training_release.json',{'status':'PASS','approved':'User approved V17 plan and two GPUs, SIX concurrent jobs per GPU',
         'registered_at':time.time(),'training_jobs':400,'fold_models':2000,'reused_result_slots':20,
         'landscape_rows':42,'max_jobs_per_gpu':6,'seeds':list(range(120,130)),
         'short_preflight':False,'reference':'existing unsigned20000 ten-seed batch; neighbor validation',
         'single_policy':'reuse V16 s4_single_n1n2; no retraining','negative_exposure_controls':['factor_P00']})
    print('RELEASE PASS: 400 formal tasks + 20 reused slots; six jobs per GPU',flush=True)


if __name__=='__main__':main()

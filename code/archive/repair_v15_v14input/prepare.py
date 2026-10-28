"""Freeze isolated legacy inputs plus independently approved algorithm repairs."""
import shutil
from common import *

def main():
    assert (ROOT/'artifacts/repair_v15/CANCELLED_BY_V14_INPUT_FREEZE.json').exists()
    if (WORK/'freeze_manifest.json').exists():raise RuntimeError('Branch already frozen; no silent replacement')
    WORK.mkdir(parents=True,exist_ok=True)
    shutil.copytree(ROOT/'code',RUNTIME/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'),dirs_exist_ok=True)
    legacy=ARCHIVE/'code/utils/features.py'
    shutil.copy2(legacy,RUNTIME/'code/utils/features.py')
    gc=RUNTIME/'code/utils/graph_cache.py'
    gc.write_text(gc.read_text().replace("SCHEMA = 'tiger_graph_ACT_v15_1'","SCHEMA = 'tiger_graph_V14_input_v15_1'"))
    # Frozen CSV inputs are separate from mutable production/package tables.
    paths={
      'training.csv':ROOT/'metadata/train_val_by_cfu_group_ug_per_mL.csv',
      'negative.csv':ROOT/'metadata/uniprot_neg_full_pool.csv',
      '2_external_testset/data/final11_peptides.csv':ROOT/'reports/methods_package_S4_v1/2_external_testset/data/final11_peptides.csv',
      '4_S4_benchmark/results/pair_keys_primary_2211.csv':ROOT/'reports/methods_package_S4_v1/4_S4_benchmark/results/pair_keys_primary_2211.csv'}
    copied={}
    for rel,src in paths.items():
        dst=PKG/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        copied[str(dst)]={'source':str(src),'sha256':sha(src)}
    assert sha(PKG/'training.csv')==sha(ROOT/'reports/S4_study_nature_v14/source_snapshot/methods_package_S4_v1/1_training_data_pairing/data/train_val_by_cfu_group_ug_per_mL.csv')
    assert sha(PKG/'negative.csv')==sha(ARCHIVE/'metadata/uniprot_neg_full_pool.csv')
    oldreg=load(ROOT/'artifacts/repair_v15/run_registry.json')
    train_gates={'S4_sim00','S4_sim03','S4_sim05','S4_sim09','s4_gate_off'}
    factors={'factor_P00','factor_P01','factor_P11','factor_S00','factor_S01'}
    registry=[]
    for r in oldreg:
        q=dict(r)
        if r['action']=='alias':registry.append(q);continue
        label,seed=r['label'],r['seed']
        if label in train_gates|factors:
            c=load(r['config_path']);c.update(train_csv=str(PKG/'training.csv'),
                neg_csv=str(PKG/'negative.csv') if c['neg_csv'] else None,
                shared_cache_dir=str(WORK/'graph_cache'),out_dir=str(RUNS/label/r['task_id']))
            dst=WORK/'configs'/f'{r["task_id"]}.json';dump(dst,c)
            q.update(action='train',config_path=str(dst),run_dir=c['out_dir'],config_hash=sha(dst))
        else:
            arm='s4_single_n2_noneg' if label=='factor_S10' else label
            run=ROOT/f'runs_ablation/{arm}/{arm}_seed{seed}'
            if not run.exists():run=ROOT/f'runs_ablation/{arm.lower()}/{arm.lower()}_seed{seed}'
            assert (run/'results/calibrator.json').exists(),run
            q.update(action='reuse',original_run_dir=str(run),run_dir=str(run),
                     config_path=str(run/'config.json'),config_hash=sha(run/'config.json'))
        registry.append(q)
    assert len(registry)==490
    assert sum(r['action']=='train' for r in registry)==100
    assert sum(r['action']=='reuse' for r in registry)==370
    assert sum(r['action']=='alias' for r in registry)==20
    dump(WORK/'run_registry.json',registry)
    dump(WORK/'source_copy_provenance.json',copied)
    # Explicitly retain already completed external jobs, with evidence hashes.
    retained=[]
    for folder in [ROOT/'artifacts/repair_v15/external_runs',ROOT/'runs_repair_v15/AMPLiT']:
        for p in sorted(folder.glob('*/seed*/DONE.json')):
            assert load(p)['status']=='PASS';retained.append({'completion':str(p),'sha256':sha(p)})
    assert len(retained)==220
    dump(WORK/'retained_external_jobs.json',retained)
    print('Prepared isolated V14 feature runtime, 100 train + 370 reuse + 20 aliases; external 220 retained',flush=True)

if __name__=='__main__':main()

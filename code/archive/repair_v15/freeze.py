"""Preserve all pre-repair source/results and hash the retained original weights."""
import shutil
import time
from common import *


def main():
    done=WORK/'freeze_complete.json'
    if done.exists():
        print('Existing pre-repair freeze retained; no overwrite.'); return
    REPORT.mkdir(parents=True,exist_ok=True); ARCHIVE.mkdir(parents=True,exist_ok=True)
    files=set()
    for rel in ['code','scripts','reports/methods_package_S4_v1']:
        for p in (ROOT/rel).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and '.ipynb_checkpoints' not in p.parts:
                if rel=='scripts' and 'repair_v15' in p.parts: continue
                files.add(p)
    for p in (ROOT/'baseline/sota_bench').rglob('*.py'):
        if '__pycache__' not in p.parts: files.add(p)
    for pat in ['methods_benchmark_*','methods_dataset_TIGERpair_*','benchmark_S4_23methods_reference.docx']:
        files.update(p for p in (ROOT/'reports').glob(pat) if p.is_file())
    for p in (ROOT/'reports/S4_study_nature_v14').iterdir():
        if p.is_file() and p.suffix in ['.py','.ipynb','.md','.docx','.json']: files.add(p)
    for rel in ['metadata/train_val_by_cfu_group_ug_per_mL.csv','metadata/uniprot_neg_full_pool.csv',
                'baseline/sota_bench/data_restricted/uniprot_negatives.csv']:
        files.add(ROOT/rel)
    manifest={}
    for p in sorted(files):
        rel=p.relative_to(ROOT);dest=ARCHIVE/rel
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        h=sha(p)
        if sha(dest)!=h: raise RuntimeError(f'Archive already differs: {rel}')
        manifest[str(rel)]={'sha256':h,'bytes':p.stat().st_size,'archived':True}
    dump(WORK/'pre_repair_source_manifest.json',manifest)
    print(f'Archived {len(manifest)} source/data/result files',flush=True)
    weights={}
    for i,arm in enumerate(arms()):
        for seed in range(120,130):
            cfgpath=run_config(arm,seed);rd=cfgpath.parent
            for p in [cfgpath,*sorted((rd/'intermediate').glob('*.json')),
                      rd/'results/calibrator.json',rd/'results/summary.json',
                      *sorted((rd/'checkpoints').glob('fold*_best.pt'))]:
                if not p.exists(): raise FileNotFoundError(p)
                weights[str(p.relative_to(ROOT))]={'sha256':sha(p),'bytes':p.stat().st_size,'retained_original':True}
        if (i+1)%5==0:print(f'Frozen weights/configs: {i+1}/40 arms',flush=True)
    dump(WORK/'pre_repair_weight_manifest.json',weights)
    dump(WORK/'resource_policy.json',{'gpus':[0,1],'max_training_processes_per_gpu':5,
         'cpu_threads_per_training_process':2,'max_active_training_processes':10,
         'timing_requires_training_idle':True,'user_authorized':True})
    dump(done,{'status':'PASS','source_files':len(manifest),'weight_and_run_files':len(weights),
               'created_unix':time.time(),'R08':'closed_by_user_confirmation',
               'endpoint':{'families':11,'peptides':208,'pairs':2211}})
    print('W0 freeze PASS',flush=True)


if __name__=='__main__':main()

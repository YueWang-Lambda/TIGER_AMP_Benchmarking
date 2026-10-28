"""Sequential training and strict external evaluation for one registered job."""
import argparse
import subprocess
from common import *


def main():
    if (WORK/'CANCELLED_BY_V14_INPUT_FREEZE.json').exists():
        raise RuntimeError('ACT-version tasks cancelled; use scripts/repair_v15_v14input instead')
    ap=argparse.ArgumentParser();ap.add_argument('--id',required=True);ap.add_argument('--gpu',type=int,default=0);a=ap.parse_args()
    jobs=load(WORK/'campaign_jobs.json');job=next(j for j in jobs if j['id']==a.id)
    def run(cmd):subprocess.run([PYTHON,*cmd],cwd=ROOT,check=True)
    if job['kind']=='tiger':
        release=load(WORK/'full_training_release.json');assert release['status']=='PASS'
        for p,h in release['training_source_hashes'].items():assert sha(ROOT/p)==h,('Training source changed after acceptance',p)
        reg=next(r for r in load(WORK/'run_registry.json') if r['task_id']==job['task_id'])
        if reg['action']=='train':
            path=Path(reg['run_dir'])
            if not (path/'DONE.json').exists():run(['scripts/repair_v15/train_one.py','--config',reg['config_path'],'--gpu',str(a.gpu)])
            assert load(path/'DONE.json')['status']=='PASS'
        else:
            assert load(WORK/'reuse_acceptance.json')['status']=='PASS'
            path=Path(reg['original_run_dir'])
        out=Path(job['output'])
        if not (out/'EVALUATED.json').exists():run(['scripts/repair_v15/tiger_external.py','--run',str(path),'--gpu',str(a.gpu),'--output',str(out)])
    elif job['kind']=='amplit':
        if job['condition']=='with_negatives':assert load(WORK/'external_training_release.json')['status']=='PASS'
        run(['scripts/repair_v15/amplit.py','--condition',job['condition'],'--seed',str(job['seed']),'--gpu',str(a.gpu)])
    elif job['kind']=='external':
        run(['scripts/repair_v15/external_one.py','--method',job['method'],'--seed',str(job['seed']),'--gpu',str(a.gpu)])
    elif job['kind']=='pilot_acceptance':run(['scripts/repair_v15/pilot_acceptance.py','--gpu',str(a.gpu)])
    else:raise ValueError(job)
    assert Path(job['completion']).exists(),('Missing final completion record',job['id'])


if __name__=='__main__':main()

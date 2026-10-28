"""Train when necessary, then export strictly checked V14-input predictions."""
import argparse
import subprocess
from common import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--id',required=True);ap.add_argument('--gpu',type=int,required=True);a=ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    check_freeze();assert load(WORK/'full_training_release.json')['status']=='PASS'
    r=next(x for x in load(WORK/'run_registry.json') if x.get('task_id')==a.id)
    assert r['action'] in ['train','reuse']
    run=Path(r['run_dir'])
    if r['action']=='train':
        if not (run/'DONE.json').exists():
            subprocess.run([PYTHON,str(SCRIPTS/'train_one.py'),'--id',a.id,'--gpu',str(a.gpu)],cwd=ROOT,check=True)
        assert load(run/'DONE.json')['status']=='PASS'
    import torch
    torch.set_num_threads(2)
    from tiger_external import predict_run
    out=WORK/'tiger_external'/a.id;out.mkdir(exist_ok=True,parents=True)
    if not (out/'EVALUATED.json').exists():
        met,fam,pred=predict_run(run)
        import numpy as np
        cal=load(run/'results/calibrator.json')
        expected=cal['slope']*pred.raw_fold_mean.to_numpy()+cal['intercept']
        assert len(pred)==2211 and np.allclose(pred.y_pred,expected,rtol=0,atol=1e-12)
        dump(out/'metrics.json',met);fam.to_csv(out/'family_metrics.csv',index=False);pred.to_csv(out/'pair_predictions.csv',index=False)
        dump(out/'EVALUATED.json',{'status':'PASS','task_id':a.id,'run':str(run),'action':r['action'],
             'n_pairs':2211,'n_families':11,'calibration_value_check':True,'input_version':'V14 frozen actual inputs',
             'config_sha256':sha(run/'config.json'),'prediction_sha256':sha(out/'pair_predictions.csv')})
    check_freeze();print('Task fully evaluated',a.id,flush=True)

if __name__=='__main__':main()

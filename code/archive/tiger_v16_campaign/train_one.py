"""One isolated V14-input five-fold training job, with manifest checks."""
import argparse
import time
from common import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--id',required=True);ap.add_argument('--gpu',type=int,required=True);a=ap.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    check_freeze()
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    from code.utils.config import load_config
    from code.train import train
    r=next(x for x in load(WORK/'run_registry.json') if x.get('task_id')==a.id)
    assert r['action']=='train'
    assert load(WORK/'full_training_release.json')['status']=='PASS'
    cfg=load_config(r['config_path'])
    cfg.device='cuda:0';out=Path(cfg.out_dir)
    if (out/'DONE.json').exists():raise RuntimeError('Refusing to overwrite completed run')
    started=time.time();summary=train(cfg,root=ROOT)
    for fold in range(1,6):
        for rel in [f'checkpoints/fold{fold}_best.pt',f'intermediate/p_feature_zscore_fold{fold}.json',f'intermediate/fold{fold}_indices.json']:
            assert (out/rel).exists(),(a.id,rel)
    import math
    cal=load(out/'results/calibrator.json')
    assert all(math.isfinite(float(v)) for v in cal.values() if isinstance(v,(float,int)))
    check_freeze()
    dump(out/'DONE.json',{'status':'PASS','smoke':False,'elapsed_seconds':time.time()-started,
         'input_version':'V14 frozen actual inputs','gpu':a.gpu,'peak_cuda_allocated_bytes':summary['peak_cuda_allocated_bytes'],
         'config_sha256':sha(out/'config.json'),'runtime_manifest_sha256':sha(WORK/'freeze_manifest.json')})
    print('Completed five-fold job',a.id,'formal P11 ablation',flush=True)

if __name__=='__main__':main()

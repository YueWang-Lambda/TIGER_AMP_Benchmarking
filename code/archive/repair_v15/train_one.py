"""One GPU job; writes DONE only after complete five-fold artifact validation."""
import argparse
import time
from common import *


def main():
    if (WORK/'CANCELLED_BY_V14_INPUT_FREEZE.json').exists():
        raise RuntimeError('ACT-version training cancelled; use the isolated V14-input branch')
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    from code.utils.config import load_config
    from code.train import train
    cfg=load_config(args.config)
    if args.smoke:
        cfg.graph_epochs=2;cfg.cls_pretrain_epochs=min(cfg.cls_pretrain_epochs,2)
        cfg.pair_balance_num=min(cfg.pair_balance_num,80);cfg.max_train_pairs=12000
        cfg.out_dir=str(WORK/'smoke'/cfg.name);cfg.early_stop_patience=2
    cfg.device='cuda:0'
    out=Path(cfg.out_dir)
    if (out/'DONE.json').exists():raise RuntimeError('Refusing to overwrite a completed run')
    t=time.time();summary=train(cfg,root=ROOT)
    for fold in range(1,6):
        for rel in [f'checkpoints/fold{fold}_best.pt',f'intermediate/p_feature_zscore_fold{fold}.json',
                    f'intermediate/fold{fold}_indices.json']:
            if not (out/rel).exists():raise RuntimeError(f'Missing completed artifact: {rel}')
    cal=load(out/'results/calibrator.json')
    import math
    if not all(math.isfinite(float(v)) for k,v in cal.items() if isinstance(v,(float,int))):raise ValueError('Non-finite calibrator')
    dump(out/'DONE.json',{'status':'PASS','smoke':args.smoke,'elapsed_seconds':time.time()-t,
         'gpu':args.gpu,'peak_cuda_allocated_bytes':summary['peak_cuda_allocated_bytes'],'config_sha256':sha(out/'config.json')})


if __name__=='__main__':main()

"""Formal five-fold V17 job; frozen scientific code and observational audits."""
import argparse
import math
import resource
import time
from common import *


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--id',required=True);ap.add_argument('--gpu',type=int,required=True);a=ap.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    check_freeze()
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    from code.utils.config import load_config
    import code.train as training
    from strict_audit import install_recording
    from validation import check_training_identity
    r=next(x for x in load(WORK/'run_registry.json') if x.get('task_id')==a.id)
    assert r['action']=='train' and load(WORK/'full_training_release.json')['status']=='PASS'
    cfg=load_config(r['config_path']);cfg.device='cuda:0';out=Path(cfg.out_dir)
    assert not (out/'DONE.json').exists(), 'Refusing to overwrite completed task'
    out.mkdir(parents=True,exist_ok=True)
    dump(out/'execution_environment.json',{'started':time.time(),'torch_version':torch.__version__,
         'cuda_version':torch.version.cuda,'cudnn_version':torch.backends.cudnn.version(),
         'gpu_name':torch.cuda.get_device_name(0),'gpu':a.gpu,'max_jobs_per_gpu':6,
         'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
         'cpu_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads()})
    if cfg.val_pair_mode=='within_fold_strict':install_recording(training,out)
    started=time.time();summary=training.train(cfg,root=ROOT)
    identity=check_training_identity(out,r);dump(out/'training_identity_acceptance.json',identity)
    for fold in range(1,6):
        assert (out/f'checkpoints/fold{fold}_best.pt').exists()
    cal=load(out/'results/calibrator.json')
    assert all(math.isfinite(float(v)) for v in cal.values() if isinstance(v,(float,int)))
    check_freeze()
    dump(out/'DONE.json',{'status':'PASS','smoke':False,'elapsed_seconds':time.time()-started,
         'input_version':'V14 frozen actual inputs','gpu':a.gpu,
         'peak_cuda_allocated_bytes':summary['peak_cuda_allocated_bytes'],
         'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
         'peak_host_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         'config_sha256':sha(out/'config.json'),'runtime_manifest_sha256':sha(WORK/'freeze_manifest.json')})
    print('Completed formal V17 five-fold job',a.id,flush=True)


if __name__=='__main__':main()

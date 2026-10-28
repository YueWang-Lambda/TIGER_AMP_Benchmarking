"""Release full TIGER training on implementation checks, never test scores."""
import argparse
from common import *

LABELS=['S4','factor_P00','factor_P11','factor_S00','s4_gate_off']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,default=1);a=ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    torch.set_num_threads(2)
    import numpy as np
    import pandas as pd
    from code.utils.config import load_config
    from code.dataloader import load_or_build_cache
    from code.utils.scaling import fit_p_f_stats
    from tiger_external import predict_run
    assert load(WORK/'queue/pilot/state.json')['status']=='COMPLETE'
    assert load(WORK/'core_acceptance.json')['status']=='PASS'
    assert load(WORK/'reuse_acceptance.json')['status']=='PASS'
    reg=[r for r in load(WORK/'run_registry.json') if r['seed']==123 and r['label'] in LABELS]
    assert len(reg)==5
    base=RUNS/'S4/S4_seed123';cfg=load_config(base/'config.json')
    payload=load_or_build_cache(cfg,ROOT);graphs=payload['graphs'];rows=payload['rows']
    assert len(rows)==7217
    evidence=[]
    for r in reg:
        run=Path(r['run_dir']);done=load(run/'DONE.json');assert done['status']=='PASS' and not done['smoke']
        assert sha(run/'config.json')==done['config_sha256']
        cfg=load_config(run/'config.json');assert cfg.validation_similarity_threshold==.7
        maxerr=0.
        for fold in range(1,6):
            ix=load(run/'intermediate'/f'fold{fold}_indices.json')
            assert ix==load(base/'intermediate'/f'fold{fold}_indices.json')
            assert not set(ix['train'])&set(ix['validation'])
            ref=fit_p_f_stats(graphs,ix['train'])
            actual=load(run/'intermediate'/f'p_feature_zscore_fold{fold}.json')
            for k in ['mean','std']:
                err=float(np.max(np.abs(np.asarray(actual[k])-ref[k])));maxerr=max(maxerr,err)
                assert err<1e-12,(r['task_id'],fold,k,err)
            if r['label'] in ['factor_P00','factor_P11']:
                pa=load(run/'intermediate'/f'fold{fold}_pair_audit.json')
                ba=load(base/'intermediate'/f'fold{fold}_pair_audit.json')
                assert pa['positive_pair_sha256']==ba['positive_pair_sha256']
                if r['label']=='factor_P00':assert pa['negative_pairs']==0 and pa['negative_pool_size']==0
        if r['label'] in ['factor_P00','factor_S00']:
            assert cfg.neg_csv is None and cfg.cls_pretrain_epochs==0 and not cfg.neg_train_inject
        if r['label']=='s4_gate_off':assert cfg.train_similarity_gate_enabled is False
        out=WORK/'pilot_external'/r['task_id'];out.mkdir(parents=True,exist_ok=True)
        if not (out/'EVALUATED.json').exists():
            met,fam,pred=predict_run(run)
            dump(out/'metrics.json',met);fam.to_csv(out/'family_metrics.csv',index=False);pred.to_csv(out/'pair_predictions.csv',index=False)
        pred=pd.read_csv(out/'pair_predictions.csv');cal=load(run/'results/calibrator.json')
        expected=cal['slope']*pred.raw_fold_mean.to_numpy()+cal['intercept']
        assert len(pred)==2211 and np.isfinite(pred.y_pred).all()
        assert np.allclose(pred.y_pred,expected,rtol=0,atol=1e-12)
        # Negative control: nonidentity calibration must differ from raw values.
        if abs(cal['slope']-1)+abs(cal['intercept'])>1e-6:assert not np.allclose(pred.y_pred,pred.raw_fold_mean,atol=1e-8)
        dump(out/'EVALUATED.json',{'status':'PASS','run':str(run),'n_pairs':2211,'n_families':11,
              'config_hash':sha(run/'config.json'),'calibrated_values_independently_verified':True})
        evidence.append({'task_id':r['task_id'],'feature_stats_max_error':maxerr,'n_pairs':len(pred),
             'runtime_seconds':done['elapsed_seconds'],'peak_cuda_allocated_bytes':done['peak_cuda_allocated_bytes']})
    dump(WORK/'full_training_release.json',{'status':'PASS','pilot_checks':evidence,
         'selection_rule':'implementation, input identity and finite predictions only; no external accuracy threshold',
         'training_source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'code/train.py',ROOT/'code/dataloader.py',ROOT/'code/utils/features.py',ROOT/'code/utils/config.py']}})
    print('Full TIGER training released after complete pilot acceptance',flush=True)


if __name__=='__main__':main()

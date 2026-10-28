"""Release formal tasks after bounded path checks, without selecting scores."""
from common import *
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from code.utils.scaling import fit_p_f_stats
from code.utils.config import load_config
from code.dataloader import load_or_build_cache

SMOKE_LABELS=['factor_P00','factor_P11','factor_S00','S4_sim03','s4_gate_off']

def main():
    check_freeze();assert load(WORK/'input_and_replay_acceptance.json')['status']=='PASS'
    torch.set_num_threads(2)
    reg=load(WORK/'run_registry.json');r=next(x for x in reg if x['label']=='factor_P11' and x['seed']==123)
    cfg=load_config(r['config_path']);p=load_or_build_cache(cfg,ROOT)
    splits=list(GroupKFold(5).split(p['rows'],groups=p['rows'].sequence));evidence=[]
    for label in SMOKE_LABELS:
        run=WORK/'smoke'/f'{label}_seed123';d=load(run/'DONE.json')
        assert d['status']=='PASS' and d['smoke'];cfg=load_config(run/'config.json')
        for fold,(tr,va) in enumerate(splits,1):
            ix=load(run/'intermediate'/f'fold{fold}_indices.json')
            assert ix=={'train':tr.tolist(),'validation':va.tolist()}
            expected=fit_p_f_stats(p['graphs'],tr);actual=load(run/'intermediate'/f'p_feature_zscore_fold{fold}.json')
            for k in ['mean','std']:assert np.allclose(actual[k],expected[k],rtol=0,atol=1e-12)
        if label in ['factor_P00','factor_S00']:
            assert cfg.neg_csv is None and cfg.cls_pretrain_epochs==0 and not cfg.neg_train_inject
        if label=='s4_gate_off':assert not cfg.train_similarity_gate_enabled
        assert cfg.validation_similarity_threshold==.7
        evidence.append({'task_id':f'{label}_seed123','five_folds':True,'matching_stats':True,'duration':d['elapsed_seconds']})
    dump(WORK/'full_training_release.json',{'status':'PASS','smoke_checks':evidence,
         'manifest_sha256':sha(WORK/'freeze_manifest.json'),
         'selection_rule':'V14 input/checkpoint replay, algebraic contracts and bounded five-fold smoke; no external accuracy selection'})
    print('Full 100-task training release PASS',flush=True)

if __name__=='__main__':main()

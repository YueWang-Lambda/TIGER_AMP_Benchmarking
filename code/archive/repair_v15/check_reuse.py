"""Actual archived checkpoint forward invariance for all 30 no-p runs."""
from copy import deepcopy
from common import *
import numpy as np
import torch
torch.set_num_threads(2)
from code.utils.config import load_config
from code.utils.features import calculate_property
from code.utils.scaling import apply_p_f_stats,apply_node_x_stats
from code.evaluation import load_checkpoint_model
from code.dataloader import make_batch


def main():
    old=ROOT/'artifacts/cache_ll37_holdout_cfu/mic_graph_cache_len50_seed123_cfu_v1_trainonly_rawprops_aa_seq1to20_nocoord.pt'
    payload=torch.load(old,map_location='cpu',weights_only=False)
    rows=payload['rows'];indices=np.flatnonzero(rows.n_terminus.fillna('').eq('ACT').to_numpy())[:16]
    graphs=[deepcopy(payload['graphs'][i]) for i in indices]
    new=deepcopy(graphs)
    for g,i in zip(new,indices):
        r=rows.iloc[i];g.p_f=torch.tensor([calculate_property(r.sequence,r.n_terminus,r.c_terminus)],dtype=torch.float32)
    assert any(not torch.equal(a.p_f,b.p_f) for a,b in zip(graphs,new))
    checks=[]
    for task in load(WORK/'run_registry.json'):
        if task['action']!='reuse_validate':continue
        run=Path(task['original_run_dir']);cfg=load_config(run/'config.json')
        assert 'p' not in cfg.feature_modalities
        for fold in range(1,6):
            a=deepcopy(graphs);b=deepcopy(new)
            for collection in [a,b]:
                apply_p_f_stats(collection,load(run/'intermediate'/f'p_feature_zscore_fold{fold}.json'))
                apply_node_x_stats(collection,load(run/'intermediate'/f'node_feature_zscore_fold{fold}.json'))
            model,_=load_checkpoint_model(run/'checkpoints'/f'fold{fold}_best.pt',cfg,torch.device('cpu'));model.eval()
            with torch.inference_mode():
                x=model(make_batch(a[:8]),make_batch(a[8:]));y=model(make_batch(b[:8]),make_batch(b[8:]))
            err=float((x-y).abs().max());assert torch.equal(x,y),(task['task_id'],fold,err)
            checks.append({'task_id':task['task_id'],'fold':fold,'max_abs_change':err})
        print('Reuse PASS',task['task_id'],flush=True)
    assert len(checks)==150
    dump(WORK/'reuse_acceptance.json',{'status':'PASS','runs':30,'checkpoints':150,'pairs_per_checkpoint':8,'checks':checks})


if __name__=='__main__':main()

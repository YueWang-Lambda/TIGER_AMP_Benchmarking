"""V14 terminal features and uniform CFU inputs on the canonical 208-peptide endpoint."""
import argparse
import copy
import fcntl
from common import *
import numpy as np
import pandas as pd
import torch
from code.utils.config import load_config
from code.utils.constants import TARGET_COL
from code.utils.features import normalize_cfu_group
from code.utils.scaling import apply_p_f_stats,apply_node_x_stats
from code.utils.metrics import apply_calibrator
from code.dataloader import build_graphs
from code.evaluation import load_checkpoint_model,eval_explicit_pairs
from external_metrics import panel,evaluate,KEYS

PDB_DIRS={'APEXGO':'data/test_external/test_activity_apexgo/pdb',
          'APEXD':'data/test_external/test_activity_apexd/pdb',
          'LL37':'data/test_external/test_activity_ll37/pdb'}


def cfu_category(value):
    try:
        number=float(value)
        if np.isfinite(number):
            if 1e4<=number<1e5:return '1E4 - 1E5'
            if 1e5<=number<1e6:return '1E5 - 1E6'
            if 1e6<=number<1e7:return '1E6 - 1E7'
            return 'unknown'
    except (ValueError,TypeError):pass
    return normalize_cfu_group(value)


def external_graphs(cfg,fresh=False):
    pep,edges=panel()
    rows=pep[['sequence','n_terminus','c_terminus','cfu_group','panel']].copy()
    rows['source_cfu']=rows.cfu_group;rows['cfu_group']='1E5 - 1E6'  # Frozen V14 actual input, not a claim of measured CFU
    rows[TARGET_COL]=pep.MIC_E_coli_ug_mL
    rows['pdb_path']=[str(ROOT/PDB_DIRS[r.panel]/f'{r.sequence}.pdb') for r in rows.itertuples()]
    assert all(Path(f).exists() for f in rows.pdb_path)
    ident={'input_rows':rows.fillna('').to_dict('records'),'pdbs':{p:sha(p) for p in rows.pdb_path},
           'options':{k:getattr(cfg,k) for k in ['include_node_coords','edge_mode','p_feature_scaling','p_feature_hmom','pair_hmom_delta','max_len','use_cfu_feature']},
           'source_hashes':{str(p):sha(p) for p in [RUNTIME/'code/dataloader.py',RUNTIME/'code/utils/features.py',SCRIPTS/'tiger_external.py']}}
    key=hashlib.sha256(json.dumps(ident,sort_keys=True).encode()).hexdigest()
    cache=WORK/'external_graphs'/f'{key}.pt';cache.parent.mkdir(exist_ok=True,parents=True)
    with cache.with_suffix('.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        if not fresh and cache.exists():
            payload=torch.load(cache,map_location='cpu',weights_only=False)
            assert payload['identity']==ident
            return payload['graphs'],payload['rows'],edges
        graphs,kept,stats=build_graphs(rows,cfg)
        assert len(graphs)==208 and set(kept.sequence)==set(pep.sequence)
        if not fresh:
            tmp=cache.with_name(cache.name+f'.{os.getpid()}.tmp')
            torch.save({'identity':ident,'graphs':graphs,'rows':kept},tmp);os.replace(tmp,cache)
        rows.to_csv(WORK/'external_input_dictionary.csv',index=False)
        return graphs,kept,edges


def predict_run(run,device='cuda:0',fresh=False):
    run=Path(run);cfg=load_config(run/'config.json');device=torch.device(device)
    graphs,rows,edges=external_graphs(cfg,fresh=fresh)
    frames=[]
    for fold in range(1,6):
        apply_p_f_stats(graphs,load(run/'intermediate'/f'p_feature_zscore_fold{fold}.json'))
        if cfg.node_feature_scaling=='zscore':apply_node_x_stats(graphs,load(run/'intermediate'/f'node_feature_zscore_fold{fold}.json'))
        model,_=load_checkpoint_model(run/'checkpoints'/f'fold{fold}_best.pt',cfg,device)
        with torch.inference_mode():pred,_=eval_explicit_pairs(model,graphs,rows,edges,cfg,device)
        pred=pred.rename(columns={'y_true_delta_log2_anchor_minus_query':'y_true','y_pred_delta_log2_anchor_minus_query':'y_pred'})
        # Evaluate validates identities and truth, rather than trusting row order.
        _,_,ordered=evaluate(pred)
        frames.append(ordered.y_pred.to_numpy(float))
        del model
    raw=np.mean(np.stack(frames),axis=0)
    result=edges.copy();result['y_pred_delta_log2_anchor_minus_query']=raw
    result=apply_calibrator(result,load(run/'results/calibrator.json'))
    result=result.rename(columns={'y_pred_delta_log2_anchor_minus_query':'y_pred'})
    met,fam,ordered=evaluate(result)
    ordered['raw_fold_mean']=raw
    return met,fam,ordered


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--output',required=True);a=ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);torch.set_num_threads(2)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    met,fam,pred=predict_run(a.run)
    dump(out/'metrics.json',met);fam.to_csv(out/'family_metrics.csv',index=False);pred.to_csv(out/'pair_predictions.csv',index=False)
    dump(out/'EVALUATED.json',{'status':'PASS','run':str(a.run),'n_pairs':2211,'n_families':11,'config_hash':sha(Path(a.run)/'config.json')})
    print('External endpoint PASS',a.run,flush=True)


if __name__=='__main__':main()

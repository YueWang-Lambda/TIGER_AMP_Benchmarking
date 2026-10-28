"""Contract checks and failure injections required before expensive training."""
import ast
import tempfile
from copy import deepcopy
from common import *
import numpy as np
import pandas as pd
from code.utils.config import Config
from code.utils.features import terminal_charge_adjustment,calculate_property
from code.utils.graph_cache import cache_identity
from code.dataloader import build_training_pairs,build_neg_pairs,precompute_neighbors
from external_metrics import panel,from_peptides,evaluate


def reject(fn):
    try:fn()
    except (ValueError,AssertionError):return
    raise AssertionError('Negative control silently accepted')


def main():
    checks=[]
    assert terminal_charge_adjustment('ACT','AMD')==terminal_charge_adjustment('acetyl','AMD')==0
    assert terminal_charge_adjustment(' ACT ','')==-1
    assert calculate_property('RFWWFRRR','ACT','AMD')==calculate_property('RFWWFRRR','acetyl','AMD')
    checks.append('ACT alias and raw feature equality')
    c=Config(similarity_threshold=0.,max_train_pairs=100,pair_balance_num=100)
    rows=pd.DataFrame({'sequence':['A'*10,'C'*10]})
    assert build_training_pairs(rows,np.array([0.,1.]),c,123)==[]
    c.train_similarity_gate_enabled=False
    assert set(build_training_pairs(rows,np.array([0.,1.]),c,123))=={(0,1),(1,0)}
    checks.append('True gate-off accepts negative NW score; threshold zero rejects')
    c=Config(similarity_threshold=.7,neg_pair_mode='joint')
    assert build_neg_pairs(pd.DataFrame({'sequence':['A'*15]}),pd.DataFrame({'sequence':['A'*10]}),c)==[(1,0)]
    assert build_neg_pairs(pd.DataFrame({'sequence':['A'*10]}),pd.DataFrame({'sequence':['A'*15]}),c)==[(0,1)]
    checks.append('Both negative-pair directions independently gated')
    c=Config(similarity_threshold=.7,validation_similarity_threshold=.7,use_cfu_protocol=False,use_neighbor_fallback=False)
    q=pd.DataFrame({'sequence':['A'*15]});a=pd.DataFrame({'sequence':['A'*10]})
    before=precompute_neighbors(a,q,c);c.similarity_threshold=.3
    assert precompute_neighbors(a,q,c)==before==[[]]
    checks.append('Training gate independent of validation gate')
    # A content edit with equal row count, a row permutation, and PDB edit must
    # independently change the cache key; no graph build needed for this probe.
    with tempfile.TemporaryDirectory() as td:
        d=Path(td);csv=d/'rows.csv';pdb=d/'pdb';pdb.mkdir();ff=d/'features.txt';ff.write_text('features')
        pd.DataFrame({'sequence':['AAAAAA','CCCCCC'],'n_terminus':['ACT','']}).to_csv(csv,index=False)
        for s in ['AAAAAA','CCCCCC']:(pdb/f'{s}.pdb').write_text('pdb-v1')
        cfg=Config(train_csv=str(csv),train_pdb_dir=str(pdb),feature_path=str(ff),shared_cache_dir=str(d/'cache'))
        original=cache_identity(cfg)[0]
        x=pd.read_csv(csv);x.loc[0,'n_terminus']='';x.to_csv(csv,index=False)
        changed=cache_identity(cfg)[0];assert original!=changed
        x.iloc[::-1].to_csv(csv,index=False);reordered=cache_identity(cfg)[0];assert changed!=reordered
        (pdb/'AAAAAA.pdb').write_text('pdb-v2');assert reordered!=cache_identity(cfg)[0]
    checks.append('Cache source content/order/PDB invalidation')
    pep,edges=panel();pred=pd.Series(np.log10(pep.MIC_E_coli_ug_mL.to_numpy()),index=pep.sequence)
    frame=from_peptides(pred,unit='log10_ug_mL');met,_,_=evaluate(frame)
    assert abs(met['macro']['PCC']-1)<1e-12 and met['macro']['RSE']<1e-25
    shuffled=frame.sample(frac=1,random_state=3);assert evaluate(shuffled)[0]==met
    reject(lambda:evaluate(frame.iloc[:-1]));reject(lambda:evaluate(pd.concat([frame,frame.iloc[:1]])))
    bad=frame.copy();bad.loc[0,'y_pred']=np.nan;reject(lambda:evaluate(bad))
    bad=frame.copy();bad['y_true']*=np.log10(2);reject(lambda:evaluate(bad))
    reject(lambda:from_peptides(pred.iloc[:-1],unit='log10_ug_mL'))
    zero=frame.copy();zero['y_pred']=0.;z,_,_=evaluate(zero)
    assert z['macro']['PCC'] is None and len(z['undefined_families']['PCC'])==11
    assert z['zero_contribution_sensitivity']['PCC']==0
    checks.append('Canonical identities/units/NaN/degenerate macros and row-order invariance')
    registry=load(WORK/'run_registry.json')
    assert sum(r['group']=='ablation40' for r in registry)==400
    assert sum(r['group']=='negative_factorial8' for r in registry)==80
    assert sum(r['action']=='reuse_validate' for r in registry)==30
    for r in registry:
        if r['label'] in ['factor_P00','factor_S00']:
            c=load(r['config_path']);assert c['neg_csv'] is None and c['cls_pretrain_epochs']==0 and not c['neg_train_inject']
    checks.append('Complete registry and strictly positive-only 00 configurations')
    for rel in ['code/train.py','code/dataloader.py','code/utils/features.py','code/utils/graph_cache.py']:
        ast.parse((ROOT/rel).read_text())
    dump(WORK/'core_acceptance.json',{'status':'PASS','checks':checks,'failed':0})
    print(json.dumps({'status':'PASS','checks':checks},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

"""Require legacy input equality, real V14 replay and independent core contracts."""
from common import *
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from code.utils.config import load_config, Config
from code.utils.features import terminal_charge_adjustment,calculate_property
from code.utils.scaling import fit_p_f_stats
from code.dataloader import load_or_build_cache,load_or_build_neg_cache,build_training_pairs,build_neg_pairs,precompute_neighbors
from external_metrics import panel,evaluate,KEYS,from_peptides


def reject(fn):
    try:fn()
    except (AssertionError,ValueError):return
    raise AssertionError('Invalid endpoint accepted')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,default=0);a=ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);torch.set_num_threads(2)
    assert Path(__import__('code').__file__).resolve()==RUNTIME/'code/__init__.py'
    assert sha(RUNTIME/'code/utils/features.py')==sha(ARCHIVE/'code/utils/features.py')
    # Legacy behaviour is deliberately frozen, not advertised as chemical truth.
    assert terminal_charge_adjustment('ACT','AMD')==1.0
    assert terminal_charge_adjustment('acetyl','AMD')==0.0
    reg=load(WORK/'run_registry.json')
    cfg=load_config(next(r['config_path'] for r in reg if r['label']=='factor_P11' and r['seed']==123))
    print('Build fresh frozen-V14 positive/negative graphs',flush=True)
    pos=load_or_build_cache(cfg,ROOT);neg=load_or_build_neg_cache(cfg,ROOT)
    oldroot=ROOT/'artifacts/cache_ll37_holdout_cfu'
    oldp=oldroot/'mic_graph_cache_len50_seed123_cfu_v1_trainonly_rawprops_aa_seq1to20_nocoord.pt'
    oldn=oldroot/'neg_graph_cache_len50_rawprops_aa_seq1to20_nocoord_uniprotnegfullpool.pt'
    refs=[]
    for new,path,gn,rn,n in [(pos,oldp,'graphs','rows',7217),(neg,oldn,'neg_graphs','neg_rows',7860)]:
        old=torch.load(path,map_location='cpu',weights_only=False)
        assert len(new[gn])==len(old[gn])==n
        cols=['sequence','n_terminus','c_terminus','cfu_group','label']
        assert new[rn][cols].fillna('').equals(old[rn][cols].fillna(''))
        for i,(x,y) in enumerate(zip(new[gn],old[gn])):
            for key in ['x_s','edge_index_s','seq','p_f','cfu_id']:
                assert torch.equal(getattr(x,key),getattr(y,key)),(path.name,i,key)
            assert x.gt==y.gt
        refs.append({'reference':str(path),'sha256':sha(path),'n_graphs':n,'all_model_tensors_equal':True})
    assert not set(pos['rows'].sequence)&set(neg['neg_rows'].sequence)
    splits=list(GroupKFold(5).split(pos['rows'],groups=pos['rows'].sequence))
    for f,(tr,va) in enumerate(splits,1):
        ref=fit_p_f_stats(pos['graphs'],tr)
        old=load(ROOT/f'runs_ablation/S4/S4_seed123/intermediate/p_feature_zscore_fold{f}.json')
        for key in ['mean','std']:assert np.allclose(ref[key],old[key],rtol=0,atol=1e-12),(f,key)
    print('All 15,077 graph inputs and five-fold property statistics match V14',flush=True)
    checks=[]
    c=Config(similarity_threshold=0.,max_train_pairs=100,pair_balance_num=100)
    rr=pd.DataFrame({'sequence':['A'*10,'C'*10]})
    assert build_training_pairs(rr,np.array([0.,1.]),c,123)==[]
    c.train_similarity_gate_enabled=False
    assert set(build_training_pairs(rr,np.array([0.,1.]),c,123))=={(0,1),(1,0)}
    checks.append('True gate-off includes negative NW scores')
    c=Config(similarity_threshold=.7,neg_pair_mode='joint')
    assert build_neg_pairs(pd.DataFrame({'sequence':['A'*15]}),pd.DataFrame({'sequence':['A'*10]}),c)==[(1,0)]
    assert build_neg_pairs(pd.DataFrame({'sequence':['A'*10]}),pd.DataFrame({'sequence':['A'*15]}),c)==[(0,1)]
    checks.append('Joint negative directions independently gated')
    c=Config(similarity_threshold=.7,validation_similarity_threshold=.7,use_cfu_protocol=False,use_neighbor_fallback=False)
    q=pd.DataFrame({'sequence':['A'*15]});anchors=pd.DataFrame({'sequence':['A'*10]})
    before=precompute_neighbors(anchors,q,c);c.similarity_threshold=.3
    assert precompute_neighbors(anchors,q,c)==before==[[]]
    checks.append('Training and validation gates independent')
    pep,edges=panel();p=pd.Series(np.log10(pep.MIC_E_coli_ug_mL.to_numpy()),index=pep.sequence)
    truth=from_peptides(p,unit='log10_ug_mL');met,_,_=evaluate(truth)
    assert abs(met['macro']['PCC']-1)<1e-12
    reject(lambda:evaluate(truth.iloc[:-1]));reject(lambda:evaluate(pd.concat([truth,truth.iloc[:1]])))
    bad=truth.copy();bad.loc[0,'y_pred']=np.nan;reject(lambda:evaluate(bad))
    bad=truth.copy();bad.y_true*=np.log10(2);reject(lambda:evaluate(bad))
    bad=truth.copy();bad.y_pred=0.;assert evaluate(bad)[0]['macro']['PCC'] is None
    assert evaluate(truth.sample(frac=1,random_state=1))[0]==met
    checks.append('Fixed endpoint, units, nonfinite/constant predictions, row ordering')
    for r in reg:
        if r['action']=='train':
            c=load(r['config_path']);assert c['validation_similarity_threshold']==.7
            if r['label'] in ['factor_P00','factor_S00']:
                assert c['neg_csv'] is None and c['cls_pretrain_epochs']==0 and not c['neg_train_inject']
    # Real five-fold inference against an independently archived V14 prediction.
    from tiger_external import predict_run
    manifest=load(ARCHIVE/'reports/S4_study_nature_v14/rse_prediction_replay_manifest.json')
    replay=[]
    for method,arm in [('TIGER-pair','S4'),('TIGER-single','s4_single_n1n2')]:
        ref=next(r for r in manifest['predictions'] if r['method']==method and r['seed']==123)
        path=ROOT/'reports/S4_study_nature_v14'/ref['cache_file'];assert sha(path)==ref['sha256']
        for source in ref['inputs']:assert sha(source['path'])==source['sha256']
        m,f,actual=predict_run(ROOT/f'runs_ablation/{arm}/{arm}_seed123')
        _,_,expected=evaluate(pd.read_csv(path))
        aligned=expected.merge(actual,on=KEYS,suffixes=('_ref','_new'),validate='one_to_one')
        err=float(np.max(np.abs(aligned.y_pred_ref-aligned.y_pred_new)))
        assert len(aligned)==2211 and err<1e-5,(method,err)
        dst=WORK/'baseline_replay'/method;dst.mkdir(parents=True,exist_ok=True)
        actual.to_csv(dst/'predictions.csv',index=False);dump(dst/'metrics.json',m)
        replay.append({'method':method,'seed':123,'max_abs_calibrated_error':err,'n_pairs':2211,'PCC':m['macro']['PCC'],'reference_sha256':sha(path)})
        print('V14 checkpoint value replay PASS',replay[-1],flush=True)
    assert pd.read_csv(WORK/'external_input_dictionary.csv').cfu_group.eq('1E5 - 1E6').all()
    dump(WORK/'input_and_replay_acceptance.json',{'status':'PASS','runtime_features_sha256':sha(RUNTIME/'code/utils/features.py'),
        'graph_references':refs,'fold_statistics_match_V14':True,'checks':checks,'replay':replay,
        'selection_rule':'input and numerical identity, no performance-based selection'})
    print('Input and replay preflight PASS',flush=True)

if __name__=='__main__':main()

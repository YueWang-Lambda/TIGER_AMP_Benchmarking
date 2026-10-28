"""Time an evaluated seed-123 checkpoint; validate every returned endpoint.

One warm-up and 20 repeats, resident models, live query featurization. No fit,
proxy checkpoint, or cached query embedding is permitted in the timed call.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/S4_study_nature_v15'
WORK=ROOT/'artifacts/repair_v15_v14input'
PKG=ROOT/'reports/methods_package_S4_v1/4_S4_benchmark/code'
TROOT=ROOT/'artifacts/repair_v15_v14input/timing'
DEFS={
 'Das':('external','t_das'),'DeepAMP':('external','t_deep_amp'),
 'BERT-AmPEP60':('external','t_bert_ampep'),'Deep-PIE':('external','t_deep_pie'),
 'LLAMP':('external','t_llamp'),'HydrAMP':('external','t_hydramp'),
 'EvoGradient':('external','t_evo'),'APEX':('external','t_apex'),
 'AMP-Designer':('external','t_ampd'),
 'HyperAMP':('new4_torch','t_hyperamp'),'SenseXAMP':('new4_torch','t_sensexamp'),
 'MBC-Attention':('new4_tf','t_mbc_attention'),'AMPlify':('new4_tf','t_amplify'),
 'AMPGen':('new7','t_ampgen'),'tsAMP':('new7','t_tsamp'),
 'ANIA':('new7','t_ania'),'GGN-ABPMIC':('new7','t_ggn'),
 'AMPpredictor':('new7','t_apppred'),'EIPpred':('new7','t_eippred')}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--method',required=True);ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--acceptance-only',action='store_true');a=ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    method=a.method;tf=method in ['MBC-Attention','AMPlify'];cpu=method in ['AMPActiPred','EIPpred']
    torch=None
    if not tf and method!='AMPActiPred':
        import torch
        torch.set_num_threads(2)
    dest=TROOT/method;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'result.json').exists() and not a.acceptance_only:raise RuntimeError('Accepted timing already exists; do not overwrite')
    edges=pd.read_csv(OUT/'source_snapshot/canonical_pairs.csv');pep=pd.read_csv(OUT/'source_snapshot/external_peptides.csv')
    assert len(edges)==2211 and len(pep)==208
    seq=pep.sequence.tolist();pairs=list(zip(edges.anchor_sequence,edges.query_sequence))
    reference=pd.read_csv(OUT/'source_snapshot/predictions'/('tiger' if method.startswith('TIGER') else 'external')/
        ((('factor_P11' if method=='TIGER-pair' else 'factor_S11')+'_seed123.csv') if method.startswith('TIGER') else method+'__with_negatives_seed123.csv'))
    reference=edges[['family','anchor_sequence','query_sequence']].merge(reference,on=['family','anchor_sequence','query_sequence'],validate='one_to_one').y_pred.to_numpy(float)
    job=ROOT/'artifacts/repair_v15/external_runs'/method/'seed123'
    if method=='AMPLiT':job=ROOT/'runs_repair_v15/AMPLiT/with_negatives/seed123'
    expected_peptides=None
    if not method.startswith('TIGER'):
        q=pd.read_csv(job/'predictions.csv');expected_peptides=q.set_index('sequence').iloc[:,0].reindex(seq).to_numpy(float)
    captured={};checks=[];times=[]
    # Inherited, prespecified numerical tolerances for live PLM/R re-extraction.
    pep_tol={'ANIA':1e-2,'GGN-ABPMIC':2e-2}.get(method,1e-3)
    def capture(preds,actual=None):
        assert set(preds)==set(seq) and len(preds)==208
        independent=np.array([(float(preds[u])-float(preds[v]))/np.log10(2) for u,v in pairs])
        if actual is None:actual=independent
        captured.update(peptides=np.array([preds[s] for s in seq],float),pairs=np.asarray(actual,float),independent=independent)
        return actual
    def validate():
        v=captured['pairs'];assert v.shape==(2211,) and np.isfinite(v).all()
        if not method.startswith('TIGER'):
            assert np.allclose(v,captured['independent'],rtol=0,atol=1e-9),'Pair algebra/unit/order failure'
            err=np.abs(captured['peptides']-expected_peptides)
            assert err.max()<=pep_tol,(method,'checkpoint mismatch',float(err.max()),pep_tol)
            pair_tol=2*pep_tol/np.log10(2)+1e-9
        else:err=np.abs(v-reference);pair_tol=1e-5
        assert np.max(np.abs(v-reference))<=pair_tol,(method,'pair mismatch',float(np.max(abs(v-reference))))
        checks.append({'max_abs_peptide_or_tiger_pair_diff':float(err.max()),'mean_abs_diff':float(err.mean()),
                       'max_abs_pair_diff':float(np.max(abs(v-reference))),'n_pair':2211,'finite':True,'independent_pair_algebra':True})
    def sync():
        if torch is not None and not cpu:torch.cuda.synchronize()
    def timed(fn,reps=20):
        fn();sync();validate();print('WARMUP PASS',method,checks[-1],flush=True)
        for rep in range(0 if a.acceptance_only else 20):
            captured.clear();sync();start=time.perf_counter();fn();sync();elapsed=time.perf_counter()-start
            validate();times.append(elapsed)
            print('REPEAT',rep+1,elapsed,flush=True)
        return {'repeats_s':times,'median_s':float(np.median(times)) if times else None}
    checkpoint_files=[]
    if method in DEFS:
        suite,function=DEFS[method];src=PKG/f'eval_cost_timing_{suite}.py'
        text=src.read_text().replace("ROOT = Path(__file__).resolve().parents[1]",f'ROOT = Path({str(ROOT)!r})')
        text=text.replace("Path('/tmp/ggn_timing_work')",f'Path({str(dest/"ggn_live")!r})')
        if method=='EIPpred':text=text.replace("'sota_models/eippred_neg/rf_s123.joblib'","'sota_models/eippred_neg/model.joblib'")
        # Generated task drivers must precede their older workspace counterparts.
        ws=job/'workspace';sb=ws/'baseline/sota_bench';br=ws/'runs_review/benchmark_restricted'
        generated=dest/'timing_driver.py';generated.write_text(text)
        mod=types.ModuleType('timing_driver');mod.__file__=str(generated)
        exec(compile(text,str(generated),'exec'),mod.__dict__)
        mod.ROOT=ROOT;mod.SOTA=sb;mod.SB=sb;mod.OLD=br/'sota_models';mod.BR=br;mod.CKPT=br/'sota_models'
        mod.PEP_CSV=OUT/'source_snapshot/external_peptides.csv'
        if tf:mod.REPO_MBC=sb/'mbc_attention_repo';mod.REPO_AMP=sb/'amplify_repo'
        sys.path.insert(0,str(sb));sys.path.insert(0,str(ws))
        if method=='HydrAMP' and 'code' in sys.modules and not hasattr(sys.modules['code'],'__path__'):
            del sys.modules['code']
        # Endpoint setup is static; all differencing still runs in each timed call.
        mod.PAIRS=pairs
        original=mod.pair_delta
        def endpoint(preds):return capture(preds,original(preds))
        mod.pair_delta=endpoint;mod.timed=timed
        checkpoint_files=[p for p in (br/'sota_models').rglob('*') if p.is_file()]
        assert checkpoint_files
        if method=='EIPpred':assert (br/'sota_models/eippred_neg/model.joblib').exists()
        getattr(mod,function)()
    elif method=='AMPLiT':
        sys.path.insert(0,str(ROOT/'scripts/repair_v15'))
        from amplit import Predictor
        model=Predictor(job)
        def run():
            sequences=pd.read_csv(OUT/'source_snapshot/external_peptides.csv').sequence.tolist()
            capture(dict(zip(sequences,model.predict(sequences))))
        checkpoint_files=[job/'model.pt',job/'word_vectors.npy',job/'encoder.json']
        timed(run)
    elif method=='AMPActiPred':
        import io
        sb=job/'workspace/baseline/sota_bench';br=job/'workspace/runs_review/benchmark_restricted/sota_models/ampacti_neg'
        sys.path.insert(0,str(sb/'AMPActiPred'));os.chdir(sb/'AMPActiPred')
        from predict_lib import construct_features
        from deepforest import CascadeForestRegressor
        model=CascadeForestRegressor();model.load(str(br/'model'))
        columns=json.loads((br/'feature_columns.json').read_text())
        def run():
            sequences=pd.read_csv(OUT/'source_snapshot/external_peptides.csv').sequence.tolist()
            feats=construct_features(pd.DataFrame({'Id':[f'p{i}' for i in range(208)],'Sequence':sequences}))
            # The evaluated driver consumes pandas CSV-round-tripped descriptors.
            # Deep-forest bin boundaries can amplify sub-ULP feature differences.
            # Reproduce that numeric representation, including its cost, without
            # substituting the cached query features or relaxing acceptance.
            feats=pd.read_csv(io.StringIO(feats.to_csv(index=False)))
            capture(dict(zip(sequences,np.asarray(model.predict(feats[columns].values)).reshape(-1))))
        checkpoint_files=[p for p in (br/'model').rglob('*') if p.is_file()]
        timed(run)
    elif method.startswith('TIGER'):
        sys.path.insert(0,str(ROOT/'scripts/repair_v15_v14input'))
        # Resolve the isolated V14 feature implementation before importing TIGER.
        sys.path.insert(0,str(WORK/'runtime'))
        if 'code' in sys.modules and not hasattr(sys.modules['code'],'__path__'):del sys.modules['code']
        import tiger_external as te
        runpath=(ROOT/'runs_repair_v15_v14input/factor_P11/factor_P11_seed123') if method=='TIGER-pair' else ROOT/'runs_ablation/s4_single_n1n2/s4_single_n1n2_seed123'
        # Registered run paths, rather than inferred directory conventions.
        label='factor_P11' if method=='TIGER-pair' else 's4_single_n1n2'
        reg=json.loads((WORK/'run_registry.json').read_text());runpath=Path(next(r['run_dir'] for r in reg if r['label']==label and r['seed']==123))
        cfg=te.load_config(runpath/'config.json');device=torch.device('cuda:0')
        models=[te.load_checkpoint_model(runpath/'checkpoints'/f'fold{f}_best.pt',cfg,device)[0] for f in range(1,6)]
        pf=[json.loads((runpath/'intermediate'/f'p_feature_zscore_fold{f}.json').read_text()) for f in range(1,6)]
        nf=[json.loads((runpath/'intermediate'/f'node_feature_zscore_fold{f}.json').read_text()) for f in range(1,6)] if cfg.node_feature_scaling=='zscore' else [None]*5
        cal=json.loads((runpath/'results/calibrator.json').read_text())
        def run():
            roster=pd.read_csv(OUT/'source_snapshot/external_peptides.csv')
            rows=roster[['sequence','n_terminus','c_terminus','panel']].copy();rows['cfu_group']='1E5 - 1E6';rows[te.TARGET_COL]=roster.MIC_E_coli_ug_mL
            rows['pdb_path']=[str(ROOT/te.PDB_DIRS[r.panel]/f'{r.sequence}.pdb') for r in rows.itertuples()]
            graphs,kept,_=te.build_graphs(rows,cfg);assert len(graphs)==208
            preds=[]
            for model,scaler,node_scaler in zip(models,pf,nf):
                te.apply_p_f_stats(graphs,scaler)
                if node_scaler is not None:te.apply_node_x_stats(graphs,node_scaler)
                with torch.inference_mode():pred,_=te.eval_explicit_pairs(model,graphs,kept,edges,cfg,device)
                pred.family=pred.family.str.rstrip('/')
                aligned=edges[['family','anchor_sequence','query_sequence']].merge(pred,on=['family','anchor_sequence','query_sequence'],validate='one_to_one')
                preds.append(aligned.y_pred_delta_log2_anchor_minus_query.to_numpy(float))
            captured['pairs']=np.mean(preds,axis=0)*cal['slope']+cal['intercept']
        checkpoint_files=[runpath/'checkpoints'/f'fold{f}_best.pt' for f in range(1,6)]+[runpath/'results/calibrator.json']
        timed(run)
    else:raise ValueError(method)
    if not a.acceptance_only:pd.DataFrame({'repeat':range(1,21),'seconds':times}).to_csv(dest/'repeats.csv',index=False)
    result={'status':'PASS','method':method,'seed':123,'condition':'with_negatives','repeats_s':times,
            'median_s':float(np.median(times)) if times else None,'min_s':float(min(times)) if times else None,'max_s':float(max(times)) if times else None,
            'device':'CPU' if cpu else f'GPU:{a.gpu} (RTX 4090 D)','python':sys.executable,
            'checkpoint_files':{str(p):sha(p) for p in checkpoint_files},'checks':checks,
            'peptide_tolerance':pep_tol if not method.startswith('TIGER') else None,
            'protocol':'1 warm-up + 20 repeats; loaded evaluated seed123; live features + 2211 log2 pairs; model loading excluded; validation outside clock'}
    result['acceptance_only']=a.acceptance_only
    dump(dest/('preflight.json' if a.acceptance_only else 'result.json'),result);print('TIMING ACCEPTED',method,result['median_s'],flush=True)

if __name__=='__main__':main()

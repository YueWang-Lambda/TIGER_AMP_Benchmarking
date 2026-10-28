"""Run a registered external driver in an isolated, reproducible workspace.

The upstream architectures and optimization schedules are retained. Generated
source overlays only change input/output paths, resource limits, negative-pool
cardinality checks, and model persistence. Each seed gets its own legacy output
tree, so drivers that use best.pt/fold1.pt cannot overwrite another seed.
"""
import argparse
import ast
import subprocess
import time
from common import *
from external_prepare import SOTA,DEST,link

EXTRA_ROOT=ROOT.parents[1]
METHODS={
 'Das':('retrain_das_regressor.py','seed','ccseg'),
 'LLAMP':('train_sota_bench_neg.py','llamp','ccseg'),
 'BERT-AmPEP60':('train_sota_bench_neg.py','bert_ampep','ccseg'),
 'Deep-PIE':('train_sota_bench_neg.py','deep_pie','ccseg'),
 'HydrAMP':('retrain_hydramp_regressor_neg.py','seed','ccseg'),
 'MBC-Attention':('retrain_mbc_attention_neg.py','tf_gpu','mbc_att_gpu_v15'),
 'HyperAMP':('retrain_hyperamp_neg.py','multi','ccseg'),
 'SenseXAMP':('retrain_sensexamp_neg.py','multi','ccseg'),
 'AMPlify':('retrain_amplify_regr_neg.py','tf_gpu','mbc_att_gpu_v15'),
 'EIPpred':('retrain_eippred_neg.py','cpu','ccseg'),
 'AMPActiPred':('retrain_ampacti_neg.py','cpu','ampacti'),
 'AMPGen':('retrain_ampgen_neg.py','seed','ccseg'),
 'AMPpredictor':('retrain_amppredictor_neg.py','seed','ccseg'),
 'ANIA':('retrain_ania_neg.py','seed','ania'),
 'tsAMP':('retrain_tsamp_neg.py','seed','ccseg'),
 'GGN-ABPMIC':('retrain_ggn_neg.py','seed','ccseg'),
 'EvoGradient':('retrain_evo_strict.py','seed','ccseg'),
 'DeepAMP':('retrain_deep_amp.py','deepamp','ccseg'),
 'APEX':('apex-main/train_apex_lrfix_neg.py','ensemble','ccseg'),
 'AMP-Designer':('AMP-Designer/train_ampd_restricted_neg.py','ensemble','ccseg'),
}


def tf_gpu_profile():
    profile=load(WORK/'tf_gpu_runtime.json')
    assert profile['status']=='PASS'
    assert profile['execution']=={'AMPlify':'eager','MBC-Attention':'graph'}
    assert profile['reservation_mib']>=profile['memory_cap_mib']+1024
    assert sha(profile['preflight'])==profile['preflight_sha256']
    assert load(profile['preflight'])['status']=='PASS'
    assert sha(ROOT/'scripts/repair_v15/tf_gpu_launcher.py')==profile['launcher_sha256']
    return profile


def change_once(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new)


def patched_source(src,jobroot):
    text=src.read_text()
    # Longest prefixes first. These are literal source-path replacements,
    # not shell interpolation or a global filesystem monkeypatch.
    prefixes=[(str(ROOT),str(jobroot)),
              (str(EXTRA_ROOT/'baseline/sota_bench'),str(jobroot/'baseline/sota_bench')),
              (str(EXTRA_ROOT/'baseline/AMP-potency-prediction-EvoGradient'),str(jobroot/'baseline/AMP-potency-prediction-EvoGradient')),
              (str(EXTRA_ROOT/'TIGER'),str(jobroot))]
    # Placeholders prevent a later legacy-prefix replacement from matching a
    # newly generated path (notably TIGER versus TIGER_v2).
    for i,(old,new) in enumerate(prefixes):text=text.replace(old,f'__V15_PATH_{i}__')
    for i,(old,new) in enumerate(prefixes):text=text.replace(f'__V15_PATH_{i}__',new)
    text=text.replace('len(neg) == 7962','len(neg) == 7949')
    text=text.replace('7962 - len(neg)','7949 - len(neg)')
    text=text.replace('torch.set_num_threads(4)','torch.set_num_threads(2)')
    text=text.replace('n_jobs=8','n_jobs=2')
    if src.name=='retrain_hydramp_regressor_neg.py':
        text=change_once(text,'    np.random.seed(SEED)','    np.random.seed(SEED)\n    torch.manual_seed(SEED)')
    if src.name=='retrain_eippred_neg.py':
        text=change_once(text,"    out.to_csv(out_dir / name)","    out.to_csv(out_dir / name)\n    import joblib\n    joblib.dump(clf, out_dir / 'model.joblib')\n    (out_dir / 'feature_columns.json').write_text(__import__('json').dumps(cols))")
    if src.name=='retrain_ampacti_neg.py':
        text=change_once(text,'CascadeForestRegressor(random_state=args.seed)','CascadeForestRegressor(random_state=args.seed, n_jobs=2)')
        text=change_once(text,'    o.to_csv(out_dir / name)',"    o.to_csv(out_dir / name)\n    model.save(str(out_dir / 'model'))\n    (out_dir / 'feature_columns.json').write_text(__import__('json').dumps(xcols))")
    if src.name=='retrain_deep_amp.py':
        # DeepAMP-CNN is the registered benchmark entry; no new RNN comparison.
        text=change_once(text,"[('cnn', CNNRegressor), ('rnn', RNNRegressor)]","[('cnn', CNNRegressor)]")
    ast.parse(text,filename=str(src))
    return text


def prepare(method,seed):
    job=WORK/'external_runs'/method/f'seed{seed}';jobroot=job/'workspace'
    sb=jobroot/'baseline/sota_bench';sb.mkdir(parents=True,exist_ok=True)
    assert load(DEST/'core_inputs.json')['status']=='PASS'
    source_hashes={}
    for src in SOTA.iterdir():
        target=sb/src.name
        if src.suffix=='.py':
            # The parent-repository EvoGradient driver contains the updated
            # seed-specific export and no-negative control; archive it here.
            if src.name=='retrain_evo_strict.py':src=EXTRA_ROOT/'baseline/sota_bench'/src.name
            code=patched_source(src,jobroot);target.write_text(code)
            source_hashes[str(src)]={'source':sha(src),'generated':sha(target)}
        elif src.name in ['apex-main','AMP-Designer']:
            target.mkdir(exist_ok=True)
            for child in src.iterdir():
                if child.name in ['train_apex_lrfix_neg.py','train_ampd_restricted_neg.py']:
                    (target/child.name).write_text(patched_source(child,jobroot))
                    source_hashes[str(child)]={'source':sha(child),'generated':sha(target/child.name)}
                else:link(child,target/child.name)
        elif (DEST/src.name).exists():link(DEST/src.name,target)
        elif src.name!='runs_review':link(src,target)
    link(EXTRA_ROOT/'baseline/AMP-potency-prediction-EvoGradient',jobroot/'baseline/AMP-potency-prediction-EvoGradient')
    link(ROOT/'code',jobroot/'code')
    rr=jobroot/'runs_review';(rr/'benchmark_restricted').mkdir(parents=True,exist_ok=True)
    link(ROOT/'runs_review/final11_peptides.csv',rr/'final11_peptides.csv')
    link(PKG/'4_S4_benchmark/data/train_restricted4929_tiger.csv',rr/'benchmark_restricted/train_restricted4929_tiger.csv')
    # Persist canonical target identity independently of each driver's export.
    import pandas as pd
    pos=pd.read_csv(PKG/'4_S4_benchmark/data/train_restricted4929_tiger.csv').groupby('sequence').MIC_Escherichia_coli.mean()
    neg=pd.read_csv(sb/'data_restricted/uniprot_negatives.csv')
    assert len(neg)==7949 and not set(neg.sequence)&set(pos.index)
    if method=='HyperAMP':assert load(DEST/'hyperamp_cache/identity.json')['status']=='PASS'
    if method=='AMPpredictor':assert load(DEST/'emb_cache/amppredictor_graphs.identity.json')['status']=='PASS'
    dump(job/'source_manifest.json',source_hashes)
    dump(job/'input_manifest.json',{'method':method,'seed':seed,'condition':'with_negatives','positive_unique':len(pos),
         'negative_pool':len(neg),'negative_sha256':sha(sb/'data_restricted/uniprot_negatives.csv'),
         'exact_overlap':0,'method_specific_filters_retained':True,'workspace':str(jobroot)})
    return job,jobroot,sb


def command(method,seed,gpu,jobroot,sb):
    script,kind,env=METHODS[method]
    cmd=[f'/home/ubuntu/anaconda3/envs/{env}/bin/python',str(sb/script)]
    if kind=='tf_gpu':
        profile=tf_gpu_profile()
        return [cmd[0],str(ROOT/'scripts/repair_v15/tf_gpu_launcher.py'),
            '--gpu',str(gpu),'--memory-mib',str(profile['memory_cap_mib']),
            '--execution',profile['execution'][method],
            '--runtime-record',str(jobroot.parent/'runtime_environment.json'),
            '--target',str(sb/script),'--','--seeds',f'{seed}-{seed}']
    elif kind in ['seed','cpu']:cmd+=['--seed',str(seed)]
    elif kind in ['llamp','bert_ampep','deep_pie']:cmd+=['--model',kind,'--mode','with_neg','--seed',str(seed)]
    elif kind in ['multi','tf']:
        cmd+=['--seeds',f'{seed}-{seed}']
        if kind=='multi':cmd+=['--gpu',str(gpu)]
        else:cmd+=['--cpu']
    elif kind=='deepamp':cmd+=['--seeds',str(seed)]
    elif kind=='ensemble':cmd+=['--mode','with_neg','--seeds','1','--seed-base',str(seed)]
    return cmd


def collect(method,seed,job,jobroot):
    import numpy as np
    import pandas as pd
    from external_metrics import panel,from_peptides,evaluate
    pep,_=panel();expected=set(pep.sequence);candidates=[]
    for p in (jobroot/'runs_review').rglob('*preds.csv'):
        d=pd.read_csv(p)
        seqcol='sequence' if 'sequence' in d else d.columns[0]
        if len(d)!=208 or set(d[seqcol])!=expected or not d[seqcol].is_unique:continue
        cols=[c for c in d if 'log10mic' in c]
        if len(cols)!=1:continue
        candidates.append((p,pd.Series(d[cols[0]].to_numpy(float),index=d[seqcol]),cols[0]))
    assert candidates,('No complete peptide prediction export',method,seed)
    values=candidates[0][1].reindex(pep.sequence)
    for p,s,c in candidates[1:]:assert np.allclose(values,s.reindex(pep.sequence),rtol=0,atol=1e-10),(p,method)
    if method=='EvoGradient':
        # Native outputs are log10 micromolar; conversion must be per peptide.
        from importlib.util import spec_from_file_location,module_from_spec
        src=jobroot/'baseline/sota_bench/retrain_evo_strict.py'
        spec=spec_from_file_location('repair_evo_unit',src);mod=module_from_spec(spec);spec.loader.exec_module(mod)
        values=values+np.log10(np.array([mod.mw(s) for s in values.index])/1000.)
    values.name='log10mic';values.index.name='sequence'
    values.to_csv(job/'predictions.csv')
    met,fam,pairs=evaluate(from_peptides(values,unit='log10_ug_mL'))
    dump(job/'metrics.json',met);fam.to_csv(job/'family_metrics.csv',index=False);pairs.to_csv(job/'pair_predictions.csv',index=False)
    return [str(p.relative_to(jobroot)) for p,_,_ in candidates]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--method',choices=list(METHODS),required=True)
    ap.add_argument('--seed',type=int,required=True);ap.add_argument('--gpu',type=int,default=0)
    ap.add_argument('--prepare-only',action='store_true');a=ap.parse_args()
    job=WORK/'external_runs'/a.method/f'seed{a.seed}'
    if (job/'DONE.json').exists():raise RuntimeError('Completed external run exists')
    job,jobroot,sb=prepare(a.method,a.seed)
    if a.prepare_only:print('Prepared',a.method,a.seed);return
    release=load(WORK/'external_training_release.json')
    assert release['status']=='PASS' and a.method in release['verified_affected_methods'],release
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
          OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1',
          PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(jobroot),R_HOME='/home/ubuntu/anaconda3/envs/ania/lib/R')
    if METHODS[a.method][1] in ['cpu','tf']:env['CUDA_VISIBLE_DEVICES']='-1'
    # TensorFlow's pdb import needs the stdlib `code` module; putting the TIGER
    # package root on PYTHONPATH would import PyTorch into this TF-only env.
    if METHODS[a.method][1] in ['tf','tf_gpu']:env['PYTHONPATH']=str(sb)
    cmd=command(a.method,a.seed,a.gpu,jobroot,sb)
    dump(job/'invocation.json',{'cmd':cmd,'cwd':str(sb),'env':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','PYTHONPATH']}})
    t=time.time()
    with (job/'training.log').open('a') as log:subprocess.run(cmd,cwd=sb,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    exports=collect(a.method,a.seed,job,jobroot)
    done={'status':'PASS','method':a.method,'seed':a.seed,'condition':'with_negatives',
          'elapsed_seconds':time.time()-t,'raw_exports':exports,'source_manifest_sha256':sha(job/'source_manifest.json')}
    if METHODS[a.method][1]=='tf_gpu':
        runtime=load(job/'runtime_environment.json')
        assert runtime['physical_gpu_index']==a.gpu
        assert runtime['execution']==tf_gpu_profile()['execution'][a.method]
        assert runtime['tf32_enabled'] is False
        done.update(training_device=f'GPU:{a.gpu}',training_environment=METHODS[a.method][2],
            tf_execution=runtime['execution'],runtime_environment_sha256=sha(job/'runtime_environment.json'))
    dump(job/'DONE.json',done)
    print('External DONE',a.method,a.seed,flush=True)


if __name__=='__main__':main()

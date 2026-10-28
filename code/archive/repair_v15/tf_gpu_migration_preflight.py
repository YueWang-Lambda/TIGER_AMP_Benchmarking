"""Accept the bounded GPU runtime before moving formal TF jobs to the GPU queue."""
import numpy as np
import pandas as pd
from common import *
from external_metrics import from_peptides, evaluate


def main():
    migration=Path(load(WORK/'tf_gpu_migration/current.json')['directory'])
    checks={}
    for method in ['AMPlify','MBC-Attention']:
        ref=WORK/'tf_gpu_repair';new=migration/'validation'
        info=load(new/f'{method}_gpu.json')
        assert info['status']=='PASS' and info['gpu_logical_memory_mib']==1024
        assert info['weights_updated'] and info['checkpoint_hashes_unchanged']
        assert info['functions_run_eagerly']==(method=='AMPlify')
        cpu=pd.read_csv(ref/f'{method}_cpu_predictions.csv').set_index('sequence').log10mic
        gpu=pd.read_csv(new/f'{method}_gpu_predictions.csv').set_index('sequence').log10mic
        assert set(cpu.index)==set(gpu.index) and len(gpu)==208 and gpu.index.is_unique
        delta=float(abs(cpu-gpu.reindex(cpu.index)).max());assert delta<=1e-4
        frames={dev:from_peptides(pred,unit='log10_ug_mL') for dev,pred in [('cpu',cpu),('gpu',gpu)]}
        metrics={dev:evaluate(frame)[0] for dev,frame in frames.items()}
        pairdiff=float(abs(frames['cpu'].y_pred-frames['gpu'].y_pred).max());assert pairdiff<=7e-4
        mdiff={m:abs(metrics['cpu']['macro'][m]-metrics['gpu']['macro'][m]) for m in ['PCC','RSE']}
        assert all(v<=1e-4 for v in mdiff.values())
        a=np.load(ref/f'{method}_cpu_gradients.npz');b=np.load(new/f'{method}_gpu_gradients.npz')
        assert a.files==b.files
        maxerr=0.;err2=0.;base2=0.
        for key in a.files:
            x=a[key].astype(float);y=b[key].astype(float);assert x.shape==y.shape
            maxerr=max(maxerr,float(abs(x-y).max()));err2+=float(((x-y)**2).sum());base2+=float((x*x).sum())
        relative=float(np.sqrt(err2/max(base2,1e-30)));assert maxerr<=1e-4 and relative<=1e-3
        checks[method]={'status':'PASS','max_log10mic_diff':delta,'max_log2_pair_diff':pairdiff,
            'macro_diff':mdiff,'gradient_max_diff':maxerr,'gradient_relative_l2':relative,
            'memory_cap_mib':1024,'tensor_peak_mib':info['tf_memory_bytes']['peak']/2**20}
    originals=load(migration/'completed_preserved_hashes.json')
    for folder,files in originals.items():
        for name,expected in files.items():assert sha(ROOT/folder/name)==expected,(folder,name)
    preflight=migration/'preflight.json'
    dump(preflight,{'status':'PASS','methods':checks,'preserved_completed_runs':len(originals),
        'all_completed_top_level_files_byte_identical':True})
    seeds=[123,120,121,122,124,125,126,127,128,129]
    first=[]
    for method in ['AMPlify','MBC-Attention']:
        unfinished=[s for s in seeds if not (WORK/f'external_runs/{method}/seed{s}/DONE.json').exists()]
        if unfinished:first.append(f'{method}_with_negatives_seed{unfinished[0]}')
    dump(WORK/'tf_gpu_runtime.json',{'status':'PASS',
        'execution':{'AMPlify':'eager','MBC-Attention':'graph'},
        'environment':'mbc_att_gpu_v15','memory_cap_mib':1024,'reservation_mib':2048,
        'preflight':str(preflight),'preflight_sha256':sha(preflight),
        'launcher_sha256':sha(ROOT/'scripts/repair_v15/tf_gpu_launcher.py'),
        'first_jobs':first,'migration_directory':str(migration),
        'completed_cpu_seeds_preserved':True,'interrupted_seeds_restart_from_initialization':True})
    from external_one import command,METHODS
    from campaign_supervisor import make_jobs
    for method in ['AMPlify','MBC-Attention']:
        jobroot=WORK/'external_runs'/method/'seed126/workspace'
        cmd=command(method,126,0,jobroot,jobroot/'baseline/sota_bench')
        assert '--cpu' not in cmd and '--runtime-record' in cmd
        assert cmd[cmd.index('--execution')+1]==('eager' if method=='AMPlify' else 'graph')
        assert METHODS[method][1]=='tf_gpu'
    jobs=make_jobs();tfjobs=[j for j in jobs if j.get('method') in ['AMPlify','MBC-Attention']]
    assert len(jobs)==691 and len(tfjobs)==20
    assert all(j['resource']=='gpu' and j['reserve_mib']==2048 for j in tfjobs)
    assert sum(not Path(j['completion']).exists() for j in tfjobs)==12
    dump(migration/'queue_preflight.json',{'status':'PASS','jobs':691,'tf_jobs':20,
         'pending_gpu_tf_jobs':12,'first_jobs':first,'no_tf_cpu_jobs':True})
    print({'status':'PASS','checks':checks,'first_jobs':first})


if __name__=='__main__':main()

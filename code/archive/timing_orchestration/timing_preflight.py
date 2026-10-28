"""Sequential idle-device timing campaign; failures remain hard failures."""
import json,os,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parents[2];D=R/'artifacts/repair_v15_v14input/timing';D.mkdir(exist_ok=True)
PY='/home/ubuntu/anaconda3/envs/ccseg/bin/python';script=R/'scripts/analysis_v15/timing_one.py'
methods=['Das','DeepAMP','APEX','AMPActiPred','EIPpred','TIGER-pair','TIGER-single','AMPLiT','MBC-Attention','AMPlify','LLAMP','BERT-AmPEP60','Deep-PIE','HydrAMP','EvoGradient','AMP-Designer','HyperAMP','SenseXAMP','AMPGen','tsAMP','AMPpredictor','ANIA']
def state(**kw):
    p=D/'preflight_state.json';t=p.with_suffix('.tmp');t.write_text(json.dumps({'time':time.time(),'pid':os.getpid(),**kw},indent=2));os.replace(t,p)
completed=[]
for method in methods:
    if (D/method/'preflight.json').exists():
        assert json.loads((D/method/'preflight.json').read_text())['status']=='PASS';completed.append(method);continue
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='')
    cmd=[PY,str(script),'--method',method,'--gpu','1','--acceptance-only']
    if method=='AMPActiPred':cmd[0]='/home/ubuntu/anaconda3/envs/ampacti/bin/python'
    if method in ['MBC-Attention','AMPlify']:
        cmd=['/home/ubuntu/anaconda3/envs/mbc_att_gpu_v15/bin/python',str(R/'scripts/repair_v15/tf_gpu_launcher.py'),'--gpu','1','--memory-mib','4096','--execution','eager' if method=='AMPlify' else 'graph','--runtime-record',str(D/(method+'_runtime.json')),'--target',str(script),'--','--method',method,'--gpu','1','--acceptance-only']
    logfile=R/f'reports/S4_study_nature_v15/preflight_{method}.log'
    with logfile.open('a') as log:
        child=subprocess.Popen(cmd,env=env,cwd='/tmp' if method in ['MBC-Attention','AMPlify'] else R,stdout=log,stderr=subprocess.STDOUT)
        state(status='RUNNING',active=method,child_pid=child.pid,completed=completed,total=22)
        rc=child.wait()
    if rc or not (D/method/'preflight.json').exists():
        state(status='FAILED',failed=method,rc=rc,completed=completed,total=22);raise RuntimeError((method,rc))
    completed.append(method)
state(status='COMPLETE',completed=completed,total=22)

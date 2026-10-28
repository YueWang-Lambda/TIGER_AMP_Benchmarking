"""Durable smoke→formal→reuse campaign, respecting foreign GPU allocations."""
import argparse
import csv
import fcntl
import signal
import subprocess
import time
from common import *

SMOKE_LABELS=['factor_P00','factor_P11','factor_S00','S4_sim03','s4_gate_off']

def running(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False

def gpu_capacity(active):
    def query(flag):return subprocess.check_output(['nvidia-smi',flag,'--format=csv,noheader,nounits'],text=True)
    gpu={};total={}
    for row in csv.reader(query('--query-gpu=index,uuid,memory.total').splitlines()):
        i=int(row[0]);gpu[row[1].strip()]=i;total[i]=int(row[2])
    foreign={0:[],1:[]};owned={p.pid for _,_,p,_ in active}
    for row in csv.reader(query('--query-compute-apps=gpu_uuid,pid,used_memory').splitlines()):
        if len(row)!=3:continue
        try:
            pid=int(row[1]);group=os.getpgid(pid);idx=gpu[row[0].strip()]
            if idx in foreign and group not in owned:foreign[idx].append((pid,int(row[2])))
        except ProcessLookupError:pass
    available=[]
    for g in [0,1]:
        own=sum(gg==g for _,gg,_,_ in active);count=own+len(foreign[g])
        reservation=own*3000+sum(max(3000,m) for _,m in foreign[g])
        if count<5 and reservation+3000<=total[g]-1200:available.append((count,reservation,g))
    return available,foreign

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['auto','smoke','formal'],default='auto');ap.add_argument('--adopt');a=ap.parse_args()
    check_freeze();assert load(WORK/'input_and_replay_acceptance.json')['status']=='PASS'
    home=WORK/'queue';home.mkdir(exist_ok=True)
    lock=(home/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    reg=load(WORK/'run_registry.json');active=[];failures=[];stopping=False
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def completion(r,stage):return WORK/'smoke'/r['task_id']/'DONE.json' if stage=='smoke' else WORK/'tiger_external'/r['task_id']/'EVALUATED.json'
    def passed(path):return path.exists() and load(path).get('status')=='PASS'
    def stage_tasks(stage):
        if stage=='smoke':return [next(r for r in reg if r['label']==label and r['seed']==123) for label in SMOKE_LABELS]
        training=[r for r in reg if r['action']=='train']
        # Launch all seed123 configurations early; each is pre-registered, not accuracy-selected.
        training.sort(key=lambda r:(r['seed']!=123,r['label']!='s4_gate_off',r['seed'],r['label']))
        return training+[r for r in reg if r['action']=='reuse']
    try:
        stages=['smoke','formal'] if a.stage=='auto' else [a.stage]
        for stage in stages:
            if stage=='formal':
                if not (WORK/'full_training_release.json').exists():
                    subprocess.run([PYTHON,str(SCRIPTS/'release.py')],cwd=ROOT,check=True)
                assert load(WORK/'full_training_release.json')['status']=='PASS'
            tasks=stage_tasks(stage);todo=[r for r in tasks if not passed(completion(r,stage))]
            if a.adopt:
                prev=load(a.adopt)
                if prev['stage']==stage:
                    class Adopted:
                        owned=False
                        def __init__(self,pid,path):self.pid=pid;self.path=path
                        def poll(self):
                            if passed(self.path):return 0
                            return None if running(self.pid) else 1
                    for x in prev['active']:
                        r=next(r for r in todo if r['task_id']==x['id']);todo.remove(r)
                        active.append((r,x['gpu'],Adopted(x['pid'],completion(r,stage)),(home/f'{stage}_{r["task_id"]}.log').open('a')))
            def update(status,foreign):
                dump(home/'state.json',{'status':status,'stage':stage,'pid':os.getpid(),'updated':time.time(),
                    'stage_jobs':len(tasks),'stage_completed':sum(passed(completion(r,stage)) for r in tasks),
                    'formal_training_total':100,'formal_training_completed':sum(passed(Path(r['run_dir'])/'DONE.json') for r in reg if r['action']=='train'),
                    'remaining_to_launch':len(todo),'active':[{'id':r['task_id'],'gpu':g,'pid':p.pid,'action':r['action']} for r,g,p,_ in active],
                    'failures':failures,'foreign':foreign,'max_processes_per_gpu':5,'reserved_mib_per_job':3000})
            while todo or active:
                for item in list(active):
                    r,g,p,log=item;rc=p.poll()
                    if rc is None:continue
                    log.close();active.remove(item)
                    if rc or not passed(completion(r,stage)):
                        failures.append({'id':r['task_id'],'returncode':rc,'stage':stage});stopping=True
                while todo and not stopping:
                    available,foreign=gpu_capacity(active)
                    if not available:break
                    g=min(available)[2];r=todo.pop(0);log=(home/f'{stage}_{r["task_id"]}.log').open('a')
                    script='train_one.py' if stage=='smoke' else 'execute_job.py'
                    command=[PYTHON,str(SCRIPTS/script),'--id',r['task_id'],'--gpu',str(g)]
                    if stage=='smoke':command.append('--smoke')
                    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                    p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                    active.append((r,g,p,log));print('START',stage,r['task_id'],'GPU',g,'pid',p.pid,flush=True)
                _,foreign=gpu_capacity(active)
                update('DRAINING_AFTER_FAILURE' if failures else 'DRAINING' if stopping else 'RUNNING',foreign)
                if stopping and not active:break
                time.sleep(5)
            update('COMPLETE' if not failures and not todo else 'INCOMPLETE',{})
            if failures or todo:raise RuntimeError('Queue incomplete; inspect retained logs before restart')
        # This marks training/inference only; reports and idle-device timing remain separate.
        if 'formal' in stages:dump(WORK/'training_and_inference_complete.json',{'status':'PASS','trained':100,'reused':370,'alias_slots':20,'external_retained':220})
    finally:
        for r,g,p,log in active:
            if p.poll() is None and getattr(p,'owned',True):os.killpg(p.pid,signal.SIGTERM)
            log.close()

if __name__=='__main__':main()

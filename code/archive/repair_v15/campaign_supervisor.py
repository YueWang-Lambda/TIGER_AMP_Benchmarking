"""Resumable experiment queue with GPU process/memory and CPU concurrency limits.

Counts existing compute processes as occupied slots, including the initial
pilot supervisor. Uses process-group identity to avoid double-counting owned
training subprocesses. Stops launching on an error and retains all evidence.
"""
import argparse
import csv
import fcntl
import signal
import subprocess
import time
from common import *
from external_one import METHODS,tf_gpu_profile


class AdoptedProcess:
    """Monitor a detached worker after a supervised controller restart."""
    owned=False
    def __init__(self,pid,completion):self.pid=pid;self.completion=Path(completion)
    def poll(self):
        if self.completion.exists():return 0 if load(self.completion).get('status')=='PASS' else 1
        try:
            stat=Path(f'/proc/{self.pid}/stat').read_text()
            if stat.rsplit(')',1)[1].split()[0]=='Z':return 1
            os.kill(self.pid,0)
        except (ProcessLookupError,FileNotFoundError):return 1
        return None


def make_jobs():
    jobs=[{'id':'pilot_acceptance','kind':'pilot_acceptance','resource':'gpu','reserve_mib':3000,
           'completion':str(WORK/'full_training_release.json')}]
    # Start seed 123 first for each implementation; no accuracy-based retry.
    seeds=[123]+[s for s in range(120,130) if s!=123]
    for seed in seeds:
        for condition in ['without_negatives','with_negatives']:
            path=RUNS/'AMPLiT'/condition/f'seed{seed}'
            jobs.append({'id':f'AMPLiT_{condition}_seed{seed}','kind':'amplit','condition':condition,'seed':seed,
                 'resource':'gpu','reserve_mib':3500,'completion':str(path/'DONE.json')})
        for method,(_,kind,_) in METHODS.items():
            resource='cpu' if kind in ['cpu','tf'] else 'gpu'
            reserve=10000 if method in ['BERT-AmPEP60','Deep-PIE'] else 8000 if method in ['AMP-Designer','SenseXAMP','HyperAMP'] else 5000 if method in ['GGN-ABPMIC','AMPpredictor','LLAMP'] else 3000
            if kind=='tf_gpu':reserve=tf_gpu_profile()['reservation_mib']
            path=WORK/'external_runs'/method/f'seed{seed}'
            jobs.append({'id':f'{method}_with_negatives_seed{seed}','kind':'external','method':method,'seed':seed,
                  'resource':resource,'reserve_mib':reserve,'completion':str(path/'DONE.json')})
    # Alternate configurations/seeds as registered; completed pilots skip fitting.
    for r in load(WORK/'run_registry.json'):
        if r['action']=='alias':continue
        out=WORK/'tiger_external'/r['task_id']
        jobs.append({'id':r['task_id'],'kind':'tiger','task_id':r['task_id'],'resource':'gpu','reserve_mib':3000,
              'output':str(out),'completion':str(out/'EVALUATED.json')})
    assert len({j['id'] for j in jobs})==len(jobs)
    # User-requested migration: give the unfinished TF jobs the next eligible
    # GPU slots. Active unrelated workers are adopted and never pre-empted.
    first_tf=tf_gpu_profile().get('first_jobs',[])
    jobs.sort(key=lambda j:(0,first_tf.index(j['id'])) if j['id'] in first_tf else
              (1,0) if j.get('method') in ['AMPlify','MBC-Attention'] else (2,0))
    dump(WORK/'campaign_jobs.json',jobs)
    return jobs


def gpu_usage(active):
    q=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total','--format=csv,noheader,nounits'],text=True)
    uuids={};total={}
    for row in csv.reader(q.splitlines()):
        idx=int(row[0]);uuids[row[1].strip()]=idx;total[idx]=int(row[2])
    q=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    foreign={i:[] for i in [0,1]};owned={p.pid for j,g,p,f in active if j['resource']=='gpu'}
    for row in csv.reader(q.splitlines()):
        if len(row)!=3:continue
        idx=uuids[row[0].strip()];pid=int(row[1]);mem=int(row[2])
        try:group=os.getpgid(pid)
        except ProcessLookupError:continue
        if idx in foreign and group not in owned:foreign[idx].append((pid,mem))
    return foreign,total


def cpu_job_limit():
    """Allow scheduling-only CPU concurrency changes without touching workers."""
    path=WORK/'queue/campaign/resource_limits.json'
    value=load(path)['cpu_job_limit'] if path.exists() else 2
    if type(value) is not int or not 1<=value<=8:
        raise ValueError('cpu_job_limit must be an integer between 1 and 8')
    return value


def main():
    if (WORK/'CANCELLED_BY_V14_INPUT_FREEZE.json').exists():
        raise RuntimeError('ACT campaign superseded by approved V14-input branch; do not restart this queue')
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--adopt-state');a=ap.parse_args()
    home=WORK/'queue/campaign';home.mkdir(parents=True,exist_ok=True)
    lock=(home/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=make_jobs()
    if a.prepare_only:print('Registered',len(jobs),'jobs');return
    assert load(WORK/'smoke_acceptance.json')['status']=='PASS'
    assert load(WORK/'amplit_query_invariance.json')['status']=='PASS'
    active=[];failures=[];stopping=False;started=time.time()
    completed={j['id'] for j in jobs if Path(j['completion']).exists()}
    pending=[j for j in jobs if j['id'] not in completed]
    if a.adopt_state:
        previous=load(a.adopt_state)
        for entry in previous['active']:
            if entry['id'] in completed:continue
            j=next(j for j in pending if j['id']==entry['id']);pending.remove(j)
            p=AdoptedProcess(entry['pid'],j['completion'])
            log=(home/f'{j["id"]}.log').open('a')
            active.append((j,entry['gpu'],p,log));print('ADOPT',j['id'],'pid',p.pid,flush=True)
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def ready(j):
        if j['kind']=='pilot_acceptance':return load(WORK/'queue/pilot/state.json')['status']=='COMPLETE'
        if j['kind']=='tiger':return (WORK/'full_training_release.json').exists() and load(WORK/'full_training_release.json')['status']=='PASS'
        if j['kind']=='external' or (j['kind']=='amplit' and j['condition']=='with_negatives'):
            return load(WORK/'external_training_release.json')['status']=='PASS'
        return True
    def state(status,foreign,cpu_limit):
        dump(home/'state.json',{'status':status,'pid':os.getpid(),'started':started,'updated':time.time(),
              'total_jobs':len(jobs),'completed_jobs':len(completed),'remaining_to_launch':len(pending),
              'active':[{'id':j['id'],'resource':j['resource'],'gpu':g,'pid':p.pid} for j,g,p,f in active],
              'foreign_compute_processes':foreign,'failures':failures,'gpu_process_limit':5,'cpu_job_limit':cpu_limit})
    try:
        while pending or active:
            cpu_limit=cpu_job_limit()
            for item in list(active):
                j,g,p,log=item;rc=p.poll()
                if rc is None:continue
                log.close();active.remove(item)
                if rc or not Path(j['completion']).exists():
                    failures.append({'id':j['id'],'returncode':rc,'log':str(home/f'{j["id"]}.log')});stopping=True
                else:completed.add(j['id']);print('DONE',j['id'],flush=True)
            foreign,total=gpu_usage(active)
            if not stopping:
                for j in list(pending):
                    if not ready(j):continue
                    chosen=None
                    if j['resource']=='cpu':
                        if sum(x['resource']=='cpu' for x,_,_,_ in active)>=cpu_limit:continue
                        chosen=-1
                    else:
                        options=[]
                        for gpu in [0,1]:
                            own=[x for x,g,_,_ in active if g==gpu and x['resource']=='gpu']
                            count=len(own)+len(foreign[gpu])
                            # Foreign jobs include the already-running TIGER pilots;
                            # reserve at least 3 GiB for each, not just current usage.
                            reserved=sum(x['reserve_mib'] for x in own)+sum(max(3000,m) for _,m in foreign[gpu])
                            if count<5 and reserved+j['reserve_mib']<=total[gpu]-1200:options.append((count,reserved,gpu))
                        if not options:continue
                        chosen=min(options)[2]
                    log=(home/f'{j["id"]}.log').open('a')
                    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                    cmd=[PYTHON,'scripts/repair_v15/execute_job.py','--id',j['id'],'--gpu',str(max(0,chosen))]
                    p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    active.append((j,chosen,p,log));pending.remove(j);print('START',j['id'],'gpu',chosen,'pid',p.pid,flush=True)
            state('DRAINING_AFTER_FAILURE' if failures else 'DRAINING' if stopping else 'RUNNING',foreign,cpu_limit)
            if stopping and not active:break
            time.sleep(5)
        state('COMPLETE' if not failures and not pending else 'INCOMPLETE',{},cpu_job_limit())
        if failures or pending:sys.exit(1)
    finally:
        for j,g,p,log in active:
            if p.poll() is None and getattr(p,'owned',True):os.killpg(p.pid,signal.SIGTERM)
            log.close()


if __name__=='__main__':main()

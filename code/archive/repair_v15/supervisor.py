"""Durable, resumable GPU queue; only owns processes it launches."""
import argparse
import fcntl
import signal
import subprocess
import time
from common import *


def main():
    if (WORK/'CANCELLED_BY_V14_INPUT_FREEZE.json').exists():
        raise RuntimeError('ACT campaign superseded by approved V14-input branch; do not restart this queue')
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['smoke','pilot','full'],required=True);a=ap.parse_args()
    assert load(WORK/'core_acceptance.json')['status']=='PASS'
    if a.stage=='full': assert load(WORK/'full_training_release.json')['status']=='PASS'
    home=WORK/'queue'/a.stage;home.mkdir(parents=True,exist_ok=True)
    lock=(WORK/'queue'/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    reg=load(WORK/'run_registry.json')
    smoke_labels=['S4','S4_sp','S4_edge_chain','S4_nodezero','factor_P00','factor_P11','factor_S00','s4_gate_off','S4_nodeemb','S4_gcn']
    pilot_labels=['S4','factor_P00','factor_P11','factor_S00','s4_gate_off']
    tasks=[r for r in reg if r['action']=='train']
    if a.stage in ['smoke','pilot']:
        labels=smoke_labels if a.stage=='smoke' else pilot_labels
        tasks=[r for r in tasks if r['seed']==123 and r['label'] in labels]
    def output(r):return WORK/'smoke'/r['task_id'] if a.stage=='smoke' else Path(r['run_dir'])
    todo=[r for r in tasks if not (output(r)/'DONE.json').exists()]
    active=[];failures=[];stopping=False;started=time.time()
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def update(status):
        dump(home/'state.json',{'stage':a.stage,'status':status,'pid':os.getpid(),'started':started,'updated':time.time(),
             'total_tasks':len(tasks),'completed_tasks':sum((output(r)/'DONE.json').exists() for r in tasks),
             'remaining_to_launch':len(todo),'active':[{'task_id':r['task_id'],'gpu':g,'pid':p.pid} for r,g,p,f in active],
             'failures':failures,'max_processes_per_gpu':5,'cpu_threads_per_process':2})
    try:
        while todo or active:
            for item in list(active):
                r,g,p,log=item;rc=p.poll()
                if rc is not None:
                    log.close();active.remove(item)
                    if rc or not (output(r)/'DONE.json').exists():
                        failures.append({'task_id':r['task_id'],'returncode':rc});stopping=True
            if not stopping:
                # Round-robin fills both GPUs before adding subsequent slots.
                for slot in range(5):
                    for gpu in [0,1]:
                        if not todo or sum(g==gpu for _,g,_,_ in active)>slot:continue
                        r=todo.pop(0);logpath=home/f'{r["task_id"]}.log';log=logpath.open('a')
                        cmd=[PYTHON,str(ROOT/'scripts/repair_v15/train_one.py'),'--config',r['config_path'],'--gpu',str(gpu)]
                        if a.stage=='smoke':cmd.append('--smoke')
                        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
                        p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                        active.append((r,gpu,p,log));print('START',r['task_id'],'gpu',gpu,'pid',p.pid,flush=True)
            update('DRAINING_AFTER_FAILURE' if failures else 'DRAINING' if stopping else 'RUNNING')
            if stopping and not active:break
            time.sleep(3)
        complete=all((output(r)/'DONE.json').exists() for r in tasks)
        update('COMPLETE' if complete and not failures else 'INCOMPLETE')
        if not complete or failures:sys.exit(1)
    finally:
        # A forced supervisor exception must not leave untracked owned workers.
        for r,g,p,log in active:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
            log.close()


if __name__=='__main__':main()

"""Formal-only persistent queue: two GPUs, six concurrent seed jobs per GPU."""
import csv,fcntl,subprocess,time,signal,traceback
from common import *

def alive(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False

def passed(path):
 return path.exists() and load(path).get('status')=='PASS'

def main():
 home=WORK/'queue';home.mkdir(exist_ok=True,parents=True)
 lock=(home/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 check_freeze();assert passed(WORK/'full_training_release.json')
 reg=[r for r in load(WORK/'run_registry.json') if r['action']=='train'];assert len(reg)==400
 # Long gate arms start early; seed123 gives early coverage of all registered arms.
 reg.sort(key=lambda r:(r['seed']!=123,r['group']!='Similarity gate',r['seed'],r['label']))
 def completion(r):return WORK/'tiger_external'/r['task_id']/'EVALUATED.json'
 todo=[r for r in reg if not passed(completion(r))];active=[];failures=[];stop=False;last_telemetry=0.
 def stopping(signum,frame):
  nonlocal stop
  stop=True
 signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
 # Recover supervised children if a previous supervisor was interrupted.
 if (home/'state.json').exists():
  for item in load(home/'state.json').get('active',[]):
   if not alive(item['pid']):continue
   r=next((r for r in todo if r['task_id']==item['id']),None)
   if r is None:continue
   class Adopted:
    def __init__(self,pid,path):self.pid=pid;self.path=path
    def poll(self):return None if alive(self.pid) else (0 if passed(self.path) else 1)
   active.append((r,item['gpu'],Adopted(item['pid'],completion(r)),None));todo.remove(r)
 def capacities():
  def query(s):return subprocess.check_output(['nvidia-smi',s,'--format=csv,noheader,nounits'],text=True)
  ids={};total={}
  for row in csv.reader(query('--query-gpu=index,uuid,memory.total').splitlines()):
   ids[row[1].strip()]=int(row[0]);total[int(row[0])]=int(row[2])
  own={p.pid for _,_,p,_ in active};foreign={0:[],1:[]};used={0:0,1:0}
  for row in csv.reader(query('--query-compute-apps=gpu_uuid,pid,used_memory').splitlines()):
   if len(row)!=3:continue
   try:
    g=ids[row[0].strip()];pid=int(row[1]);mem=int(row[2]);pg=os.getpgid(pid)
    if g not in foreign:continue
    used[g]+=mem
    if pg not in own:foreign[g].append((pid,mem))
   except (ProcessLookupError,ValueError):pass
  avail=[]
  for g in [0,1]:
   count=sum(g==gg for _,gg,_,_ in active)
   reserve=count*3072+sum(max(3072,m) for _,m in foreign[g])
   if count+len(foreign[g])<6 and max(reserve,used[g])+3072<=total[g]-1536:avail.append((count,reserve,g))
  return avail,foreign
 def update(status,foreign=None):
  finished=sum(passed(completion(r)) for r in reg)
  trained=sum(passed(Path(r['run_dir'])/'DONE.json') for r in reg)
  dump(home/'state.json',dict(status=status,pid=os.getpid(),updated=time.time(),formal_training_total=400,
    formal_training_completed=trained,evaluated=finished,remaining_to_launch=len(todo),
    active=[dict(id=r['task_id'],label=r['label'],seed=r['seed'],gpu=g,pid=p.pid) for r,g,p,_ in active],
    failures=failures,foreign=foreign or {},max_processes_per_gpu=6,cpu_threads_per_job=2,
    postprocess=str(ROOT/'scripts/analysis_v17/run_after_training.py'),short_preflight=False))
 def telemetry():
  nonlocal last_telemetry
  if time.time()-last_telemetry<30:return
  last_telemetry=time.time()
  try:
   gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu','--format=csv,noheader,nounits'],text=True)
   record=dict(time=time.time(),gpu=list(csv.reader(gpu.splitlines())),loadavg=list(os.getloadavg()),
    meminfo=Path('/proc/meminfo').read_text(),cpu_stat=Path('/proc/stat').read_text().splitlines()[0],
    active=[dict(id=r['task_id'],gpu=g,pid=p.pid) for r,g,p,_ in active])
   with (home/'resource_telemetry.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  except Exception as exc:print('TELEMETRY WARNING',str(exc),flush=True)
 while todo or active:
  for item in list(active):
   r,g,p,log=item;rc=p.poll()
   if rc is None:continue
   if log:log.close()
   active.remove(item)
   if rc or not passed(completion(r)):
    failures.append(dict(id=r['task_id'],returncode=rc));stop=True
    print('FAILED',r['task_id'],rc,flush=True)
   else:print('COMPLETE',r['task_id'],flush=True)
  while todo and not stop:
   available,foreign=capacities()
   if not available:break
   g=min(available)[2];r=todo.pop(0);log=(home/(r['task_id']+'.log')).open('a')
   env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
   p=subprocess.Popen([PYTHON,str(SCRIPTS/'execute_job.py'),'--id',r['task_id'],'--gpu',str(g)],cwd=ROOT,
     stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
   active.append((r,g,p,log));print('START',r['task_id'],'GPU',g,'PID',p.pid,flush=True)
  _,foreign=capacities();update('DRAINING_AFTER_FAILURE' if failures else 'DRAINING' if stop else 'TRAINING',foreign);telemetry()
  if stop and not active:break
  time.sleep(10)
 if failures or todo:
  update('NEEDS_REPAIR' if failures else 'STOPPED');return
 dump(WORK/'training_and_inference_complete.json',dict(status='PASS',trained=400,evaluated=400,reuse_slots=20))
 update('POSTPROCESSING')
 pipeline=ROOT/'scripts/analysis_v17/run_after_training.py'
 # Postprocessing is registered independently of the frozen training runtime.
 if not pipeline.exists():
  update('WAITING_FOR_POSTPROCESS_SCRIPT');return
 log=(home/'postprocess.log').open('a')
 result=subprocess.run([PYTHON,str(pipeline)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 log.close()
 if result.returncode or not passed(WORK/'analysis_and_delivery_complete.json'):
  failures.append(dict(stage='postprocess',returncode=result.returncode));update('POSTPROCESS_NEEDS_REPAIR')
 else:update('COMPLETE')

if __name__=='__main__':
 try:main()
 except Exception:
  traceback.print_exc();raise

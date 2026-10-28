"""Read queue, fold products and telemetry; optionally write matching MD/Word status."""
import argparse,json,time,os,sys,importlib.util
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT/'artifacts/ablation_v17_u20000'
def load(p):return json.loads(Path(p).read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--write-report',action='store_true');args=ap.parse_args()
 state=load(WORK/'queue/state.json');reg=load(WORK/'run_registry.json');jobs=[r for r in reg if r['action']=='train'];done=[];running=[]
 for r in jobs:
  path=Path(r['run_dir']);d=path/'DONE.json'
  if d.exists():done.append(dict(arm=r['label'],seed=r['seed'],minutes=load(d)['elapsed_seconds']/60))
 for x in state['active']:
  p=Path(next(r['run_dir'] for r in jobs if r['task_id']==x['id']));logs=sorted((p/'logs').glob('fold*_history.csv'))
  latest=logs[-1] if logs else None
  running.append(dict(**x,last_fold=latest.stem if latest else 'pretraining/pair preparation',epochs=len(latest.read_text().splitlines())-1 if latest else 0))
 stamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds');elapsed=(time.time()-load(WORK/'queue/launch.json')['started_at'])/3600
 out=dict(timestamp=stamp,status=state['status'],pid=state['pid'],elapsed_hours=elapsed,total=400,trained=len(done),evaluated=state['evaluated'],running=len(running),pending=state['remaining_to_launch'],failures=state['failures'],max_per_gpu=6,reused_seed_results=20,completed_jobs=done,active=running,postprocess=state['postprocess'],pipeline_validation_complete=(WORK/'pipeline_validation_complete.json').exists())
 (WORK/'progress.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
 print(json.dumps({k:v for k,v in out.items() if k not in ['completed_jobs','active']},ensure_ascii=False,indent=2))
 if args.write_report:
  spec=importlib.util.spec_from_file_location('report_writer',ROOT/'reports/S4_study_nature_v16/write_documents.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  blocks=[('h1','V17 unsigned 20,000 消融执行进度'),('p','更新时间：'+stamp),('p',f'正式队列已启动，监督进程 PID {state["pid"]}。双 RTX 4090 D，每卡最多 6 个任务，总并发 12；ccseg，每任务 CPU 线程 2、interop 1。'),('table',[['阶段','数量／状态'],['正式五折任务',400],['训练完成',len(done)],['外部评估通过',state['evaluated']],['正在执行',len(running)],['尚未启动',state['remaining_to_launch']],['失败',len(state['failures'])],['复用结果',20],['运行时长（小时）',f'{elapsed:.2f}'],['监督状态',state['status']]]),('p','复用 unsigned quota 20,000、邻居验证的主模型全部十种子，以及已确认单肽头全部十种子。40 个对照重训 400 个种子任务，共 2,000 折；42 行全景包括主参考与复用单肽头。负样本消融仅 P00，不训练 P10/P01。保留 V14 实际末端特征与外测 CFU 编码。'),('p','已完成输入、配置和 31,032 个结构／特征源文件核验；训练、逐肽／逐边输出和校准均保留可审计来源。每个完成任务独立核验五折产物、规范 2,211 边、校准值和四指标。'),('p','后处理在全部 400 个任务训练及外评通过后自动接续：独立数值复算、空闲 GPU 计时、164 项消融比较、基准与同种子集成、绘图、Notebook Run All、中英文 Word/Markdown、字体和布局检查。'+('基于旧消融值的完整 CPU 流程验证已经通过；临时产物不属于正式 V17 结果。' if out['pipeline_validation_complete'] else '当前正在以归档值验证 CPU 分析与绘图流程，临时验证产物独立存放。')),('p','初始预算：正式训练及外评 20–36 小时，最终计时／图文验收 3–6 小时，开发并行开展；总体约 1–2 天。首批 10–20 个正式任务完成后用实测吞吐修订；宽门／无门任务不能按普通组均时估计。'),('p','状态：artifacts/ablation_v17_u20000/queue/state.json；日志：同目录 supervisor.log 与逐任务日志；资源采样：resource_telemetry.jsonl。正式输出：reports/S4_study_nature_v17/。原 V16 文件保持不变。')]
  if running:blocks.append(('table',[['任务','GPU','当前折／阶段','已记录 epoch']]+[[x['id'],x['gpu'],x['last_fold'],x['epochs']] for x in running]))
  m.write_pair(ROOT/'reports/V17_training_progress_20261015',blocks,blocks[0][1])
if __name__=='__main__':main()

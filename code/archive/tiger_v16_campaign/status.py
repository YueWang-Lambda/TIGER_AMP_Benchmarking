"""Read formal campaign progress; optionally save a matched Word/Markdown snapshot."""
from pathlib import Path
import argparse,collections,json,sys,time,math,statistics
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'artifacts/ablation_v16_p11'
def load(path):return json.loads(path.read_text())
def passed(path):return path.exists() and load(path).get('status')=='PASS'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--write-report',action='store_true');args=ap.parse_args()
 state=load(WORK/'queue/state.json');reg=[r for r in load(WORK/'run_registry.json') if r['action']=='train']
 groups=collections.defaultdict(lambda:dict(total=0,trained=0,evaluated=0,running=0));seconds=[]
 arm_seconds=collections.defaultdict(list);pending=[];seed123_completed=0
 active={x['id'] for x in state['active']}
 for row in reg:
  group=groups[row['group']];group['total']+=1;done=Path(row['run_dir'])/'DONE.json'
  if passed(done):
   elapsed=load(done)['elapsed_seconds'];group['trained']+=1;seconds.append(elapsed)
   arm_seconds[row['label']].append(elapsed);seed123_completed+=int(row['seed']==123)
  else:pending.append(row)
  group['evaluated']+=int(passed(WORK/'tiger_external'/row['task_id']/'EVALUATED.json'))
  group['running']+=int(row['task_id'] in active)
 now=datetime.now(ZoneInfo('Asia/Shanghai'))
 eta=None
 if pending and len(arm_seconds)==40:
  ideal=sum(statistics.median(arm_seconds[r['label']]) for r in pending)/36000
  lower=math.ceil(ideal);upper=math.ceil(ideal*1.4)
  eta=dict(remaining_ideal_training_hours=round(ideal,2),remaining_training_budget_hours=[lower,upper],
   postprocess_budget_hours=[3,5],delivery_window=[(now+timedelta(hours=lower+3)).isoformat(timespec='minutes'),(now+timedelta(hours=upper+5)).isoformat(timespec='minutes')],
   basis='Per-arm median completed duration; 10 concurrent jobs; currently running jobs conservatively counted as full jobs; not a guarantee.')
 snapshot=dict(as_of=now.isoformat(timespec='seconds'),
  state_age_seconds=round(time.time()-state['updated'],1),status=state['status'],
  training_total=400,trained=sum(g['trained'] for g in groups.values()),evaluated=sum(g['evaluated'] for g in groups.values()),
  running=len(active),failures=state['failures'],groups=groups,
  completed_mean_minutes=sum(seconds)/len(seconds)/60 if seconds else None,
  arms_with_completed_seed=len(arm_seconds),seed123_completed=seed123_completed,eta=eta,
  short_preflight=False,postprocess_cpu_validation=passed(WORK/'pipeline_validation_complete.json'),
  formal_delivery_complete=passed(WORK/'analysis_and_delivery_complete.json'))
 if snapshot['formal_delivery_complete']:
  finish=load(WORK/'analysis_and_delivery_complete.json')['finished']
  start=(WORK/'queue/launch.json').stat().st_mtime
  training_end=(WORK/'training_and_inference_complete.json').stat().st_mtime
  snapshot.update(completed_at=datetime.fromtimestamp(finish,ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
   training_wall_hours=(training_end-start)/3600,total_wall_hours=(finish-start)/3600)
 print(json.dumps(snapshot,ensure_ascii=False,indent=2))
 (WORK/'progress_snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2))
 if not args.write_report:return
 sys.path.insert(0,str(ROOT/'scripts/mic128_classifier_v2'));from document_helpers import write_pair
 completed=snapshot['formal_delivery_complete']
 blocks=[('h1','V16 P11 消融正式队列：执行进度快照'),('p','记录时间：'+snapshot['as_of']+('。全部任务及正式交付已完成，科学结果见 reports/S4_study_nature_v16/。' if completed else '。这是运行进度，不是完成后的 V16 科学结果报告。')),
  ('p','按用户最新批准，取消代表性短程预检，直接运行双 GPU、每卡最多 5 个任务的正式队列。每个任务为一个种子的完整五折训练及固定外测终点评估。'),
  ('table',[['项目','状态'],['新训练总任务','400（40 个消融 × 10 种子；2,000 个折模型）'],['完成训练',str(snapshot['trained'])],['完成外测验收',str(snapshot['evaluated'])],['当前运行',str(snapshot['running'])],['失败记录',str(len(snapshot['failures']))],['种子','120–129'],['主参考','复用 P11'],['额外复用','P00、P10、P01；不重训练'],['最终 landscape','44 行，每行 10 种子'],['后处理程序 CPU 验证','通过' if snapshot['postprocess_cpu_validation'] else '进行中'],['正式 V16 交付','完成' if snapshot['formal_delivery_complete'] else '等待全部正式结果']]),
  ('h2','分组进度'),('table',[['组别','总任务','训练完成','外测完成','运行中']]+[[k,*[str(v[x]) for x in ['total','trained','evaluated','running']]] for k,v in groups.items()]),
  ('h2','自动分析和交付'),('p',('400 个正式任务已全部通过外测验收，覆盖 208 条肽、11 个家族、2,211 条规范配对；身份、单位、有限性和校准数值检查通过。自动后处理已生成统计表、44 行全景、其他消融图、独立子图、已执行 Notebook 和中英文 Markdown/Word 报告。正式交付验收为 PASS。' if completed else '每个正式任务训练完成后自动外评，核对 208 条肽、11 个家族、2,211 条配对的身份、单位、有限性和校准数值。全部 400 个任务通过后，队列自动执行 scripts/analysis_v16/run_after_training.py，重建统计表、44 行全景、其他消融图、独立子图、已执行 Notebook 和中英文 Markdown/Word 报告。')),
  ('p','原 V15 四指标基准、PCC/RSE Pareto 和同种子 P11/S11 的 TIGER-ensemble 保留。新 single-head 消融不替换原基准或集成中的 S11。保留 V14 实际末端特征和外测 CFU 输入。'),
  ('p','后处理验证使用 artifacts/ablation_v16_p11/pipeline_validation_legacy_values 中的隔离历史数据，只用于验证代码、字体、版式和 Word 导出；不是短程训练，也不是 V16 实验结果。正式输出路径为 reports/S4_study_nature_v16/。'),
  ('h2','时间与持久化'),('p',
   (f"正式交付完成于 {snapshot['completed_at']}。400 个任务平均训练耗时 {snapshot['completed_mean_minutes']:.1f} 分钟/种子，包含完整五折；双卡并行训练及外评总墙钟时间 {snapshot['training_wall_hours']:.2f} 小时，含自动图文交付共 {snapshot['total_wall_hours']:.2f} 小时。队列已正常退出，没有剩余训练任务。" if completed else
   f"40 个消融配置均已有正式种子完成，其中 seed123 完成 {seed123_completed}/40。已完成任务平均训练耗时 {snapshot['completed_mean_minutes']:.1f} 分钟/种子，包含完整五折。按各臂已完成种子的中位耗时及十并发估算，剩余训练理想值 {eta['remaining_ideal_training_hours']:.2f} 小时；当前运行任务保守地按完整任务计入。考虑后续种子及调度波动，剩余训练预算 {eta['remaining_training_budget_hours'][0]}–{eta['remaining_training_budget_hours'][1]} 小时，再留 3–5 小时完成图文验收。预计完整交付窗口：{eta['delivery_window'][0]} 至 {eta['delivery_window'][1]}，并非完成承诺。" if eta else
    '正式训练及逐任务评估原预算约 18–30 小时，最终统计图文验收约 3–5 小时，后处理开发与训练并行；总体按 1–2 天安排。后续根据实际吞吐更新预计完成时间。')),
  ('p','状态归档：artifacts/ablation_v16_p11/queue/state.json；主日志：queue/supervisor.log；后处理日志：queue/postprocess.log。'+('状态为 COMPLETE，训练和监督进程均已正常退出。' if completed else '监督进程与训练子进程已脱离终端运行。遇到失败时停止分配新任务并保留日志，不删种子、不自动改变科学配置。'))]
 write_pair(ROOT/'reports/V16_P11_ablation_execution_status_20261014',blocks,blocks[0][1])

if __name__=='__main__':main()

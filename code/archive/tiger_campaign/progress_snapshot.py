"""Observe the formal queue and write matching progress reports; never schedule jobs."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta
import heapq
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'artifacts/ablation_v17_u20000'


def load(path):
    return json.loads(Path(path).read_text())


def passed(path):
    return path.exists() and load(path).get('status') == 'PASS'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write-report', action='store_true')
    args = ap.parse_args()
    now = time.time()
    stamp = datetime.fromtimestamp(now, ZoneInfo('Asia/Shanghai'))
    state = load(WORK / 'queue/state.json')
    launch = load(WORK / 'queue/launch.json')
    jobs = [r for r in load(WORK / 'run_registry.json') if r['action'] == 'train']
    by_id = {r['task_id']: r for r in jobs}
    done, trained, completion_times = set(), set(), []
    durations = defaultdict(list)
    for r in jobs:
        run = Path(r['run_dir'])
        if passed(run / 'DONE.json'):
            trained.add(r['task_id'])
        external = WORK / 'tiger_external' / r['task_id'] / 'EVALUATED.json'
        if r['task_id'] in trained and passed(external):
            done.add(r['task_id'])
            start = load(run / 'execution_environment.json')['started']
            durations[r['label']].append(external.stat().st_mtime - start + 15)
            completion_times.append(external.stat().st_mtime)
    active = [a for a in state['active'] if a['id'] not in done]
    active_ids = {a['id'] for a in active}
    pending = [r for r in jobs if r['task_id'] not in done | active_ids]
    pending.sort(key=lambda r: (r['seed'] != 123, r['group'] != 'Similarity gate', r['seed'], r['label']))
    running = []
    for a in active:
        run = Path(by_id[a['id']]['run_dir'])
        env = run / 'execution_environment.json'
        age = max(0, now - load(env)['started']) if env.exists() else 0
        folds = len(list((run / 'checkpoints').glob('fold*_last.pt')))
        histories = sorted((run / 'logs').glob('fold*_history.csv'))
        last_fold, last_epoch = 0, 0
        if histories:
            last_fold = int(histories[-1].stem.split('_')[0].replace('fold', ''))
            with histories[-1].open() as f:
                rows = list(csv.DictReader(f))
            if rows:
                last_epoch = max(int(row['epoch']) for row in rows)
        stage = '外测／输出验收' if folds == 5 else (
            f'第 {last_fold} 折，epoch {last_epoch}' if last_fold > folds
            else f'第 {folds + 1} 折，预训练或配对准备')
        running.append(dict(**a, elapsed_minutes=age / 60, completed_folds=folds,
                            stage=stage, last_recorded_fold=last_fold, last_recorded_epoch=last_epoch))
    means = {arm: statistics.mean(values) for arm, values in durations.items()}

    def forecast(scale):
        slots = [max(60, means[a['label']] * scale - a['elapsed_minutes'] * 60) for a in running]
        slots += [0] * max(0, 12 - len(slots))
        heapq.heapify(slots)
        for r in pending:
            t = heapq.heappop(slots)
            heapq.heappush(slots, t + means[r['label']] * scale + 10)
        return max(slots) / 3600

    eta = None
    if state['status'] == 'TRAINING' and all(r['label'] in means for r in pending + [by_id[a['id']] for a in active]):
        eta = dict(central_hours=forecast(1), sensitivity_low_hours=forecast(.8),
                   sensitivity_high_hours=forecast(1.25),
                   method='Same-arm observed mean durations, subtract active elapsed time, simulate 12 slots; sensitivity uses 0.8–1.25x durations, not a confidence interval; postprocessing excluded')
        eta['central_finish'] = (stamp + timedelta(hours=eta['central_hours'])).isoformat(timespec='minutes')
    groups = Counter(r['group'] for r in jobs)
    group_done = Counter(by_id[k]['group'] for k in done)
    seed_done = Counter(str(by_id[k]['seed']) for k in done)
    gpu_text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu',
                                        '--format=csv,noheader,nounits'], text=True)
    gpu = [dict(index=int(r[0]), memory_used_mib=int(r[1]), memory_total_mib=int(r[2]),
                utilization_percent=int(r[3]), temperature_c=int(r[4]))
           for r in csv.reader(gpu_text.splitlines())]
    result = dict(timestamp=stamp.isoformat(timespec='seconds'), status=state['status'],
        supervisor_pid=state['pid'], runtime=launch.get('backend', 'original_frozen'),
        total=len(jobs), trained=len(trained), evaluated=len(done), percent_complete=len(done)/len(jobs)*100,
        active=len(active), pending=len(pending), remaining_total=len(jobs)-len(done),
        failures=state['failures'], active_by_gpu=dict(Counter(str(a['gpu']) for a in active)),
        resumed_elapsed_hours=(now-launch['started_at'])/3600,
        completed_since_resume=len(done)-launch.get('completed_before_resume', 0),
        groups={g: dict(completed=group_done[g], total=n) for g, n in groups.items()},
        completed_by_seed=dict(sorted(seed_done.items())), fully_completed_arms=sum(len(v)==10 for v in durations.values()),
        recent_completions={str(h)+'h':sum(t>=now-h*3600 for t in completion_times) for h in [1,2,4]},
        forecast=eta, gpu=gpu, running=running, postprocess=state['postprocess'],
        final_delivery_complete=passed(WORK/'analysis_and_delivery_complete.json'))
    target = WORK / 'progress_current.json'
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    tmp.replace(target)
    if args.write_report:
        title = 'V17 unsigned 20,000 消融执行进度'
        blocks = [('h1', title), ('p', '快照时间：'+result['timestamp']+'（北京时间）。'),
          ('p', '正式队列持续使用冻结的原版训练与评估代码，双 RTX 4090 D，每卡最多 6 个任务；数据流优化版未部署。'),
          ('table', [['阶段', '数量／状态'], ['正式五折任务总数', len(jobs)], ['训练完成', len(trained)],
            ['训练及外评通过', f'{len(done)}（{result["percent_complete"]:.2f}%）'], ['正在执行', len(active)],
            ['尚未启动', len(pending)], ['剩余任务合计', len(jobs)-len(done)], ['当前失败', len(state['failures'])],
            ['监督状态', state['status']], ['监督 PID', state['pid']], ['本次恢复后运行时间', f'{result["resumed_elapsed_hours"]:.2f} 小时'],
            ['本次恢复后新增通过任务', result['completed_since_resume']]]),
          ('p', '上述数量仅统计 400 个需新训练的种子任务，每个任务完整五折；另有主模型与单肽头共 20 个既有种子结果按计划复用，不计入这 400 个任务。继续保留 V14 实际末端/MW 输入及外测 CFU 编码。')]
        if eta:
            blocks += [('h2', '剩余时间估计'), ('p', f'根据同一消融臂已完成任务的实测耗时，扣除当前任务已运行时间，并模拟 12 个任务槽，训练及逐任务外评的中央估计还需 {eta["central_hours"]:.2f} 小时，约 {eta["central_finish"]} 完成。将各臂耗时缩放至 0.8–1.25 倍，范围约 {eta["sensitivity_low_hours"]:.2f}–{eta["sensitivity_high_hours"]:.2f} 小时；这是调度敏感性范围，不是统计置信区间。早停、资源竞争和队尾任务长短会改变实际结束时间。'),
              ('p', f'最近 1、2、4 小时分别完成 {result["recent_completions"]["1h"]}、{result["recent_completions"]["2h"]}、{result["recent_completions"]["4h"]} 个任务。耗时较长的 similarity gate 组完成 {group_done["Similarity gate"]}/{groups["Similarity gate"]}；剩余时间估算未直接套用全队列历史平均吞吐。')]
        blocks += [('h2', '各实验组完成量'), ('table', [['实验组', '外评通过／任务数']]+[[g, f'{group_done[g]}/{n}'] for g,n in groups.items()]),
          ('h2', 'GPU 即时状态'), ('table', [['GPU', '运行任务', '显存 GiB', '利用率', '温度 °C']]+[[g['index'], result['active_by_gpu'].get(str(g['index']),0), f'{g["memory_used_mib"]/1024:.2f} / {g["memory_total_mib"]/1024:.2f}', f'{g["utilization_percent"]}%', g['temperature_c']] for g in gpu]),
          ('p', 'GPU 利用率为单次采样，随数据准备、训练和验证阶段变化，不代表整段运行均值。'),
          ('h2', '后续分析与交付'), ('p', '全部 400 个任务训练和外评通过后，监督程序将自动接续独立数值复算、GPU 推断计时、消融比较、基准与集成更新、绘图、Notebook 执行、中英文 Word/Markdown 和交付验收。上述剩余时间估计不含这些后处理步骤；V17 最终分析和图文交付当前尚未完成。'),
          ('h2', '运行中任务'), ('table', [['任务', 'GPU', '已运行分钟', '阶段']]+[[a['id'], a['gpu'], f'{a["elapsed_minutes"]:.1f}', a['stage']] for a in running]),
          ('p', '实时队列：artifacts/ablation_v17_u20000/queue/state.json。该快照与估算：artifacts/ablation_v17_u20000/progress_current.json。统计脚本：scripts/ablation_v17/progress_snapshot.py。')]
        spec = importlib.util.spec_from_file_location('writer', ROOT/'reports/S4_study_nature_v16/write_documents.py')
        writer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(writer)
        check = writer.write_pair(ROOT/'reports/V17_training_progress_20261015', blocks, title)
        (WORK/'progress_report_consistency.json').write_text(json.dumps(check,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['running','groups']},indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()

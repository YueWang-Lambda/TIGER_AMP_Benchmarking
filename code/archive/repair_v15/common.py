"""Shared paths and atomic metadata for the approved v15 repair campaign."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
WORK = ROOT / 'artifacts/repair_v15'
REPORT = ROOT / 'reports/repair_campaign_v15'
ARCHIVE = WORK / 'archive_pre_repair'
RUNS = ROOT / 'runs_repair_v15'
PKG = ROOT / 'reports/methods_package_S4_v1'
PYTHON = '/home/ubuntu/anaconda3/envs/ccseg/bin/python'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
    os.replace(temp,path)


def load(path): return json.loads(Path(path).read_text())


def run_config(arm,seed):
    paths=[ROOT/f'runs_ablation/{d}/{d}_seed{seed}/config.json' for d in dict.fromkeys([arm,arm.lower()])]
    matches=[p for p in paths if p.exists()]
    if len(matches)!=1: raise RuntimeError((arm,seed,matches))
    return matches[0]


def arms():
    return [r['arm'] for r in load(ROOT/'reports/code_methodology_review_20261011/repair_runtime_inventory.json')['by_arm']]

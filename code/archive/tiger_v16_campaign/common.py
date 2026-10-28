"""Paths and identity contract for the approved, isolated V14-input branch."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'artifacts/ablation_v16_p11'
RUNTIME=WORK/'runtime'
RUNS=ROOT/'runs_ablation_v16_p11'
REPORT=ROOT/'reports/S4_study_nature_v16'
PKG=WORK/'source_snapshot'
ARCHIVE=ROOT/'artifacts/repair_v15/archive_pre_repair'
SCRIPTS=ROOT/'scripts/ablation_v16'
PYTHON='/home/ubuntu/anaconda3/envs/ccseg/bin/python'
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(RUNTIME))

def load(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(path,obj):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False));os.replace(temp,path)
def check_freeze():
    m=load(WORK/'freeze_manifest.json')
    for path,h in m['file_hashes'].items():
        if sha(path)!=h:raise RuntimeError(f'Frozen runtime/input changed: {path}')
    import code
    assert Path(code.__file__).resolve()==RUNTIME/'code/__init__.py',code.__file__
    return m

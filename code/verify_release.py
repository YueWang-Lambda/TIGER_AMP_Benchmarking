"""Check SHA-256 inventory, model-file exclusions, syntax and local links."""
import ast
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
DENIED={'.pt','.pth','.ckpt','.safetensors','.pkl','.pickle','.joblib','.h5','.hdf5','.keras','.onnx','.cbm','.bin','.est','.npy','.npz','.pyc'}


def verify():
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    errors=[]
    entries={r['path']:r for r in manifest['files']}
    actual={str(p.relative_to(ROOT)):p for p in ROOT.rglob('*') if p.is_file()
            and not any(s in p.relative_to(ROOT).parts for s in ['.git','.venv','venv','reproduced','__pycache__','.ipynb_checkpoints'])
            and p.name!='MANIFEST.json'}
    if set(entries)!=set(actual):errors.append({'inventory_missing':sorted(set(entries)-set(actual)),'inventory_extra':sorted(set(actual)-set(entries))})
    for name,p in actual.items():
        if p.is_symlink():errors.append({'symlink':name})
        if p.suffix.lower() in DENIED or re.search(r'\.data-\d+-of-\d+$',name):errors.append({'forbidden_artifact':name})
        if p.stat().st_size>=100_000_000:errors.append({'large_file':name})
        if name in entries and hashlib.sha256(p.read_bytes()).hexdigest()!=entries[name]['sha256']:errors.append({'checksum':name})
        # Only public entry points are promised portable Python 3.10 syntax.
        if p.suffix=='.py' and p.parent==ROOT/'code':
            try:ast.parse(p.read_text())
            except SyntaxError as exc:errors.append({'syntax':name,'error':str(exc)})
    for p in [ROOT/'README.md',*sorted((ROOT/'docs').glob('*GUIDE*.md'))]:
        if not p.exists():continue
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if link.startswith(('http:','https:','mailto:','#')):continue
            path=unquote(link.split('#')[0].split('?')[0])
            if not (p.parent/path).exists():errors.append({'broken_link':str(p.relative_to(ROOT)),'target':link})
    source=json.loads((ROOT/'provenance/source_files.json').read_text())
    for row in source:
        path=ROOT/row['path']
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            errors.append({'source_snapshot_changed':row['path']})
    result=dict(status='FAIL' if errors else 'PASS',files=len(actual),source_records=len(source),weights=0 if not any('forbidden_artifact' in e for e in errors) else 'FOUND',errors=errors)
    print(json.dumps(result,indent=2))
    if errors:raise SystemExit(1)
    return result


if __name__=='__main__':verify()

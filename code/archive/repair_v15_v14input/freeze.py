"""Lock this branch to a reviewable runtime and immutable source tables."""
from common import *
import ast
from datetime import datetime
from zoneinfo import ZoneInfo

def main():
    if (WORK/'freeze_manifest.json').exists():raise RuntimeError('Already frozen; create an explicit revision to change it')
    assert load(WORK/'input_and_replay_acceptance.json')['status']=='PASS'
    files=list((RUNTIME/'code').rglob('*.py'))+list(SCRIPTS.glob('*.py'))+list((WORK/'configs').glob('*.json'))+list(PKG.rglob('*.csv'))
    files += [WORK/'run_registry.json',WORK/'source_copy_provenance.json',WORK/'retained_external_jobs.json']
    for p in files:
        if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
    dump(WORK/'freeze_manifest.json',{'status':'FROZEN','time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
         'input_policy':'V14 sequences/labels/terminal features and external CFU preserved; independent methodology fixes retained',
         'file_hashes':{str(p):sha(p) for p in sorted(set(files))}})
    print('Frozen',len(files),'runtime, config and input files')

if __name__=='__main__':main()

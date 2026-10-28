"""Run the V17 ensemble timer after formal training, with exclusive GPU checks."""
from pathlib import Path
import os,sys,subprocess

def main():
 here=Path(__file__).resolve();parent=here.parents[2]
 candidates=[p/'scripts/analysis_v17/time_tiger.py' for p in here.parents]
 script=next((p for p in candidates if p.is_file()),None)
 if script is None:raise RuntimeError('Run this helper in the registered TIGER workspace; see the run registry.')
 env=os.environ.copy();env['V17_DELIVERY_DIR']=str(parent)
 subprocess.run([sys.executable,str(script),*sys.argv[1:]],check=True,env=env)
if __name__=='__main__':main()

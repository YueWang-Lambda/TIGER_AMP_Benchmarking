"""Run the data-dependent ensemble delivery gate after document generation."""
from reconstruct import OUT
import subprocess,sys
if __name__=='__main__':subprocess.run([sys.executable,str(OUT/'TIGER_ensemble_addendum/code/validate_delivery.py')],check=True)

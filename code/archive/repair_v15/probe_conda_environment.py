"""Read-only interpreter/framework probe; does not allocate GPU tensors."""
import argparse
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import sys

ROOT=Path(__file__).resolve().parents[2]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True)
    ap.add_argument('--framework',choices=['torch','tensorflow','deepforest'],required=True)
    a=ap.parse_args()
    sys.path=[p for p in sys.path if p and Path(p).resolve()!=ROOT]
    versions={}
    for name in ['numpy','pandas','scipy','scikit-learn','torch','torch-geometric',
                 'transformers','tensorflow','tensorflow-cpu','keras','deep-forest',
                 'rpy2','gensim','propy3']:
        try:versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError:pass
    out={'name':a.name,'executable':sys.executable,'real_executable':str(Path(sys.executable).resolve()),
         'prefix':sys.prefix,'python':platform.python_version(),'versions':versions,
         'cuda_visible_devices_probe':os.environ.get('CUDA_VISIBLE_DEVICES'),
         'conda_meta_exists':(Path(sys.prefix)/'conda-meta').is_dir(),
         'scope':'read-only imports and device discovery; no GPU tensor/model allocation'}
    try:
        mod=importlib.import_module(a.framework)
        out['framework_import']={'status':'PASS','module_file':mod.__file__,
                                  'version':getattr(mod,'__version__',None)}
        if a.framework=='torch':
            out['cuda']={'compiled_version':mod.version.cuda,'available':mod.cuda.is_available(),
                         'device_count':mod.cuda.device_count()}
        elif a.framework=='tensorflow':
            out['cuda']={'built_with_cuda':mod.test.is_built_with_cuda(),
                         'build_info':dict(mod.sysconfig.get_build_info()),
                         'visible_gpus':[d.name for d in mod.config.list_physical_devices('GPU')]}
    except Exception as exc:
        out['framework_import']={'status':'ERROR','type':type(exc).__name__,'message':str(exc)}
    dest=ROOT/'artifacts/repair_v15/environment_audit'/f'{a.name}.json'
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(out,indent=2,ensure_ascii=False))
    print(json.dumps(out,ensure_ascii=False))


if __name__=='__main__':main()

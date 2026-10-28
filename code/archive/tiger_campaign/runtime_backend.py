"""Select a hash-registered dataflow runtime without altering frozen model code.

The release file is absent until the independent GPU validation has passed.
All math/model/data modules continue to resolve from the original frozen code
package. Only train.py and the dataflow helpers can be substituted.
"""
import importlib.util
from common import *

RELEASE=WORK/'dataflow_optimization_release.json'


def verify_release(path=RELEASE):
    release=load(path)
    if release.get('status')!='PASS':raise RuntimeError('Dataflow runtime has no PASS release')
    if release['frozen_manifest_sha256']!=sha(WORK/'freeze_manifest.json'):
        raise RuntimeError('Dataflow release targets a different frozen recipe')
    for name,expected in release['file_hashes'].items():
        if sha(name)!=expected:raise RuntimeError('Released code/evidence changed: '+name)
    for name in release['required_pass_evidence']:
        if load(name).get('status')!='PASS':raise RuntimeError('Required dataflow evidence failed: '+name)
    if release['production_backend_flags']!={'deterministic_algorithms':False,'cudnn_deterministic':False,'cudnn_benchmark':False,'tf32_matmul':False}:
        raise RuntimeError('Unexpected production numerical-policy change')
    return release


def _load(name,path):
    if name in sys.modules:raise RuntimeError('Runtime module was already loaded: '+name)
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def select_training_backend(task_id):
    if not RELEASE.exists():
        import code.train as training
        return training,{'backend':'original_frozen','release_sha256':None}
    release=verify_release()
    if task_id not in release['eligible_task_ids']:raise RuntimeError('Task not registered for this release: '+task_id)
    import torch
    flags={'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
           'cudnn_deterministic':torch.backends.cudnn.deterministic,
           'cudnn_benchmark':torch.backends.cudnn.benchmark,
           'tf32_matmul':torch.backends.cuda.matmul.allow_tf32}
    if flags!=release['production_backend_flags']:raise RuntimeError('Production numerical flags differ: '+str(flags))
    _load('code.flow',release['flow_module'])
    training=_load('code.optimized_train',release['training_module'])
    check_freeze()
    return training,{'backend':release['backend'],'release_sha256':sha(RELEASE),
                     'training_module_sha256':sha(release['training_module']),
                     'flow_module_sha256':sha(release['flow_module']),
                     'production_backend_flags':flags}


def verify_backend_at_end(provenance):
    if provenance['backend']=='original_frozen':return
    verify_release()
    if sha(RELEASE)!=provenance['release_sha256']:raise RuntimeError('Release changed during training')

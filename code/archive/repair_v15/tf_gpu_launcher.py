"""Launch TF 2.12 in the isolated CUDA-11 environment; never edit the active env.

Example: python tf_gpu_launcher.py --gpu 0 --memory-mib 2048 --target
    /absolute/path/to/diagnostic.py -- --method AMPlify
Use --gpu -1 for a CPU reference. Do not pass the target's --cpu flag on GPU.
Used by diagnostics and the campaign scheduler; preserves its process group.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PREFIX = Path('/home/ubuntu/anaconda3/envs/mbc_att_gpu_v15')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu', type=int, choices=[-1, 0, 1], required=True)
    ap.add_argument('--memory-mib', type=int, default=2048)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--execution', choices=['eager', 'graph'], default='eager',
        help='AMPlify requires eager to avoid the masked backward-LSTM graph defect; MBC graph is validated')
    ap.add_argument('--target', type=Path, required=True)
    ap.add_argument('--runtime-record', type=Path)
    ap.add_argument('--_inside', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('target_args', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    target = a.target.resolve(strict=True)
    extra = a.target_args[1:] if a.target_args[:1] == ['--'] else a.target_args
    if a.gpu >= 0 and '--cpu' in extra:
        raise ValueError('--cpu conflicts with an explicitly requested GPU')
    is_amplify = target.name.startswith('retrain_amplify') or any(
        extra[i:i+2] == ['--method', 'AMPlify'] for i in range(len(extra)-1))
    if a.gpu >= 0 and is_amplify and a.execution != 'eager':
        raise ValueError('AMPlify GPU requires --execution eager; graph parity is not reliable in TF 2.12.1')
    if a.memory_mib < 512 or a.threads < 1:
        raise ValueError('memory-mib >= 512 and threads >= 1 required')
    if not a._inside:
        libdirs = [PREFIX/'lib'] + sorted((PREFIX/'lib/python3.9/site-packages/nvidia').glob('*/lib'))
        assert all(p.is_dir() for p in libdirs)
        env = os.environ.copy()
        env.update(CONDA_PREFIX=str(PREFIX), CONDA_DEFAULT_ENV=PREFIX.name,
            PATH=str(PREFIX/'bin') + os.pathsep + env.get('PATH', ''),
            # Deliberately do not inherit the CUDA-12 runtime or an empty path.
            LD_LIBRARY_PATH=os.pathsep.join(map(str, libdirs)),
            PYTHONPATH='', PYTHONDONTWRITEBYTECODE='1',
            NVIDIA_TF32_OVERRIDE='0',
            CUDA_VISIBLE_DEVICES=str(a.gpu),
            OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads),
            OPENBLAS_NUM_THREADS=str(a.threads), NUMEXPR_NUM_THREADS=str(a.threads),
            TF_NUM_INTRAOP_THREADS=str(a.threads), TF_NUM_INTEROP_THREADS='1')
        env.pop('PYTHONHOME', None)
        env.pop('TF_FORCE_GPU_ALLOW_GROWTH', None)
        record_args=['--runtime-record',str(a.runtime_record.resolve())] if a.runtime_record else []
        os.chdir('/tmp')
        os.execve(str(PREFIX/'bin/python'), [str(PREFIX/'bin/python'),
            str(Path(__file__).resolve()), '--_inside', '--gpu', str(a.gpu),
            '--memory-mib', str(a.memory_mib), '--threads', str(a.threads),
            '--execution', a.execution,
            *record_args, '--target', str(target), '--', *extra], env)

    assert Path(sys.prefix) == PREFIX
    # Avoid deltaMIC/code shadowing Python's stdlib code module during TF import.
    sys.path = [p for p in sys.path if p and Path(p).resolve() != ROOT]
    if a.gpu >= 0:
        raw = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
            '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
        pids = [int(x.strip()) for x in raw.splitlines() if x.strip().isdigit()]
        if len(pids) >= 5:
            raise RuntimeError(f'GPU {a.gpu} already has {len(pids)} compute processes; five-process limit')
        free = int(subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
            '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        if free < a.memory_mib + 1536:
            raise RuntimeError(f'Insufficient headroom: free {free} MiB, cap {a.memory_mib} MiB')
    import tensorflow as tf
    # Match the FP32 arithmetic baseline, rather than allowing TensorFloat-32.
    tf.config.experimental.enable_tensor_float_32_execution(False)
    # AMPlify: arithmetic-off alone is insufficient on fresh model.predict
    # traces; eager execution avoids the unstable graph backend and keeps cuDNN.
    # Controls: graph_options_probe.json, amplify_stability_*.json.
    tf.config.optimizer.set_experimental_options({'arithmetic_optimization': False})
    tf.config.run_functions_eagerly(a.execution == 'eager')
    tf.config.threading.set_intra_op_parallelism_threads(a.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    physical = tf.config.list_physical_devices('GPU')
    if a.gpu >= 0:
        if len(physical) != 1:
            raise RuntimeError(f'GPU required; refusing silent CPU fallback: {physical}')
        tf.config.set_logical_device_configuration(physical[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=a.memory_mib)])
        # Register the small allocation promptly so the campaign scheduler sees it.
        with tf.device('/GPU:0'):
            assert 'GPU:0' in tf.constant(1.).device
    elif physical:
        raise RuntimeError('CPU reference must not see a GPU')
    record={'runtime': 'isolated_tf_cuda11', 'prefix': sys.prefix,
        'executable':sys.executable,'pid':os.getpid(),'target':str(target),
        'physical_gpu_index': a.gpu, 'memory_cap_mib': a.memory_mib if a.gpu >= 0 else None,
        'tensorflow': tf.__version__, 'threads': a.threads,
        'execution': a.execution,
        'tf32_enabled': tf.config.experimental.tensor_float_32_execution_enabled(),
        'graph_options': tf.config.optimizer.get_experimental_options()}
    if a.runtime_record:
        a.runtime_record.parent.mkdir(parents=True,exist_ok=True)
        temporary=a.runtime_record.with_suffix('.tmp.json')
        temporary.write_text(json.dumps(record,indent=2))
        os.replace(temporary,a.runtime_record)
    print(json.dumps(record), flush=True)
    sys.argv = [str(target), *extra]
    sys.path.insert(0, str(target.parent))
    runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__':
    main()

"""CPU/GPU parity and real training-step probe; no formal weights are written.

Always use tf_gpu_launcher.py. Both devices read the same completed v15
checkpoint. External queries are used only for forward parity; gradient steps
use 256 sampled training-corpus peptides. Diagnostic weights are discarded.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/repair_v15/tf_gpu_repair'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['AMPlify', 'MBC-Attention'], required=True)
    ap.add_argument('--device', choices=['cpu', 'gpu'], required=True)
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args()
    OUT = a.out.resolve()
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    assert bool(tf.config.list_physical_devices('GPU')) == (a.device == 'gpu')
    OUT.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(987654)
    pep = pd.read_csv(ROOT/'reports/methods_package_S4_v1/2_external_testset/data/final11_peptides.csv')
    pos = pd.read_csv(ROOT/'reports/methods_package_S4_v1/4_S4_benchmark/data/train_restricted4929_tiger.csv')
    pos = pos.groupby('sequence', as_index=False).MIC_Escherichia_coli.mean()
    neg = pd.read_csv(ROOT/'artifacts/repair_v15/external_inputs/data_restricted/uniprot_negatives.csv')
    assert len(pos) == 4929 and len(neg) == 7949
    seqs = pos.sequence.tolist() + neg.sequence.tolist()
    logmic = np.r_[np.log10(pos.MIC_Escherichia_coli.to_numpy()), np.full(len(neg), 4.)]
    idx = np.random.RandomState(987654).permutation(len(seqs))[:256]
    trainseq = [seqs[i] for i in idx]
    target = logmic[idx]
    sota = ROOT/'baseline/sota_bench'
    result = {'method': a.method, 'device': a.device, 'tensorflow': tf.__version__,
        'prefix': sys.prefix, 'formal_outputs_modified': False,
        'tf32_enabled': tf.config.experimental.tensor_float_32_execution_enabled(),
        'functions_run_eagerly': tf.config.functions_run_eagerly(),
        'graph_options': tf.config.optimizer.get_experimental_options(),
        'n_training_probe': len(idx), 'sampled_training_indices': idx.tolist(),
        'n_external': len(pep), 'seed': 987654,
        'timing_scope': 'busy shared device; diagnostic batch timing only, not publication cost timing'}
    if a.method == 'AMPlify':
        spec = importlib.util.spec_from_file_location('amplify_gpu_probe', sota/'retrain_amplify_regr_neg.py')
        ra = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ra)
        model = ra.build_model()
        checkpoint = ROOT/'artifacts/repair_v15/external_runs/AMPlify/seed120/workspace/runs_review/benchmark_restricted/sota_models/amplify_regr_neg/amplify_s120.h5'
        model.load_weights(checkpoint)
        xext = ra.one_hot_padding(pep.sequence.tolist(), ra.MAXLEN)
        xtrain = ra.one_hot_padding(trainseq, ra.MAXLEN)
        ytrain = (target+4.)/9.
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(.001))
        inverse = lambda p: p*9.-4.
    else:
        repo = sota/'mbc_attention_repo'
        sys.path.insert(0, str(repo))
        os.chdir(repo)
        from tools.MultiBranchCNN import (CNNimportFtsDataSetsPoNe,
            CNNstandardInputOutput, best14, lrParas)
        checkpoint = ROOT/'artifacts/repair_v15/external_runs/MBC-Attention/seed123/workspace/runs_review/benchmark_restricted/sota_models/mbc_attention_neg/mbc_s123'
        model = tf.keras.models.load_model(checkpoint, compile=False)
        ext = pd.DataFrame({'ID': [f'e{i}' for i in range(len(pep))],
            'SEQUENCE': pep.sequence.tolist(), 'EC_pMIC': 0.})
        tr = pd.DataFrame({'ID': [f't{i}' for i in range(len(trainseq))],
            'SEQUENCE': trainseq, 'EC_pMIC': -target})
        xext, _ = CNNstandardInputOutput(CNNimportFtsDataSetsPoNe(ext, best14, None))
        xtrain, ytrain = CNNstandardInputOutput(CNNimportFtsDataSetsPoNe(tr, best14, 'EC_pMIC'))
        schedule = tf.keras.optimizers.schedules.ExponentialDecay(lrParas['lr'],
            lrParas['decay_steps'], lrParas['decay_rate'])
        model.compile(loss='mean_squared_logarithmic_error', optimizer=tf.keras.optimizers.Adam(schedule))
        inverse = lambda p: -(p*6.-3.8)
    files = [checkpoint] if checkpoint.is_file() else sorted(p for p in checkpoint.rglob('*') if p.is_file())
    before = {str(p): digest(p) for p in files}
    result.update(checkpoint=str(checkpoint), checkpoint_sha256=before,
        parameters=model.count_params(), variable_devices=sorted(set(v.device for v in model.trainable_variables)))
    assert all(('GPU:0' if a.device == 'gpu' else 'CPU:0') in v.device for v in model.trainable_variables)
    predicted = inverse(model.predict(xext, batch_size=32, verbose=0).ravel())
    assert len(predicted) == 208 and np.isfinite(predicted).all()
    pd.DataFrame({'sequence': pep.sequence, 'log10mic': predicted}).to_csv(OUT/f'{a.method}_{a.device}_predictions.csv', index=False)
    inference = []
    for _ in range(3):
        start = time.perf_counter()
        check = model.predict(xext, batch_size=32, verbose=0)
        inference.append(time.perf_counter()-start)
        assert np.isfinite(check).all()
        # Value parity on repeated queries, not only finite outputs.
        assert np.max(np.abs(inverse(check.ravel())-predicted)) <= 1e-4
    result['inference_forward_only_seconds'] = inference
    # Deterministic value-level backward check (dropout disabled in this one
    # diagnostic calculation). Actual fit below still uses the original dropout.
    loss_fn = tf.keras.losses.MeanSquaredError() if a.method == 'AMPlify' else tf.keras.losses.MeanSquaredLogarithmicError()
    xb = [v[:32] for v in xtrain] if isinstance(xtrain, list) else xtrain[:32]
    yb = tf.convert_to_tensor(np.asarray(ytrain[:32]).reshape(-1, 1), dtype=tf.float32)
    @tf.function
    def gradient_reference(inputs, labels):
        with tf.GradientTape() as tape:
            output = model(inputs, training=False)
            loss = loss_fn(labels, output)
        return loss, tape.gradient(loss, model.trainable_variables)
    ref_loss, gradients = gradient_reference(xb, yb)
    assert all(g is not None and np.isfinite(g.numpy()).all() for g in gradients)
    np.savez_compressed(OUT/f'{a.method}_{a.device}_gradients.npz',
        **{str(i): g.numpy() for i, g in enumerate(gradients)})
    result['deterministic_gradient_loss'] = float(ref_loss.numpy())
    result['gradient_variable_names'] = [v.name for v in model.trainable_variables]
    # The model's existing weights/architecture, loss, LR, dropout and batch size
    # are retained. Fresh diagnostic optimizer state; no promise of resumed training.
    initial = model.trainable_variables[-1].numpy().copy()
    train_times = []
    class Timing(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None): self.epoch = epoch
        def on_train_batch_begin(self, batch, logs=None): self.started = time.perf_counter()
        def on_train_batch_end(self, batch, logs=None):
            if self.epoch > 0: train_times.append(time.perf_counter()-self.started)
    history = model.fit(xtrain, ytrain, batch_size=32, epochs=3, shuffle=False,
        callbacks=[Timing()], verbose=2)
    result.update(train_batch_seconds=train_times,
        train_batch_median_seconds=float(np.median(train_times)),
        training_losses=[float(v) for v in history.history['loss']],
        weights_updated=bool(not np.array_equal(initial, model.trainable_variables[-1].numpy())),
        loss_finite=bool(np.isfinite(history.history['loss']).all()),
        checkpoint_hashes_unchanged=before == {str(p): digest(p) for p in files})
    assert result['weights_updated'] and result['loss_finite'] and result['checkpoint_hashes_unchanged']
    if a.device == 'gpu':
        result['gpu_logical_memory_mib'] = tf.config.get_logical_device_configuration(
            tf.config.list_physical_devices('GPU')[0])[0].memory_limit
        result['tf_memory_bytes'] = tf.config.experimental.get_memory_info('GPU:0')
        # Resolve actual loaded libraries, not merely installed distributions.
        result['loaded_cuda_libraries'] = sorted(set(line.split()[-1]
            for line in Path('/proc/self/maps').read_text().splitlines()
            if any(s in line for s in ['libcudnn', 'libcublas', 'libcudart'])))
        assert any('libcudnn' in s for s in result['loaded_cuda_libraries'])
    result['status'] = 'PASS'
    (OUT/f'{a.method}_{a.device}.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items()
        if k not in ['checkpoint_sha256', 'sampled_training_indices', 'train_batch_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()

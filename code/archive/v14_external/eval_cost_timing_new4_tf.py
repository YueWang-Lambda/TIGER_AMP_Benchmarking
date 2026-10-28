#!/usr/bin/env python3
"""Cost timing for the 4 retrained benchmark additions (TF/Keras side:
MBC-Attention, AMPlify-regr) on the actual external workload (208 peptides).

Protocol identical to scripts/eval_cost_timing_external.py: reconstruct the
submitted model (seed-123 weights from runs_review/benchmark_restricted/
sota_models/<method>/), warm up once, time 3 repeats of the full 208-peptide
inference, report the median. Per-query feature computation (iFeature
descriptors for MBC, one-hot for AMPlify) IS included; one-time model
loading is NOT. TF predict returns host arrays, so wall-clock timing is
already synchronized.

Sister script: eval_cost_timing_new4_torch.py (HyperAMP, SenseXAMP-Ecoli,
ccseg env) — both write disjoint method keys into the same output json.

Usage (conda mbc_att): cd <repo>/baseline/sota_bench && \
       CUDA_VISIBLE_DEVICES=0 python ../../scripts/eval_cost_timing_new4_tf.py
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOTA = ROOT / 'baseline/sota_bench'
OUT = ROOT / 'runs_review/benchmark_restricted/new4_cost_timing.json'
REPO_MBC = SOTA / 'mbc_attention_repo'
REPO_AMP = SOTA / 'amplify_repo'
# NOTE: ROOT (the v2 project root) must NOT go on sys.path - its `code/`
# package shadows the stdlib `code` module that TF's absl->pdb chain imports.
for p in (str(SOTA), str(ROOT / 'scripts'), str(REPO_MBC)):
    sys.path.insert(0, p)

PEP_CSV = ROOT / 'runs_review/final11_peptides.csv'


def pep():
    """Fresh read of the 208 query sequences inside every timed repeat."""
    return list(dict.fromkeys(pd.read_csv(PEP_CSV).sequence))
CKPT = ROOT / 'runs_review/benchmark_restricted/sota_models'


def timed(fn, reps=3):
    fn()  # warm-up
    ts = []
    for _ in range(reps):
        t0 = time.time()
        fn()
        ts.append(round(time.time() - t0, 4))
    return {'repeats_s': ts, 'median_s': sorted(ts)[1]}


# pair endpoint (v13b): the PRIMARY benchmark panel — 11 families, 2,211
# edges whose endpoints are exactly the 208 test peptides (full table spans
# 21 families / 2,748 edges; extra 537 non-benchmark-family edges NOT
# differenced). Canonical keys: pair_keys_primary_2211.csv; unit = LOG2.
PAIRS = None


def pair_delta(preds):
    """preds: {seq: log10-MIC}; returns the 2,211 ordered LOG2 deltas
    (evaluator semantics); non-finite => featuriser drop => loud assertion."""
    global PAIRS
    if PAIRS is None:
        K = pd.read_csv(ROOT / 'runs_review/benchmark_restricted/pair_keys_primary_2211.csv')
        PAIRS = list(zip(K.anchor_sequence, K.query_sequence))
        assert len(PAIRS) == 2211
    d = np.array([(preds.get(a, np.nan) - preds.get(b, np.nan)) / math.log10(2)
                  for a, b in PAIRS])
    assert np.isfinite(d).all(), 'primary-panel endpoint missing from preds'
    return d


def t_mbc_attention():
    """Per-query: best14 iFeature featurization (their inference path,
    pure-CPU python) + MultiBranchCNN forward. One-time: TF model load."""
    os.chdir(REPO_MBC)
    from tensorflow import keras
    from tools.MultiBranchCNN import (CNNimportFtsDataSetsPoNe,
                                      CNNstandardInputOutput, best14)
    model = keras.models.load_model(CKPT / 'mbc_attention_neg/mbc_s123')

    def run():
        P = pep()                                            # input re-read
        df = pd.DataFrame({'ID': [f'e{i}' for i in range(len(P))],
                           'SEQUENCE': P, 'EC_pMIC': 0.0})
        sets = CNNimportFtsDataSetsPoNe(df, ft_list=best14, target=None)
        X, _ = CNNstandardInputOutput(sets)
        y = model.predict(X, verbose=0)
        # per-query label inversion: sigmoid (pMIC+3.8)/6 -> pMIC -> log10 MIC
        mic = -(y * 6.0 - 3.8)
        pr = {s: float(mic[j]) for j, s in enumerate(P)}
        pair_delta(pr)                   # 2,211-edge endpoint (v13)
    return timed(run)


def t_amplify():
    """Per-query: one-hot padding + BiLSTM-attention forward (single seed-123
    model; the submitted configuration is one model per seed)."""
    os.chdir(REPO_AMP)
    import retrain_amplify_regr_neg as ra
    model = ra.build_model()
    model.load_weights(CKPT / 'amplify_regr_neg/amplify_s123.h5')

    def run():
        P = pep()
        X = ra.one_hot_padding(P, ra.MAXLEN)                # input re-read
        y = model.predict(X, verbose=0)
        # per-query label inversion: sigmoid (log10MIC+4)/9 -> log10 MIC
        mic = y * 9.0 - 4.0
        pr = {s: float(mic[j]) for j, s in enumerate(P)}
        pair_delta(pr)                   # 2,211-edge endpoint (v13)
    return timed(run)


METHODS = {'MBC-Attention+neg': t_mbc_attention,
           'AMPlify-regr (ours)': t_amplify}


def main():
    result = {}
    if OUT.exists():
        result = json.loads(OUT.read_text())
    result["n_peptides"] = len(pep())
    import tensorflow as tf
    result['device'] = (tf.config.list_physical_devices('GPU')[0].name
                        if tf.config.list_physical_devices('GPU') else 'cpu')
    result['batch_note'] = ('deployment-style; per-query feature computation '
                            'included, one-time model loading excluded '
                            '(protocol of eval_cost_timing_external.py)')
    result.setdefault('methods', {})
    for name, fn in METHODS.items():
        try:
            r = fn()
            # per-method device record (v13b): record the runtime this suite
            # actually used (TF may fall back to CPU in this env); do NOT
            # clobber the sister suite's top-level device field
            import tensorflow as tf_
            r['device'] = (tf_.config.list_physical_devices('GPU')[0].name
                           if tf_.config.list_physical_devices('GPU')
                           else 'cpu (TensorFlow)')
            result['methods'][name] = r
            print(f'[{name}] median {r["median_s"]}s  {r["repeats_s"]}',
                  flush=True)
        except Exception as e:
            result['methods'][name] = {'error': f'{type(e).__name__}: {e}'}
            print(f'[{name}] ERROR {type(e).__name__}: {e}', flush=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f'saved {OUT}', flush=True)


if __name__ == '__main__':
    main()

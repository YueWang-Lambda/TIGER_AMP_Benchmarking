#!/usr/bin/env python3
"""Retrain AMPlify (Li et al., BMC Genomics 2022, 10.1186/s12864-022-08634-9;
official code bcgsc/AMPlify, BirolLab fork) as a regressor on the shared
restricted corpus under the unified +neg protocol, seeds 120-129.
Reported as "AMPlify-regr (ours)".

Architecture (authors' build_amplify, verbatim; one-hot 20 -> Masking ->
BiLSTM(512, dropout 0.5) -> MultiHeadAttention(head_num=32, relu) ->
Dropout(0.2) -> Attention (Yang hierarchical) -> Dense(1, sigmoid)).
The published head is already Dense(1)+sigmoid (AMP probability); for
regression it is trained with MSE on affine-scaled labels instead of
binary cross-entropy - the same conversion pattern as BERT-AmPEP+neg /
Das2021-regr in this benchmark.

Training (authors' train_amplify.py): Adam lr 0.001, batch 32,
EarlyStopping(min_delta=0.001, patience=50, restore_best_weights=True,
monitor val_acc -> val_loss for regression), max 1000 epochs.

Adaptations (documented for METHODS):
  - regression conversion (loss/labels); single model per seed (the
    published 5-submodel bagging is replaced by our 10-seed protocol used
    for every benchmark method)
  - MAX_LEN 200 -> 50: masking (Masking + supports_masking in both custom
    attention layers) makes padded timesteps numerically inert, so this is
    exactly equivalent for sequences <= 49 aa (all corpus/external peptides)
    and ~4x faster
  - label scaling y = (log10MIC + 4) / 9 maps our label range
    [-3.31, 4.0] into (0.08, 0.89); inverted exactly at prediction
  - train/val split 90/10 per seed

Original author weights (models/*.h5) are NOT loaded.

Outputs (per seed):
  runs_review/benchmark_restricted/sota_amplify_regr_neg_s<seed>_preds.csv
  runs_review/benchmark_restricted/sota_models/amplify_regr_neg/amplify_s<seed>.h5

Usage (conda mbc_att): python baseline/sota_bench/retrain_amplify_regr_neg.py \
          [--seeds 120-129] [--smoke] [--cpu]
"""
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SOTA = Path(__file__).resolve().parent
REPO = SOTA / 'amplify_repo'
ROOT = SOTA.parent.parent
sys.path.insert(0, str(REPO / 'src'))

OUT = ROOT / 'runs_review/benchmark_restricted'
TRAIN_CSV = OUT / 'train_restricted4929_tiger.csv'
NEG_CSV = SOTA / 'data_restricted/uniprot_negatives.csv'
NEG_LOG10 = 4.0
PEP_CSV = ROOT / 'runs_review/final11_peptides.csv'

MAXLEN = 50  # see header: masking makes this equivalent to 200 for <=49 aa
HP = dict(lr=0.001, batch_size=32, epochs=1000, patience=50, min_delta=0.001)
SCALE_A, SCALE_B = 1.0 / 9.0, 4.0  # y = (log10mic + 4) / 9

AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
      'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


def one_hot_padding(seq_list, padding):
    """Authors' featurization, verbatim."""
    feat_list = []
    one_hot = {}
    for i in range(len(AA)):
        one_hot[AA[i]] = [0] * 20
        one_hot[AA[i]][i] = 1
    for s in seq_list:
        assert len(s) <= padding and set(s) <= set(AA)
        feat = [one_hot[c] for c in s]
        feat = feat + [[0] * 20] * (padding - len(s))
        feat_list.append(feat)
    return np.array(feat_list)


def build_model():
    """Authors' build_amplify, verbatim (MAX_LEN adapted, see header)."""
    from tensorflow import keras
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Masking, Dense, LSTM,
                                         Bidirectional, Input, Dropout)
    from layers import Attention, MultiHeadAttention
    inputs = Input(shape=(MAXLEN, 20), name='Input')
    masking = Masking(mask_value=0.0, input_shape=(MAXLEN, 20),
                      name='Masking')(inputs)
    hidden = Bidirectional(LSTM(512, use_bias=True, dropout=0.5,
                                return_sequences=True),
                           name='Bidirectional-LSTM')(masking)
    hidden = MultiHeadAttention(head_num=32, activation='relu',
                                use_bias=True,
                                return_multi_attention=False,
                                name='Multi-Head-Attention')(hidden)
    hidden = Dropout(0.2, name='Dropout_1')(hidden)
    hidden = Attention(name='Attention')(hidden)
    prediction = Dense(1, activation='sigmoid', name='Output')(hidden)
    model = Model(inputs=inputs, outputs=prediction)
    return model


def load_universe():
    tr = pd.read_csv(TRAIN_CSV)
    pos = tr.groupby('sequence', as_index=False).MIC_Escherichia_coli.mean()
    neg = pd.read_csv(NEG_CSV)
    seqs = pos.sequence.tolist() + neg.sequence.tolist()
    mics = np.concatenate([pos.MIC_Escherichia_coli.values.astype(float),
                           np.full(len(neg), 10.0 ** NEG_LOG10)])
    assert len(pos) == 4929 and len(neg) == 7962
    return seqs, np.log10(mics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='120-129')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()
    a, b = (int(x) for x in args.seeds.split('-'))
    seeds = list(range(a, b + 1))
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    import tensorflow as tf
    from tensorflow import keras
    print('TF', tf.__version__, 'GPU:', tf.config.list_physical_devices('GPU'),
          flush=True)

    seqs, log10mic = load_universe()
    pep = pd.read_csv(PEP_CSV)
    X_all = one_hot_padding(seqs, MAXLEN)
    X_ext = one_hot_padding(pep.sequence.tolist(), MAXLEN)
    y_all = (log10mic + SCALE_B) * SCALE_A

    epochs = 2 if args.smoke else HP['epochs']
    ckpt_dir = OUT / 'sota_models/amplify_regr_neg'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    n = len(seqs)

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        n_val = int(n * 0.1)
        va_i, tr_i = perm[:n_val], perm[n_val:]
        model = build_model()
        model.compile(loss='mse',
                      optimizer=keras.optimizers.Adam(learning_rate=HP['lr']))
        cb = keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=HP['min_delta'],
            patience=HP['patience'], restore_best_weights=True)
        hist = model.fit(X_all[tr_i], y_all[tr_i], epochs=epochs,
                         batch_size=HP['batch_size'],
                         validation_data=(X_all[va_i], y_all[va_i]),
                         verbose=2, callbacks=[cb])
        pred = model.predict(X_ext, verbose=0)[:, 0]
        log10mic_pred = pred / SCALE_A - SCALE_B
        out = pd.DataFrame({'log10mic': log10mic_pred},
                           index=pep.sequence.tolist())
        out.index.name = ''
        fn = OUT / f'sota_amplify_regr_neg_s{seed}_preds.csv'
        out.to_csv(fn)
        if not args.smoke:
            model.save_weights(ckpt_dir / f'amplify_s{seed}.h5')
        print(f'seed {seed}: epochs run {len(hist.history["loss"])}, '
              f'preds -> {fn.name} (head {np.round(log10mic_pred[:3], 3)})',
              flush=True)


if __name__ == '__main__':
    main()

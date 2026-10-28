#!/usr/bin/env python3
"""Retrain MBC-Attention (Yan et al., mSystems 2023, 10.1128/msystems.00345-23)
on the shared restricted corpus under the unified +neg protocol, seeds 120-129.

Recipe = the authors' final-model entry (tools/MultiBranchCNN.py::
DevelopfitAndSaveMBCAttenionMdl) kept verbatim:
  - best14 iFeature descriptors (PseKRAAC + QSOrder) -> one 2D feature image
    per descriptor -> 14-branch CNN, local self-attention per branch +
    global cross-branch attention (cnnArchtest)
  - sigmoid head on (pMIC + 3.8) * (1/6); MSLE loss
  - Adam lr 5e-4, ExponentialDecay(0.92, 25 steps); EarlyStopping on train
    loss, patience 15, validation_split 0.2; max 200 epochs; batch 32 (Keras
    default), dropout 0.4, filter 32, layer_num 1

Label universe (project standard): pMIC = -log10(MIC ug/mL) fed via target
column 'EC_pMIC'; negatives anchored at pseudo MIC 10,000 ug/mL (pMIC -4.0,
scaled label floor -0.0333 — protocol constant shared by every +neg method).
Predictions are written as log10 MIC ug/mL (benchmark consumer convention).
Original author weights (model/whole_train.mdl) are NOT loaded.

Outputs (per seed s):
  runs_review/benchmark_restricted/sota_mbc_attention_neg_s{s}_preds.csv
  runs_review/benchmark_restricted/sota_models/mbc_attention_neg/mbc_s{s}/ (SavedModel)

Usage: conda activate mbc_att && python retrain_mbc_attention_neg.py \
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
REPO = SOTA / 'mbc_attention_repo'
ROOT = SOTA.parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)  # author code writes log/ and split_data/ under its root

OUT = ROOT / 'runs_review/benchmark_restricted'
TRAIN_CSV = OUT / 'train_restricted4929_tiger.csv'
NEG_CSV = SOTA / 'data_restricted/uniprot_negatives.csv'
NEG_LOG10 = 4.0  # pseudo MIC 10,000 ug/mL
PEP_CSV = ROOT / 'runs_review/final11_peptides.csv'


def load_universe(no_neg=False):
    """4,929 positives (per-sequence MEAN) + 7,962 shared negatives, as
    train_sota_bench_neg.load_universe('with_neg')."""
    tr = pd.read_csv(TRAIN_CSV)
    pos = tr.groupby('sequence', as_index=False).MIC_Escherichia_coli.mean()
    if no_neg:                      # [noneg campaign] positives-only training
        assert len(pos) == 4929, len(pos)
        mics = pos.MIC_Escherichia_coli.values.astype(float)
        return pd.DataFrame({'ID': [f'p{i}' for i in range(len(pos))],
                             'SEQUENCE': pos.sequence.tolist(),
                             'EC_pMIC': -np.log10(mics).astype(float)})
    neg = pd.read_csv(NEG_CSV)
    seqs = pos.sequence.tolist() + neg.sequence.tolist()
    mics = np.concatenate([pos.MIC_Escherichia_coli.values.astype(float),
                           np.full(len(neg), 10.0 ** NEG_LOG10)])
    assert len(pos) == 4929 and len(neg) == 7949, (len(pos), len(neg))
    return pd.DataFrame({'ID': [f'p{i}' for i in range(len(pos))]
                                + [f'n{i}' for i in range(len(neg))],
                         'SEQUENCE': seqs,
                         'EC_pMIC': -np.log10(mics).astype(float)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='120-129')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--no-neg', action='store_true',
                    help='[noneg] train on the 4,929 positives only')
    args = ap.parse_args()
    a, b = (int(x) for x in args.seeds.split('-'))
    seeds = list(range(a, b + 1))

    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    import tensorflow as tf
    from tensorflow import keras
    from tools.MultiBranchCNN import (CNNimportFtsDataSetsPoNe,
                                      CNNstandardInputOutput, best14,
                                      cnnArchtest, hyperParaDict, lrParas)
    print('TF', tf.__version__, 'GPU:', tf.config.list_physical_devices('GPU'),
          flush=True)
    # final-model hyper-parameters shipped with the repo (module level of
    # MultiBranchCNN.py): lr 5e-4 / decay 0.92@25 / patience 15,
    # layer 1 / dropout 0.4 / filter 32 / 200 epochs
    assert lrParas == {'lr': 5e-4, 'decay_rate': 0.92, 'decay_steps': 25,
                       'patience': 15, 'monitor': 'loss'}, lrParas
    assert hyperParaDict == {'layer_num': 1, 'dropOutRate': 0.4,
                             'filter_num': 32, 'epoch': 200}, hyperParaDict

    train_df = load_universe(args.no_neg)
    pep = pd.read_csv(PEP_CSV)
    ext_df = pd.DataFrame({'ID': [f'e{i}' for i in range(len(pep))],
                           'SEQUENCE': pep.sequence, 'EC_pMIC': 0.0})

    print(f'featurizing {len(train_df)} train sequences (best14) ...',
          flush=True)
    train_sets = CNNimportFtsDataSetsPoNe(train_df, ft_list=best14,
                                          target='EC_pMIC')
    print('featurizing 208 external peptides ...', flush=True)
    ext_sets = CNNimportFtsDataSetsPoNe(ext_df, ft_list=best14, target=None)
    X, Y = CNNstandardInputOutput(train_sets)  # Y = (pMIC + 3.8) / 6
    Xe, _ = CNNstandardInputOutput(ext_sets)
    print('train X shapes:', [x.shape for x in X], 'Y:', Y.shape, flush=True)

    epochs = 2 if args.smoke else hyperParaDict['epoch']
    ckpt_dir = OUT / ('sota_models/mbc_attention_noneg' if args.no_neg
                      else 'sota_models/mbc_attention_neg')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lrParas['lr'],
            decay_steps=lrParas['decay_steps'],
            decay_rate=lrParas['decay_rate'])
        opt = keras.optimizers.Adam(learning_rate=lr_schedule)
        mdl = cnnArchtest(train_sets, hyperParaDict, opt)
        cb = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                              patience=lrParas['patience'])
        hist = mdl.fit(X, Y, epochs=epochs, validation_split=0.2,
                       callbacks=[cb], verbose=2)
        pred = mdl.predict(Xe, verbose=0)[:, 0]
        # inverse of (pMIC + 3.8) * (1/6), then pMIC -> log10 MIC (ug/mL)
        pmic = pred / (1.0 / 6.0) - 3.8
        log10mic = -pmic
        out = pd.DataFrame({'log10mic': log10mic},
                           index=pep.sequence.tolist())
        out.index.name = ''
        fn = OUT / (f'sota_mbc_attention_noneg_s{seed}_preds.csv' if args.no_neg
                    else f'sota_mbc_attention_neg_s{seed}_preds.csv')
        out.to_csv(fn)
        if not args.smoke:
            mdl.save(ckpt_dir / f'mbc_s{seed}')
        print(f'seed {seed}: epochs run {len(hist.history["loss"])}, '
              f'preds -> {fn.name} (head {log10mic[:3].round(3)})', flush=True)


if __name__ == '__main__':
    main()

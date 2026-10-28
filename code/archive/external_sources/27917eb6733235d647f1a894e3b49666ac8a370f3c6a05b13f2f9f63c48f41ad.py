#!/usr/bin/env python3
"""AMPActiPred third-stage regressor (Protein Science 2024, 10.1002/pro.5006)
retrained under the unified "+neg" protocol (v2-9). Run in the `ampacti` env.

Fidelity notes:
- The repo ships no training code; we train its published feature pipeline +
  model family: construct_features() from predict.py (AAC20 + DPC400 +
  CKSAAGP(gap=3)100 + PAAC(lambda=4,w=0.5)24 + PHYC9 = 553 dims, vendor-
  imported) -> deep-forest CascadeForestRegressor (library defaults; the paper
  gives no released hyperparameters — documented).
- Third-stage E. coli continuous-activity module; label domain native
  log10(uM) -> unified log10(ug/mL) fed directly (MBC precedent).
- Original three-stage model files (external links) never loaded; the
  first/second-stage classification gates are bypassed (regression-only arm).
- Full-length input (repo has no truncation; min length 6 aa satisfied).
- Seeds: numpy seed + deep-forest random_state (0.1.5 exposes set_random_state).

Usage:
  once:  python retrain_ampacti_neg.py --features
  xseed: python retrain_ampacti_neg.py --seed 120
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE / 'AMPActiPred'
V2 = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC/artifacts/repair_v15/external_runs/AMPActiPred/seed121/workspace')
CACHE = HERE / 'emb_cache'
OUT = V2 / 'runs_review/benchmark_restricted/sota_models/ampacti_neg'
OUT_NONEG = V2 / 'runs_review/benchmark_restricted/sota_models/ampacti_noneg'
FEAT = {'corpus': CACHE / 'ampacti_feat_corpus.csv', 'ext': CACHE / 'ampacti_feat_ext.csv'}


def corpus(no_neg=False):
    tr = pd.read_csv(V2 / 'runs_review/benchmark_restricted/train_restricted4929_tiger.csv')
    pos = tr.groupby('sequence', as_index=False).MIC_Escherichia_coli.mean()
    if no_neg:                      # [noneg campaign] positives-only training
        assert len(pos) == 4929, len(pos)
        return pos
    neg = pd.read_csv(HERE / 'data_restricted/uniprot_negatives.csv')[['sequence']]
    neg['MIC_Escherichia_coli'] = 10000.0
    return pd.concat([pos, neg], ignore_index=True)


def ext_table():
    return pd.read_csv(V2 / 'runs_review/final11_peptides.csv')[
        ['sequence', 'MIC_E_coli_ug_mL']].dropna()


def run_features():
    sys.path.insert(0, str(REPO))
    os.chdir(REPO)          # PAAC.txt 等 data/ 相对路径依赖
    from predict_lib import construct_features
    for tag, df in (('corpus', corpus()), ('ext', ext_table())):
        seqs = df.sequence.astype(str).str.upper().str.strip()
        d = pd.DataFrame({'Id': [f'p{i}' for i in range(len(seqs))],
                          'Sequence': seqs})
        feats = construct_features(d)
        feats.to_csv(FEAT[tag], index=False)
        print(f'features[{tag}]:', feats.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', action='store_true')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--no-neg', action='store_true',
                    help='[noneg] train on the 4,929 positives only; reuses the cached '
                         'corpus features filtered to positive sequences')
    args = ap.parse_args()
    if args.features:
        run_features()
        return
    assert args.seed is not None
    np.random.seed(args.seed)
    import deepforest
    from deepforest import CascadeForestRegressor
    ftr = pd.read_csv(FEAT['corpus'])
    fex = pd.read_csv(FEAT['ext'])
    tr = corpus(args.no_neg)
    if args.no_neg:                 # filter the cached +neg corpus features to pos rows
        # (drop_duplicates first: 13 sequences sit in BOTH pools — see the
        #  eippred note; positive row first in the pos-then-neg order)
        ftr = (ftr.drop_duplicates('Sequence', keep='first')
               .set_index('Sequence').loc[tr.sequence].reset_index())
    assert list(ftr.Sequence) == list(tr.sequence)
    assert (not args.no_neg) or len(ftr) == 4929, len(ftr)
    xcols = [c for c in ftr.columns if c not in ('Sequence', 'Id')]
    assert len(xcols) == 553, f'特征列 {len(xcols)} != 553'
    X = ftr[xcols].values
    y = np.log10(tr.MIC_Escherichia_coli.values.astype(float))
    model = CascadeForestRegressor(random_state=args.seed, n_jobs=2)
    try:
        model.set_random_state(args.seed)
    except Exception:
        pass
    model.fit(X, y)
    pred = np.asarray(model.predict(fex[xcols].values)).ravel()
    out_dir = OUT_NONEG if args.no_neg else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    o = pd.DataFrame({'log10mic': pred})
    o.index = ext_table().sequence.values
    o.index.name = 'sequence'
    name = (f'sota_AMPActiPred_noneg_s{args.seed}_preds.csv' if args.no_neg
            else f'sota_AMPActiPred+neg_s{args.seed}_preds.csv')
    o.to_csv(out_dir / name)
    model.save(str(out_dir / 'model'))
    (out_dir / 'feature_columns.json').write_text(__import__('json').dumps(xcols))
    print(f'seed {args.seed}: preds range [{pred.min():.2f},{pred.max():.2f}] -> saved '
          f'(deep-forest 0.1.5)')


if __name__ == '__main__':
    main()

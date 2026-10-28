#!/usr/bin/env python3
"""EIPpred (Sci Rep 2025, 10.1038/s41598-025-86638-z) retrained as an E. coli
MIC regressor under the unified "+neg" protocol (v2-9).

Fidelity notes (adaptations, all documented):
- The repo ships no training code; we retrain its published recipe verbatim:
  19 Pfeature descriptor groups (vendored eippred.features_calulate) ->
  the repo's mRMR1000 column set (Data/selected_features_mrmr1000_new.csv) ->
  sklearn RandomForestRegressor(random_state=seed) with default parameters.
- Label domain: native EIPpred is -log10(MIC uM); under the unified protocol we
  feed -log10(MIC ug/mL) directly (unit redefined by training labels, MBC
  precedent); predictions are inverted exactly (log10mic = -pred).
- Full-length input: the repo's lenchk() 30-aa truncation (predict path only)
  is bypassed — descriptors are length-normalized and have no length cap.
- Negative pool anchors at log10 = 4.0 ug/mL -> label -4.0.
- Original author weights (Data/model2.pkl) never loaded; the repo's
  `2**x` inverse in eippred.py is buggy and not used (we invert with -x).

Usage:
  step1 (once):  python retrain_eippred_neg.py --features
  step2 (xseed): python retrain_eippred_neg.py --seed 120
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE / 'EIPpred'
V2 = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC/artifacts/repair_v15/external_runs/EIPpred/seed127/workspace')
CACHE = HERE / 'emb_cache'
OUT = V2 / 'runs_review/benchmark_restricted/sota_models/eippred_neg'
OUT_NONEG = V2 / 'runs_review/benchmark_restricted/sota_models/eippred_noneg'
PY = sys.executable

FEAT_TRAIN = CACHE / 'eippred_feat_corpus.csv'
FEAT_EXT = CACHE / 'eippred_feat_ext.csv'


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
    e = pd.read_csv(V2 / 'runs_review/final11_peptides.csv')
    return e[['sequence', 'MIC_E_coli_ug_mL']].dropna()


def run_features():
    CACHE.mkdir(exist_ok=True)
    for tag, df in (('corpus', corpus()), ('ext', ext_table())):
        seqs = df.sequence.astype(str).str.upper().str.strip()
        # EIPpred feature funcs expect a no-header one-sequence-per-line file
        tmp = CACHE / f'eippred_in_{tag}.txt'
        tmp.write_text('\n'.join(seqs) + '\n')
        # features_calulate redirects sys.stdout internally -> isolate in a
        # subprocess so the driver's own stdout survives
        code = (f"import sys; sys.path.insert(0, r'{REPO}');"
                f"import eippred; eippred.features_calulate(r'{tmp}', r'{CACHE}/eippred_feat_{tag}.csv')")
        subprocess.run([PY, '-c', code], check=True, cwd=str(REPO))
        f = pd.read_csv(CACHE / f'eippred_feat_{tag}.csv')
        f = f.loc[:, ~f.columns.str.startswith('Unnamed')]
        f.insert(0, 'sequence', seqs.values)
        f.to_csv(CACHE / f'eippred_feat_{tag}.csv', index=False)
        print(f'features[{tag}]:', f.shape)


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
    from sklearn.ensemble import RandomForestRegressor
    mrmr = pd.read_csv(REPO / 'Data/selected_features_mrmr1000_new.csv')['SelectedFeatures'].tolist()
    tr = corpus(args.no_neg)
    ftr = pd.read_csv(FEAT_TRAIN)
    fex = pd.read_csv(FEAT_EXT)
    if args.no_neg:                 # filter the cached +neg corpus features to pos rows
        # (drop_duplicates first: 13 sequences sit in BOTH pools and the cache
        #  holds two rows each — .loc on a duplicated index would expand rows;
        #  the positive row comes first in the pos-then-neg corpus order)
        ftr = (ftr.drop_duplicates('sequence', keep='first')
               .set_index('sequence').loc[tr.sequence].reset_index())
    assert list(ftr.sequence) == list(tr.sequence)
    assert (not args.no_neg) or len(ftr) == 4929, len(ftr)
    cols = [c for c in mrmr if c in ftr.columns]
    assert len(cols) == len(mrmr), f'mRMR 列缺失: {set(mrmr) - set(cols)}'
    X = ftr[cols].values
    y = -np.log10(tr.MIC_Escherichia_coli.values.astype(float))
    Xe = fex[cols].values
    clf = RandomForestRegressor(random_state=args.seed, n_jobs=2)
    clf.fit(X, y)
    pred = clf.predict(Xe)
    out_dir = OUT_NONEG if args.no_neg else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({'log10mic': -pred})
    out.index = ext_table().sequence.values
    out.index.name = 'sequence'
    name = (f'sota_EIPpred_noneg_s{args.seed}_preds.csv' if args.no_neg
            else f'sota_EIPpred+neg_s{args.seed}_preds.csv')
    out.to_csv(out_dir / name)
    import joblib
    joblib.dump(clf, out_dir / 'model.joblib')
    (out_dir / 'feature_columns.json').write_text(__import__('json').dumps(cols))
    print(f'seed {args.seed}: preds saved, range [{pred.min():.2f},{pred.max():.2f}] '
          f'(-log10 domain)')


if __name__ == '__main__':
    main()

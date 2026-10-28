#!/usr/bin/env python3
"""Ablation: HydrAMP M_MIC architecture retrained as a log10MIC regressor.

Question this answers: how much of HydrAMP-MIC's benchmark gap to TIGER-pair is
due to the paper's deliberate binarization (P(MIC<=32) threshold) versus the
representation ceiling of the architecture itself?

Held fixed vs the binary M_MIC retrain (retrain_hydramp_mic.py):
  - identical architecture: Emb(21->128) - Conv1d(64,k16,same) - MaxPool(5)
    - LSTM(100) - Dense(1); the only change is linear output + MSE
  - identical training set: 4,929 restricted AMPs (labels: log10 of the
    per-sequence mean MIC) + the same 2,936 UniProt non-AMPs <=25 aa
    (labels: log10MIC = 4, the DeepAMP/Pandi anchoring convention)
  - identical MAX_LENGTH 50 padding, 10-fold CV at seed 36 (KFold: stratified
    splits need discrete labels), ensemble = mean of the 10 fold models

Binary-specific recipe elements dropped by construction (disclosed): the x5
positive oversampling and the class-stratified folds are class-imbalance
mechanics without a regression analogue; epochs capped at 100 with the same
EarlyStopping(patience 7, val MSE, last weights) to let MSE converge.

Output: runs_review/benchmark_restricted/sota_models/hydramp_regr/fold*.pt +
        sota_hydramp_regr_preds.csv (log10MIC, 208 peptides) + per-family PCC.
"""
import argparse
import os
import sys
from pathlib import Path

# Import the project package BEFORE numpy/pandas/torch: their import chains
# pull in the stdlib `code` module, and once sys.modules['code'] is the stdlib
# module our package can no longer be resolved regardless of sys.path order.
TIGER = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC/artifacts/repair_v15/external_runs/HydrAMP/seed123/workspace')
sys.path.insert(0, str(TIGER))
from code.utils.metrics import regression_metrics  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from retrain_hydramp_mic import (MICClassifier, NEG_CSV, PEP_CSV, TRAIN_CSV,  # noqa: E402
                                 encode, pad_batch)

TIGER = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC/artifacts/repair_v15/external_runs/HydrAMP/seed123/workspace')
CKPT_DIR = TIGER / 'runs_review/benchmark_restricted/sota_models/hydramp_regr'
OUT_DIR = TIGER / 'runs_review/benchmark_restricted'
SEED = 36
NEG_LOG10MIC = 4.0


def fit_one(x_tr, y_tr, x_val, y_val, device, epochs=100, batch=64, patience=7):
    model = MICClassifier().to(device)      # forward() returns the logit; used raw
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    best, bad = np.inf, 0
    n = len(x_tr)
    for ep in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = pad_batch([x_tr[j] for j in idx]).to(device)
            yb = torch.tensor([y_tr[j] for j in idx], dtype=torch.float32).to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([
                lossf(model(pad_batch(x_val[i:i + 4096]).to(device)).cpu(),
                      torch.tensor(y_val[i:i + 4096], dtype=torch.float32)).item()
                for i in range(0, len(x_val), 4096)]))
        bad = 0 if vl < best else bad + 1
        best = min(best, vl)
        if bad >= patience:
            break
    return model, best, ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--seed', type=int, default=SEED,
                    help='[noneg campaign] protocol seed (default 36 = historical); '
                         'names the per-seed preds CSV')
    ap.add_argument('--no-neg', action='store_true',
                    help='[noneg] train on the 4,929 positives only (drops the '
                         '<=25 aa negative subset); skips the eleven_way column '
                         'write so the +neg archive column is untouched')
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    print(f'device={device}, seed={args.seed}', flush=True)

    tr = pd.read_csv(TRAIN_CSV)
    pos = tr.groupby('sequence', as_index=False).MIC_Escherichia_coli.mean()
    neg = pd.read_csv(NEG_CSV)
    neg = neg[neg.sequence.str.len() <= 25].reset_index(drop=True)

    pos_rows = encode(pos.sequence.tolist())
    keep = [i for i, r in enumerate(pos_rows) if r is not None]
    x_pos = [pos_rows[i] for i in keep]
    y_pos = np.log10(pos.MIC_Escherichia_coli.values.astype(float))[keep]
    if args.no_neg:                 # [noneg campaign] positives-only training
        assert len(pos) == 4929, len(pos)
        x_all, y_all = x_pos, y_pos.astype(np.float32)
        print(f'regression set [noneg]: {len(x_pos)} AMPs only '
              f'(log10MIC [{y_pos.min():.2f},{y_pos.max():.2f}])', flush=True)
    else:
        neg_rows = encode(neg.sequence.tolist())
        x_neg = [r for r in neg_rows if r is not None]
        y_neg = np.full(len(x_neg), NEG_LOG10MIC)
        x_all = x_pos + x_neg
        y_all = np.concatenate([y_pos, y_neg]).astype(np.float32)
        print(f'regression set: {len(x_pos)} AMPs (log10MIC [{y_pos.min():.2f},{y_pos.max():.2f}]) '
              f'+ {len(x_neg)} negatives @ log10MIC={NEG_LOG10MIC}', flush=True)

    from sklearn.model_selection import KFold
    cv = KFold(n_splits=10, shuffle=True, random_state=args.seed)

    pep = pd.read_csv(PEP_CSV)
    uniq = list(dict.fromkeys(pep.sequence))
    q_rows = encode(uniq)
    kept = [(s, r) for s, r in zip(uniq, q_rows) if r is not None]

    ckpt_dir = (TIGER / 'runs_review/benchmark_restricted/sota_models/hydramp_regr_noneg'
                ) if args.no_neg else CKPT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fold_preds, val_mses = [], []
    for f, (tri, vai) in enumerate(cv.split(np.zeros(len(x_all))), 1):
        x_tr = [x_all[i] for i in tri]
        y_tr = y_all[tri]
        x_va = [x_all[i] for i in vai]
        y_va = y_all[vai]
        model, vl, ep = fit_one(x_tr, y_tr, x_va, y_va, device)
        torch.save({'model': model.state_dict(), 'fold': f}, ckpt_dir / f'fold{f}.pt')
        model.eval()
        with torch.no_grad():
            ps = []
            for i in range(0, len(kept), 4096):
                xb = pad_batch([r for _, r in kept[i:i + 4096]]).to(device)
                ps.append(model(xb).cpu().numpy())          # raw linear output
            fold_preds.append(np.concatenate(ps))
        val_mses.append(vl)
        print(f'[fold {f}] stopped ep{ep} val_MSE={vl:.4f}', flush=True)

    ckpt_dir = (TIGER / 'runs_review/benchmark_restricted/sota_models/hydramp_regr_noneg'
                ) if args.no_neg else CKPT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ens = np.mean(fold_preds, axis=0)
    preds = dict(zip([s for s, _ in kept], ens.tolist()))
    pd.Series(preds, name='log10mic').to_csv(
        OUT_DIR / (f'sota_hydramp_regr_noneg_s{args.seed}_preds.csv' if args.no_neg
                   else 'sota_hydramp_regr_preds.csv'))
    print(f'ensemble saved; mean val MSE={np.mean(val_mses):.4f}; '
          f'test log10MIC std={ens.std():.3f} range [{ens.min():.2f},{ens.max():.2f}]', flush=True)

    # --- pair-delta evaluation on the 11-family benchmark ---
    import math
    # merge onto eleven_way_predictions.csv when present so the other new-method
    # columns survive; fall back to eight_way on a fresh checkout
    base = OUT_DIR / 'eleven_way_predictions.csv'
    A = pd.read_csv(base if base.exists() else OUT_DIR / 'eight_way_predictions.csv')
    if 'hydramp_regr_delta' in A.columns:
        A = A.drop(columns=['hydramp_regr_delta'])
    A['hydramp_regr_delta'] = [
        (preds.get(a, np.nan) - preds.get(q, np.nan)) / math.log10(2)
        if a in preds and q in preds else np.nan
        for a, q in zip(A.anchor_sequence, A.query_sequence)]
    YT = 'y_true_delta_log2_anchor_minus_query_03'
    KEEP = ['Mammuthusin-3', 'F', 'Hesperelin-3', 'Hydrodamin-3', 'Hydrodamin-2', 'Repeat',
            'LL37', 'Mammuthusin-2', 'Arctoterin-1', 'HAL-2/', 'HAL-1/']
    order = ['Mammuthusin-3', 'F', 'Hesperelin-3', 'Hydrodamin-3', 'Hydrodamin-2', 'Repeat',
             'HAL-2', 'LL37', 'Mammuthusin-2', 'Arctoterin-1', 'HAL-1']
    S = A[A.family.isin(KEEP)]
    ps = []
    for fam in order:
        g = S[S.family == (fam + '/' if fam.startswith('HAL') else fam)].dropna(subset=['hydramp_regr_delta'])
        m = regression_metrics(g[YT], g.hydramp_regr_delta) if len(g) >= 4 else {'PCC': float('nan')}
        ps.append(m['PCC'])
        print(f'{fam:<13} n={len(g):>3}  PCC={m["PCC"]:.3f}', flush=True)
    print(f'MACRO PCC = {np.nanmean(ps):.4f}', flush=True)
    if not args.no_neg:
        A.to_csv(OUT_DIR / 'eleven_way_predictions.csv', index=False)
    else:
        print('[noneg] eleven_way column write SKIPPED (+neg archive column preserved)',
              flush=True)


if __name__ == '__main__':
    main()

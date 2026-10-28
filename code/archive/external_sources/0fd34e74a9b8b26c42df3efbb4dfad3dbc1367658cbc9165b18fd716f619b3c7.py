#!/usr/bin/env python3
"""HydrAMP M_MIC architecture as a log10MIC regressor on the UNIFIED negative
universe (experiment 12 supplement, user-requested).

Identical to retrain_hydramp_regressor.py in every respect (architecture:
Emb(21->128) -> Conv1d(64,k16,same) -> MaxPool(5) -> LSTM(100) -> Dense(1,
linear); MSE on log10MIC; 10-fold KFold(shuffle, seed 36) ensemble = mean of
the 10 fold models; Adam 1e-3, batch 64, epochs<=100, EarlyStopping(7) on val
MSE; positives 4,929 with per-sequence mean MIC) — except the negative
universe:

  original: 2,936 UniProt non-AMPs <=25 aa (the paper's length rule)
  here    : the FULL shared pool (7,962) at log10MIC = 4.0 — the
            experiment-12 unified protocol (disclosed)

Output: runs_review/benchmark_restricted/sota_models/hydramp_regr_neg/fold*.pt
      + sota_hydramp_regr_neg_preds.csv (log10MIC, 208 peptides).
"""
import argparse
import sys
from pathlib import Path

TIGER = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC/artifacts/repair_v15/external_runs/HydrAMP/seed121/workspace')
sys.path.insert(0, str(TIGER))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import retrain_hydramp_regressor as base  # noqa: E402  (verbatim pieces)
from retrain_hydramp_mic import encode, pad_batch  # noqa: E402

OUT_DIR = TIGER / 'runs_review/benchmark_restricted'
CKPT_DIR = OUT_DIR / 'sota_models/hydramp_regr_neg'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--seed', type=int, default=36)
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(f'device={device}, seed={SEED} — UNIFIED FULL POOL variant', flush=True)

    tr = pd.read_csv(base.TRAIN_CSV)
    pos = tr.groupby('sequence', as_index=False).MIC_Escherichia_coli.mean()
    neg = pd.read_csv(base.NEG_CSV)  # full 7,962 — NO <=25 aa filter (disclosed)

    pos_rows = encode(pos.sequence.tolist())
    keep = [i for i, r in enumerate(pos_rows) if r is not None]
    x_pos = [pos_rows[i] for i in keep]
    y_pos = np.log10(pos.MIC_Escherichia_coli.values.astype(float))[keep]
    neg_rows = encode(neg.sequence.tolist())
    x_neg = [r for r in neg_rows if r is not None]
    y_neg = np.full(len(x_neg), base.NEG_LOG10MIC)
    x_all = x_pos + x_neg
    y_all = np.concatenate([y_pos, y_neg]).astype(np.float32)
    print(f'regression set: {len(x_pos)} AMPs (log10MIC [{y_pos.min():.2f},{y_pos.max():.2f}]) '
          f'+ {len(x_neg)} FULL-POOL negatives @ log10MIC={base.NEG_LOG10MIC}', flush=True)

    from sklearn.model_selection import KFold
    cv = KFold(n_splits=10, shuffle=True, random_state=SEED)

    pep = pd.read_csv(base.PEP_CSV)
    uniq = list(dict.fromkeys(pep.sequence))
    q_rows = encode(uniq)
    kept = [(s, r) for s, r in zip(uniq, q_rows) if r is not None]

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    fold_preds, val_mses = [], []
    for f, (tri, vai) in enumerate(cv.split(np.zeros(len(x_all))), 1):
        model, vl, ep = base.fit_one([x_all[i] for i in tri], y_all[tri],
                                     [x_all[i] for i in vai], y_all[vai], device)
        torch.save({'model': model.state_dict(), 'fold': f}, CKPT_DIR / f'fold{f}.pt')
        model.eval()
        with torch.no_grad():
            ps = []
            for i in range(0, len(kept), 4096):
                xb = pad_batch([r for _, r in kept[i:i + 4096]]).to(device)
                ps.append(model(xb).cpu().numpy())          # raw linear output
            fold_preds.append(np.concatenate(ps))
        val_mses.append(vl)
        print(f'[fold {f}] stopped ep{ep} val_MSE={vl:.4f}', flush=True)

    ens = np.mean(fold_preds, axis=0)
    suffix = '' if SEED == 36 else f'_s{SEED}'
    pd.Series(dict(zip([s for s, _ in kept], ens.tolist())),
              name='log10mic').to_csv(OUT_DIR / f'sota_hydramp_regr_neg{suffix}_preds.csv')
    print(f'saved sota_hydramp_regr_neg_preds.csv; mean val MSE={np.mean(val_mses):.4f}; '
          f'test log10MIC std={ens.std():.3f} range [{ens.min():.2f},{ens.max():.2f}]', flush=True)


if __name__ == '__main__':
    main()

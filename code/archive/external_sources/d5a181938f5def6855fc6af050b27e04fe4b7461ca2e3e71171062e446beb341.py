#!/usr/bin/env python3
"""Retrain the Das et al. 2021 BiLSTM encoder as a log10MIC REGRESSOR on the
TIGER restricted training universe (experiment 12 / negative-protocol
unification; ours variant — NOT the paper's classifier, do not list as their
method).

Same encoder and training recipe as retrain_das_2021.py (BiLSTM hidden 100,
dropout 0.3, one-hot 50x20, Adam 1e-3, ~80/10/10 split rng 42, early stopping
patience 7 on val loss); only the head and loss change (Linear(200->1), MSE on
log10MIC), mirroring the HydrAMP-arch-regr convention:

  positives = 4,929 restricted AMPs, label = log10(per-sequence mean MIC)
  negatives = the shared UniProt non-AMP pool (7,962), pseudo log10MIC = 4.0
              (= 10,000 ug/mL; the DeepAMP/TIGER cap anchoring convention)

Output: runs_review/benchmark_restricted/sota_das_regr_preds.csv (log10MIC,
208 external test peptides).
"""
import sys
from pathlib import Path

ROOT = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC/artifacts/repair_v15/external_runs/Das/seed123/workspace')
TIGER = ROOT
sys.path.insert(0, str(TIGER))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

OUT = TIGER / 'runs_review/benchmark_restricted'
CKPT_DIR = OUT / 'sota_models/das_regr'
AA = 'ACDEFGHIKLMNPQRSTVWY'
AA2I = {c: i for i, c in enumerate(AA)}
MAX_LEN = 50
NEG_LOG10 = 4.0  # pseudo MIC 10,000 ug/mL


def encode(seqs):
    X = np.zeros((len(seqs), MAX_LEN, 20), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s[:MAX_LEN]):
            if c in AA2I:
                X[i, j, AA2I[c]] = 1.0
    return X


class BiLSTMRegressor(nn.Module):
    """Das et al. 2021 sequence-level encoder with a regression head (ours)."""

    def __init__(self, hidden=100, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=20, hidden_size=hidden, batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)                      # (B, L, 2H)
        pooled = out.max(dim=1).values
        return self.head(self.drop(pooled)).squeeze(-1)  # log10MIC


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--no-neg', action='store_true',
                    help='[noneg] train on the 4,929 positives only')
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'device={device} seed={args.seed}', flush=True)

    tr = pd.read_csv(OUT / 'train_restricted4929_tiger.csv')
    pos = tr.groupby('sequence', as_index=False).MIC_Escherichia_coli.mean()
    pos_seqs = pos.sequence.tolist()
    pos_y = np.log10(pos.MIC_Escherichia_coli.values.astype(float))
    if args.no_neg:                 # [noneg campaign] positives-only training
        assert len(pos_seqs) == 4929, len(pos_seqs)
        seqs, y = pos_seqs, pos_y.astype(np.float32)
        print(f'positives={len(seqs)} negatives=0 [noneg]; '
              f'label range [{y.min():.2f},{y.max():.2f}]', flush=True)
    else:
        neg = pd.read_csv(ROOT / 'baseline/sota_bench/data_restricted/uniprot_negatives.csv')
        seqs = pos_seqs + neg.sequence.tolist()
        y = np.concatenate([pos_y, np.full(len(neg), NEG_LOG10)]).astype(np.float32)
        print(f'positives={len(pos_seqs)} negatives={len(neg)}; '
              f'label range [{y.min():.2f},{y.max():.2f}]', flush=True)
    X = encode(seqs)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    n_tr, n_va = int(0.8 * len(X)), int(0.1 * len(X))
    tr_i, va_i = idx[:n_tr], idx[n_tr:n_tr + n_va]

    model = BiLSTMRegressor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    Xs = {k: torch.from_numpy(X[v]) for k, v in [('tr', tr_i), ('va', va_i)]}
    ys = {k: torch.from_numpy(y[v]) for k, v in [('tr', tr_i), ('va', va_i)]}

    best_val, best_state, stale = float('inf'), None, 0
    for ep in range(1, 51):
        model.train()
        perm = torch.randperm(len(tr_i))
        losses = []
        for i in range(0, len(tr_i), 256):
            b = perm[i:i + 256]
            xb, yb = Xs['tr'][b].to(device), ys['tr'][b].to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            vl = float(lossf(model(Xs['va'].to(device)), ys['va'].to(device)).cpu())
        print(f'ep{ep} train_loss={np.mean(losses):.4f} val_loss={vl:.4f}', flush=True)
        if vl < best_val - 1e-4:
            best_val, stale = vl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 7:
            break
    model.load_state_dict(best_state)
    ckpt_dir = (OUT / 'sota_models/das_regr_noneg') if args.no_neg else CKPT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict()}, ckpt_dir / 'bilstm_regr_best.pt')

    pep = pd.read_csv(TIGER / 'runs_review/final11_peptides.csv')
    uniq = list(dict.fromkeys(pep.sequence))
    Xq = torch.from_numpy(encode(uniq))
    with torch.no_grad():
        preds = []
        for i in range(0, len(uniq), 512):
            preds.append(model(Xq[i:i + 512].to(device)).cpu().numpy())
    preds = np.concatenate(preds)
    fn = ('sota_das_regr_noneg_s%d_preds.csv' % args.seed if args.no_neg else
          ('sota_das_regr_preds.csv' if args.seed == 123
           else f'sota_das_regr_s{args.seed}_preds.csv'))
    pd.Series(dict(zip(uniq, preds.tolist())), name='log10mic').to_csv(OUT / fn)
    print(f'saved sota_das_regr_preds.csv; log10MIC mean={preds.mean():.3f} '
          f'std={preds.std():.3f} range [{preds.min():.3f},{preds.max():.3f}]', flush=True)


if __name__ == '__main__':
    main()

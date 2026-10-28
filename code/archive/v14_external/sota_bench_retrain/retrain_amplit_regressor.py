#!/usr/bin/env python3
"""Retrain AMPLiT (Chen et al., Nat Commun 17:1788, 2026) as a log10MIC
REGRESSOR on the TIGER restricted training universe (experiment 12 /
negative-protocol unification; ours variant — NOT the paper's classifier).

Architecture and training recipe are imported verbatim from retrain_amplit.py
(three-branch DenseNet, Adam lr 1e-4 betas 0.9/0.999 eps 1e-8 wd 1e-6,
batch 512, 21 epochs, grad clip 5.0, 10% validation split rng 42); only the
head and loss change (Dense(16->1) linear, MSE), mirroring the
HydrAMP-arch-regr convention:

  positives = 4,929 restricted AMPs, label = log10(per-sequence mean MIC)
  negatives = the shared UniProt non-AMP pool (7,962), pseudo log10MIC = 4.0
              (= 10,000 ug/mL; the DeepAMP/TIGER cap anchoring convention)

The /tmp/amplit_feats.npz feature cache built for the classifier (same
pos+neg sequence order) is reused when present.

Output: runs_review/benchmark_restricted/sota_amplit_regr_preds.csv (log10MIC,
208 external test peptides).
"""
import sys
from pathlib import Path

ROOT = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC')
TIGER = ROOT
sys.path.insert(0, str(TIGER))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from retrain_amplit import (  # noqa: E402  (verbatim port of the official code)
    EPOCHS, LR, BATCH, WD, K1, K2, K3, FP_W, Branch, featurize,
)
import retrain_amplit as ra  # noqa: E402

OUT = TIGER / 'runs_review/benchmark_restricted'
CKPT_DIR = OUT / 'sota_models/amplit_regr'
NEG_LOG10 = 4.0  # pseudo MIC 10,000 ug/mL


class AMPLiTRegressor(nn.Module):
    """AMPLiT with a linear regression head (ours variant)."""

    def __init__(self):
        super().__init__()
        self.b1 = Branch(20, K1, second_block=True)
        self.b2 = Branch(100, K2, second_block=True)
        self.b3 = Branch(FP_W, K3, second_block=True)
        self.head = None  # built lazily on first forward

    def _build_head(self, ch, length):
        self.cbam = ra.CBAM(ch)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(ch * length, 16), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(16, 1))

    def forward(self, x1, x2, x3):
        h = torch.cat([self.b1(x1), self.b2(x2), self.b3(x3)], dim=1)
        if self.head is None:
            self._build_head(h.size(1), h.size(2))
            self.to(h.device)
        return self.head(self.cbam(h)).squeeze(-1)  # log10MIC


def main():
    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}', flush=True)

    tr = pd.read_csv(OUT / 'train_restricted4929_tiger.csv')
    pos = tr.groupby('sequence', as_index=False).MIC_Escherichia_coli.mean()
    pos_seqs = pos.sequence.tolist()
    pos_y = np.log10(pos.MIC_Escherichia_coli.values.astype(float))
    neg = pd.read_csv(ROOT / 'baseline/sota_bench/data_restricted/uniprot_negatives.csv')
    seqs = pos_seqs + neg.sequence.tolist()
    y = np.concatenate([pos_y, np.full(len(neg), NEG_LOG10)]).astype(np.float32)
    print(f'positives={len(pos_seqs)} negatives={len(neg)}; '
          f'label range [{y.min():.2f},{y.max():.2f}]', flush=True)

    cache = Path('/tmp/amplit_feats.npz')
    if cache.exists():
        z = np.load(cache)
        if z['X1'].shape[0] == len(seqs):
            X1, X2, X3 = z['X1'], z['X2'], z['X3']
            print('reused /tmp/amplit_feats.npz', flush=True)
        else:
            X1, X2, X3 = featurize(seqs)
            np.savez(cache, X1=X1, X2=X2, X3=X3)
    else:
        X1, X2, X3 = featurize(seqs)
        np.savez(cache, X1=X1, X2=X2, X3=X3)
    print('featurized', flush=True)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(seqs))
    n_va = int(0.1 * len(seqs))
    va_i, tr_i = idx[:n_va], idx[n_va:]

    model = AMPLiTRegressor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=WD)
    lossf = nn.MSELoss()

    def batchify(ids):
        return (torch.from_numpy(X1[ids]), torch.from_numpy(X2[ids]),
                torch.from_numpy(X3[ids]), torch.from_numpy(y[ids]))

    Xv = batchify(va_i)
    best_val, best_state = float('inf'), None
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = np.random.permutation(len(tr_i))
        losses = []
        for i in range(0, len(tr_i), BATCH):
            b = tr_i[perm[i:i + BATCH]]
            xb1, xb2, xb3, yb = batchify(b)
            xb1, xb2, xb3, yb = xb1.to(device), xb2.to(device), xb3.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb1, xb2, xb3), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            vl = float(lossf(model(*[t.to(device) for t in Xv[:3]]),
                             Xv[3].to(device)).cpu())
        print(f'ep{ep} train_loss={np.mean(losses):.4f} val_loss={vl:.4f}', flush=True)
        if vl < best_val - 1e-4:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict()}, CKPT_DIR / 'amplit_regr_best.pt')

    pep = pd.read_csv(TIGER / 'runs_review/final11_peptides.csv')
    uniq = list(dict.fromkeys(pep.sequence))
    Q1, Q2, Q3 = featurize(uniq)
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(uniq), 256):
            preds.append(model(torch.from_numpy(Q1[i:i + 256]).to(device),
                               torch.from_numpy(Q2[i:i + 256]).to(device),
                               torch.from_numpy(Q3[i:i + 256]).to(device)).cpu().numpy())
    preds = np.concatenate(preds)
    pd.Series(dict(zip(uniq, preds.tolist())), name='log10mic').to_csv(
        OUT / 'sota_amplit_regr_preds.csv')
    print(f'saved sota_amplit_regr_preds.csv; log10MIC mean={preds.mean():.3f} '
          f'std={preds.std():.3f} range [{preds.min():.3f},{preds.max():.3f}]', flush=True)


if __name__ == '__main__':
    main()

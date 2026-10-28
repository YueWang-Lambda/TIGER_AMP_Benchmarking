#!/usr/bin/env python3
"""Retrain the Das et al. 2021 sequence-level AMP classifier on the TIGER
restricted training universe.

Source: Das P. et al., "Accelerated antimicrobial discovery via deep
generative models and molecular dynamics simulations", Nat Biomed Eng 5:613-623
(2021); architecture per the arXiv full text (2005.11248v2), Methods 6.2.1
"Sequence-Level Attribute Classifiers":

  "For each attribute, we trained a bidirectional LSTM-based classifier on the
   labeled dataset. We used a hidden layer size of 100 and a dropout of 0.3."

  AMP labels (Methods 5.1.1): positives from AMP databases; negatives =
  UniProt sequences after discarding those annotated as antimicrobial,
  membrane, toxic, secretory, defensive, antibiotic, anticancer, antiviral and
  antifungal. Binary cross-entropy; train/valid/test split ~80/10/10 (6489/
  811/812 in the original).

Disclosed defaults (not stated in the paper text): per-amino-acid one-hot
input (20 + pad), Adam lr 1e-3, early stopping on validation loss. Retrained
on our universe: positives = the 4,929 restricted AMPs, negatives = the shared
UniProt non-AMP pool (7,962; same exclusion protocol).

Output: runs_review/benchmark_restricted/sota_das2021_preds.csv (P(AMP), 208
external test peptides).
"""
import sys
from pathlib import Path

ROOT = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC')
TIGER = ROOT
sys.path.insert(0, str(TIGER))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

OUT = TIGER / 'runs_review/benchmark_restricted'
CKPT_DIR = OUT / 'sota_models/das2021'
AA = 'ACDEFGHIKLMNPQRSTVWY'
AA2I = {c: i for i, c in enumerate(AA)}
MAX_LEN = 50


def encode(seqs):
    X = np.zeros((len(seqs), MAX_LEN, 20), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s[:MAX_LEN]):
            if c in AA2I:
                X[i, j, AA2I[c]] = 1.0
    return X


class BiLSTMClassifier(nn.Module):
    """Das et al. 2021 sequence-level attribute classifier (Methods 6.2.1)."""

    def __init__(self, hidden=100, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=20, hidden_size=hidden, batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)                      # (B, L, 2H)
        pooled = out.max(dim=1).values             # max over positions
        return self.head(self.drop(pooled)).squeeze(-1)  # logit


def main():
    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}', flush=True)

    tr = pd.read_csv(OUT / 'train_restricted4929_tiger.csv')
    pos = tr.groupby('sequence', as_index=False).MIC_Escherichia_coli.mean().sequence.tolist()
    neg = pd.read_csv(ROOT / 'baseline/sota_bench/data_restricted/uniprot_negatives.csv').sequence.tolist()
    X = encode(pos + neg)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.float32)
    print(f'positives={len(pos)} negatives={len(neg)}', flush=True)

    # ~80/10/10 split (paper's Data-Split proportions)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    n_tr, n_va = int(0.8 * len(X)), int(0.1 * len(X))
    tr_i, va_i, te_i = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

    model = BiLSTMClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    Xs = {k: torch.from_numpy(X[v]) for k, v in [('tr', tr_i), ('va', va_i), ('te', te_i)]}
    ys = {k: torch.from_numpy(y[v]) for k, v in [('tr', tr_i), ('va', va_i), ('te', te_i)]}

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

    # held-out test accuracy (paper reports 88.0% on their data)
    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(Xs['te'].to(device))).cpu().numpy()
    acc = float(((pred > 0.5) == ys['te'].numpy()).mean())
    print(f'held-out accuracy = {acc:.4f} (paper: 0.880 on their data)', flush=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict()}, CKPT_DIR / 'bilstm_best.pt')

    # 208 external peptides
    pep = pd.read_csv(TIGER / 'runs_review/final11_peptides.csv')
    uniq = list(dict.fromkeys(pep.sequence))
    Xq = torch.from_numpy(encode(uniq))
    with torch.no_grad():
        probs = []
        for i in range(0, len(uniq), 512):
            probs.append(torch.sigmoid(model(Xq[i:i + 512].to(device))).cpu().numpy())
    probs = np.concatenate(probs)
    pd.Series(dict(zip(uniq, probs.tolist())), name='p_amp').to_csv(OUT / 'sota_das2021_preds.csv')
    print(f'saved sota_das2021_preds.csv; P(AMP) mean={probs.mean():.3f} '
          f'range [{probs.min():.3f},{probs.max():.3f}]', flush=True)


if __name__ == '__main__':
    main()

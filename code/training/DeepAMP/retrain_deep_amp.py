#!/usr/bin/env python3
"""Retrain the Pandi et al. (Nat Commun 14:7197, 2023) MIC regressors on the
TIGER restricted 4,929-peptide set (unified benchmark protocol).

Faithful PyTorch port of github.com/amirpandi/Deep_AMP code/regressor.py +
train_regressor.py (paper Methods "Regressor convolutional and recurrent
neural networks"):
  - one-hot over the 22-token alphabet 'BCDSQKIPTFNGHLRWAVEYM-', post-padded
    with '-' to length 48; sequences longer than 48 aa are dropped (chopping)
  - CNN regressor: ZeroPad(5) - Conv1D(64,k5,relu) - MaxPool(2,s2) - Drop(0.5)
    - Conv1D(64,k5,relu) - MaxPool(2) - Flatten - Drop(0.5) - Dense(100) -
    Dense(100) - Dense(20) - Dense(1)
  - RNN regressor: LSTM(500) - Dense(100,relu) - Dense(1)
  - MSE loss, Adam (default lr 1e-3), 200 epochs, batch 50
  - labels: log10(MIC) per unique AMP sequence (per-sequence mean MIC, the
    GRAMPA averaging convention); UniProt non-AMP negatives labelled
    log10MIC = 4 (paper Methods)

Evaluation protocol addition (disclosed): 5-seed ensemble averaging, matching
the AMP-Designer treatment in this benchmark.

Output: runs_review/benchmark_restricted/sota_models/deep_amp_ensemble/
        {cnn,rnn}_seed*.pt  +  sota_deep_amp_{cnn,rnn}_preds.csv (log10MIC)
"""
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SOTA = Path(os.path.dirname(os.path.abspath(__file__)))
TIGER = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC/artifacts/repair_v15/external_runs/DeepAMP/seed123/workspace')
TRAIN_CSV = TIGER / 'runs_review/benchmark_restricted/train_restricted4929_tiger.csv'
NEG_CSV = SOTA / 'data_restricted/uniprot_negatives.csv'
PEP_CSV = TIGER / 'runs_review/final11_peptides.csv'
CKPT_DIR = TIGER / 'runs_review/benchmark_restricted/sota_models/deep_amp_ensemble'
OUT_DIR = TIGER / 'runs_review/benchmark_restricted'

ALPHABET = 'BCDSQKIPTFNGHLRWAVEYM-'  # Deep_AMP utils.AAalphabet (22 tokens)
MAX_LEN = 48
AA2I = {aa: i for i, aa in enumerate(ALPHABET)}


def encode(seqs):
    """Deep_AMP utils: chopping + '-' post-padding + one-hot."""
    rows = []
    for s in seqs:
        if len(s) > MAX_LEN or any(aa not in AA2I for aa in s):
            rows.append(None)
            continue
        rows.append([AA2I[aa] for aa in s + '-' * (MAX_LEN - len(s))])
    kept = [r for r in rows if r is not None]
    X = np.zeros((len(kept), MAX_LEN, len(ALPHABET)), dtype=np.float32)
    for i, r in enumerate(kept):
        X[i, np.arange(MAX_LEN), r] = 1.0
    return X, rows


class CNNRegressor(nn.Module):
    """Deep_AMP code/regressor.py cnn_regressor, layer for layer."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(len(ALPHABET), 64, 5)   # after ZeroPadding1D(5)
        self.conv2 = nn.Conv1d(64, 64, 5)
        self.drop = nn.Dropout(0.5)
        self.pool1 = nn.MaxPool1d(2, 2)
        self.pool2 = nn.MaxPool1d(2)
        n = self._flat_dim()
        self.fc = nn.Sequential(nn.Linear(n, 100), nn.ReLU(),
                                nn.Linear(100, 100), nn.ReLU(),
                                nn.Linear(100, 20), nn.ReLU(),
                                nn.Linear(20, 1))

    def _flat_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, len(ALPHABET), MAX_LEN)
            x = torch.nn.functional.pad(x, (5, 5))     # ZeroPadding1D(5)
            x = self.pool1(torch.relu(self.conv1(x)))
            x = self.pool2(torch.relu(self.conv2(self.drop(x))))
            return x.numel()

    def forward(self, x):
        x = x.transpose(1, 2)                          # (B, tokens, len)
        x = torch.nn.functional.pad(x, (5, 5))
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(self.drop(x))))
        x = torch.flatten(x, 1)
        return self.fc(self.drop(x)).squeeze(-1)


class RNNRegressor(nn.Module):
    """Deep_AMP code/regressor.py rnn_regressor."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(len(ALPHABET), 500, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(500, 100), nn.ReLU(), nn.Linear(100, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def run_epochs(model, X, y, epochs, batch, device, seed):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters())          # keras default lr=1e-3
    lossf = nn.MSELoss()
    n = len(X)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g).numpy()
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = torch.from_numpy(X[idx]).to(device)
            yb = torch.from_numpy(y[idx]).to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if ep % 20 == 0 or ep == 1:
            print(f'  ep{ep} train_MSE={tot / n:.4f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch', type=int, default=50)
    ap.add_argument('--seeds', type=str, default='42,43,44,45,46')
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--no-neg', action='store_true',
                    help='[noneg] train on the 4,929 positives only')
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}', flush=True)

    # --- data: positives = restricted 4,929 (per-sequence mean MIC -> log10) ---
    tr = pd.read_csv(TRAIN_CSV)
    pos = tr.groupby('sequence', as_index=False).MIC_Escherichia_coli.mean()
    neg = pd.read_csv(NEG_CSV)
    pos_seqs = pos.sequence.tolist()
    pos_y = np.log10(pos.MIC_Escherichia_coli.values.astype(float))
    neg_seqs = neg.sequence.tolist()
    neg_y = np.full(len(neg_seqs), 4.0)
    print(f'positives={len(pos_seqs)} (len<=48: {sum(len(s) <= MAX_LEN for s in pos_seqs)}), '
          f'negatives={len(neg_seqs)}', flush=True)

    Xp, rows_p = encode(pos_seqs)
    Xn, rows_n = encode(neg_seqs)
    keep_p = [i for i, r in enumerate(rows_p) if r is not None]
    if args.no_neg:                 # [noneg campaign] positives-only training
        assert len(pos_seqs) == 4929, len(pos_seqs)
        X = Xp.astype(np.float32)
        y = pos_y[keep_p].astype(np.float32)
        print(f'[noneg] positives-only n={len(X)}', flush=True)
    else:
        X = np.concatenate([Xp, Xn]).astype(np.float32)
        y = np.concatenate([pos_y[keep_p], neg_y]).astype(np.float32)
    Xt = torch.from_numpy(X)

    # --- test peptides (208) ---
    pep = pd.read_csv(PEP_CSV)
    uniq = list(dict.fromkeys(pep.sequence))
    Xq, rows_q = encode(uniq)
    dropped = [uniq[i] for i, r in enumerate(rows_q) if r is None]
    if dropped:
        print(f'WARNING: dropped {len(dropped)} test peptides >48aa/non-canonical: {dropped}', flush=True)
    # encode() returns rows for kept sequences only, in the same order as kept_seqs
    kept_seqs = [uniq[i] for i, r in enumerate(rows_q) if r is not None]
    Xq_t = torch.from_numpy(Xq)

    ckpt_dir = (TIGER / 'runs_review/benchmark_restricted/sota_models/'
                'deep_amp_noneg') if args.no_neg else CKPT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for arch, Cls in [('cnn', CNNRegressor)]:
        all_preds = []
        for seed in [int(s) for s in args.seeds.split(',')]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = Cls().to(device)
            print(f'[{arch} seed {seed}] training {args.epochs} epochs '
                  f'batch {args.batch} on n={len(X)}', flush=True)
            run_epochs(model, X, y, args.epochs, args.batch, device, seed)
            torch.save({'model': model.state_dict(), 'arch': arch, 'seed': seed},
                       ckpt_dir / f'{arch}_seed{seed}.pt')
            model.eval()
            with torch.no_grad():
                pred = []
                for i in range(0, len(Xq_t), 256):
                    pred.append(model(Xq_t[i:i + 256].to(device)).cpu().numpy())
                all_preds.append(np.concatenate(pred))
            print(f'  [{arch} seed {seed}] test log10MIC std={all_preds[-1].std():.3f} '
                  f'range [{all_preds[-1].min():.2f},{all_preds[-1].max():.2f}]', flush=True)
        ens = np.mean(all_preds, axis=0)
        sd = pd.Series(dict(zip(kept_seqs, ens.tolist())), name=f'{arch}_log10mic')
        sd.to_csv(OUT_DIR / f'sota_deep_amp_{arch}_preds.csv')
        print(f'[{arch}] ensemble saved, {len(ens)} preds', flush=True)


if __name__ == '__main__':
    main()

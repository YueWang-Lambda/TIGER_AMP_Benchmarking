#!/usr/bin/env python3
"""Causal test for the APEX-restr gap: paper-recipe learning rate.

Hypothesis: the released best_key_list lr values (1e-5/1e-6) are the DECAYED
endpoint of the paper's schedule (Adam 1e-4, /10 every 1,000 epochs, batch 128,
5,000 epochs). Retraining from scratch AT the endpoint lr skips the paper's
dominant high-lr phase -> colder, poorer optima.

This arm: ONE config (2&128&2048&...&wd 0.01 — the strongest family in the
restricted retrain), 5 seeds, lr=1e-4 constant (the paper's dominant phase;
on our 21-step epochs the first decay would land near the end of training),
batch 128 as the paper. Everything else as train_apex_restricted.py.

Usage: python train_apex_lrfix.py --gpu 0
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from AMP_DL_model_twohead import AMP_model  # noqa: E402
from utils import make_vocab, AAindex, onehot_encoding  # noqa: E402

DATA = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/baseline/sota_bench/data_restricted/deep_pie_input.csv')
OUT = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC/runs_review/benchmark_restricted/sota_models/apex_lrfix')
MAX_LEN = 52
KEY = '2&128&2048&1e-05&0.01&1.0'  # config identity (lr field overridden below)
WD = 0.01


class SeqDS(Dataset):
    def __init__(self, seqs, ys, word2idx):
        self.x, _, _ = onehot_encoding(list(seqs), MAX_LEN, word2idx)
        self.y = torch.tensor(ys, dtype=torch.float)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]).long(), self.y[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--max-epochs', type=int, default=400)
    ap.add_argument('--patience', type=int, default=40)
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    OUT.mkdir(parents=True, exist_ok=True)

    word2idx, _ = make_vocab()
    emb, _ = AAindex(str(HERE / 'aaindex1.csv'), word2idx)
    df = pd.read_csv(DATA)
    y = (6.0 - np.log10(df.MIC.values.astype(float))).astype(np.float32)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(df))
    n_val = int(round(0.15 * len(df)))
    va_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr_loader = DataLoader(SeqDS(df.sequence.values[tr_idx], y[tr_idx], word2idx), batch_size=128, shuffle=True)
    va_loader = DataLoader(SeqDS(df.sequence.values[va_idx], y[va_idx], word2idx), batch_size=1024)

    report = []
    for rep in range(args.seeds):
        seed = 3000 + rep
        torch.manual_seed(seed)
        np.random.seed(seed % (2**31))
        model = AMP_model(emb, emb.shape[1], 2, 128, 2048, 4, 1).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WD)
        best_mse, best_state, stale = float('inf'), None, 0
        for ep in range(1, args.max_epochs + 1):
            model.train()
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = nn.MSELoss()(model(xb).squeeze(-1), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            preds, tgts = [], []
            with torch.no_grad():
                for xb, yb in va_loader:
                    preds.append(model(xb.to(device)).squeeze(-1).cpu())
                    tgts.append(yb)
            mse = float(nn.functional.mse_loss(torch.cat(preds), torch.cat(tgts)))
            if mse < best_mse - 1e-5:
                best_mse, stale = mse, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= args.patience:
                break
        torch.save({'model': best_state, 'key': KEY, 'rep': rep, 'seed': seed,
                    'val_mse': best_mse, 'last_epoch': ep, 'lr': args.lr}, OUT / f'apexlr_{rep}.pt')
        report.append({'rep': rep, 'lr': args.lr, 'val_mse': round(best_mse, 4), 'epochs': ep})
        print(f'[apex-lr{args.lr:g}] rep{rep} val_MSE={best_mse:.4f} ep={ep}', flush=True)
    pd.DataFrame(report).to_csv(OUT / 'training_report.csv', index=False)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()

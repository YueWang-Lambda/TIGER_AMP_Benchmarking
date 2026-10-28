#!/usr/bin/env python3
"""Retrain APEX (Wan & de la Fuente labs, gitlab.com/machine-biology-group-public/apex)
on the restricted 4,929-sequence training set, single-task (E. coli MIC).

Faithful to the released architecture (AMP_DL_model_twohead.AMP_model: AAindex frozen
embedding -> BiGRU -> two-stage attention -> 4-layer MLP -> ReLU) and the released
ensemble recipe (best_key_list: 8 hyperparameter configs x 5 repeats = 40 models).

The repo ships predict.py only (no training script); the training loop below is a
reasonable reconstruction: Adam(lr, wd) from the config key, MSE on the APEX activity
scale y = 6 - log10(MIC) (predict.py inverts via MIC = 10**(6 - y)), early stopping on
a fixed 15% validation split. Labels in the restricted set span y in [1.73, 7.33], all
ReLU-representable.

Usage: python train_apex_restricted.py --gpu 0
"""
import argparse, json, math, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from AMP_DL_model_twohead import AMP_model  # noqa: E402
from utils import make_vocab, AAindex, onehot_encoding  # noqa: E402

SOTA = HERE.parent
DATA = SOTA / 'data_restricted' / 'deep_pie_input.csv'
OUT = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC/runs_review/benchmark_restricted/sota_models/apex_ensemble')
MAX_LEN = 52
NUM_TASK = 1  # single-task retrain: E. coli only (the 35-species head has no labels here)


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
    ap.add_argument('--max-epochs', type=int, default=400)
    ap.add_argument('--patience', type=int, default=40)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--configs', default='', help='comma-separated config indices to (re)train, e.g. 1,4,5,7; empty = all')
    ap.add_argument('--seed-base', type=int, default=1000, help='seed offset so retrained models can be told apart')
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    OUT.mkdir(parents=True, exist_ok=True)

    word2idx, _ = make_vocab()
    emb, _ = AAindex(str(HERE / 'aaindex1.csv'), word2idx)
    emb_size = emb.shape[1]

    df = pd.read_csv(DATA)
    y = (6.0 - np.log10(df.MIC.values.astype(float))).astype(np.float32)
    print(f'train rows={len(df)} uniq_seq={df.sequence.nunique()} y[{y.min():.2f},{y.max():.2f}]', flush=True)

    # Fixed val split for early stopping (same policy as the other bench baselines).
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(df))
    n_val = int(round(0.15 * len(df)))
    va_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr_ds = SeqDS(df.sequence.values[tr_idx], y[tr_idx], word2idx)
    va_ds = SeqDS(df.sequence.values[va_idx], y[va_idx], word2idx)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=1024)

    configs = [ln.strip() for ln in (HERE / 'best_key_list').read_text().splitlines() if ln.strip()]
    sel = {int(x) for x in args.configs.split(',') if x.strip()} if args.configs else None
    report = []
    for ci, key in enumerate(configs):
        if sel is not None and ci not in sel:
            continue
        rl, dh, dl, lr, wd, _ = key.split('&')
        rl, dh, dl, lr, wd = int(rl), int(dh), int(dl), float(lr), float(wd)
        for rep in range(5):
            seed = args.seed_base + ci * 10 + rep
            torch.manual_seed(seed)
            np.random.seed(seed % (2**31))
            model = AMP_model(emb, emb_size, rl, dh, dl, 4, NUM_TASK).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
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
            torch.save({'model': best_state, 'key': key, 'rep': rep, 'seed': seed,
                        'val_mse': best_mse, 'last_epoch': ep}, OUT / f'apex_{ci}_{rep}.pt')
            report.append({'config': key, 'rep': rep, 'val_mse': round(best_mse, 4), 'epochs': ep})
            print(f'[apex] cfg{ci}({key}) rep{rep} val_MSE={best_mse:.4f} ep={ep}', flush=True)
    pd.DataFrame(report).to_csv(OUT / 'training_report.csv', index=False)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()

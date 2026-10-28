#!/usr/bin/env python3
"""AMPGen MIC scorer (Commun Biol 2025, 10.1038/s42003-025-08282-7) retrained
under the unified "+neg" protocol (v2-9).

Fidelity notes (from MIC_scorer source, all adaptations documented):
- Frozen ESM-2 esm2_t36_3B_UR50D layer-36 mean representation (2560d), cached
  once in emb_cache/esm2_3b_L36.npz (frozen backbone -> seed-independent).
- Head/recipe verbatim from tools/lstm_train.py: LSTM(2560->128, 2 layers,
  dropout 0.7) + Dropout + Linear(128->1); MSE; Adam lr 1e-4 wd 1e-4;
  ReduceLROnPlateau(patience 5, factor 0.1); batch 64; 200 epochs; best
  checkpoint by validation R^2 (early stopping commented out upstream).
- The published ecoliscaler.pkl is a FEATURE StandardScaler (verified); we
  refit it on the training split of our corpus (label domain unchanged).
- Label domain: native log10(uM); unified protocol feeds log10(ug/mL)
  directly (MBC precedent); prediction = log10mic as-is.
- Seeds: upstream only fixes train_test_split(random_state=42); we parametrize
  split rs + torch/np seeds by the protocol seed.
- Original checkpoints (2ecoli_best_model_checkpoint.pth) never loaded.

Usage: python retrain_ampgen_neg.py --seed 120 [--cpu]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
V2 = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC/artifacts/repair_v15/external_runs/AMPGen/seed128/workspace')
OUT = V2 / 'runs_review/benchmark_restricted/sota_models/ampgen_neg'
OUT_NONEG = V2 / 'runs_review/benchmark_restricted/sota_models/ampgen_noneg'


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
        c0 = torch.zeros_like(h0)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(self.dropout(out[:, -1, :]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--no-neg', action='store_true',
                    help='[noneg] train on the 4,929 positives only')
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')

    emb = np.load(HERE / 'emb_cache/esm2_3b_L36.npz')
    tr = pd.read_csv(V2 / 'runs_review/benchmark_restricted/train_restricted4929_tiger.csv')
    pos = tr.groupby('sequence', as_index=False).MIC_Escherichia_coli.mean()
    if args.no_neg:                 # [noneg campaign] positives-only training
        assert len(pos) == 4929, len(pos)
        corpus = pos
    else:
        neg = pd.read_csv(HERE / 'data_restricted/uniprot_negatives.csv')[['sequence']]
        neg['MIC_Escherichia_coli'] = 10000.0
        corpus = pd.concat([pos, neg], ignore_index=True)
    X = np.stack([emb[s] for s in corpus.sequence])
    y = np.log10(corpus.MIC_Escherichia_coli.values.astype(float))
    ext = pd.read_csv(V2 / 'runs_review/final11_peptides.csv')[['sequence', 'MIC_E_coli_ug_mL']].dropna()
    Xe = np.stack([emb[s] for s in ext.sequence])

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=args.seed)
    scaler = StandardScaler().fit(Xtr)          # feature scaler, refit (upstream: ecoliscaler.pkl)
    Xtr, Xva, Xe = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xe)

    def ds(a, b=None):
        return DataLoader(TensorDataset(torch.tensor(a, dtype=torch.float32).unsqueeze(1),
                                         *([] if b is None else [torch.tensor(b, dtype=torch.float32)])),
                          batch_size=64, shuffle=b is not None)
    tl, vl = ds(Xtr, ytr), ds(Xva, yva)   # upstream: val loader unshuffled; fixed below
    vl = DataLoader(TensorDataset(torch.tensor(Xva, dtype=torch.float32).unsqueeze(1),
                                   torch.tensor(yva, dtype=torch.float32)),
                    batch_size=64, shuffle=False)
    model = LSTMModel(Xtr.shape[1], 128, 2, 1, 0.7).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=5, factor=0.1)
    crit = nn.MSELoss()
    best_r2, best_state = -np.inf, None
    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb).squeeze(-1), yb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = np.concatenate([model(xb.to(device)).squeeze(-1).cpu().numpy()
                                 for xb, _ in vl])
        r2 = 1 - np.sum((yva - pv) ** 2) / np.sum((yva - yva.mean()) ** 2)
        sch.step(float(np.mean((yva - pv) ** 2)))
        if r2 > best_r2:
            best_r2, best_state = r2, {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 20 == 0:
            print(f's{args.seed} ep{ep} val_R2={r2:.3f} best={best_r2:.3f}', flush=True)
    model.load_state_dict(best_state)
    model.eval()
    # [v2-12 patch] persist the evaluated task weights (+ fitted feature scaler)
    # so inference-cost timing can load the exact accuracy-evaluated model.
    out_dir = OUT_NONEG if args.no_neg else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(),
                'sc_mean': scaler.mean_, 'sc_scale': scaler.scale_},
               out_dir / f'task_s{args.seed}.pt')
    with torch.no_grad():
        pe = np.concatenate([model(xb.to(device)).squeeze(-1).cpu().numpy()
                             for xb, _ in DataLoader(
                                 TensorDataset(torch.tensor(Xe, dtype=torch.float32).unsqueeze(1),
                                               torch.zeros(len(Xe))), batch_size=64)])
    o = pd.DataFrame({'log10mic': pe})
    o.index = ext.sequence.values
    o.index.name = 'sequence'
    name = (f'sota_AMPgen_noneg_s{args.seed}_preds.csv' if args.no_neg
            else f'sota_AMPgen+neg_s{args.seed}_preds.csv')
    o.to_csv(out_dir / name)
    print(f'seed {args.seed}: best val R2 {best_r2:.3f}; preds range '
          f'[{pe.min():.2f},{pe.max():.2f}] -> saved')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""tsAMP-CS strain-level MIC module (Front Microbiol 2026,
10.3389/fmicb.2026.1842380) retrained under the unified "+neg" protocol (v2-9).

Fidelity notes (from the released code + checkpoint dissection):
- Input = concat(peptide ESM-1v esm1v_t33_650M_UR90S_1 layer-33 mean 1280d,
  strain ESM-1v layer-33 mean 1280d) = 2560d. Peptide side cached in
  emb_cache/esm1v_L33.npz; strain side = the repo's shipped embedding for a
  single fixed E. coli strain, ATCC 25922 (user-approved protocol choice;
  tsAMP/data_strain/Ec_ATCC_25922.pt).
- model.py in the repo (1280->128->1) does NOT match the released checkpoint
  (fc1 2560->128, fc3 128->64, fc4 64->1); we follow the CHECKPOINT structure.
- Data augmentation kept verbatim (noise 0.1*std x2, dropout 0.1, scale
  0.9-1.1, mixup 0.2-0.8; x3 rounds -> 16 copies per peptide) but applied
  AFTER the train/val split (upstream augments before splitting -> leak;
  fixed) and seeded by the protocol seed. Validation set unaugmented.
- Recipe per train.py: MSE, Adam lr 1e-4 wd 1e-5, StepLR(step 50, gamma 0.5),
  grad clip 1.0, batch 32, 70 epochs, final-epoch model (upstream has no
  best-checkpoint selection).
- Label domain: native 'log' column = log10(uM); unified protocol feeds
  log10(ug/mL) directly. Strain-level granularity collapsed to one fixed
  strain. Released checkpoints never loaded.

Usage: python retrain_tsamp_neg.py --seed 120 [--cpu]
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

HERE = Path(__file__).resolve().parent
V2 = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC/artifacts/repair_v15/external_runs/tsAMP/seed125/workspace')
OUT = V2 / 'runs_review/benchmark_restricted/sota_models/tsamp_neg'
OUT_NONEG = V2 / 'runs_review/benchmark_restricted/sota_models/tsamp_noneg'


class TSAMPCS(nn.Module):
    """Structure per the released checkpoint (see docstring)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2560, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


def augment(X, y, rounds=3):
    """Upstream 5-op augmentation, replicated verbatim (dataloader.py)."""
    std = X.std()
    out_x, out_y = list(X), list(y)
    for xi, yi in zip(X, y):
        for _ in range(rounds):
            for _ in range(2):
                out_x.append(xi + torch.randn_like(xi) * 0.1 * std)
                out_y.append(yi)
            m = (torch.rand_like(xi) > 0.1).float()
            out_x.append(xi * m)
            out_x.append(xi * random.uniform(0.9, 1.1))
            lam = random.uniform(0.2, 0.8)
            other = X[random.randrange(len(X))]
            out_x.append(lam * xi + (1 - lam) * other)
            out_y.extend([yi] * 3)
    return torch.stack(out_x), torch.tensor(out_y, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--no-neg', action='store_true',
                    help='[noneg] train on the 4,929 positives only')
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')

    emb = np.load(HERE / 'emb_cache/esm1v_L33.npz')
    strain = torch.load(HERE / 'tsAMP/data_strain/Ec_ATCC_25922.pt',
                        map_location='cpu')['mean_representations'][33].float()

    def vec(seq):
        return torch.cat((torch.tensor(emb[seq]), strain))

    tr = pd.read_csv(V2 / 'runs_review/benchmark_restricted/train_restricted4929_tiger.csv')
    pos = tr.groupby('sequence', as_index=False).MIC_Escherichia_coli.mean()
    if args.no_neg:                 # [noneg campaign] positives-only training
        assert len(pos) == 4929, len(pos)
        corpus = pos
    else:
        neg = pd.read_csv(HERE / 'data_restricted/uniprot_negatives.csv')[['sequence']]
        neg['MIC_Escherichia_coli'] = 10000.0
        corpus = pd.concat([pos, neg], ignore_index=True)
    X = torch.stack([vec(s) for s in corpus.sequence])
    y = np.log10(corpus.MIC_Escherichia_coli.values.astype(float))

    idx = torch.randperm(len(X), generator=torch.Generator().manual_seed(args.seed)).tolist()
    n_tr = int(0.8 * len(idx))
    Xtr, ytr = X[idx[:n_tr]], torch.tensor(y[idx[:n_tr]], dtype=torch.float32)
    Xva, yva = X[idx[n_tr:]], torch.tensor(y[idx[n_tr:]], dtype=torch.float32)
    Xtr, ytr = augment(Xtr, ytr)          # after split (leak fixed), seeded
    print(f'seed {args.seed}: train {len(Xtr)} (augmented) + val {len(Xva)}')

    model = TSAMPCS().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    sch = optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)
    crit = nn.MSELoss()
    for ep in range(1, 71):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            xb, yb = Xtr[b].to(device), ytr[b].to(device)
            opt.zero_grad()
            loss = crit(model(xb).squeeze(-1), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                pv = model(Xva.to(device)).squeeze(-1).cpu()
            print(f's{args.seed} ep{ep} val_MSE={crit(pv, yva).item():.4f}', flush=True)
    model.eval()
    # [v2-12 patch] persist the evaluated task weights (final-epoch model: the
    # published recipe trains a fixed 70 epochs with no checkpoint selection)
    out_dir = OUT_NONEG if args.no_neg else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict()}, out_dir / f'task_s{args.seed}.pt')
    ext = pd.read_csv(V2 / 'runs_review/final11_peptides.csv')[
        ['sequence', 'MIC_E_coli_ug_mL']].dropna()
    with torch.no_grad():
        pe = model(torch.stack([vec(s) for s in ext.sequence]).to(device)).squeeze(-1).cpu().numpy()
    o = pd.DataFrame({'log10mic': pe})
    o.index = ext.sequence.values
    o.index.name = 'sequence'
    name = (f'sota_tsAMP-CS_noneg_s{args.seed}_preds.csv' if args.no_neg
            else f'sota_tsAMP-CS+neg_s{args.seed}_preds.csv')
    o.to_csv(out_dir / name)
    print(f'seed {args.seed}: preds range [{pe.min():.2f},{pe.max():.2f}] -> saved')


if __name__ == '__main__':
    main()

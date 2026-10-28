#!/usr/bin/env python3
"""Retrain the HydrAMP M_MIC activity classifier (Szymczak et al., Nat Commun
14:1453, 2023) on the TIGER restricted 4,929-peptide set.

Faithful PyTorch port of the released training pipeline
(github.com/szczurek-lab/hydramp, scripts/mic_classifier_training_prodecure.ipynb
+ amp/models/discriminators/veltri_amp_classifier.py):

  - input: integer AA codes 1..20 (alphabetical ACDEFGHIKLMNPQRSTVWY),
    post-padded with 0
  - Embedding(21 -> 128) - Conv1D(64, k=16, padding=same, relu) -
    MaxPooling1D(5) - LSTM(100, dropout 0.1) - Dense(1, sigmoid)
  - Adam lr 1e-3, binary cross-entropy, epochs 32, batch 64,
    EarlyStopping(patience 7) on the held-out fold (final weights kept, i.e.
    Keras default restore_best_weights=False)
  - labels: AMPs with log10(mean MIC) < 1.5 (MIC <= 10^1.5 ~ 32 ug/mL) are
    positive (paper Methods "MIC data"); UniProt non-AMPs labelled 0
  - 10-fold StratifiedKFold(shuffle, seed 36); positives oversampled x5 inside
    each training fold; final predictor = mean of the 10 fold models (the
    notebook's ensemble evaluation protocol)

Disclosed adaptations:
  - MAX_LENGTH 25 -> 50: the paper's models serve their <=25 aa generator; our
    benchmark families contain peptides up to 42 aa. Architecture, kernel sizes
    and strides are otherwise identical.
  - Their unlabelled_negative.csv (UniProt-derived) is replaced by the shared
    UniProt-reviewed non-AMP pool built for this benchmark (same construction).

Output: runs_review/benchmark_restricted/sota_models/hydramp_mic/fold*.pt +
        sota_hydramp_mic_preds.csv (P(active), 208 peptides)
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SOTA = Path(os.path.dirname(os.path.abspath(__file__)))
TIGER = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC/artifacts/repair_v15/external_runs/HydrAMP/seed124/workspace')
TRAIN_CSV = TIGER / 'runs_review/benchmark_restricted/train_restricted4929_tiger.csv'
NEG_CSV = SOTA / 'data_restricted/uniprot_negatives.csv'
PEP_CSV = TIGER / 'runs_review/final11_peptides.csv'
CKPT_DIR = TIGER / 'runs_review/benchmark_restricted/sota_models/hydramp_mic'
OUT_DIR = TIGER / 'runs_review/benchmark_restricted'

ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
AA2I = {aa: i + 1 for i, aa in enumerate(ALPHABET)}
MAX_LENGTH = 50          # paper: 25 (adapted, see docstring)
THRESH_LOG10 = 1.5       # MIC <= 10^1.5 ug/mL  ->  active
SEED = 36                # notebook seed


def encode(seqs):
    rows = []
    for s in seqs:
        if any(aa not in AA2I for aa in s) or len(s) > MAX_LENGTH:
            rows.append(None)
            continue
        r = [AA2I[aa] for aa in s] + [0] * (MAX_LENGTH - len(s))
        rows.append(np.asarray(r, dtype=np.int64))
    return rows


class MICClassifier(nn.Module):
    """VeltriAMPClassifierFactory.get_default(max_length), layer for layer."""

    def __init__(self, max_length=MAX_LENGTH):
        super().__init__()
        self.emb = nn.Embedding(21, 128)
        self.conv = nn.Conv1d(128, 64, 16)             # 'same' via asymmetric pad
        self.pool = nn.MaxPool1d(5)
        self.drop = nn.Dropout(0.1)                    # keras LSTM(dropout=0.1)
        self.lstm = nn.LSTM(64, 100, batch_first=True)
        self.out = nn.Linear(100, 1)

    def forward(self, x):
        h = self.emb(x).transpose(1, 2)                # (B,128,L)
        h = torch.nn.functional.pad(h, (8, 7))         # keras same, even k=16
        h = torch.relu(self.conv(h))                   # (B,64,L)
        h = self.pool(h)                               # (B,64,L/5)
        h = self.drop(h)
        out, _ = self.lstm(h.transpose(1, 2))
        return self.out(out[:, -1, :]).squeeze(-1)     # logit


def pad_batch(rows):
    return torch.from_numpy(np.stack(rows))


def fit_one(x_tr, y_tr, x_val, y_val, device, epochs=32, batch=64, patience=7):
    model = MICClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    # Keras EarlyStopping default: monitor val_loss, keep LAST weights
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
        improved = vl < best
        bad = 0 if improved else bad + 1
        best = min(best, vl)
        if bad >= patience:
            break
    return model, best, ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='0')
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    np.random.seed(SEED)
    print(f'device={device}, seed={SEED}', flush=True)

    tr = pd.read_csv(TRAIN_CSV)
    pos = tr.groupby('sequence', as_index=False).MIC_Escherichia_coli.mean()
    neg = pd.read_csv(NEG_CSV)
    neg = neg[neg.sequence.str.len() <= 25].reset_index(drop=True)  # paper: <=25 aa

    pos_rows = encode(pos.sequence.tolist())
    keep = [i for i, r in enumerate(pos_rows) if r is not None]
    mic_x = [pos_rows[i] for i in keep]
    mic_y = (np.log10(pos.MIC_Escherichia_coli.values.astype(float)) < THRESH_LOG10).astype(int)[keep]
    neg_x = encode(neg.sequence.tolist())
    neg_x = [r for r in neg_x if r is not None]
    print(f'MIC set: {len(mic_x)} AMPs ({int(mic_y.sum())} active <=32 u/mL), '
          f'{len(neg_x)} UniProt negatives <=25aa', flush=True)

    x_all = mic_x + neg_x
    y_all = np.concatenate([mic_y, np.zeros(len(neg_x), dtype=int)])

    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

    pep = pd.read_csv(PEP_CSV)
    uniq = list(dict.fromkeys(pep.sequence))
    q_rows = encode(uniq)
    kept = [(s, r) for s, r in zip(uniq, q_rows) if r is not None]
    print(f'test peptides kept: {len(kept)}/{len(uniq)}', flush=True)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    fold_preds, oof_auc = [], []
    for f, (tri, vai) in enumerate(cv.split(np.zeros(len(x_all)), y_all), 1):
        x_tr = [x_all[i] for i in tri]
        y_tr = y_all[tri]
        # positives oversampled x5 (notebook cell 16)
        pidx = [i for i, yy in enumerate(y_tr) if yy == 1]
        x_tr = x_tr + [x_tr[i] for i in pidx] * 5
        y_tr = np.concatenate([y_tr, np.ones(len(pidx) * 5, dtype=int)])
        x_va = [x_all[i] for i in vai]
        y_va = y_all[vai]
        model, vl, ep = fit_one(x_tr, y_tr, x_va, y_va, device)
        torch.save({'model': model.state_dict(), 'fold': f}, CKPT_DIR / f'fold{f}.pt')
        model.eval()
        with torch.no_grad():
            ps = []
            for i in range(0, len(kept), 4096):
                xb = pad_batch([r for _, r in kept[i:i + 4096]]).to(device)
                ps.append(torch.sigmoid(model(xb)).cpu().numpy())
            fold_preds.append(np.concatenate(ps))
            vp = np.concatenate([
                torch.sigmoid(model(pad_batch(x_va[i:i + 4096]).to(device))).cpu().numpy()
                for i in range(0, len(x_va), 4096)])
        from sklearn.metrics import roc_auc_score
        oof_auc.append(roc_auc_score(y_va, vp))
        print(f'[fold {f}] stopped ep{ep} val_loss={vl:.4f} val_AUC={oof_auc[-1]:.4f}', flush=True)

    ens = np.mean(fold_preds, axis=0)
    pd.Series(dict(zip([s for s, _ in kept], ens.tolist())), name='p_active').to_csv(
        OUT_DIR / 'sota_hydramp_mic_preds.csv')
    print(f'ensemble saved; mean fold AUC={np.mean(oof_auc):.4f}; '
          f'P(active) range [{ens.min():.3f},{ens.max():.3f}]', flush=True)


if __name__ == '__main__':
    main()

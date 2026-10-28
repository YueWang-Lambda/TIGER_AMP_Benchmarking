#!/usr/bin/env python3
"""APEX-restr(lrfix) with a UNIFIED training-set scale (experiment 12 /
negative-protocol unification), cloning train_apex_lrfix.py verbatim (single
strongest config, AAindex frozen embedding + BiGRU, Adam lr 1e-4 constant,
batch 128, wd 0.01, clip 1.0, 15% val split rng 42, 5 seeds):

  mode pos4929 : positives only, 4,929 UNIQUE sequences (per-sequence mean MIC)
  mode with_neg: 4,929 positives + the full shared UniProt non-AMP pool
                 (7,962), negatives anchored at pseudo log10MIC = 4.0
                 (= 10,000 ug/mL) -> APEX activity scale y = 6 - 4 = 2.0

Outputs checkpoints + the 5-seed-mean per-peptide prediction CSV in log10MIC
(runs_review/benchmark_restricted/sota_apex_{neg,pos4929}_preds.csv;
log10MIC = 6 - y).

Usage: python train_apex_lrfix_neg.py --mode {pos4929,with_neg} --gpu 0
"""
import argparse
import sys
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
TRAIN_CSV = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC/artifacts/repair_v15/external_runs/APEX/seed123/workspace/runs_review/'
                 'benchmark_restricted/train_restricted4929_tiger.csv')
NEG_CSV = SOTA / 'data_restricted/uniprot_negatives.csv'
PEP_CSV = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC/artifacts/repair_v15/external_runs/APEX/seed123/workspace/runs_review/final11_peptides.csv')
OUT = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC/artifacts/repair_v15/external_runs/APEX/seed123/workspace/runs_review/benchmark_restricted')
MAX_LEN = 52
KEY = '2&128&2048&1e-05&0.01&1.0'  # config identity (lr field overridden below)
WD = 0.01
NEG_LOG10 = 4.0


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
    ap.add_argument('--mode', required=True, choices=['pos4929', 'with_neg'])
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--seed-base', type=int, default=3000,
                    help='rep r uses seed seed_base+r (default 3000 = original protocol)')
    ap.add_argument('--max-epochs', type=int, default=400)
    ap.add_argument('--patience', type=int, default=40)
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    tag = 'neg' if args.mode == 'with_neg' else 'pos4929'
    ck = OUT / f'sota_models/apex_{tag}'
    ck.mkdir(parents=True, exist_ok=True)

    word2idx, _ = make_vocab()
    emb, _ = AAindex(str(HERE / 'aaindex1.csv'), word2idx)
    tr = pd.read_csv(TRAIN_CSV)
    pos = tr.groupby('sequence', as_index=False).MIC_Escherichia_coli.mean()
    seqs = pos.sequence.tolist()
    mics = pos.MIC_Escherichia_coli.values.astype(float)
    if args.mode == 'with_neg':
        neg = pd.read_csv(NEG_CSV)
        seqs = seqs + neg.sequence.tolist()
        mics = np.concatenate([mics, np.full(len(neg), 10.0 ** NEG_LOG10)])
    y = (6.0 - np.log10(mics)).astype(np.float32)
    print(f'[{args.mode}] rows={len(seqs)} y[{y.min():.2f},{y.max():.2f}]', flush=True)

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(seqs))
    n_val = int(round(0.15 * len(seqs)))
    va_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr_loader = DataLoader(SeqDS([seqs[i] for i in tr_idx], y[tr_idx], word2idx),
                           batch_size=128, shuffle=True)
    va_loader = DataLoader(SeqDS([seqs[i] for i in va_idx], y[va_idx], word2idx),
                           batch_size=1024)

    pep = pd.read_csv(PEP_CSV)
    uniq = list(dict.fromkeys(pep.sequence))
    q_loader = DataLoader(SeqDS(uniq, np.zeros(len(uniq)), word2idx), batch_size=1024)

    report, pred_sum = [], np.zeros(len(uniq))
    for rep in range(args.seeds):
        seed = args.seed_base + rep
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
                    'val_mse': best_mse, 'last_epoch': ep, 'lr': args.lr,
                    'mode': args.mode}, ck / f'apexlr_{seed}.pt')
        report.append({'rep': rep, 'lr': args.lr, 'val_mse': round(best_mse, 4), 'epochs': ep})
        print(f'[apex-lrfix/{args.mode}] rep{rep} val_MSE={best_mse:.4f} ep={ep}', flush=True)
        model.load_state_dict(best_state)
        model.to(device).eval()
        qs = []
        with torch.no_grad():
            for xb, _ in q_loader:
                qs.extend(model(xb.to(device)).squeeze(-1).cpu().tolist())
        pred_sum += np.asarray(qs)
    pd.DataFrame(report).to_csv(ck / 'training_report.csv', index=False)

    preds = 6.0 - pred_sum / args.seeds  # y-scale -> log10MIC
    pd.Series(dict(zip(uniq, preds.tolist())), name='log10mic').to_csv(
        OUT / f'sota_apex_{tag}_preds.csv')
    print(f'saved sota_apex_{tag}_preds.csv; log10MIC mean={preds.mean():.3f} '
          f'std={preds.std():.3f}', flush=True)


if __name__ == '__main__':
    main()

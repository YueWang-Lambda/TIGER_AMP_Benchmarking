#!/usr/bin/env python3
"""Retrain AMP-Designer's MIC regressor with a UNIFIED training-set scale
(experiment 12 / negative-protocol unification), cloning
train_ampd_restricted.py verbatim (AMP-GPT backbone + MLP head, AdamW 1e-4
warmup 500 clip 1.0 batch 16, 15% val split rng 42, 5-seed ensemble,
46-residue cap disclosed):

  mode pos4929 : positives only, 4,929 UNIQUE sequences (per-sequence mean MIC)
  mode with_neg: 4,929 positives + the full shared UniProt non-AMP pool
                 (7,962), negatives anchored at pseudo log10MIC = 4.0
                 (= 10,000 ug/mL)

Outputs checkpoints + the 5-seed-mean per-peptide prediction CSV
(runs_review/benchmark_restricted/sota_amp_designer_{neg,pos4929}_preds.csv).

Usage: python train_ampd_restricted_neg.py --mode {pos4929,with_neg} --gpu 0
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, GPT2Config, GPT2LMHeadModel, get_scheduler

HERE = Path(__file__).resolve().parent
SOTA = HERE.parent
TIGER = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC')
WEIGHTS = SOTA / 'models/amp_designer'
TRAIN_CSV = TIGER / 'runs_review/benchmark_restricted/train_restricted4929_tiger.csv'
NEG_CSV = SOTA / 'data_restricted/uniprot_negatives.csv'
PEP_CSV = TIGER / 'runs_review/final11_peptides.csv'
OUT = TIGER / 'runs_review/benchmark_restricted'
NEG_LOG10 = 4.0


class SeqDS:
    """Space-separated char tokens (46-residue cap), one trailing [PAD]."""

    def __init__(self, seqs, ys, tokenizer):
        self.enc = []
        for s in seqs:
            ids = tokenizer(' '.join(list(s[:46])))['input_ids']
            ids.extend([0])
            self.enc.append(ids)
        self.y = torch.tensor(ys, dtype=torch.float)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.enc[i], self.y[i]


def collate(batch):
    ids, ys = zip(*batch)
    m = max(len(x) for x in ids)
    x = torch.full((len(ids), m), 0, dtype=torch.long)
    for i, seq in enumerate(ids):
        x[i, :len(seq)] = torch.tensor(seq)
    return x, torch.stack(ys)


class CombinedModel(nn.Module):
    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt
        self.regressor = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x):
        out = self.gpt(x, output_hidden_states=True)
        return self.regressor(out.hidden_states[-1][:, -1, :]).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['pos4929', 'with_neg'])
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--seed-base', type=int, default=2000,
                    help='rep r uses seed seed_base+r (default 2000 = original protocol)')
    ap.add_argument('--max-epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=8)
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    tr = pd.read_csv(TRAIN_CSV)
    pos = tr.groupby('sequence', as_index=False).MIC_Escherichia_coli.mean()
    seqs = pos.sequence.tolist()
    y = np.log10(pos.MIC_Escherichia_coli.values.astype(float))
    if args.mode == 'with_neg':
        neg = pd.read_csv(NEG_CSV)
        seqs = seqs + neg.sequence.tolist()
        y = np.concatenate([y, np.full(len(neg), NEG_LOG10)])
    print(f'[{args.mode}] rows={len(seqs)} log10MIC[{y.min():.2f},{y.max():.2f}]', flush=True)

    tokenizer = BertTokenizer(vocab_file=str(WEIGHTS / 'vocab.txt'))
    cfg = GPT2Config.from_pretrained(str(WEIGHTS / 'config.json'))

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(seqs))
    n_val = int(round(0.15 * len(seqs)))
    va_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr_loader = DataLoader(SeqDS([seqs[i] for i in tr_idx], y[tr_idx], tokenizer),
                           batch_size=16, shuffle=True, collate_fn=collate)
    va_loader = DataLoader(SeqDS([seqs[i] for i in va_idx], y[va_idx], tokenizer),
                           batch_size=64, collate_fn=collate)

    pep = pd.read_csv(PEP_CSV)
    uniq = list(dict.fromkeys(pep.sequence))
    q_loader = DataLoader(SeqDS(uniq, np.zeros(len(uniq)), tokenizer),
                          batch_size=64, collate_fn=collate)

    ck_dir = OUT / f'sota_models/amp_designer_{args.mode}'
    ck_dir.mkdir(parents=True, exist_ok=True)
    report, pred_sum = [], np.zeros(len(uniq))
    for rep in range(args.seeds):
        seed = args.seed_base + rep
        torch.manual_seed(seed)
        np.random.seed(seed % (2**31))
        gpt = GPT2LMHeadModel.from_pretrained(str(WEIGHTS / 'pytorch_model.bin'), config=cfg)
        model = CombinedModel(gpt).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        sched = get_scheduler('linear', optimizer=opt, num_warmup_steps=500,
                              num_training_steps=len(tr_loader) * args.max_epochs)
        best_mae, best_state, stale = float('inf'), None, 0
        for ep in range(1, args.max_epochs + 1):
            model.train()
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = nn.MSELoss()(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
            model.eval()
            preds, tgts = [], []
            with torch.no_grad():
                for xb, yb in va_loader:
                    preds.append(model(xb.to(device)).cpu())
                    tgts.append(yb)
            mae = float((torch.cat(preds) - torch.cat(tgts)).abs().mean())
            if mae < best_mae - 1e-4:
                best_mae, stale = mae, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            print(f'[ampd/{args.mode}] rep{rep} ep{ep} val_MAE={mae:.4f} best={best_mae:.4f}', flush=True)
            if stale >= args.patience:
                break
        torch.save({'model': best_state, 'rep': rep, 'seed': seed,
                    'val_mae': best_mae}, ck_dir / f'ampd_{seed}.pt')
        report.append({'rep': rep, 'val_mae': round(best_mae, 4), 'epochs': ep})
        model.load_state_dict(best_state)
        model.to(device).eval()
        qs = []
        with torch.no_grad():
            for xb, _ in q_loader:
                qs.extend(model(xb.to(device)).cpu().tolist())
        pred_sum += np.asarray(qs)
        del model, opt
        torch.cuda.empty_cache()
    pd.DataFrame(report).to_csv(ck_dir / 'training_report.csv', index=False)

    preds = pred_sum / args.seeds
    tag = 'neg' if args.mode == 'with_neg' else 'pos4929'
    pd.Series(dict(zip(uniq, preds.tolist())), name='log10mic').to_csv(
        OUT / f'sota_amp_designer_{tag}_preds.csv')
    print(f'saved sota_amp_designer_{tag}_preds.csv; log10MIC mean={preds.mean():.3f} '
          f'std={preds.std():.3f}', flush=True)


if __name__ == '__main__':
    main()

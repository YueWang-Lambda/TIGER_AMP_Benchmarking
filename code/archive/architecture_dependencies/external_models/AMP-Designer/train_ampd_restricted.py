#!/usr/bin/env python3
"""Retrain AMP-Designer's MIC regressor (AMP-GPT + MLP head) on the restricted
4,929-sequence training set (Wang et al., Science Advances 11:eads8932, 2025).

Faithful to the released AMP-MIC/AMP-MIC.py: BertTokenizer over the 25-token
character vocab, GPT2LMHeadModel backbone, last-position hidden state ->
Linear(768->256) -> ReLU -> Linear(256->1), predicting log10(MIC). Deviations
from the released inference template, for benchmark fairness:
  * trained on the restricted 5,454-row set (no test leakage)
  * the <=32 aa filter is dropped (our peptides are 6-50 aa; GPT-2 handles them)
  * 5-seed ensemble instead of the single released checkpoint
Optimizer follows their template (AdamW, lr=1e-4, warmup 500, clip 1.0, batch 16)
with per-epoch early stopping on a fixed 15% validation split.

Usage: python train_ampd_restricted.py --gpu 0 [--seeds 5]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, GPT2Config, GPT2LMHeadModel, get_scheduler

HERE = Path(__file__).resolve().parent
WEIGHTS = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/baseline/sota_bench/models/amp_designer')
DATA = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/baseline/sota_bench/data_restricted/deep_pie_input.csv')
OUT = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC/runs_review/benchmark_restricted/sota_models/amp_designer_ensemble')


class SeqDS(Dataset):
    """Space-separated character tokens, padded per batch; y = log10(MIC).

    The released config has n_positions=50: [CLS]+seq+[SEP]+one trailing [PAD]
    must fit, so residues are capped at 46 (2 of 5,454 training rows exceed it;
    all 208 external test peptides are <=42 aa and pass untruncated).
    """

    def __init__(self, seqs, ys, tokenizer):
        self.enc = []
        for s in seqs:
            ids = tokenizer(' '.join(list(s[:46])))['input_ids']
            ids.extend([0])  # released recipe: one trailing [PAD], regressor reads position -1
            self.enc.append(ids)
        self.y = torch.tensor(ys, dtype=torch.float)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.enc[i], self.y[i]


def collate(batch):
    ids, ys = zip(*batch)
    lens = [len(x) for x in ids]
    m = max(lens)
    x = torch.full((len(ids), m), 0, dtype=torch.long)
    for i, seq in enumerate(ids):
        x[i, :len(seq)] = torch.tensor(seq)
    return x, torch.stack(ys)


class CombinedModel(nn.Module):
    """Released architecture: GPT-2 hidden state at the last position -> MLP."""

    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt
        self.regressor = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x):
        out = self.gpt(x, output_hidden_states=True)
        h = out.hidden_states[-1]
        return self.regressor(h[:, -1, :]).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--max-epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=8)
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    OUT.mkdir(parents=True, exist_ok=True)

    tokenizer = BertTokenizer(vocab_file=str(WEIGHTS / 'vocab.txt'))
    cfg = GPT2Config.from_pretrained(str(WEIGHTS / 'config.json'))

    df = pd.read_csv(DATA)
    y = np.log10(df.MIC.values.astype(float))
    print(f'train rows={len(df)} uniq_seq={df.sequence.nunique()} log10MIC[{y.min():.2f},{y.max():.2f}]', flush=True)

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(df))
    n_val = int(round(0.15 * len(df)))
    va_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr_ds = SeqDS(df.sequence.values[tr_idx], y[tr_idx], tokenizer)
    va_ds = SeqDS(df.sequence.values[va_idx], y[va_idx], tokenizer)
    tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True, collate_fn=collate)
    va_loader = DataLoader(va_ds, batch_size=64, collate_fn=collate)

    report = []
    for rep in range(args.seeds):
        seed = 2000 + rep
        torch.manual_seed(seed)
        np.random.seed(seed % (2**31))
        gpt = GPT2LMHeadModel.from_pretrained(str(WEIGHTS / 'pytorch_model.bin'), config=cfg)
        model = CombinedModel(gpt).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        total_steps = len(tr_loader) * args.max_epochs
        sched = get_scheduler('linear', optimizer=opt, num_warmup_steps=500, num_training_steps=total_steps)
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
            print(f'[ampd] rep{rep} ep{ep} val_MAE(log10)={mae:.4f} best={best_mae:.4f}', flush=True)
            if stale >= args.patience:
                break
        torch.save({'model': best_state, 'rep': rep, 'seed': seed,
                    'val_mae': best_mae, 'last_epoch': ep}, OUT / f'ampd_{rep}.pt')
        report.append({'rep': rep, 'val_mae': round(best_mae, 4), 'epochs': ep})
        del model, opt
        torch.cuda.empty_cache()
    pd.DataFrame(report).to_csv(OUT / 'training_report.csv', index=False)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()

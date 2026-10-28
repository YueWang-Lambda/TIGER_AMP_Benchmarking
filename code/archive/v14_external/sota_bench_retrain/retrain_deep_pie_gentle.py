#!/usr/bin/env python3
"""Causal retest arm 3 for the Deep-PIE failure.

Arms so far:
  A (benchmark, = released recipe): z-scored linear MIC, Adam 1e-4, no clip,
     full finetune  -> constant predictor (76.15 ug/mL for all 208 peptides)
  B (retrain_deep_pie_log.py): log10 MIC labels, everything else as released
     -> still constant (pred_std ~ 0 after 4+ epochs)

This arm C changes the OPTIMIZATION instead: log10 MIC labels + AdamW 2e-5,
grad clip 1.0, embeddings + encoder layers 0-1 frozen (BERT-AmPEP-style
finetuning hygiene on the same ProtBERT backbone). If C recovers, the failure
is the released training recipe acting on a degenerate readout head.

Usage: python retrain_deep_pie_gentle.py --gpu 0
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer

SOTA = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/baseline/sota_bench')
DATA = SOTA / 'data_restricted' / 'deep_pie_input.csv'
OUT = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC/runs_review/benchmark_restricted/sota_models')


class DS(Dataset):
    def __init__(self, seqs, ys, tokenizer):
        self.seqs, self.ys = [' '.join(s) for s in seqs], ys
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        enc = self.tokenizer(self.seqs[i], padding='max_length', truncation=True, max_length=60, return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(), 'attention_mask': enc['attention_mask'].squeeze()}, torch.tensor(self.ys[i], dtype=torch.float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--epochs', type=int, default=15)
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained(str(SOTA / 'models/prot_bert'), do_lower_case=False)
    bert = BertModel.from_pretrained(str(SOTA / 'models/prot_bert'))

    df = pd.read_csv(DATA)
    y = np.log10(df.MIC.values.astype(float))
    tr_s, te_s, tr_y, te_y = train_test_split(df.sequence.tolist(), y, test_size=0.2, random_state=42)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = bert
            self.fc1 = nn.Linear(bert.config.hidden_size, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 1)
            self.relu = nn.ReLU()
            self.drop = nn.Dropout(0.1)

        def forward(self, input_ids, attention_mask):
            p = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
            x = self.drop(self.relu(self.fc1(p)))
            x = self.drop(self.relu(self.fc2(x)))
            return self.fc3(x).squeeze(-1)

    net = Net().to(device)
    # BERT-AmPEP-style freezing: embeddings + first two encoder layers
    for name, p in net.bert.named_parameters():
        if 'embeddings' in name or 'encoder.layer.0.' in name or 'encoder.layer.1.' in name:
            p.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=2e-5, weight_decay=0.01)

    tr_loader = DataLoader(DS(tr_s, tr_y, tokenizer), batch_size=16, shuffle=True)
    te_loader = DataLoader(DS(te_s, te_y, tokenizer), batch_size=64)
    best_mae, best_state = float('inf'), None
    for ep in range(1, args.epochs + 1):
        net.train()
        for batch, yb in tr_loader:
            ii, am = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = nn.MSELoss()(net(ii, am), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        errs, preds_all = [], []
        with torch.no_grad():
            for batch, yb in te_loader:
                ii, am = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                out = net(ii, am).cpu()
                errs.extend((out - yb).abs().tolist())
                preds_all.extend(out.tolist())
        mae = float(np.mean(errs))
        if mae < best_mae:
            best_mae, best_state = mae, {k: v.cpu().clone() for k, v in net.state_dict().items()}
        print(f'[deep_pie_gentle] ep{ep} val_MAE(log10)={mae:.4f} pred_std={np.std(preds_all):.4f}', flush=True)
    net.load_state_dict(best_state)
    net.to(device)
    torch.save({'model': net.state_dict(), 'type': 'deep_pie_gentle'}, OUT / 'deep_pie_gentle_best.pt')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()

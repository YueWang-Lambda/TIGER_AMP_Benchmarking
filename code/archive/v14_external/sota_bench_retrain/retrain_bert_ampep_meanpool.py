#!/usr/bin/env python3
"""Sensitivity arm: BERT-AmPEP with masked mean pooling instead of pooler_output.

Single-variable change vs train_sota_bench.py::train_bert_ampep — the readout.
Everything else (data, labels EC_pMIC_scale, AdamW 2e-5 + wd 0.01, frozen
embeddings + encoder layers 0-1, grad clip, batch 16/64, 20 epochs, split 42)
is identical. Motivation: ProtBERT pooler_output is near-constant across
peptides (cosine 1.0000 at init); masked mean over last_hidden_state is the
officially recommended readout (used by LLAMP).

Usage: python retrain_bert_ampep_meanpool.py --gpu 0
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
DATA = SOTA / 'data_restricted' / 'bert_ampep_input.csv'
OUT = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC/runs_review/benchmark_restricted/sota_models')


class DS(Dataset):
    def __init__(self, d, tokenizer):
        self.seqs = d.SEQUENCE_space.tolist()
        self.ys = d.EC_pMIC_scale.values
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        enc = self.tokenizer(self.seqs[i], padding='max_length', truncation=True, max_length=60, return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(), 'attention_mask': enc['attention_mask'].squeeze()}, torch.tensor(self.ys[i], dtype=torch.float)


class REG(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(bert.config.hidden_size, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask):
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        p = (h * m).sum(dim=1) / m.sum(dim=1)  # masked mean pooling (THE change)
        x = self.dropout(torch.relu(self.fc1(self.dropout(p))))
        return self.fc2(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='0')
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained(str(SOTA / 'models/prot_bert'), do_lower_case=False)
    bert = BertModel.from_pretrained(str(SOTA / 'models/prot_bert'))
    df = pd.read_csv(DATA)

    net = REG(bert).to(device)
    for name, p in net.bert.named_parameters():
        if 'encoder.layer.0.' in name or 'encoder.layer.1.' in name or 'embeddings' in name:
            p.requires_grad = False

    tr, te = train_test_split(df, test_size=0.2, random_state=42)
    tr_loader = DataLoader(DS(tr, tokenizer), batch_size=16, shuffle=True)
    te_loader = DataLoader(DS(te, tokenizer), batch_size=64)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=2e-5, weight_decay=0.01)
    best_mae, best_state = float('inf'), None
    for ep in range(1, 21):
        net.train()
        for batch, y in tr_loader:
            ii, am = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            y = y.to(device)
            opt.zero_grad()
            loss = nn.MSELoss()(net(ii, am), y)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        errs, preds_all = [], []
        with torch.no_grad():
            for batch, y in te_loader:
                ii, am = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                out = net(ii, am).cpu()
                errs.extend((out - y).abs().tolist())
                preds_all.extend(out.tolist())
        mae = float(np.mean(errs))
        if mae < best_mae:
            best_mae, best_state = mae, {k: v.cpu().clone() for k, v in net.state_dict().items()}
        print(f'[bert_ampep_meanpool] ep{ep} val_MAE(scale)={mae:.4f} pred_std={np.std(preds_all):.4f}', flush=True)
    net.load_state_dict(best_state)
    net.to(device)
    torch.save({'model': net.state_dict(), 'type': 'bert_ampep_meanpool'}, OUT / 'bert_ampep_meanpool_best.pt')

    # Inference on the 208 external peptides; invert scale -> pMIC -> MIC (same
    # convention as sota_bert_ampep_preds.csv, whose delta formula is
    # log2(MIC_a) - log2(MIC_q), verified to machine precision).
    a_, b_ = np.polyfit(df.EC_pMIC_scale.values, df.EC_pMIC.values, 1)  # pMIC = a_*scale + b_
    pep = pd.read_csv('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC/runs_review/final11_peptides.csv')
    uniq = list(dict.fromkeys(pep.sequence))
    mic_preds = {}
    net.eval()
    with torch.no_grad():
        for s in uniq:
            enc = tokenizer(' '.join(s), padding='max_length', truncation=True, max_length=60, return_tensors='pt')
            sc = float(net(enc['input_ids'].to(device), enc['attention_mask'].to(device)).item())
            mic_preds[s] = float(10 ** (-(a_ * sc + b_)))
    pd.Series(mic_preds, name='mic').to_csv(
        '/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC/runs_review/benchmark_restricted/sota_bert_ampep_meanpool_preds.csv')
    lg = np.log10(list(mic_preds.values()))
    print(f'外测 {len(uniq)} 条: log10MIC std={lg.std():.3f} 范围[{lg.min():.2f},{lg.max():.2f}]', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Train three SOTA models (Deep-PIE, BERT-AmPEP60, LLAMP) on the restricted 4,929-peptide set.

All architectures are used verbatim from their released code; only data paths and
local model weight paths are changed. Hyperparameters follow each paper's defaults.
Usage: python train_sota_bench.py --model {deep_pie,bert_ampep,llamp} --gpu N
"""
import argparse, json, math, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SOTA = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/baseline/sota_bench')
DATA = SOTA / 'data_restricted'
OUT = Path('runs_review/benchmark_restricted/sota_models')
os.environ['TRANSFORMERS_OFFLINE'] = '1'


def train_deep_pie(gpu):
    """Deep-PIE: ProtBERT + 3-layer FC head, 15 epochs, lr=1e-4, batch=8."""
    from transformers import BertTokenizer, BertModel
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    pb = str(SOTA / 'models/prot_bert')
    tokenizer = BertTokenizer.from_pretrained(pb, do_lower_case=False)
    bert = BertModel.from_pretrained(pb)
    df = pd.read_csv(DATA / 'deep_pie_input.csv')
    scaler = StandardScaler()
    labels = scaler.fit_transform(df.MIC.values.reshape(-1, 1)).flatten()
    tr_s, te_s, tr_y, te_y = train_test_split(df.sequence.tolist(), labels, test_size=0.2, random_state=42)

    class DS(Dataset):
        def __init__(self, seqs, ys):
            self.seqs, self.ys = [' '.join(s) for s in seqs], ys
        def __len__(self): return len(self.seqs)
        def __getitem__(self, i):
            enc = tokenizer(self.seqs[i], padding='max_length', truncation=True, max_length=60, return_tensors='pt')
            return {'input_ids': enc['input_ids'].squeeze(), 'attention_mask': enc['attention_mask'].squeeze()}, torch.tensor(self.ys[i], dtype=torch.float)

    model = nn.Sequential()
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = bert
            self.fc1 = nn.Linear(bert.config.hidden_size, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 1)
            self.relu = nn.ReLU(); self.drop = nn.Dropout(0.1)
        def forward(self, input_ids, attention_mask):
            p = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
            x = self.drop(self.relu(self.fc1(p)))
            x = self.drop(self.relu(self.fc2(x)))
            return self.fc3(x).squeeze(-1)
    net = Net().to(device)
    tr_loader = DataLoader(DS(tr_s, tr_y), batch_size=8, shuffle=True)
    te_loader = DataLoader(DS(te_s, te_y), batch_size=8)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    best_mae, best_state = float('inf'), None
    for ep in range(1, 16):
        net.train()
        for batch, y in tr_loader:
            ii, am = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            y = y.to(device)
            opt.zero_grad(); loss = nn.MSELoss()(net(ii, am), y); loss.backward(); opt.step()
        net.eval(); errs = []
        with torch.no_grad():
            for batch, y in te_loader:
                ii, am = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                errs.extend((net(ii, am).cpu() - y).abs().tolist())
        mae = float(np.mean(errs))
        if mae < best_mae: best_mae, best_state = mae, {k: v.cpu().clone() for k, v in net.state_dict().items()}
        print(f'[deep_pie] ep{ep} val_MAE(z)={mae:.4f}', flush=True)
    net.load_state_dict(best_state); net.to(device)
    return net, tokenizer, scaler, device


def train_bert_ampep(gpu):
    """BERT-AmPEP60: ProtBERT + REG head with frozen layers, AdamW."""
    from transformers import BertTokenizer, BertModel
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    pb = str(SOTA / 'models/prot_bert')
    tokenizer = BertTokenizer.from_pretrained(pb, do_lower_case=False)
    bert = BertModel.from_pretrained(pb)
    df = pd.read_csv(DATA / 'bert_ampep_input.csv')

    class REG(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = bert
            self.dropout = nn.Dropout(0.1)
            self.fc1 = nn.Linear(bert.config.hidden_size, 256)
            self.fc2 = nn.Linear(256, 1)
        def forward(self, input_ids, attention_mask):
            p = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
            x = self.dropout(torch.relu(self.fc1(self.dropout(p))))
            return self.fc2(x).squeeze(-1)

    net = REG().to(device)
    for name, p in net.bert.named_parameters():
        if 'encoder.layer.0.' in name or 'encoder.layer.1.' in name or 'embeddings' in name:
            p.requires_grad = False
    tr, te = train_test_split(df, test_size=0.2, random_state=42)

    class DS(Dataset):
        def __init__(self, d):
            self.seqs = d.SEQUENCE_space.tolist(); self.ys = d.EC_pMIC_scale.values
        def __len__(self): return len(self.seqs)
        def __getitem__(self, i):
            enc = tokenizer(self.seqs[i], padding='max_length', truncation=True, max_length=60, return_tensors='pt')
            return {'input_ids': enc['input_ids'].squeeze(), 'attention_mask': enc['attention_mask'].squeeze()}, torch.tensor(self.ys[i], dtype=torch.float)

    tr_loader = DataLoader(DS(tr), batch_size=16, shuffle=True)
    te_loader = DataLoader(DS(te), batch_size=64)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=2e-5, weight_decay=0.01)
    best_mae, best_state = float('inf'), None
    for ep in range(1, 21):
        net.train()
        for batch, y in tr_loader:
            ii, am = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            y = y.to(device)
            opt.zero_grad(); loss = nn.MSELoss()(net(ii, am), y); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval(); errs = []
        with torch.no_grad():
            for batch, y in te_loader:
                ii, am = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                errs.extend((net(ii, am).cpu() - y).abs().tolist())
        mae = float(np.mean(errs))
        if mae < best_mae: best_mae, best_state = mae, {k: v.cpu().clone() for k, v in net.state_dict().items()}
        print(f'[bert_ampep] ep{ep} val_MAE(scaled)={mae:.4f}', flush=True)
    net.load_state_dict(best_state); net.to(device)
    return net, tokenizer, None, device


def train_llamp(gpu):
    """LLAMP: peptide-tuned ESM-2 + FC head."""
    from transformers import AutoTokenizer, AutoModel
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    pe = str(SOTA / 'models/peptide_tuned_ESM-2')
    tokenizer = AutoTokenizer.from_pretrained(pe)
    esm = AutoModel.from_pretrained(pe)
    df = pd.read_csv(DATA / 'llamp_input.csv')

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.esm = esm
            h = esm.config.hidden_size
            self.fc1 = nn.Linear(h, 256); self.fc2 = nn.Linear(256, 1)
            self.drop = nn.Dropout(0.1)
        def forward(self, input_ids, attention_mask):
            p = self.esm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.mean(dim=1)
            return self.fc2(self.drop(torch.relu(self.fc1(self.drop(p))))).squeeze(-1)

    net = Net().to(device)
    tr, te = train_test_split(df, test_size=0.2, random_state=42)

    class DS(Dataset):
        def __init__(self, d):
            self.seqs = d.sequence.tolist(); self.ys = d.label.values
        def __len__(self): return len(self.seqs)
        def __getitem__(self, i):
            enc = tokenizer(self.seqs[i], padding='max_length', truncation=True, max_length=60, return_tensors='pt')
            return {'input_ids': enc['input_ids'].squeeze(), 'attention_mask': enc['attention_mask'].squeeze()}, torch.tensor(self.ys[i], dtype=torch.float)

    tr_loader = DataLoader(DS(tr), batch_size=16, shuffle=True)
    te_loader = DataLoader(DS(te), batch_size=64)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)
    best_mae, best_state = float('inf'), None
    for ep in range(1, 21):
        net.train()
        for batch, y in tr_loader:
            ii, am = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            y = y.to(device)
            opt.zero_grad(); loss = nn.MSELoss()(net(ii, am), y); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval(); errs = []
        with torch.no_grad():
            for batch, y in te_loader:
                ii, am = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                errs.extend((net(ii, am).cpu() - y).abs().tolist())
        mae = float(np.mean(errs))
        if mae < best_mae: best_mae, best_state = mae, {k: v.cpu().clone() for k, v in net.state_dict().items()}
        print(f'[llamp] ep{ep} val_MAE(log10)={mae:.4f}', flush=True)
    net.load_state_dict(best_state); net.to(device)
    return net, tokenizer, None, device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['deep_pie', 'bert_ampep', 'llamp'])
    ap.add_argument('--gpu', default='0')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.model == 'deep_pie':
        net, tok, scaler, device = train_deep_pie(args.gpu)
    elif args.model == 'bert_ampep':
        net, tok, scaler, device = train_bert_ampep(args.gpu)
    else:
        net, tok, scaler, device = train_llamp(args.gpu)
    torch.save({'model': net.state_dict(), 'type': args.model}, OUT / f'{args.model}_best.pt')
    print(f'DONE: saved to {OUT}/{args.model}_best.pt', flush=True)


if __name__ == '__main__':
    main()

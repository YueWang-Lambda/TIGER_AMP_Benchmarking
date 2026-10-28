#!/usr/bin/env python3
"""Retrain BERT-AmPEP / LLAMP / Deep-PIE(gentle) with a UNIFIED training-set
scale (experiment 12 / negative-protocol unification):

  mode pos4929 : positives only, 4,929 UNIQUE sequences (per-sequence mean MIC)
                 — the deduplication control
  mode with_neg: 4,929 positives + the full shared UniProt non-AMP pool
                 (7,962), negatives anchored at pseudo MIC 10,000 ug/mL

Architectures, optimizers and schedules are cloned verbatim from
train_sota_bench.py / retrain_deep_pie_gentle.py; only the training universe
and label universe change. Per-model native label scales (negatives included):

  bert_ampep : EC_pMIC = -log10(MIC); scale = 0.178545*pMIC + 0.762956
               (affine least-squares fit of the shipped bert_ampep_input.csv,
               r = 1.0000 — the paper's fixed standardisation). Negatives:
               pMIC = -4 -> scale = 0.048776. Inverted back to log10MIC at
               inference for the benchmark join.
  llamp      : label = log10MIC; negatives 4.0
  deep_pie   : label = log10MIC; negatives 4.0 (gentle-arm recipe: AdamW 2e-5,
               clip 1.0, frozen embeddings + layers 0-1)

Output: runs_review/benchmark_restricted/sota_{model}_{mode}_preds.csv
(log10MIC per external test peptide; mode in {pos4929, neg}).

Usage: python train_sota_bench_neg.py --model {bert_ampep,llamp,deep_pie}
                                     --mode {pos4929,with_neg} --gpu 0
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC')
TIGER = ROOT
sys.path.insert(0, str(TIGER))
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

SOTA = Path(__file__).resolve().parent
OUT = TIGER / 'runs_review/benchmark_restricted'
TRAIN_CSV = OUT / 'train_restricted4929_tiger.csv'
NEG_CSV = SOTA / 'data_restricted/uniprot_negatives.csv'
NEG_LOG10 = 4.0  # pseudo MIC 10,000 ug/mL
# BERT-AmPEP fixed standardisation: scale = A*pMIC + B (fit to the shipped
# bert_ampep_input.csv, r = 1.0000; matches the paper's (pMIC+4)/6 convention
# up to the dataset's own affine constants)
BA_A, BA_B = 0.178545, 0.762956


def load_universe(mode):
    tr = pd.read_csv(TRAIN_CSV)
    pos = tr.groupby('sequence', as_index=False).MIC_Escherichia_coli.mean()
    seqs = pos.sequence.tolist()
    mics = pos.MIC_Escherichia_coli.values.astype(float)
    if mode == 'with_neg':
        neg = pd.read_csv(NEG_CSV)
        seqs = seqs + neg.sequence.tolist()
        mics = np.concatenate([mics, np.full(len(neg), 10.0 ** NEG_LOG10)])
    return seqs, mics


def make_labels(model, mics):
    if model == 'bert_ampep':
        pmic = -np.log10(mics)
        return (BA_A * pmic + BA_B).astype(np.float32), 'scale'
    return np.log10(mics).astype(np.float32), 'log10mic'


class DS(Dataset):
    def __init__(self, seqs, ys, tokenizer, spaced):
        self.seqs = [' '.join(s) for s in seqs] if spaced else list(seqs)
        self.ys, self.tok, self.spaced = ys, tokenizer, spaced

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        enc = self.tok(self.seqs[i], padding='max_length', truncation=True,
                       max_length=60, return_tensors='pt')
        return ({'input_ids': enc['input_ids'].squeeze(),
                 'attention_mask': enc['attention_mask'].squeeze()},
                torch.tensor(self.ys[i], dtype=torch.float))


def build_bert_ampep(bert):
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
    return REG()


def build_llamp(esm):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.esm = esm
            h = esm.config.hidden_size
            self.fc1 = nn.Linear(h, 256)
            self.fc2 = nn.Linear(256, 1)
            self.drop = nn.Dropout(0.1)

        def forward(self, input_ids, attention_mask):
            p = self.esm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.mean(dim=1)
            return self.fc2(self.drop(torch.relu(self.fc1(self.drop(p))))).squeeze(-1)
    return Net()


def build_deep_pie(bert):
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
    return Net()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['bert_ampep', 'llamp', 'deep_pie'])
    ap.add_argument('--mode', required=True, choices=['pos4929', 'with_neg'])
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--gpu', default='0')
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    seqs, mics = load_universe(args.mode)
    y, yscale = make_labels(args.model, mics)
    print(f'[{args.model}/{args.mode}] n={len(seqs)} label_scale={yscale} '
          f'y in [{y.min():.3f},{y.max():.3f}]', flush=True)

    if args.model == 'llamp':
        from transformers import AutoTokenizer, AutoModel
        pe = str(SOTA / 'models/peptide_tuned_ESM-2')
        tokenizer, backbone = AutoTokenizer.from_pretrained(pe), AutoModel.from_pretrained(pe)
        net, spaced, epochs, lr = build_llamp(backbone), False, 20, 1e-4
    else:
        from transformers import BertTokenizer, BertModel
        pb = str(SOTA / 'models/prot_bert')
        tokenizer, backbone = BertTokenizer.from_pretrained(pb, do_lower_case=False), \
            BertModel.from_pretrained(pb)
        spaced = True  # ProtBERT expects space-separated residues
        if args.model == 'bert_ampep':
            net, epochs, lr = build_bert_ampep(backbone), 20, 2e-5
        else:
            net, epochs, lr = build_deep_pie(backbone), 15, 2e-5
        # BERT-AmPEP-style finetuning hygiene for both ProtBERT models
        for name, p in net.bert.named_parameters():
            if 'embeddings' in name or 'encoder.layer.0.' in name or 'encoder.layer.1.' in name:
                p.requires_grad = False
    net = net.to(device)

    tr_s, te_s, tr_y, te_y = train_test_split(seqs, y, test_size=0.2, random_state=42)
    tr_loader = DataLoader(DS(tr_s, tr_y, tokenizer, spaced), batch_size=16, shuffle=True)
    te_loader = DataLoader(DS(te_s, te_y, tokenizer, spaced), batch_size=64)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),
                            lr=lr, weight_decay=0.01)

    best_mae, best_state = float('inf'), None
    for ep in range(1, epochs + 1):
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
        print(f'[{args.model}/{args.mode}] ep{ep} val_MAE({yscale})={mae:.4f} '
              f'pred_std={np.std(preds_all):.4f}', flush=True)
    net.load_state_dict(best_state)
    net.to(device)

    ck = OUT / f'sota_models/{args.model}_{args.mode}'
    ck.mkdir(parents=True, exist_ok=True)
    torch.save({'model': net.state_dict()}, ck / 'best.pt')

    # 208 external peptides -> log10MIC
    pep = pd.read_csv(TIGER / 'runs_review/final11_peptides.csv')
    uniq = list(dict.fromkeys(pep.sequence))
    qloader = DataLoader(DS(uniq, np.zeros(len(uniq), dtype=np.float32), tokenizer, spaced),
                         batch_size=64)
    net.eval()
    preds = []
    with torch.no_grad():
        for batch, _ in qloader:
            ii, am = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            preds.extend(net(ii, am).cpu().tolist())
    preds = np.asarray(preds, dtype=np.float64)
    if yscale == 'scale':  # invert scale -> pMIC -> log10MIC
        preds = -((preds - BA_B) / BA_A)
    tag = 'neg' if args.mode == 'with_neg' else 'pos4929'
    suffix = '' if args.seed == 123 else f'_s{args.seed}'
    pd.Series(dict(zip(uniq, preds.tolist())), name='log10mic').to_csv(
        OUT / f'sota_{args.model}_{tag}{suffix}_preds.csv')
    print(f'saved sota_{args.model}_{tag}_preds.csv; log10MIC mean={preds.mean():.3f} '
          f'std={preds.std():.3f} range [{preds.min():.3f},{preds.max():.3f}]', flush=True)


if __name__ == '__main__':
    main()

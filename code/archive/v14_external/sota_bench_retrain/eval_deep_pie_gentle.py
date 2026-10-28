#!/usr/bin/env python3
"""External 11-family evaluation of the gentle-recipe Deep-PIE retest (arm C).

Loads deep_pie_gentle_best.pt, predicts log10(MIC) for the 208 test peptides,
builds pair deltas on the benchmark scale and reports per-family PCC + macro
against the original (collapsed) Deep-PIE numbers.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import BertModel, BertTokenizer

ROOT = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC')
SOTA = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/baseline/sota_bench')
sys.path.insert(0, str(ROOT))
from code.utils.metrics import regression_metrics  # noqa: E402

CKPT = ROOT / 'runs_review/benchmark_restricted/sota_models/deep_pie_gentle_best.pt'
YT = 'y_true_delta_log2_anchor_minus_query_03'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self, bert):
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


tok = BertTokenizer.from_pretrained(str(SOTA / 'models/prot_bert'), do_lower_case=False)
net = Net(BertModel.from_pretrained(str(SOTA / 'models/prot_bert')))
ck = torch.load(CKPT, map_location='cpu')
net.load_state_dict(ck['model'])
net.to(device).eval()

pep = pd.read_csv(ROOT / 'runs_review/final11_peptides.csv')
uniq = list(dict.fromkeys(pep.sequence))
preds = {}
with torch.no_grad():
    for s in uniq:
        e = tok(' '.join(s), padding='max_length', truncation=True, max_length=60, return_tensors='pt')
        preds[s] = float(net(e['input_ids'].to(device), e['attention_mask'].to(device)).item())
ser = pd.Series(preds, name='log10mic')
ser.to_csv(ROOT / 'runs_review/benchmark_restricted/sota_deep_pie_gentle_preds.csv')
print(f'推理 {len(uniq)} 条, 预测log10MIC: min {ser.min():.2f} / max {ser.max():.2f} / std {ser.std():.3f}')

A = pd.read_csv(ROOT / 'runs_review/benchmark_restricted/eight_way_predictions.csv')
A['deep_pie_gentle_delta'] = [
    (preds.get(a, np.nan) - preds.get(q, np.nan)) / math.log10(2)
    if a in preds and q in preds else np.nan
    for a, q in zip(A.anchor_sequence, A.query_sequence)
]
A.to_csv(ROOT / 'runs_review/benchmark_restricted/eight_way_predictions.csv', index=False)

KEEP_RAW = ['Mammuthusin-3', 'F', 'Hesperelin-3', 'Hydrodamin-3', 'Hydrodamin-2', 'Repeat',
            'LL37', 'Mammuthusin-2', 'Arctoterin-1', 'HAL-2/', 'HAL-1/']
order = ['Mammuthusin-3', 'F', 'Hesperelin-3', 'Hydrodamin-3', 'Hydrodamin-2', 'Repeat',
         'HAL-2', 'LL37', 'Mammuthusin-2', 'Arctoterin-1', 'HAL-1']
S = A[A.family.isin(KEEP_RAW)]
print(f'\n{"家族":<14}{"n":>5}  Deep-PIE(原配方)  Deep-PIE(温和配方)')
vals = []
for fam in order:
    g = S[S.family == (fam + '/' if fam.startswith('HAL') else fam)]
    r1 = regression_metrics(g.dropna(subset=['deep_pie_delta'])[YT], g.dropna(subset=['deep_pie_delta']).deep_pie_delta)['PCC']
    gg = g.dropna(subset=['deep_pie_gentle_delta'])
    r2 = regression_metrics(gg[YT], gg.deep_pie_gentle_delta)['PCC'] if len(gg) >= 4 else float('nan')
    vals.append(r2)
    print(f'{fam:<14}{len(g):>5}     {r1:+.3f}            {r2:+.3f}')
print(f'\n宏PCC: 原配方 -0.012   温和配方 {np.mean(vals):+.3f}')

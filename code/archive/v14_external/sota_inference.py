#!/usr/bin/env python3
"""Inference for trained SOTA models (deep_pie, llamp) on the 11-family testset."""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SOTA = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/baseline/sota_bench')
CKPT = SOTA / 'runs_review/benchmark_restricted/sota_models'
YT = 'y_true_delta_log2_anchor_minus_query_03'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
device = torch.device('cpu')  # GPU unavailable

pep = pd.read_csv(ROOT / 'runs_review/final11_peptides.csv')
uniq = list(dict.fromkeys(pep.sequence))
print(f'推理目标: {len(uniq)} 条测试肽', flush=True)

def predict_deep_pie(seqs):
    from transformers import BertTokenizer, BertModel
    pb = str(SOTA / 'models/prot_bert')
    tok = BertTokenizer.from_pretrained(pb, do_lower_case=False)
    bert = BertModel.from_pretrained(pb)
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = bert; self.fc1 = nn.Linear(bert.config.hidden_size, 512)
            self.fc2 = nn.Linear(512, 256); self.fc3 = nn.Linear(256, 1)
            self.relu = nn.ReLU(); self.drop = nn.Dropout(0.1)
        def forward(self, ii, am):
            p = self.bert(input_ids=ii, attention_mask=am).pooler_output
            x = self.drop(self.relu(self.fc1(p))); x = self.drop(self.relu(self.fc2(x)))
            return self.fc3(x).squeeze(-1)
    net = Net()
    ck = torch.load(CKPT / 'deep_pie_best.pt', map_location='cpu')
    net.load_state_dict(ck['model']); net.eval()
    # scaler from training
    tr = pd.read_csv(SOTA / 'data_restricted/deep_pie_input.csv')
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(tr.MIC.values.reshape(-1, 1))
    preds = {}
    with torch.no_grad():
        for s in seqs:
            enc = tok(' '.join(s), padding='max_length', truncation=True, max_length=60, return_tensors='pt')
            z = net(enc['input_ids'], enc['attention_mask']).item()
            preds[s] = float(sc.inverse_transform([[z]])[0][0])  # raw MIC
    return preds  # raw MIC

def predict_llamp(seqs):
    from transformers import AutoTokenizer, AutoModel
    pe = str(SOTA / 'models/peptide_tuned_ESM-2')
    tok = AutoTokenizer.from_pretrained(pe)
    esm = AutoModel.from_pretrained(pe)
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.esm = esm; h = esm.config.hidden_size
            self.fc1 = nn.Linear(h, 256); self.fc2 = nn.Linear(256, 1); self.drop = nn.Dropout(0.1)
        def forward(self, ii, am):
            p = self.esm(input_ids=ii, attention_mask=am).last_hidden_state.mean(dim=1)
            return self.fc2(self.drop(torch.relu(self.fc1(self.drop(p))))).squeeze(-1)
    net = Net()
    ck = torch.load(CKPT / 'llamp_best.pt', map_location='cpu')
    net.load_state_dict(ck['model']); net.eval()
    preds = {}
    with torch.no_grad():
        for s in seqs:
            enc = tok(s, padding='max_length', truncation=True, max_length=60, return_tensors='pt')
            preds[s] = float(net(enc['input_ids'], enc['attention_mask']).item())  # log10 MIC
    return preds

# Run both
print('Loading deep_pie...', flush=True)
dp = predict_deep_pie(uniq)
pd.Series(dp).to_csv(ROOT / 'runs_review/benchmark_restricted/sota_deep_pie_preds.csv')
print(f'deep_pie: {len(dp)} preds done', flush=True)

print('Loading llamp...', flush=True)
ll = predict_llamp(uniq)
pd.Series(ll).to_csv(ROOT / 'runs_review/benchmark_restricted/sota_llamp_preds.csv')
print(f'llamp: {len(ll)} preds done', flush=True)

# Assemble pair deltas + evaluate
A = pd.read_csv(ROOT / 'runs_review/benchmark_restricted/five_way_predictions.csv')
import math
A['deep_pie_delta'] = [math.log2(dp.get(a, np.nan)) - math.log2(dp.get(q, np.nan)) if dp.get(a) and dp.get(q) and dp.get(a) > 0 and dp.get(q) > 0 else np.nan for a, q in zip(A.anchor_sequence, A.query_sequence)]
A['llamp_delta'] = [ll.get(a, np.nan) - ll.get(q, np.nan) if not (np.isnan(ll.get(a, np.nan)) or np.isnan(ll.get(q, np.nan))) else np.nan for a, q in zip(A.anchor_sequence, A.query_sequence)]
A['llamp_delta'] = [v / math.log10(2) if v == v else np.nan for v in A.llamp_delta]  # log10→log2

sys.path.insert(0, str(ROOT))
from code.utils.metrics import regression_metrics
KEEP_RAW = ['Mammuthusin-3','F','Hesperelin-3','Hydrodamin-3','Hydrodamin-2','Repeat','LL37','Mammuthusin-2','Arctoterin-1','HAL-2/','HAL-1/']
S = A[A.family.isin(KEEP_RAW)]
METH = [('pair(0.7)', 'y_pred_delta_log2_anchor_minus_query_07'), ('single-restr', 'single_r'), ('evo-restr', 'evo_r'), ('Deep-PIE', 'deep_pie_delta'), ('LLAMP', 'llamp_delta')]
order = ['Mammuthusin-3','F','Hesperelin-3','Hydrodamin-3','Hydrodamin-2','Repeat','HAL-2','LL37','Mammuthusin-2','Arctoterin-1','HAL-1']
rows = []
for fam in order:
    g = S[S.family == (fam + '/' if fam.startswith('HAL') else fam)]
    r = {'family': fam, 'n': len(g)}
    for tag, col in METH:
        gg = g.dropna(subset=[col])
        m = regression_metrics(gg[YT], gg[col]) if len(gg) >= 4 else {'PCC': float('nan'), 'KCC': float('nan'), 'log10MAE': float('nan')}
        r[f'{tag}|PCC'] = round(m['PCC'], 3) if m['PCC'] == m['PCC'] else None
    rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(ROOT / 'runs_review/benchmark_restricted/sota_per_family.csv', index=False)
print(f'\n{"家族":<14}{"n":>5}' + ''.join(f'{t[0]:>14}' for t in METH), flush=True)
for r in rows:
    print(f'{r["family"]:<14}{r["n"]:>5}' + ''.join(f'{r[f"{t[0]}|PCC"] if r[f"{t[0]}|PCC"] is not None else "—":>14}' for t in METH), flush=True)
summary = {}
for tag, _ in METH:
    v = [r[f'{tag}|PCC'] for r in rows if r[f'{tag}|PCC'] is not None]
    summary[tag] = {'macro_PCC': round(float(np.mean(v)), 4), 'n_families': len(v)}
json.dump(summary, open(ROOT / 'runs_review/benchmark_restricted/sota_summary.json', 'w'), indent=2)
print('\n' + json.dumps(summary, indent=2), flush=True)

#!/usr/bin/env python3
"""Per-seed inference + 11-family macro PCC for EvoGradient retrain checkpoints.

Works for both recipes (old = scripts/train_evo_restricted.py whole-module
pickles in <dir>/{CNN,Transformer,Attention,LSTM}.pth; strict = state-dict
checkpoints handled by retrain_evo_strict.py itself).

Usage: python eval_evo_ckpt_dir.py <ckpt_dir> [--label NAME]
Writes runs_review/benchmark_restricted/sota_models/evo_eval/<label>_preds.csv
and prints the per-family table + macro PCC.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC')
EVO = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/baseline/AMP-potency-prediction-EvoGradient')
sys.path.insert(0, str(EVO))
from eval_tiger_style_pair_delta import SeqDataset  # noqa: E402

from torch.utils.data import DataLoader  # noqa: E402

OUT = ROOT / 'runs_review/benchmark_restricted'
YT = 'y_true_delta_log2_anchor_minus_query_03'
KEEP = ['Mammuthusin-3', 'F', 'Hesperelin-3', 'Hydrodamin-3', 'Hydrodamin-2', 'Repeat',
        'LL37', 'Mammuthusin-2', 'Arctoterin-1', 'HAL-2/', 'HAL-1/']
order = ['Mammuthusin-3', 'F', 'Hesperelin-3', 'Hydrodamin-3', 'Hydrodamin-2', 'Repeat',
         'HAL-2', 'LL37', 'Mammuthusin-2', 'Arctoterin-1', 'HAL-1']


def infer(ckpt_dir: Path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pep = pd.read_csv(ROOT / 'runs_review/final11_peptides.csv')
    uniq = list(dict.fromkeys(pep.sequence))
    loader = DataLoader(SeqDataset(uniq), batch_size=256, shuffle=False)
    outs = {}
    for name in ('CNN', 'Transformer', 'Attention', 'LSTM'):
        model = torch.load(ckpt_dir / f'{name}.pth', map_location='cpu')
        model.to(device).eval()
        vals = []
        with torch.no_grad():
            for inputs, masks, _ in loader:
                inputs, masks = inputs.float().to(device), masks.float().to(device)
                o = model(inputs, masks) if name == 'Attention' else model(inputs)
                vals.extend(float(v) for v in torch.squeeze(o).cpu().numpy().ravel())
        outs[name] = np.asarray(vals)
        del model
        torch.cuda.empty_cache()
    ens = np.mean([outs[k] for k in outs], axis=0)
    return dict(zip(uniq, ens.tolist()))


def macro(preds: dict):
    A = pd.read_csv(OUT / 'eleven_way_predictions.csv')
    delta = [(preds.get(a, np.nan) - preds.get(q, np.nan)) / math.log10(2)
             if a in preds and q in preds else np.nan
             for a, q in zip(A.anchor_sequence, A.query_sequence)]
    col = '_tmp'
    A[col] = delta
    ps = []
    for fam in order:
        raw = fam + '/' if fam.startswith('HAL') else fam
        g = A[A.family == raw].dropna(subset=[col])
        ps.append(np.corrcoef(g[YT], g[col])[0, 1] if len(g) >= 4 else np.nan)
    return float(np.nanmean(ps)), ps


def main():
    ckpt_dir = Path(sys.argv[1])
    label = sys.argv[sys.argv.index('--label') + 1] if '--label' in sys.argv else ckpt_dir.name
    preds = infer(ckpt_dir)
    m, ps = macro(preds)
    outdir = OUT / 'sota_models/evo_eval'
    outdir.mkdir(parents=True, exist_ok=True)
    pd.Series(preds, name='log10mic').to_csv(outdir / f'{label}_preds.csv')
    print(f'[{label}] macro PCC = {m:.4f}')
    print('  ' + '  '.join(f'{f}={p:+.3f}' for f, p in zip(order, ps)))


if __name__ == '__main__':
    main()

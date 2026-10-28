#!/usr/bin/env python3
"""Evaluate the ORIGINAL published APEX checkpoints (40-model ensemble, trained on
the authors' large multi-species corpus) on our 11-family external testset.

Multi-task output is 34 species; column 0 = 'E. coli ATCC11775' (per predict.py).
Protocol mirrors the released predict.py: equal-weight ensemble average of the
activity scale, then MIC = 10**(6 - y). Pair delta on the benchmark scale:
delta_log2 = (y_q - y_a) / log10(2).

Usage: python eval_apex_original.py [--gpu 0] [--min-models 30]
"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

ROOT = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC')
HERE = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/baseline/sota_bench/apex-main')
sys.path.insert(0, str(ROOT))
from code.utils.metrics import regression_metrics  # noqa: E402
sys.path.insert(0, str(HERE))
from utils import make_vocab, AAindex, onehot_encoding  # noqa: E402

OUT_DIR = ROOT / 'runs_review/benchmark_restricted'
YT = 'y_true_delta_log2_anchor_minus_query_03'
MAX_LEN = 52
KEEP_RAW = ['Mammuthusin-3', 'F', 'Hesperelin-3', 'Hydrodamin-3', 'Hydrodamin-2', 'Repeat',
            'LL37', 'Mammuthusin-2', 'Arctoterin-1', 'HAL-2/', 'HAL-1/']
order = ['Mammuthusin-3', 'F', 'Hesperelin-3', 'Hydrodamin-3', 'Hydrodamin-2', 'Repeat',
         'HAL-2', 'LL37', 'Mammuthusin-2', 'Arctoterin-1', 'HAL-1']
KD = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4,
      'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8,
      'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}


def charge(s):
    return s.count('K') + s.count('R') + 0.5 * s.count('H') - s.count('D') - s.count('E')


def hydro(s):
    return float(np.mean([KD[c] for c in s]))


def macro_eval(A, col):
    ps, ks, ms = [], [], []
    worst = (2.0, '')
    for fam in order:
        g = A[A.family == (fam + '/' if fam.startswith('HAL') else fam)].dropna(subset=[col])
        if len(g) < 4:
            continue
        m = regression_metrics(g[YT], g[col])
        ps.append(m['PCC'])
        ks.append(m['KCC'])
        ms.append(m['log10MAE'])
        if m['PCC'] < worst[0]:
            worst = (m['PCC'], fam)
    return {'macro_PCC': round(float(np.mean(ps)), 4), 'macro_KCC': round(float(np.mean(ks)), 4),
            'macro_MAE': round(float(np.mean(ms)), 4), 'worst': f'{worst[1]}({worst[0]:+.3f})',
            'per_family': {f: round(p, 3) for f, p in zip(order, ps)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--min-models', type=int, default=30)
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    files = [p for p in (HERE / 'trained_models').iterdir()
             if p.is_file() and p.stat().st_size > 1_000_000]
    print(f'可用原版权重: {len(files)}/40', flush=True)
    if len(files) < args.min_models:
        print(f'不足 {args.min_models} 个，退出（下载未完成）', flush=True)
        sys.exit(1)

    word2idx, _ = make_vocab()
    pep = pd.read_csv(ROOT / 'runs_review/final11_peptides.csv')
    uniq = list(dict.fromkeys(pep.sequence))
    X, _, _ = onehot_encoding(uniq, MAX_LEN, word2idx)
    Xt = torch.from_numpy(X).long().to(device)

    sums = np.zeros(len(uniq))
    n_used = 0
    for pth in sorted(files):
        m = torch.load(pth, map_location=device)
        m.to(device).eval()
        with torch.no_grad():
            pr = []
            for i in range(0, len(uniq), 512):
                pr.append(m(Xt[i:i + 512])[:, 0].cpu().numpy())  # col 0 = E. coli
        sums += np.concatenate(pr)
        n_used += 1
        del m
        torch.cuda.empty_cache()
    y_act = sums / n_used
    print(f'集成 {n_used} 模型; 活性范围 [{y_act.min():.2f},{y_act.max():.2f}]', flush=True)
    pd.Series(dict(zip(uniq, y_act)), name='apex_original_activity').to_csv(
        OUT_DIR / 'sota_apex_original_preds.csv')

    A = pd.read_csv(OUT_DIR / 'eight_way_predictions.csv')
    act = dict(zip(uniq, y_act))
    A['apex_original_delta'] = [(act.get(q, np.nan) - act.get(a, np.nan)) / math.log10(2)
                                if a in act and q in act else np.nan
                                for a, q in zip(A.anchor_sequence, A.query_sequence)]

    res = {'n_models': n_used, 'peptide_log10MIC_std': round(float(np.std(6 - y_act)), 3)}
    res['original'] = macro_eval(A, 'apex_original_delta')
    res['retrained_v0'] = macro_eval(A, 'apex_delta')       # benchmark primary (original 40-retrain preds)
    res['retrained_v1'] = macro_eval(A, 'apex_v1')          # tuned subset
    # 组成规则相关（机制检查）
    pl = 6 - y_act
    ch = np.array([charge(s) for s in uniq])
    hy = np.array([hydro(s) for s in uniq])
    ln = np.array([len(s) for s in uniq])
    res['composition_rule'] = {
        'corr_pred_charge': round(float(pearsonr(pl, ch)[0]), 3),
        'corr_pred_hydro': round(float(pearsonr(pl, hy)[0]), 3),
        'corr_pred_len': round(float(pearsonr(pl, ln)[0]), 3)}
    A.to_csv(OUT_DIR / 'eight_way_predictions.csv', index=False)
    json.dump(res, open(OUT_DIR / 'apex_original_vs_retrained.json', 'w'), indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

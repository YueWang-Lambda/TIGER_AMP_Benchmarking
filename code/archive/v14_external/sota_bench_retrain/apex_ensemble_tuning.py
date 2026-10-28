#!/usr/bin/env python3
"""Stage 2a: inference-side ensemble tuning for APEX-restr (no retraining).

Runs each of the 40 retrained checkpoints on the 208 external peptides to get a
40x208 activity matrix, then evaluates five ensemble variants on the 11-family
benchmark protocol:
  v0  all 40 equal (current benchmark, macro PCC 0.323)
  v1  only the 4 lr=1e-5 configs x 5 reps = 20 models
  v2  best rep per config by val MSE = 8 models
  v3  softmax(val-MSE) weights, T in {0.05, 0.1}
  v4  global top-10 by val MSE
Pair-delta scale matches the benchmark: delta_log2 = (y_q - y_a) / log10(2).

Usage: python apex_ensemble_tuning.py --gpu 1
"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC')
sys.path.insert(0, str(ROOT))
# Import the project package FIRST: once an apex-module dependency pulls in the
# stdlib `code` module, sys.modules['code'] would shadow this package.
from code.utils.metrics import regression_metrics  # noqa: E402

sys.path.insert(0, '/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/baseline/sota_bench/apex-main')
from AMP_DL_model_twohead import AMP_model  # noqa: E402
from utils import make_vocab, AAindex, onehot_encoding  # noqa: E402

CKPT_DIR = ROOT / 'runs_review/benchmark_restricted/sota_models/apex_ensemble'
OUT_DIR = ROOT / 'runs_review/benchmark_restricted'
YT = 'y_true_delta_log2_anchor_minus_query_03'
MAX_LEN = 52
KEEP_RAW = ['Mammuthusin-3', 'F', 'Hesperelin-3', 'Hydrodamin-3', 'Hydrodamin-2', 'Repeat',
            'LL37', 'Mammuthusin-2', 'Arctoterin-1', 'HAL-2/', 'HAL-1/']
order = ['Mammuthusin-3', 'F', 'Hesperelin-3', 'Hydrodamin-3', 'Hydrodamin-2', 'Repeat',
         'HAL-2', 'LL37', 'Mammuthusin-2', 'Arctoterin-1', 'HAL-1']


def macro_eval(A, col):
    vals, worst, rows = [], None, []
    S = A[A.family.isin(KEEP_RAW)]
    for fam in order:
        g = S[S.family == (fam + '/' if fam.startswith('HAL') else fam)].dropna(subset=[col])
        r = regression_metrics(g[YT], g[col])['PCC'] if len(g) >= 4 else float('nan')
        vals.append(r)
        rows.append({'family': fam, 'PCC': None if r != r else round(r, 3)})
        if worst is None or r < worst[0]:
            worst = (r, fam)
    return round(float(np.nanmean(vals)), 4), worst, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='1')
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    word2idx, _ = make_vocab()
    emb, _ = AAindex('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/baseline/sota_bench/apex-main/aaindex1.csv', word2idx)
    emb_size = emb.shape[1]
    pep = pd.read_csv(ROOT / 'runs_review/final11_peptides.csv')
    uniq = list(dict.fromkeys(pep.sequence))
    X, _, _ = onehot_encoding(uniq, MAX_LEN, word2idx)
    Xt = torch.from_numpy(X).long().to(device)

    ckpts, acts, mses, keys = [], [], [], []
    for pth in sorted(CKPT_DIR.glob('apex_*.pt')):
        ck = torch.load(pth, map_location=device)
        rl, dh, dl = ck['key'].split('&')[:3]
        m = AMP_model(emb, emb_size, int(rl), int(dh), int(dl), 4, 1).to(device)
        m.load_state_dict(ck['model'])
        m.eval()
        with torch.no_grad():
            pr = []
            for i in range(0, len(uniq), 1024):
                pr.append(m(Xt[i:i + 1024]).squeeze(-1).cpu().numpy())
        acts.append(np.concatenate(pr))
        mses.append(ck['val_mse'])
        keys.append(ck['key'])
        del m
    acts = np.array(acts)  # (40, 208) activity scale y = 6 - log10 MIC
    mses = np.array(mses)
    lr15 = np.array(['1e-05' in k for k in keys])
    print(f'40 模型推理完成; lr1e-5 模型数={lr15.sum()}; valMSE范围[{mses.min():.3f},{mses.max():.3f}]', flush=True)

    A = pd.read_csv(OUT_DIR / 'eight_way_predictions.csv')
    variants = {'v0_40等权': np.ones(len(acts)) / len(acts),
                'v1_lr1e-5x20': lr15 / max(lr15.sum(), 1),
                'v2_每配置最优rep': None,
                'v3_软加权T0.05': np.exp(-mses / 0.05), 'v3_软加权T0.1': np.exp(-mses / 0.1)}
    best_per_cfg = {}
    for i, k in enumerate(keys):
        if k not in best_per_cfg or mses[i] < mses[best_per_cfg[k]]:
            best_per_cfg[k] = i
    w2 = np.zeros(len(acts))
    w2[list(best_per_cfg.values())] = 1.0
    variants['v2_每配置最优rep'] = w2
    top10 = np.argsort(mses)[:10]
    w4 = np.zeros(len(acts))
    w4[top10] = 1.0
    variants['v4_top10'] = w4

    report = []
    for name, w in variants.items():
        w = w / w.sum()
        y_act = (w[:, None] * acts).sum(axis=0)
        ampd = dict(zip(uniq, y_act.tolist()))
        col = 'apex_' + name.split('_')[0]
        A[col] = [(ampd.get(q, np.nan) - ampd.get(a, np.nan)) / math.log10(2)
                  if a in ampd and q in ampd else np.nan
                  for a, q in zip(A.anchor_sequence, A.query_sequence)]
        m, worst, rows = macro_eval(A, col)
        peptide_log10mic_std = float(np.std(6 - y_act))
        report.append({'variant': name, 'macro_PCC': m, 'worst_family': worst[1],
                       'worst_PCC': round(worst[0], 3), 'peptide_log10MIC_std': round(peptide_log10mic_std, 3)})
        print(f'{name:<18} 宏PCC={m:+.4f}  最差={worst[1]}({worst[0]:+.3f})  逐肽std={peptide_log10mic_std:.3f}', flush=True)
    rep = pd.DataFrame(report)
    rep.to_csv(OUT_DIR / 'sensitivity_apex_tuned.csv', index=False)
    A.to_csv(OUT_DIR / 'eight_way_predictions.csv', index=False)
    best = rep.loc[rep.macro_PCC.idxmax()]
    json.dump({'best_variant': best.variant, 'best_macro_PCC': best.macro_PCC,
               'trigger_2b': bool(best.macro_PCC < 0.38)},
              open(OUT_DIR / 'sensitivity_apex_tuned_decision.json', 'w'), ensure_ascii=False, indent=2)
    print('\n最优:', best.variant, best.macro_PCC, '| 触发2b重训:', best.macro_PCC < 0.38, flush=True)


if __name__ == '__main__':
    main()

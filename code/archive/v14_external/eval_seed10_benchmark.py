#!/usr/bin/env python3
"""Seed-10 benchmark: 16 methods x seeds 120-129, macro PCC per seed, raw and
trimmed (drop max+min) statistics. Final table caliber for the paper.
(2026-10-04: +4 retrained methods — MBC-Attention / HyperAMP / SenseXAMP-Ecoli
/ AMPlify-regr — unified +neg protocol, no author weights loaded.)

Sources per method:
  TIGER-pair(E) / TIGER-single(n1n2): run dirs (external eval_run inference)
  CSV methods (per-seed preds files; base file = seed 123):
    Das / LLAMP / BERT-AmPEP / Deep-PIE  sota_<m>{,_s<seed>}_preds.csv
    HydrAMP-arch-regr                     sota_hydramp_regr_neg_s<seed>_preds.csv (36-base excluded)
    AMPLiT-regr vB                        sota_amplit_regr_vB{,_s<seed>}_preds.csv
    MBC-Attention+neg                     sota_mbc_attention_neg_s<seed>_preds.csv
    HyperAMP+neg                          sota_hyperamp_neg_s<seed>_preds.csv
    SenseXAMP-Ecoli+neg                   sota_sensexamp_neg_s<seed>_preds.csv
    AMPlify-regr (ours)                   sota_amplify_regr_neg_s<seed>_preds.csv
    EvoGrad-strict                        sota_models/evo_strict/preds_seed<seed>.csv
  ckpt methods (re-inference on 208 peptides):
    DeepAMP-CNN   sota_models/deep_amp_ensemble/cnn_seed<seed>.pt
    APEX lrfix    sota_models/apex_neg/apexlr_<seed>.pt
    AMPD+neg      sota_models/amp_designer_with_neg/ampd_<seed>.pt

Outputs: runs_review/benchmark_restricted/benchmark_fullpool_seed10.json (+csv table)
Usage: CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. python scripts/eval_seed10_benchmark.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / 'baseline' / 'sota_bench'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(BENCH))

BR = ROOT / 'runs_review/benchmark_restricted'
SEEDS = tuple(range(120, 130))
YT = 'y_true_delta_log2_anchor_minus_query_03'
ORDER = ['Mammuthusin-3', 'F', 'Hesperelin-3', 'Hydrodamin-3', 'Hydrodamin-2',
         'Repeat', 'HAL-2', 'LL37', 'Mammuthusin-2', 'Arctoterin-1', 'HAL-1']

A0 = pd.read_csv(BR / 'eleven_way_predictions.csv')
PEP = list(dict.fromkeys(pd.read_csv(ROOT / 'runs_review/final11_peptides.csv').sequence))


def macro(d):
    delta = [(d.get(a, np.nan) - d.get(q, np.nan)) / math.log10(2)
             if a in d and q in d else np.nan
             for a, q in zip(A0.anchor_sequence, A0.query_sequence)]
    A0['_t'] = delta
    vs = []
    for f in ORDER:
        r = f + '/' if f.startswith('HAL') else f
        g = A0[A0.family == r].dropna(subset=['_t'])
        vs.append(np.corrcoef(g[YT], g['_t'])[0, 1] if len(g) >= 4 else np.nan)
    return round(float(np.nanmean(vs)), 4)


def macro_csv(fn):
    d = pd.read_csv(BR / fn, index_col=0).iloc[:, 0].to_dict()
    return macro(d)


def csv_methods():
    out = {}
    for m, base in [('Das2021-regr (ours)', 'sota_das_regr'),
                    ('LLAMP+neg', 'sota_llamp_neg'),
                    ('BERT-AmPEP+neg', 'sota_bert_ampep_neg'),
                    ('Deep-PIE+neg', 'sota_deep_pie_neg'),
                    ('HydrAMP-arch-regr (ours)', 'sota_hydramp_regr_neg'),
                    ('AMPLiT-regr vB (ours)', 'sota_amplit_regr_vB'),
                    ('MBC-Attention+neg', 'sota_mbc_attention_neg'),
                    ('HyperAMP+neg', 'sota_hyperamp_neg'),
                    ('SenseXAMP-Ecoli+neg', 'sota_sensexamp_neg'),
                    ('AMPlify-regr (ours)', 'sota_amplify_regr_neg'),
                    # v2-9 additions (2026-10-08, eight retrained methods)
                    ('EIPpred+neg', 'sota_EIPpred+neg'),
                    ('AMPActiPred+neg', 'sota_AMPActiPred+neg'),
                    ('AMPGen+neg', 'sota_AMPgen+neg'),
                    ('AMPpredictor+neg', 'sota_AMPpredictor+neg'),
                    ('ANIA+neg', 'sota_ANIA+neg'),
                    ('tsAMP-CS+neg', 'sota_tsAMP-CS+neg'),
                    ('GGN-ABPMIC+neg', 'sota_GGN-ABPMIC+neg'),
                    ('EvoGrad-strict', None)]:
        per = {}
        if m == 'EvoGrad-strict':
            # [v2-11 fix] preds are log10 MIC in µM (censored at log10(8192));
            # harmonize per peptide to log10 µg/mL before differencing
            # (log10(µg/mL) = log10(µM) + log10(MW/1000), MW per retrain_evo_strict.mw)
            from retrain_evo_strict import mw as evo_mw
            for s in SEEDS:
                f = BR / f'sota_models/evo_strict/preds_seed{s}.csv'
                if f.exists():
                    d = pd.read_csv(f, index_col=0).iloc[:, 0].to_dict()
                    d = {seq: v + float(np.log10(evo_mw(seq) / 1000.0)) for seq, v in d.items()}
                    per[s] = macro(d)
        else:
            for s in SEEDS:
                f = (BR / f'{base}_s{s}_preds.csv') if (base != 'sota_das_regr' or s != 123) else None
                if m in ('Das2021-regr (ours)', 'LLAMP+neg', 'BERT-AmPEP+neg',
                         'Deep-PIE+neg', 'AMPLiT-regr vB (ours)') and s == 123:
                    f = BR / f'{base}_preds.csv'
                if f is not None and f.exists():
                    per[s] = macro_csv(f.relative_to(BR))
        out[m] = per
        print(f'{m}: {len(per)}/10 seeds', flush=True)
    return out


def ckpt_methods(device):
    out = {}
    # DeepAMP-CNN
    import retrain_deep_amp as da
    import torch
    X = da.encode(PEP)[0]
    x = torch.from_numpy(X[:len(PEP)]).float().to(device)
    per = {}
    for s in SEEDS:
        f = BR / f'sota_models/deep_amp_ensemble/cnn_seed{s}.pt'
        if not f.exists():
            continue
        m_ = da.CNNRegressor().to(device)
        m_.load_state_dict(torch.load(f, map_location='cpu')['model'])
        m_.eval()
        with torch.no_grad():
            p = m_(x).squeeze(-1).cpu().numpy()
        per[s] = macro(dict(zip(PEP, p.tolist())))
    out['DeepAMP-CNN'] = per
    print(f'DeepAMP-CNN: {len(per)}/10 seeds', flush=True)

    # APEX lrfix+neg
    sys.path.insert(0, str(BENCH / 'apex-main'))
    from AMP_DL_model_twohead import AMP_model
    from utils import make_vocab, AAindex, onehot_encoding
    word2idx, _ = make_vocab()
    emb, _ = AAindex(str(BENCH / 'apex-main' / 'aaindex1.csv'), word2idx)
    xa, _, _ = onehot_encoding(PEP, 52, word2idx)
    xt = torch.from_numpy(xa).long().to(device)
    per = {}
    for s in SEEDS:
        f = BR / f'sota_models/apex_neg/apexlr_{s}.pt'
        if not f.exists():
            continue
        m_ = AMP_model(emb, emb.shape[1], 2, 128, 2048, 4, 1).to(device)
        m_.load_state_dict(torch.load(f, map_location='cpu')['model'])
        m_.eval()
        with torch.no_grad():
            p = m_(xt).squeeze(-1).cpu().numpy()
        per[s] = macro(dict(zip(PEP, (6.0 - p).tolist())))
    out['APEX-restr(lrfix)+neg'] = per
    print(f'APEX: {len(per)}/10 seeds', flush=True)

    # AMPD+neg
    from transformers import BertTokenizer, GPT2Config, GPT2LMHeadModel
    sys.path.insert(0, str(BENCH / 'AMP-Designer'))
    import train_ampd_restricted_neg as an
    W = BENCH / 'models' / 'amp_designer'
    tok = BertTokenizer(vocab_file=str(W / 'vocab.txt'))
    cfg = GPT2Config.from_pretrained(str(W / 'config.json'))
    ds = an.SeqDS(PEP, np.zeros(len(PEP)), tok)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=64, collate_fn=an.collate)
    per = {}
    for s in SEEDS:
        f = BR / f'sota_models/amp_designer_with_neg/ampd_{s}.pt'
        if not f.exists():
            continue
        gpt = GPT2LMHeadModel.from_pretrained(str(W / 'pytorch_model.bin'), config=cfg)
        m_ = an.CombinedModel(gpt).to(device)
        m_.load_state_dict(torch.load(f, map_location='cpu')['model'])
        m_.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                preds.extend(m_(xb.to(device)).cpu().tolist())
        per[s] = macro(dict(zip(PEP, preds)))
    out['AMP-Designer+neg'] = per
    print(f'AMPD: {len(per)}/10 seeds', flush=True)
    return out


def tiger_arms(device):
    # submission-caliber TIGER rows, read from the frozen S4 panels table
    # (per-seed macro PCC; 2026-10-05: replaces the historical companion-
    # regime E rows, whose archived checkpoints predate the v2 renaming and
    # are deliberately not consumed by this project)
    panels = json.load(open(BR / 's4_panels_seed10.json'))
    src = (('TIGER-pair (S4)', panels['模态']['S4(gps)']),
           ('TIGER-single', panels['头类型']['single(N2预训练+N1行注入)']))
    out = {}
    for name, ent in src:
        per = {int(s): v for s, v in ent['per_seed'].items()}
        out[name] = {s: per[s] for s in SEEDS if s in per}
        print(f'{name}: {len(out[name])}/10 seeds (from s4_panels_seed10.json)',
              flush=True)
    return out


def stats(per):
    v = sorted(per.values())
    if not v:
        return {}
    row = {'n': len(v), 'per_seed': {str(s): per[s] for s in sorted(per)}}
    row['mean'] = round(float(np.mean(v)), 4)
    row['std'] = round(float(np.std(v)), 4)
    if len(v) >= 3:
        t = v[1:-1]  # drop max + min
        row['trimmed_mean'] = round(float(np.mean(t)), 4)
        row['trimmed_std'] = round(float(np.std(t)), 4)
    return row


def main():
    import torch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    res = {}
    res.update(csv_methods())
    res.update(ckpt_methods(device))
    res.update(tiger_arms(device))
    table = {m: stats(per) for m, per in res.items()}
    json.dump(table, open(BR / 'benchmark_fullpool_seed10.json', 'w'),
              ensure_ascii=False, indent=2)

    rows = sorted(table.items(), key=lambda kv: -kv[1].get('mean', -9))
    print('\n=== 16 方法 × 10 种子（120–129）终表：原始均值±σ（副口径：截尾） ===')
    for i, (m, r) in enumerate(rows, 1):
        tm = r.get('trimmed_mean')
        ts = r.get('trimmed_std')
        print(f"{i:2d}. {m:34s} {r.get('mean', float('nan')):.4f} ± {r.get('std', float('nan')):.4f}"
              f"   截尾 {tm:.4f} ± {ts:.4f}   n={r['n']}")
    print(f'saved {BR}/benchmark_fullpool_seed10.json', flush=True)


if __name__ == '__main__':
    main()

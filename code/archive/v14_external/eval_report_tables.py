#!/usr/bin/env python3
"""Assemble all numeric tables for the advisor report.

Outputs runs_review/benchmark_restricted/report_tables.json with:
  metrics11 : {method: {PCC/KCC/log10MAE/RSE: {mean,std}}} — 11-family full metrics,
              mean±std over seeds (deterministic single value for APEX-original).
  perfam11  : {method: {family: mean-over-seeds PCC, '宏PCC': ...}}
  tierB_perfam : {method: {family: PCC}} for the 4-method decontaminated showdown.

Prediction sources: per-seed CSVs (CSV_METHODS), local v2 ckpts (deep_amp cnn),
old-project ckpts (apexlr, ampd), TIGER eval_run, saved APEX-original activity.
Usage: PYTHONPATH=. python scripts/eval_report_tables.py [--gpu 0]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'baseline/sota_bench'))

from eval_neg_twostage_arms import eval_run, ORDER  # noqa: E402
from code.utils.metrics import regression_metrics  # noqa: E402

BR = ROOT / 'runs_review/benchmark_restricted'
RA = ROOT / 'runs_ablation'
OLD = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER/'
           'runs_review/benchmark_restricted/sota_models')
YT = 'y_true_delta_log2_anchor_minus_query_03'
METRICS = ('PCC', 'KCC', 'log10MAE', 'RSE')
SEEDS = tuple(range(120, 130))
TIER_B = ['Mammuthusin-3', 'F', 'Hesperelin-3', 'Hydrodamin-3', 'Hydrodamin-2',
          'Repeat', 'Mammuthusin-2', 'Arctoterin-1']

A0 = pd.read_csv(BR / 'eleven_way_predictions.csv')


def per_family(A, fams, base):
    per, met = {}, {m: [] for m in METRICS}
    for fam in fams:
        r = fam + '/' if fam.startswith('HAL') else fam
        g = A[A.family == r].dropna(subset=['_p'])
        if len(g) < 4:
            per[fam] = None
            continue
        mm = regression_metrics(g[YT], g['_p'], input_log_base=base)
        per[fam] = round(mm['PCC'], 3)
        for m in METRICS:
            met[m].append(mm.get(m, np.nan))
    per['宏PCC'] = round(float(np.nanmean([v for v in per.values() if v is not None])), 4)
    return per, {m: round(float(np.nanmean(met[m])), 4) for m in METRICS}


def attach_delta(acts, invert=False):
    """acts: {seq: log10MIC prediction (higher activity models: pass 6-y already)."""
    A = A0.copy()
    sgn = -1.0 if invert else 1.0
    A['_p'] = [(sgn * (acts.get(a, np.nan) - acts.get(q, np.nan))) / 0.3010299956639812
               if a in acts and q in acts else np.nan
               for a, q in zip(A.anchor_sequence, A.query_sequence)]
    return A


def stats(per_seed_metrics):
    return {m: {'mean': round(float(np.mean([r[m] for r in per_seed_metrics])), 4),
                'std': round(float(np.std([r[m] for r in per_seed_metrics])), 4)}
            for m in METRICS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='0')
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    out = {'metrics11': {}, 'perfam11': {}, 'tierB_perfam': {}}
    uniq = list(dict.fromkeys(pd.read_csv(ROOT / 'runs_review/final11_peptides.csv').sequence))

    # ---------- TIGER arms (eval_run, calibrated log2 deltas) ----------
    for label, arm in (('TIGER-pair(S4)', 'S4'), ('TIGER-single', 's4_single_n1n2')):
        pers, mets = [], []
        for s in SEEDS:
            ens = eval_run(RA / arm / f'{arm}_seed{s}')
            A = ens.rename(columns={'y_true': YT, 'y_pred': '_p'})
            pf, mm = per_family(A, ORDER, base=2)
            pers.append(pf)
            mets.append(mm)
            print(f'[{label} s{s}] {mm["PCC"]}', flush=True)
        out['metrics11'][label] = stats(mets)
        out['perfam11'][label] = {f: round(float(np.mean([p[f] for p in pers])), 3)
                                  if pers[0][f] is not None else None for f in ORDER}
        out['perfam11'][label]['宏PCC'] = out['metrics11'][label]['PCC']['mean']

    # ---------- CSV-based external methods ----------
    CSV_METHODS = {
        'Das2021-regr': [('sota_das_regr_preds.csv' if s == 123 else f'sota_das_regr_s{s}_preds.csv') for s in SEEDS],
        'LLAMP': [('sota_llamp_neg_preds.csv' if s == 123 else f'sota_llamp_neg_s{s}_preds.csv') for s in SEEDS],
        'BERT-AmPEP60': [('sota_bert_ampep_neg_preds.csv' if s == 123 else f'sota_bert_ampep_neg_s{s}_preds.csv') for s in SEEDS],
        'Deep-PIE': [('sota_deep_pie_neg_preds.csv' if s == 123 else f'sota_deep_pie_neg_s{s}_preds.csv') for s in SEEDS],
        'HydrAMP': [f'sota_hydramp_regr_neg_s{s}_preds.csv' for s in SEEDS],
        'AMPLiT': [('sota_amplit_regr_vB_preds.csv' if s == 123 else f'sota_amplit_regr_vB_s{s}_preds.csv') for s in SEEDS],
        'EvoGradient': [f'sota_models/evo_strict/preds_seed{s}.csv' for s in SEEDS],  # OLD-project dir, see CSV_ROOT
        # 2026-10-05: four retrained additions (unified +neg protocol, no
        # author weights; see baseline/sota_bench/retrain_*_neg.py headers)
        'MBC-Attention': [f'sota_mbc_attention_neg_s{s}_preds.csv' for s in SEEDS],
        'HyperAMP': [f'sota_hyperamp_neg_s{s}_preds.csv' for s in SEEDS],
        'SenseXAMP-Ecoli': [f'sota_sensexamp_neg_s{s}_preds.csv' for s in SEEDS],
        'AMPlify-regr': [f'sota_amplify_regr_neg_s{s}_preds.csv' for s in SEEDS],
        # v2-9 additions (2026-10-08, eight retrained methods)
        'EIPpred': [f'sota_EIPpred+neg_s{s}_preds.csv' for s in SEEDS],
        'AMPActiPred': [f'sota_AMPActiPred+neg_s{s}_preds.csv' for s in SEEDS],
        'AMPGen': [f'sota_AMPgen+neg_s{s}_preds.csv' for s in SEEDS],
        'AMPpredictor': [f'sota_AMPpredictor+neg_s{s}_preds.csv' for s in SEEDS],
        'ANIA': [f'sota_ANIA+neg_s{s}_preds.csv' for s in SEEDS],
        'tsAMP-CS': [f'sota_tsAMP-CS+neg_s{s}_preds.csv' for s in SEEDS],
        'GGN-ABPMIC': [f'sota_GGN-ABPMIC+neg_s{s}_preds.csv' for s in SEEDS],
    }
    CSV_ROOTS = [BR, OLD.parent]  # v2 first, then the legacy benchmark_restricted
    def read_pred(fn):
        for root in CSV_ROOTS:
            p = root / fn
            if p.exists():
                return pd.read_csv(p, index_col=0).iloc[:, 0].to_dict()
        raise FileNotFoundError(fn)

    # [v2-11 fix] EvoGradient driver labels/predictions are log10 MIC in µM
    # (censored at log10(8192)); harmonize to the unified log10 µg/mL domain
    # per peptide before differencing: log10(µg/mL) = log10(µM) + log10(MW/1000),
    # MW = sum of average residue masses + water (retrain_evo_strict.mw).
    from retrain_evo_strict import mw as evo_mw

    for label, files in CSV_METHODS.items():
        pers, mets = [], []
        for fn in files:
            d = read_pred(fn)
            if label == 'EvoGradient':
                d = {s: v + float(np.log10(evo_mw(s) / 1000.0)) for s, v in d.items()}
            pf, mm = per_family(attach_delta(d), ORDER, base=2)
            pers.append(pf)
            mets.append(mm)
        out['metrics11'][label] = stats(mets)
        out['perfam11'][label] = {f: round(float(np.mean([p[f] for p in pers])), 3)
                                  if pers[0][f] is not None else None for f in ORDER}
        out['perfam11'][label]['宏PCC'] = out['metrics11'][label]['PCC']['mean']
        print(f'[{label}] {out["metrics11"][label]["PCC"]}', flush=True)

    # ---------- ckpt-based external methods ----------
    sys.path.insert(0, str(ROOT / 'baseline/sota_bench/apex-main'))
    from AMP_DL_model_twohead import AMP_model
    from utils import make_vocab, AAindex, onehot_encoding
    word2idx, _ = make_vocab()
    emb, _ = AAindex(str(ROOT / 'baseline/sota_bench/apex-main/aaindex1.csv'), word2idx)
    X, _, _ = onehot_encoding(uniq, 52, word2idx)
    Xt = torch.from_numpy(X).long().to(device)

    def apex_preds(state_path):
        sd = torch.load(state_path, map_location='cpu')
        m = AMP_model(emb, emb.shape[1], 2, 128, 2048, 4, 1).to(device)
        m.load_state_dict(sd['model'])
        m.eval()
        with torch.no_grad():
            pr = [m(Xt[i:i + 512])[:, 0].cpu().numpy() for i in range(0, len(uniq), 512)]
        y = np.concatenate(pr)
        return dict(zip(uniq, (6.0 - y)))  # activity -> log10MIC

    pers, mets = [], []
    for s in SEEDS:
        pf, mm = per_family(attach_delta(apex_preds(OLD / 'apex_neg' / f'apexlr_{s}.pt')), ORDER, base=2)
        pers.append(pf)
        mets.append(mm)
    out['metrics11']['APEX-retrained'] = stats(mets)
    out['perfam11']['APEX-retrained'] = {f: round(float(np.mean([p[f] for p in pers])), 3) for f in ORDER}
    out['perfam11']['APEX-retrained']['宏PCC'] = out['metrics11']['APEX-retrained']['PCC']['mean']
    print('[APEX-retrained]', out['metrics11']['APEX-retrained']['PCC'], flush=True)

    import retrain_deep_amp as da
    Xd = da.encode(uniq)[0]
    Xdt = torch.from_numpy(Xd[:len(uniq)]).float().to(device)
    pers, mets = [], []
    for s in SEEDS:
        m = da.CNNRegressor().to(device)
        dd = (OLD / 'deep_amp_ensemble' / f'cnn_seed{s}.pt') if (OLD / 'deep_amp_ensemble' / f'cnn_seed{s}.pt').exists() \
            else (BR / f'sota_models/deep_amp_ensemble/cnn_seed{s}.pt')
        m.load_state_dict(torch.load(dd, map_location='cpu')['model'])
        m.eval()
        with torch.no_grad():
            p = m(Xdt).squeeze(-1).cpu().numpy()
        pf, mm = per_family(attach_delta(dict(zip(uniq, p))), ORDER, base=2)
        pers.append(pf)
        mets.append(mm)
    out['metrics11']['Deep_AMP-CNN'] = stats(mets)
    out['perfam11']['Deep_AMP-CNN'] = {f: round(float(np.mean([p[f] for p in pers])), 3) for f in ORDER}
    out['perfam11']['Deep_AMP-CNN']['宏PCC'] = out['metrics11']['Deep_AMP-CNN']['PCC']['mean']
    print('[Deep_AMP-CNN]', out['metrics11']['Deep_AMP-CNN']['PCC'], flush=True)

    from transformers import BertTokenizer, GPT2Config, GPT2LMHeadModel
    sys.path.insert(0, str(ROOT / 'baseline/sota_bench/AMP-Designer'))
    import train_ampd_restricted_neg as an
    W = ROOT / 'baseline/sota_bench/models/amp_designer'
    tok = BertTokenizer(vocab_file=str(W / 'vocab.txt'))
    cfg = GPT2Config.from_pretrained(str(W / 'config.json'))
    from torch.utils.data import DataLoader
    loader = DataLoader(an.SeqDS(uniq, np.zeros(len(uniq)), tok), batch_size=64, collate_fn=an.collate)
    pers, mets = [], []
    for s in SEEDS:
        f = OLD / 'amp_designer_with_neg' / f'ampd_{s}.pt'
        if not f.exists():
            f = BR / f'sota_models/amp_designer_with_neg/ampd_{s}.pt'
        gpt = GPT2LMHeadModel.from_pretrained(str(W / 'pytorch_model.bin'), config=cfg)
        m = an.CombinedModel(gpt).to(device)
        m.load_state_dict(torch.load(f, map_location='cpu')['model'])
        m.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                preds.extend(m(xb.to(device)).cpu().tolist())
        pf, mm = per_family(attach_delta(dict(zip(uniq, preds))), ORDER, base=2)
        pers.append(pf)
        mets.append(mm)
    out['metrics11']['AMP-Designer'] = stats(mets)
    out['perfam11']['AMP-Designer'] = {f: round(float(np.mean([p[f] for p in pers])), 3) for f in ORDER}
    out['perfam11']['AMP-Designer']['宏PCC'] = out['metrics11']['AMP-Designer']['PCC']['mean']
    print('[AMP-Designer]', out['metrics11']['AMP-Designer']['PCC'], flush=True)

    # ---------- APEX original (deterministic 40-ckpt ensemble) ----------
    act = pd.read_csv(BR / 'sota_apex_original_preds_v2.csv', index_col=0).iloc[:, 0].to_dict()
    A_orig = attach_delta({k: 6.0 - v for k, v in act.items()})
    pf, mm = per_family(A_orig, ORDER, base=2)
    out['metrics11']['APEX-original'] = {m: {'mean': mm[m], 'std': None} for m in METRICS}
    out['perfam11']['APEX-original'] = pf
    print('[APEX-original]', mm, flush=True)

    # ---------- Tier-B per-family for the 4-method showdown ----------
    for label, arm in (('TIGER-pair(S4)', 'S4'), ('TIGER-single', 's4_single_n1n2')):
        pers = []
        for s in SEEDS:
            ens = eval_run(RA / arm / f'{arm}_seed{s}')
            A = ens.rename(columns={'y_true': YT, 'y_pred': '_p'})
            pf, _ = per_family(A, TIER_B, base=2)
            pers.append(pf)
        out['tierB_perfam'][label] = {f: round(float(np.mean([p[f] for p in pers])), 3) for f in TIER_B}
        print(f'[tierB {label}] done', flush=True)
    out['tierB_perfam']['APEX-retrained'] = {
        f: round(float(np.mean([per_family(attach_delta(apex_preds(OLD / 'apex_neg' / f'apexlr_{s}.pt')),
                                TIER_B, base=2)[0][f] for s in SEEDS])), 3) for f in TIER_B}
    out['tierB_perfam']['APEX-original'] = per_family(A_orig, TIER_B, base=2)

    json.dump(out, open(BR / 'report_tables.json', 'w'), ensure_ascii=False, indent=2)
    print(f'saved {BR}/report_tables.json', flush=True)


if __name__ == '__main__':
    main()

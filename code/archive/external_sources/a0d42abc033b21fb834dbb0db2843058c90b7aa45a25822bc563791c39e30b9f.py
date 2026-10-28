#!/usr/bin/env python3
"""Paper-fidelity retrain of the EvoGradient regression ensemble (Wang et al.,
Nat Microbiol 10:332-347, 2025) on the TIGER restricted 4,929-peptide set.

The previous retrain (scripts/train_evo_restricted.py, macro PCC 0.461) used
declared defaults. The paper-specific elements it missed, restored here:

  1. Two-stage recipe (Extended Data Fig. 2c): the four regression models are
     pretrained on AMP classification, then fine-tuned on MIC regression.
  2. Label convention of the released data pipeline (AMP_regression.py):
     value = log10(MIC) in uM, censored at MAX_MIC = log10(8192).
  3. Regression training set contains non-AMPs: released train.csv is
     3,748 AMPs (true MIC) + 3,686 non-AMPs all anchored at the cap.
  4. Ensemble with the released fixed weights
     {CNN .25000594, Transformer .2500046, Attention .25000825, LSTM .24998219}.

Held identical to the previous run for a clean A/B (schedule is not in the
released code and the paper Methods are paywalled - declared as such):
optimizer Adam lr 1e-3, batch 256, <=100 epochs, EarlyStopping(10) on a 10%
sequence-disjoint split, grad-clip 5.0, architectures/encoding copied verbatim
(one-hot 20, pad 100).

Leakage safety (per protocol): training universe = 4,929 restricted AMPs (the
0.7-expert training pool; no final11/LL37/APEXGO/APEXD sequences) + the fresh
UniProt reviewed non-AMP pool; the authors' own AMP-CLIP/AMP-READ corpora are
NOT used (their train/test CSVs contain 68 of our test peptides).

Output: runs_review/benchmark_restricted/sota_models/evo_strict/ +
        sota_evo_strict_preds.csv (log10 MIC uM, 208 peptides).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC/artifacts/repair_v15/external_runs/EvoGradient/seed121/workspace')
SOTA = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC/artifacts/repair_v15/external_runs/EvoGradient/seed121/workspace/baseline/sota_bench')
EVO = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC/artifacts/repair_v15/external_runs/EvoGradient/seed121/workspace/baseline/AMP-potency-prediction-EvoGradient')
sys.path.insert(0, str(EVO))
import otherModelsRegression as omr  # noqa: E402  (pickle classes resolve at load time)
from eval_tiger_style_pair_delta import AttentionNetwork  # noqa: E402

MAX_MIC = math.log10(8192)
AA = 'ACDEFGHIKLMNPQRSTVWY'
AA_IDX = {c: i for i, c in enumerate(AA)}
NUM_TOKENS = 100
# average residue masses (Da) + water, for ug/mL -> uM conversion
RES_MASS = {'A': 71.0788, 'R': 156.1875, 'N': 114.1038, 'D': 115.0886, 'C': 103.1388,
            'E': 129.1155, 'Q': 128.1307, 'G': 57.0519, 'H': 137.1411, 'I': 113.1594,
            'L': 113.1594, 'K': 128.1741, 'M': 131.1926, 'F': 147.1766, 'P': 97.1167,
            'S': 87.0782, 'T': 101.1051, 'W': 186.2132, 'Y': 163.1760, 'V': 99.1326}
WEIGHTS = {'CNN': 0.25000594, 'Transformer': 0.2500046,
           'Attention': 0.25000825, 'LSTM': 0.24998219}


def encode(seq: str):
    ids = [AA_IDX[c] for c in seq.upper() if c in AA_IDX][:NUM_TOKENS]
    onehot = np.zeros((NUM_TOKENS, 20), dtype=np.float32)
    mask = np.ones(NUM_TOKENS, dtype=np.float32)
    for i, v in enumerate(ids):
        onehot[i, v] = 1.0
    mask[len(ids):] = 0.0
    return onehot, mask


def mw(seq: str) -> float:
    return sum(RES_MASS[c] for c in seq) + 18.0152


class DS(Dataset):
    """Two-target dataset: cls (0/1) and reg (log10 MIC uM, capped)."""

    def __init__(self, seqs, cls, reg):
        self.seqs, self.cls, self.reg = seqs, cls, reg

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        x, m = encode(self.seqs[i])
        return x, m, self.cls[i], self.reg[i]


def run_stage(model, name, loader_tr, loader_va, device, *, task, epochs=100, patience=10):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, stale, best_state = float('inf'), 0, None
    for ep in range(1, epochs + 1):
        model.train()
        for x, m, ycls, yreg in loader_tr:
            x, m = x.to(device), m.to(device)
            if task == 'cls':
                y = ycls.to(device)
                crit = nn.BCEWithLogitsLoss()
            else:
                y = yreg.to(device)
                crit = nn.MSELoss()
            opt.zero_grad()
            out = model(x, m) if name == 'Attention' else model(x)
            loss = crit(out.squeeze(-1).float(), y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if not torch.isfinite(loss):
                return model, float('inf'), ep
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for x, m, ycls, yreg in loader_va:
                x, m = x.to(device), m.to(device)
                y = (ycls if task == 'cls' else yreg).to(device)
                out = model(x, m) if name == 'Attention' else model(x)
                crit = nn.BCEWithLogitsLoss() if task == 'cls' else nn.MSELoss()
                tot += float(crit(out.squeeze(-1).float(), y.float())) * len(y)
                n += len(y)
        vl = tot / max(n, 1)
        if math.isfinite(vl) and vl < best - 1e-4:
            best, stale = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best, ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--no-neg', action='store_true',
                    help='[noneg, plan A] positives-only training: negatives '
                         'dropped AND the cls pre-training stage SKIPPED (its '
                         '0/1 labels require non-AMPs); regression-only '
                         'end-to-end, registered as the strict-noneg variant')
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # --- data: 4,929 restricted AMPs (-> log10 uM, capped) + UniProt negatives at cap
    tr = pd.read_csv(ROOT / 'runs_review/benchmark_restricted/train_restricted4929_tiger.csv')
    pos = tr.groupby('sequence', as_index=False).MIC_Escherichia_coli.mean()
    pos = pos[pos.sequence.str.len() <= NUM_TOKENS]
    log10_um = np.log10(pos.MIC_Escherichia_coli.values * 1000.0 / np.array([mw(s) for s in pos.sequence]))
    log10_um = np.clip(log10_um, None, MAX_MIC)
    if args.no_neg:                 # [noneg, plan A] positives only
        assert len(pos) == 4929, len(pos)
        seqs = pos.sequence.tolist()
        cls = np.ones(len(pos)).astype(np.float32)   # unused: cls stage skipped
        reg = log10_um.astype(np.float32)
        neg = pos.iloc[:0]           # empty frame for the report/log lines
    else:
        neg = pd.read_csv(SOTA / 'data_restricted/uniprot_negatives.csv')
        neg = neg[neg.sequence.str.len() <= NUM_TOKENS].reset_index(drop=True)
        seqs = pos.sequence.tolist() + neg.sequence.tolist()
        cls = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.float32)
        reg = np.concatenate([log10_um, np.full(len(neg), MAX_MIC)]).astype(np.float32)

    # --- leakage audit: no final11 sequence may appear anywhere in training
    final11 = pd.read_csv(ROOT / 'runs_review/final11_peptides.csv')
    test_seqs = set(final11.sequence)
    assert not (test_seqs & set(seqs)), 'LEAKAGE: test peptide in training pool'
    print(f'[leakage] OK: 0/{len(test_seqs)} test peptides in train ({len(pos)} AMP + {len(neg)} neg)', flush=True)
    print(f'regression labels (uM): AMP range [{log10_um.min():.2f},{log10_um.max():.2f}] '
          f'(cap {MAX_MIC:.3f}, {(log10_um >= MAX_MIC - 1e-9).sum()} at cap)', flush=True)

    df = pd.DataFrame({'Sequence': seqs})
    uniq = df.drop_duplicates('Sequence').sample(frac=1.0, random_state=args.seed)
    n_val = int(0.1 * len(uniq))
    val_seqs = set(uniq.Sequence.head(n_val))
    is_val = df.Sequence.isin(val_seqs).to_numpy()
    tr_ds = DS(list(np.array(seqs)[~is_val]), cls[~is_val], reg[~is_val])
    va_ds = DS(list(np.array(seqs)[is_val]), cls[is_val], reg[is_val])
    loader_tr = DataLoader(tr_ds, batch_size=256, shuffle=True)
    loader_va = DataLoader(va_ds, batch_size=256)
    print(f'train {len(tr_ds)} / val {len(va_ds)} rows', flush=True)

    outdir = ROOT / ('runs_review/benchmark_restricted/sota_models/evo_strict_noneg'
                     if args.no_neg else
                     'runs_review/benchmark_restricted/sota_models/evo_strict')
    outdir.mkdir(parents=True, exist_ok=True)

    pep = pd.read_csv(ROOT / 'runs_review/final11_peptides.csv')
    test_uniq = list(dict.fromkeys(pep.sequence))
    Xq = torch.from_numpy(np.stack([encode(s)[0] for s in test_uniq]))
    Mq = torch.from_numpy(np.stack([encode(s)[1] for s in test_uniq]))

    preds, report = {}, {}
    for name, model in {'CNN': omr.CNN(), 'Transformer': omr.Transformer(),
                        'Attention': AttentionNetwork(), 'LSTM': omr.lstm_att()}.items():
        model.to(device)
        if args.no_neg:              # [noneg, plan A] skip the cls pre-training
            vl1, ep1 = float('nan'), 0
            print(f'[{name}] stage1 cls SKIPPED [noneg plan A]', flush=True)
        else:
            model, vl1, ep1 = run_stage(model, name, loader_tr, loader_va, device, task='cls')
            print(f'[{name}] stage1 cls: val_loss={vl1:.4f} stopped@{ep1}', flush=True)
        model, vl2, ep2 = run_stage(model, name, loader_tr, loader_va, device, task='reg')
        print(f'[{name}] stage2 reg: val_MSE={vl2:.4f} stopped@{ep2}', flush=True)
        torch.save({'model': model.state_dict(), 'arch': name, 'seed': args.seed},
                   outdir / f'{name}_seed{args.seed}.pt')
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(test_uniq), 256):
                x, m = Xq[i:i + 256].to(device), Mq[i:i + 256].to(device)
                o = model(x, m) if name == 'Attention' else model(x)
                outs.append(o.squeeze(-1).cpu().numpy())
        preds[name] = np.concatenate(outs)
        report[name] = {'stage1_val': round(vl1, 4), 'stage2_val_MSE': round(vl2, 4)}
        print(f'[{name}] test log10MIC(uM) std={preds[name].std():.3f}', flush=True)

    ens = sum(WEIGHTS[k] * preds[k] for k in WEIGHTS)
    out_csv = (ROOT / 'runs_review/benchmark_restricted/'
               f'sota_evo_strict_noneg_s{args.seed}_preds.csv' if args.no_neg else
               ROOT / 'runs_review/benchmark_restricted/sota_evo_strict_preds.csv')
    pd.Series(dict(zip(test_uniq, ens.tolist())), name='log10mic_um').to_csv(out_csv)
    # also the per-seed layout the evaluator consumes (+neg convention)
    pd.Series(dict(zip(test_uniq, ens.tolist())), name='log10mic_um').to_csv(
        outdir / f'preds_seed{args.seed}.csv')
    (outdir / f'train_report_seed{args.seed}.json').write_text(json.dumps(
        {'seed': args.seed, 'n_amp': int(len(pos)), 'n_neg': int(len(neg)),
         'no_neg': bool(args.no_neg), 'results': report}, indent=2))
    print(f'ensemble saved (fixed released weights); test std={ens.std():.3f}', flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Recover the collapsed AMPLiT regression variant (experiment 12 follow-up).

Diagnosis of the v1 collapse (macro PCC 0.1009, per-peptide std 0.088): the
regression head starts near zero against labels spanning [-1.33, 4.27] with
62% of mass pinned at the 4.0 anchor; a uniform lr 1e-4 for their 21-epoch
classification schedule lets the anchor-dominated MSE squash outputs toward a
near-constant. The classifier trunk itself is fine (0.327 as a classifier).

Fix ladder (arch/featurization/training set all unchanged; feature cache reused):
  A: z-scored labels (fit on the train split, inverted at inference)
     + 60 epochs with ES(7) on val MSE
     + param groups: fresh regression head (head+cbam) lr 1e-3, trunk lr 1e-4
  B: A + SmoothL1 (robust to the anchor mass)
  C: B recipe + warm start from the trained CLASSIFIER trunk
     (sota_models/amplit/amplit_best.pt, strict=False — branches transfer,
     fresh regression head)

Usage: python retrain_amplit_regressor_fix.py --variant {A,B,C} --gpu 0
Output: sota_amplit_regr_v{A,B,C}_preds.csv (log10MIC, 208 test peptides)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC')
TIGER = ROOT
sys.path.insert(0, str(TIGER))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from retrain_amplit import EPOCHS as _, LR, BATCH, WD, featurize  # noqa: E402
from retrain_amplit_regressor import AMPLiTRegressor, OUT, CKPT_DIR, NEG_LOG10  # noqa: E402

FIX_EPOCHS, PATIENCE = 60, 7
HEAD_LR, TRUNK_LR = 1e-3, 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=['A', 'B', 'C'])
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--gpu', default='0')
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'device={device} variant={args.variant} seed={args.seed}', flush=True)

    tr = pd.read_csv(OUT / 'train_restricted4929_tiger.csv')
    pos = tr.groupby('sequence', as_index=False).MIC_Escherichia_coli.mean()
    pos_seqs = pos.sequence.tolist()
    pos_y = np.log10(pos.MIC_Escherichia_coli.values.astype(float))
    neg = pd.read_csv(ROOT / 'baseline/sota_bench/data_restricted/uniprot_negatives.csv')
    seqs = pos_seqs + neg.sequence.tolist()
    y = np.concatenate([pos_y, np.full(len(neg), NEG_LOG10)]).astype(np.float32)

    cache = Path('/tmp/amplit_feats.npz')
    if cache.exists():
        z = np.load(cache)
        assert z['X1'].shape[0] == len(seqs), 'cache/seq mismatch'
        X1, X2, X3 = z['X1'], z['X2'], z['X3']
        print('reused /tmp/amplit_feats.npz', flush=True)
    else:
        X1, X2, X3 = featurize(seqs)
        np.savez(cache, X1=X1, X2=X2, X3=X3)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(seqs))
    n_va = int(0.1 * len(seqs))
    va_i, tr_i = idx[:n_va], idx[n_va:]

    # A: z-score labels with TRAIN-split statistics only
    mu, sd = float(y[tr_i].mean()), float(y[tr_i].std())
    yz = (y - mu) / sd

    model = AMPLiTRegressor().to(device)
    if args.variant == 'C':
        cls_ckpt = OUT / 'sota_models/amplit/amplit_best.pt'
        state = torch.load(cls_ckpt, map_location='cpu')['model']
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f'[C] warm start from classifier trunk: {len(unexpected)} head keys '
              f'skipped, {len(missing)} fresh keys', flush=True)
    # build the lazy head so the optimizer can see it (param groups need it)
    with torch.no_grad():
        model(torch.zeros(1, 50, 20, device=device),
              torch.zeros(1, 50, 100, device=device),
              torch.zeros(1, 50, 31, device=device))
    head_params = list(model.head.parameters()) + list(model.cbam.parameters())
    head_ids = {id(p) for p in head_params}
    trunk_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([
        {'params': trunk_params, 'lr': TRUNK_LR},
        {'params': head_params, 'lr': HEAD_LR},
    ], lr=LR, weight_decay=WD)
    lossf = nn.SmoothL1Loss() if args.variant == 'B' else nn.MSELoss()

    def batchify(ids):
        return (torch.from_numpy(X1[ids]), torch.from_numpy(X2[ids]),
                torch.from_numpy(X3[ids]), torch.from_numpy(yz[ids]))

    Xv = batchify(va_i)
    best_val, best_state, stale = float('inf'), None, 0
    for ep in range(1, FIX_EPOCHS + 1):
        model.train()
        perm = np.random.permutation(len(tr_i))
        losses = []
        for i in range(0, len(tr_i), BATCH):
            b = tr_i[perm[i:i + BATCH]]
            xb1, xb2, xb3, yb = batchify(b)
            xb1, xb2, xb3, yb = (xb1.to(device), xb2.to(device),
                                 xb3.to(device), yb.to(device))
            opt.zero_grad()
            loss = lossf(model(xb1, xb2, xb3), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            vl = float(lossf(model(*[t.to(device) for t in Xv[:3]]),
                             Xv[3].to(device)).cpu())
        print(f'ep{ep} train_loss={np.mean(losses):.4f} val_loss={vl:.4f}', flush=True)
        if vl < best_val - 1e-4:
            best_val, stale = vl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    ck = CKPT_DIR.parent / 'amplit_regr_fix'
    ck.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'variant': args.variant,
                'zscore': [mu, sd]}, ck / f'best_{args.variant}.pt')

    pep = pd.read_csv(TIGER / 'runs_review/final11_peptides.csv')
    uniq = list(dict.fromkeys(pep.sequence))
    Q1, Q2, Q3 = featurize(uniq)
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(uniq), 256):
            preds.append(model(torch.from_numpy(Q1[i:i + 256]).to(device),
                               torch.from_numpy(Q2[i:i + 256]).to(device),
                               torch.from_numpy(Q3[i:i + 256]).to(device)).cpu().numpy())
    preds = np.concatenate(preds) * sd + mu  # invert z-score
    suffix = '' if args.seed == 123 else f'_s{args.seed}'
    pd.Series(dict(zip(uniq, preds.tolist())), name='log10mic').to_csv(
        OUT / f'sota_amplit_regr_v{args.variant}{suffix}_preds.csv')
    print(f'saved sota_amplit_regr_v{args.variant}_preds.csv; log10MIC '
          f'mean={preds.mean():.3f} std={preds.std():.3f} '
          f'range [{preds.min():.3f},{preds.max():.3f}]', flush=True)


if __name__ == '__main__':
    main()

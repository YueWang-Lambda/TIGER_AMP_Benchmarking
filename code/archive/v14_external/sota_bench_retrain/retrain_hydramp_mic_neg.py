#!/usr/bin/env python3
"""Retrain HydrAMP-MIC (Szymczak et al., Nat Commun 14:1453, 2023) on the
UNIFIED negative universe (experiment 12 supplement, user-requested).

Identical to retrain_hydramp_mic.py in every respect (architecture per the
released M_MIC classifier; labels binarized at MIC <= 10^1.5 ug/mL; Adam 1e-3,
BCE, 10-fold StratifiedKFold(shuffle, seed 36), positives oversampled x5,
EarlyStopping patience 7, 10-fold ensemble) — except the negative universe:

  original: UniProt negatives restricted to <=25 aa (2,936; the paper's
            length rule for its generator-serving models)
  here    : the FULL shared pool (7,962) — the experiment-12 unified protocol
            (the MAX_LENGTH 25->50 adaptation of experiment 6 already
            accommodates longer peptides; disclosed)

Reuses encode/fit_one/pad_batch and all constants from the original arm via
import; the training loop below is a verbatim replication of its main().

Output: runs_review/benchmark_restricted/sota_hydramp_mic_neg_preds.csv
(P(active), 208 external test peptides).
"""
import sys
from pathlib import Path

ROOT = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC')
TIGER = ROOT
sys.path.insert(0, str(TIGER))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

import retrain_hydramp_mic as base  # noqa: E402  (original arm, verbatim pieces)

OUT = TIGER / 'runs_review/benchmark_restricted'
CKPT_DIR = OUT / 'sota_models/hydramp_mic_neg'


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    SEED = base.SEED
    np.random.seed(SEED)
    print(f'device={device}, seed={SEED} — UNIFIED FULL POOL variant', flush=True)

    tr = pd.read_csv(base.TRAIN_CSV)
    pos = tr.groupby('sequence', as_index=False).MIC_Escherichia_coli.mean()
    neg = pd.read_csv(base.NEG_CSV)  # full 7,962 — NO <=25 aa filter (disclosed)

    pos_rows = base.encode(pos.sequence.tolist())
    keep = [i for i, r in enumerate(pos_rows) if r is not None]
    mic_x = [pos_rows[i] for i in keep]
    mic_y = (np.log10(pos.MIC_Escherichia_coli.values.astype(float))
             < base.THRESH_LOG10).astype(int)[keep]
    neg_x = [r for r in base.encode(neg.sequence.tolist()) if r is not None]
    print(f'MIC set: {len(mic_x)} AMPs ({int(mic_y.sum())} active <=32 u/mL), '
          f'{len(neg_x)} UniProt negatives FULL POOL', flush=True)

    x_all = mic_x + neg_x
    y_all = np.concatenate([mic_y, np.zeros(len(neg_x), dtype=int)])

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    pep = pd.read_csv(base.PEP_CSV)
    uniq = list(dict.fromkeys(pep.sequence))
    q_rows = base.encode(uniq)
    kept = [(s, r) for s, r in zip(uniq, q_rows) if r is not None]
    print(f'test peptides kept: {len(kept)}/{len(uniq)}', flush=True)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    fold_preds, oof_auc = [], []
    for f, (tri, vai) in enumerate(cv.split(np.zeros(len(x_all)), y_all), 1):
        x_tr = [x_all[i] for i in tri]
        y_tr = y_all[tri]
        # positives oversampled x5 (notebook cell 16, verbatim)
        pidx = [i for i, yy in enumerate(y_tr) if yy == 1]
        x_tr = x_tr + [x_tr[i] for i in pidx] * 5
        y_tr = np.concatenate([y_tr, np.ones(len(pidx) * 5, dtype=int)])
        model, vl, ep = base.fit_one(x_tr, y_tr, [x_all[i] for i in vai], y_all[vai], device)
        torch.save({'model': model.state_dict(), 'fold': f}, CKPT_DIR / f'fold{f}.pt')
        model.eval()
        with torch.no_grad():
            ps = []
            for i in range(0, len(kept), 4096):
                xb = base.pad_batch([r for _, r in kept[i:i + 4096]]).to(device)
                ps.append(torch.sigmoid(model(xb)).cpu().numpy())
            fold_preds.append(np.concatenate(ps))
            va_x = [x_all[i] for i in vai]
            vp = np.concatenate([
                torch.sigmoid(model(base.pad_batch(va_x[i:i + 4096]).to(device))).cpu().numpy()
                for i in range(0, len(va_x), 4096)])
        oof_auc.append(roc_auc_score(y_all[vai], vp))
        print(f'[fold {f}] stopped ep{ep} val_loss={vl:.4f} val_AUC={oof_auc[-1]:.4f}', flush=True)

    probs = np.mean(fold_preds, axis=0)
    seqs = [s for s, _ in kept]
    pd.Series(dict(zip(seqs, probs.tolist())), name='p_active').to_csv(
        OUT / 'sota_hydramp_mic_neg_preds.csv')
    print(f'saved sota_hydramp_mic_neg_preds.csv; P(active) mean={probs.mean():.3f} '
          f'std={probs.std():.3f} OOF-AUC={np.mean(oof_auc):.4f}', flush=True)


if __name__ == '__main__':
    main()

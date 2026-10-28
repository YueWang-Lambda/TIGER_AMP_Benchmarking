#!/usr/bin/env python3
"""Retrain M3-CAD's multimodal identification model as an MIC regressor
(experiment 12 supplement; ours adaptation — the paper's multitask heads are
multilabel classification, we regress log10MIC).

Faithful to the official code (github.com/haifangong/M3CAD):
  model   network.MMPeptide(v_encoder='resnet26', q_encoder='mlp', classes=1)
          — ResNet3D 3D-conv trunk with FiLM-style sequence gating on the
          (3,64,64,64) voxel grid + SEQ-mlp sequence encoder (256-d) +
          Linear(2048+256 -> 1)
  data    voxels from build_m3cad_voxels.py (official pdb_parser replication);
          sequence tokens = their AMAs indices (A1..H19, G20, X21) padded to 50
  aug     train-time RandomHorizontalFlip(0.5)+RandomVerticalFlip(0.5)+
          RandomRotation(90) on the voxel (their transforms.Compose, verbatim)
  optim   AdamW lr 2e-3, weight_decay 5e-4 (paper), batch 16, 50 epochs;
          regression addition (disclosed): best-val checkpoint on a fixed 15%
          split (rng 42) — their loop kept last-epoch weights for classifiers
Universe: 4,929 restricted positives + 7,860 structure-bearing negatives
(pseudo log10MIC = 4.0, the shared cap-anchoring convention).

Output: runs_review/benchmark_restricted/sota_m3cad_mic_preds.csv (log10MIC,
208 external test peptides).

Usage: python retrain_m3cad_mic.py --gpu 0
"""
import argparse
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'm3cad_official' / 'identification'))
sys.path.insert(0, str(HERE))

# The host's wandb install is broken (proto mismatch) and gets pulled in by
# timm via the official swinunetr import chain; the model math never uses it,
# so stub it out before importing the official identification module.
if 'wandb' not in sys.modules:
    _w = types.ModuleType('wandb')
    _w.__version__ = '0.0.0-stub'
    sys.modules['wandb'] = _w

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import transforms  # noqa: E402

from network import MMPeptide  # noqa: E402  (official identification module)

TIGER = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC')
CACHE = HERE / 'm3cad_cache'
OUT = TIGER / 'runs_review/benchmark_restricted'
MAX_LEN = 50
# their AMAs token dict (demo.py, verbatim)
AMAS = {'G': 20, 'A': 1, 'V': 2, 'L': 3, 'I': 4, 'P': 5, 'F': 6, 'Y': 7, 'W': 8,
        'S': 9, 'T': 10, 'C': 11, 'M': 12, 'N': 13, 'Q': 14, 'D': 15, 'E': 16,
        'K': 17, 'R': 18, 'H': 19, 'X': 21}


def tokens(seq: str) -> np.ndarray:
    code = [AMAS.get(c, 21) for c in seq[:MAX_LEN]]
    return np.array(code + [0] * max(0, MAX_LEN - len(code)), dtype=np.int64)


class VoxelDS(Dataset):
    def __init__(self, vox, meta, train, aug=True, yz=None):
        self.vox, self.meta, self.train = vox, meta, train
        self.aug = aug
        self.yz = yz  # optional (mu, sd) for z-scored labels
        self.im_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90)])

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        v = torch.from_numpy(np.asarray(self.vox[i])).float()
        if self.train and self.aug:
            v = self.im_aug(v)
        y = self.meta.y_log10mic[i]
        if self.yz is not None:
            y = (y - self.yz[0]) / self.yz[1]
        return v, torch.from_numpy(tokens(self.meta.sequence[i])), \
            torch.tensor(y, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--no-aug', action='store_true',
                    help='v2: drop the flip/rotation augmentation (it mirrors '
                         'chirality — label-invariant for their classification '
                         'task, harmful for regression)')
    ap.add_argument('--zscore', action='store_true',
                    help='v2: z-score labels with train-split statistics')
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    vox = np.load(CACHE / 'vox_train.npy', mmap_mode='r')
    meta = pd.read_csv(CACHE / 'meta_train.csv')
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(meta))
    n_val = int(round(0.15 * len(meta)))
    va_i, tr_i = perm[:n_val], perm[n_val:]
    tr_meta, va_meta = meta.iloc[tr_i].reset_index(drop=True), meta.iloc[va_i].reset_index(drop=True)
    yz = (float(meta.y_log10mic.iloc[tr_i].mean()), float(meta.y_log10mic.iloc[tr_i].std())) \
        if args.zscore else None
    tr_loader = DataLoader(VoxelDS(vox, tr_meta, train=True, aug=not args.no_aug, yz=yz),
                           batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    va_loader = DataLoader(VoxelDS(vox, va_meta, train=False, yz=yz), batch_size=64,
                           num_workers=2, pin_memory=True)
    print(f'universe={len(meta)} train={len(tr_meta)} val={len(va_meta)} '
          f'y=[{meta.y_log10mic.min():.2f},{meta.y_log10mic.max():.2f}]', flush=True)

    model = MMPeptide(v_encoder='resnet26', q_encoder='mlp', classes=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=5e-4)
    lossf = nn.MSELoss()

    best_val, best_state = float('inf'), None
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for v, s, y in tr_loader:
            v, s, y = v.to(device), s.to(device), y.to(device)
            opt.zero_grad()
            pred, _ = model((v, s))
            loss = lossf(pred.squeeze(-1), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        errs = []
        with torch.no_grad():
            for v, s, y in va_loader:
                pred, _ = model((v.to(device), s.to(device)))
                errs.extend((pred.squeeze(-1).cpu() - y).abs().tolist())
        mae = float(np.mean(errs))
        if mae < best_val:
            best_val, best_state = mae, {k: v2.cpu().clone() for k, v2 in model.state_dict().items()}
        print(f'ep{ep} train_loss={np.mean(losses):.4f} val_MAE(log10)={mae:.4f}', flush=True)
    model.load_state_dict(best_state)
    model.to(device)
    ck = OUT / 'sota_models/m3cad_mic'
    ck.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict()}, ck / 'best.pt')

    vox_t = np.load(CACHE / 'vox_test.npy', mmap_mode='r')
    meta_t = pd.read_csv(CACHE / 'meta_test.csv')
    q_loader = DataLoader(VoxelDS(vox_t, meta_t, train=False), batch_size=64)
    preds = []
    model.eval()
    with torch.no_grad():
        for v, s, _ in q_loader:
            pred, _ = model((v.to(device), s.to(device)))
            preds.extend(pred.squeeze(-1).cpu().tolist())
    preds = np.asarray(preds)
    if yz is not None:
        preds = preds * yz[1] + yz[0]  # invert z-score
    suffix = '_v2' if (args.no_aug or args.zscore) else ''
    suffix += '' if args.seed == 123 else f'_s{args.seed}'
    pd.Series(dict(zip(meta_t.sequence, preds.tolist())), name='log10mic').to_csv(
        OUT / f'sota_m3cad_mic{suffix}_preds.csv')
    print(f'saved sota_m3cad_mic_preds.csv; log10MIC mean={preds.mean():.3f} '
          f'std={preds.std():.3f} range [{preds.min():.3f},{preds.max():.3f}]', flush=True)


if __name__ == '__main__':
    main()

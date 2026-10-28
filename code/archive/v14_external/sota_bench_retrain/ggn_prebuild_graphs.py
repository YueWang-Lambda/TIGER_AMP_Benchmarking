#!/usr/bin/env python3
"""Prebuild the GGN graph cache (ggn_work/graph_cache.pt).

Iterates ProteinGraphDataset over BOTH the training corpus and the external
208 once, forcing the [v2-9 patch] per-sequence cache fills in myData, then
saves one merged {seq: graph-tensors} dict. Training runs then take the
fast path (no PDB parsing / knn rebuild per __getitem__).

Usage: python ggn_prebuild_graphs.py   (ccseg env, CPU-only ok)
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'GGN-ABPMIC'))
os.environ['GGN_SEEDS'] = '1'

import torch  # noqa: E402
from myData import ProteinGraphDataset  # noqa: E402

WORK = HERE / 'ggn_work'
OUT = WORK / 'graph_cache.pt'


class A: pass


def ds_of(csv):
    a = A()
    a.train_csv = str(WORK / csv)
    a.val_csv = str(WORK / csv)
    a.pdb_dir = str(WORK / 'pdb')
    a.prottrans_dir = str(WORK / 'prottrans')
    a.dssp_dir = str(WORK / 'dssp')
    a.device = 'cpu'
    return ProteinGraphDataset(a, mode='train')


merged = {}
for csv in ('train.csv', 'ext.csv'):
    ds = ds_of(csv)
    print(f'{csv}: {len(ds)} samples')
    for i in range(len(ds)):
        _ = ds[i]                      # forces cache fill
        if i % 2000 == 0:
            print(f'  {i}/{len(ds)} (cache {len(ds._gcache)})', flush=True)
    merged.update(ds._gcache)
torch.save(merged, OUT)
print(f'saved {OUT}: {len(merged)} graphs')

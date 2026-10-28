#!/usr/bin/env python3
"""GGN-ABPMIC E. coli arm (Brief Bioinform 2026, 10.1093/bib/bbag170)
retrained under the unified "+neg" protocol (v2-9). Runs in ccseg.

Fidelity notes:
- Architecture/training verbatim from the repo (GVP-style geometric graph
  network over CA kNN graphs; node_s = 6 dihedral + 1024 ProtT5 + 13 DSSP =
  1043d; AdaptiveHybridLoss; AdamW wd 0.01; 100 epochs, early stop 30, fp16,
  BatchSampler by node count; 80:20 split random_state=seed; best-val ckpt).
  Upstream hardcodes 10 random splits at seed 42+i -> env-patched to a single
  protocol seed per run (GGN_SEEDS=1 GGN_SEED_BASE=<seed>); our 10 protocol
  seeds replace their split-ensemble (single-model-per-seed, like every other
  method here). Vendor patches marked "[v2-9 patch]".
- Structure source: this project's Rosetta-relaxed library (user-approved;
  upstream used ESMFold) — {seq}.pdb under ggn_work/pdb.
- ProtT5-XL-U50 per-residue embeddings and mkdssp features precomputed by
  ggn_prepare.py in the repo's .npy formats. The dataset's silent zero-fill
  fallback is disabled by ggn_prepare's no-structure drop + no-NaN assertion.
- Label domain: native value = log10(uM); unified protocol feeds log10(ug/mL).
- 39/13,086 universe sequences without Rosetta structures are excluded
  (logged by ggn_prepare.py). No repo checkpoints exist/loaded.
- [v2-13 patch] torch.set_num_threads(4) below is MANDATORY for parallel
  seed batches: uncapped, each process spawns ~nproc OMP threads, so a 5+5
  dual-GPU launch put 10 procs x 84 threads on 32 cores (load ~102, GPU util
  3-27% starved, 4.2 min/epoch); capped -> 0.6 min/epoch and GPUs fed.
  Do NOT remove when retraining multiple seeds concurrently. Launcher-level
  OMP_NUM_THREADS=4 works too; this in-code cap makes it unconditional.

Usage: python retrain_ggn_neg.py --seed 120 [--predict-only]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# [v2-13 patch] thread cap — required for concurrent multi-seed launches
# (see fidelity note above); harmless for single-process runs.
torch.set_num_threads(4)

HERE = Path(__file__).resolve().parent
GGN = HERE / 'GGN-ABPMIC'
WORK = HERE / 'ggn_work'
V2 = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC')
OUT = V2 / 'runs_review/benchmark_restricted/sota_models/ggn_neg'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--predict-only', action='store_true')
    args = ap.parse_args()
    sys.path.insert(0, str(GGN))
    os.environ['GGN_SEEDS'] = '1'
    os.environ['GGN_SEED_BASE'] = str(args.seed)
    # [v2-9 patch] prebuilt graph cache (built by ggn_prebuild_graphs.py)
    os.environ['GGN_GRAPH_CACHE'] = str(WORK / 'graph_cache.pt')
    import config as ggn_config
    gcn = ggn_config.config
    gcn.train_csv = str(WORK / 'train.csv')
    gcn.pdb_dir = str(WORK / 'pdb')
    gcn.prottrans_dir = str(WORK / 'prottrans')
    gcn.dssp_dir = str(WORK / 'dssp')
    gcn.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gcn.save_dir = str(OUT / f's{args.seed}')
    os.makedirs(gcn.save_dir, exist_ok=True)

    if not args.predict_only:
        import train as ggn_train
        ggn_train.train_kfold(gcn)

    # ---- prediction on the external 208 ----
    from model import AMPRegressor
    from myData import ProteinGraphDataset
    from torch_geometric.loader import DataLoader
    ext = pd.read_csv(WORK / 'ext.csv')
    ds_args = type('A', (), {'train_csv': str(WORK / 'ext.csv'),
                             'val_csv': str(WORK / 'ext.csv'),   # mode='test' reads val_csv
                             'pdb_dir': gcn.pdb_dir,
                             'prottrans_dir': gcn.prottrans_dir,
                             'dssp_dir': gcn.dssp_dir,
                             'device': gcn.device})()
    ds = ProteinGraphDataset(ds_args, mode='test')
    model = AMPRegressor(device=gcn.device).to(gcn.device)
    ck = torch.load(Path(gcn.save_dir) / 'random_split1_best.pth',
                    map_location=gcn.device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    preds = []
    loader = DataLoader(ds, batch_size=16)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(gcn.device)
            out, _ = model(batch)          # AMPRegressor returns (pred, embedding)
            preds.append(out.squeeze(-1).cpu().numpy())
    pe = np.concatenate(preds)
    o = pd.DataFrame({'log10mic': pe})
    o.index = ext.sequence.values
    o.index.name = 'sequence'
    o.to_csv(OUT / f'sota_GGN-ABPMIC+neg_s{args.seed}_preds.csv')
    print(f'seed {args.seed}: preds range [{pe.min():.2f},{pe.max():.2f}] -> saved')


if __name__ == '__main__':
    main()

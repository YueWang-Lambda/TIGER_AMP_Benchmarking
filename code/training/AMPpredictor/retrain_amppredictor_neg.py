#!/usr/bin/env python3
"""AMPredictor (eLife 2025, 10.7554/eLife.97330.3) retrained under the unified
"+neg" protocol (v2-9). Runs in ccseg (torch 2.2 + PyG 2.5 + rdkit).

Fidelity notes (from repo source):
- Frozen ESM-1b layer-33 per-token features + ESM contact head maps, cached
  once in emb_cache/esm1b_tok/{seq}.pt (upstream preprocess.py; its batch
  `break` bug that drops half the sequences is bypassed by our extractor).
- Graph: contact>=0.5 edges + self-loops; BOTH feature rows and the contact
  map are zero-padded to 65 then eye-added (upstream embedding_65/contact_65
  semantics: padded nodes carry self-loops and enter global_mean_pool).
- Morgan fingerprint per residue from the repo's Fmoc-AA smiles_file.json
  (radius 3, nBits 2048), padded to 65x2048, mean(axis=1) -> 65d.
- Model GNNPredictor verbatim: GCNConv 1280->1280->640->320, global_mean_pool,
  FC 320->1024->128; concat with fp(65) -> 193->1024->512->1.
- Recipe per train.py: batch 128, Adam lr 1e-3, MSE, 100 epochs, best-val-MSE
  checkpoint; upstream has NO seeds -> we add torch/np seeds; upstream fixed
  80/10/10 split -> our protocol 80/20 random_state=seed.
- No species column in the method (generic AMP model); our corpus is E. coli.
- Released checkpoints under ./models never loaded.
- Label domain: native logMIC (log10 uM per repo data); unified protocol feeds
  log10(ug/mL) directly.

Usage:
  once:  python retrain_amppredictor_neg.py --fingerprint
  xseed: python retrain_amppredictor_neg.py --seed 120
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader as PyGLoader
from torch_geometric.nn import GCNConv, global_mean_pool

HERE = Path(__file__).resolve().parent
REPO = HERE / 'AMPredictor'
V2 = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC/artifacts/repair_v15/external_runs/AMPpredictor/seed123/workspace')
CACHE = HERE / 'emb_cache'
OUT = V2 / 'runs_review/benchmark_restricted/sota_models/amppredictor_neg'
OUT_NONEG = V2 / 'runs_review/benchmark_restricted/sota_models/amppredictor_noneg'
SEQ_MAX = 65


class GNNPredictor(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=1280, output_dim=128, dropout=0.1):
        super().__init__()
        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro_conv2 = GCNConv(num_features_pro, 640)
        self.pro_conv3 = GCNConv(640, 320)
        self.pro_fc_g1 = nn.Linear(320, 1024)
        self.pro_fc_g2 = nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(193, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, n_output)

    def forward(self, data_pro, data_fp):
        xf = data_fp.float()
        xt = relu(self.pro_conv1(data_pro.x, data_pro.edge_index))
        xt = relu(self.pro_conv2(xt, data_pro.edge_index))
        xt = relu(self.pro_conv3(xt, data_pro.edge_index))
        xt = global_mean_pool(xt, data_pro.batch)
        xt = relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)
        xc = torch.cat((xf, xt), 1)
        xc = relu(self.fc1(xc))
        xc = self.dropout(xc)
        xc = relu(self.fc2(xc))
        xc = self.dropout(xc)
        return self.out(xc)


def corpus(no_neg=False):
    tr = pd.read_csv(V2 / 'runs_review/benchmark_restricted/train_restricted4929_tiger.csv')
    pos = tr.groupby('sequence', as_index=False).MIC_Escherichia_coli.mean()
    if no_neg:                      # [noneg campaign] positives-only training
        assert len(pos) == 4929, len(pos)
        return pos
    neg = pd.read_csv(HERE / 'data_restricted/uniprot_negatives.csv')[['sequence']]
    neg['MIC_Escherichia_coli'] = 10000.0
    return pd.concat([pos, neg], ignore_index=True)


def ext_table():
    return pd.read_csv(V2 / 'runs_review/final11_peptides.csv')[
        ['sequence', 'MIC_E_coli_ug_mL']].dropna()


def run_fingerprint():
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    smiles = json.load(open(REPO / 'data/smiles_file.json'))
    lookup = {}
    for k, v in smiles.items():
        lookup[k] = np.array(rdMolDescriptors.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(v), radius=3, nBits=2048))
    lookup[' '] = np.zeros(2048, dtype=int)
    lookup['0'] = np.zeros(2048, dtype=int)
    for tag, df in (('corpus', corpus()), ('ext', ext_table())):
        out = {}
        for s in df.sequence.astype(str).str.upper().str.strip():
            if s in out:
                continue
            fp = np.asarray([lookup[c] for c in s])
            fp = np.concatenate([fp, np.zeros((SEQ_MAX - len(s), 2048))], 0)
            out[s] = fp.mean(axis=1).astype(np.float32)
        np.savez(CACHE / f'amppredictor_fp_{tag}.npz', **out)
        print(f'fingerprint[{tag}]:', len(out))


def graph_of(seq):
    d = torch.load(CACHE / 'esm1b_tok' / f'{seq}.pt', map_location='cpu')
    feat = d['feature'].float()
    L = feat.shape[0]
    x = torch.zeros(SEQ_MAX, 1280)
    x[:L] = feat                                    # pad to 65 (upstream)
    c = torch.zeros(SEQ_MAX, SEQ_MAX)
    c[:L, :L] = d['contacts'].float()
    c += torch.eye(SEQ_MAX)                          # +eye over ALL 65 nodes
    ei = torch.nonzero(c >= 0.5).t().contiguous()
    return Data(x=x, edge_index=ei)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fingerprint', action='store_true')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--no-neg', action='store_true',
                    help='[noneg] train on the 4,929 positives only (the shared graph '
                         'cache is prefix-filtered: it was built in pos-then-neg order)')
    args = ap.parse_args()
    if args.fingerprint:
        run_fingerprint()
        return
    assert args.seed is not None
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    df = corpus(args.no_neg)
    y = np.log10(df.MIC_Escherichia_coli.values.astype(float))
    fp = np.load(CACHE / 'amppredictor_fp_corpus.npz')
    # graph building is seed-independent -> build once, cache to disk
    gcache = CACHE / 'amppredictor_graphs.pt'
    if gcache.exists():
        items = torch.load(gcache)
        if args.no_neg:
            # the cache was built in corpus() order = positives FIRST then
            # negatives -> the first 4,929 entries are exactly the positives
            items = items[:4929]
    else:
        items = [(graph_of(s), torch.tensor(fp[s])) for s in df.sequence]
        torch.save(items, gcache)
    assert len(items) == len(df), (len(items), len(df))
    idx = np.random.RandomState(args.seed).permutation(len(items))
    n_tr = int(0.8 * len(idx))
    tr = [items[i] for i in idx[:n_tr]]
    va = [items[i] for i in idx[n_tr:]]
    ytr = torch.tensor(y[idx[:n_tr]], dtype=torch.float32)
    yva = torch.tensor(y[idx[n_tr:]], dtype=torch.float32)

    def loader(lst):
        return PyGLoader(lst, batch_size=128, shuffle=False)

    model = GNNPredictor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    best, best_state = np.inf, None
    for ep in range(1, 101):
        model.train()
        perm = torch.randperm(len(tr))
        for i in range(0, len(perm), 128):
            b = [tr[j] for j in perm[i:i + 128].tolist()]
            bg = Batch.from_data_list([g for g, _ in b]).to(device)
            bf = torch.stack([f for _, f in b]).to(device)
            yb = ytr[perm[i:i + 128]].to(device)
            opt.zero_grad()
            loss = crit(model(bg, bf).squeeze(-1), yb)
            loss.backward()
            opt.step()
        model.eval()
        pv = []
        with torch.no_grad():
            for i in range(0, len(va), 128):
                b = va[i:i + 128]
                bg = Batch.from_data_list([g for g, _ in b]).to(device)
                bf = torch.stack([f for _, f in b]).to(device)
                pv.append(model(bg, bf).squeeze(-1).cpu())
        mse = crit(torch.cat(pv), yva).item()
        if mse < best:
            best, best_state = mse, {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0:
            print(f's{args.seed} ep{ep} val_MSE={mse:.4f} best={best:.4f}', flush=True)
    model.load_state_dict(best_state)
    model.eval()
    out_dir = OUT_NONEG if args.no_neg else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    # [v2-12 patch] persist the evaluated task weights for cost timing
    torch.save({'model': model.state_dict()}, out_dir / f'task_s{args.seed}.pt')
    ext = ext_table()
    fpe = np.load(CACHE / 'amppredictor_fp_ext.npz')
    pe = []
    with torch.no_grad():
        for i in range(0, len(ext), 128):
            b = ext.sequence.iloc[i:i + 128].tolist()
            bg = Batch.from_data_list([graph_of(s) for s in b]).to(device)
            bf = torch.stack([torch.tensor(fpe[s]) for s in b]).to(device)
            pe.append(model(bg, bf).squeeze(-1).cpu())
    pe = torch.cat(pe).numpy()
    o = pd.DataFrame({'log10mic': pe})
    o.index = ext.sequence.values
    o.index.name = 'sequence'
    name = (f'sota_AMPpredictor_noneg_s{args.seed}_preds.csv' if args.no_neg
            else f'sota_AMPpredictor+neg_s{args.seed}_preds.csv')
    o.to_csv(out_dir / name)
    print(f'seed {args.seed}: best val MSE {best:.4f}; preds range '
          f'[{pe.min():.2f},{pe.max():.2f}] -> saved')


if __name__ == '__main__':
    main()

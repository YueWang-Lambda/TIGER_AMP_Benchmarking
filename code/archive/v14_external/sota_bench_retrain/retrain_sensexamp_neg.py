#!/usr/bin/env python3
"""Retrain SenseXAMP-Ecoli (W Zhang et al., Brief Bioinform 2023, bbad336;
official code William-Zhanng/SenseXAMP) on the shared restricted corpus under
the unified +neg protocol, seeds 120-129.

Recipe (authors' configs/reg_task/ecoli_SenseXAMP.py, kept verbatim):
  - modality 1: frozen ESM-1b per-token embeddings (repr layer 33, incl.
    BOS/EOS/pad rows, each sequence padded to the corpus max length when
    embedded - exactly tools/esm_emb_gen.py 'all' mode)
  - modality 2: 676-dim handcrafted sequence descriptors (AAC/DIP/Moreau-
    Broto/CTD/QSO/PAAC/APAAC/Basic via their structure_data_generate),
    z-scored with train-corpus statistics (their StcProcessor; stats fitted
    on the TRAIN split here and applied to all splits)
  - MultiModalFusionModel: Linear(676->1280) single stc token + self-attn
    (2 layers) over ESM tokens + cross-attn (1 layer) fusion, three heads
    (stc_pred / slf_pred / final_pred), Ensemble_MSELoss 0.25/0.25/0.5
  - Adam lr 1e-4 eps 1e-8, grad clip 1.0, batch 64, 80 epochs, no early
    stop (final-epoch model used, like the authors' config)

Adaptations (documented for METHODS):
  - the published 6-25 aa applicability domain is a data/prefilter property,
    not architectural: both modalities are length-agnostic; retrained here
    on the project corpus (6-49 aa)
  - label universe: MIC column = log10 MIC ug/mL (project standard;
    negatives at pseudo MIC 10,000 ug/mL -> 4.0)
  - train/val split 90/10 per seed (authors used fixed csv splits)

Original author weights (Google-Drive checkpoints) are NOT loaded.

Outputs (per seed):
  runs_review/benchmark_restricted/sota_sensexamp_neg_s<seed>_preds.csv
  runs_review/benchmark_restricted/sota_models/sensexamp_neg/sx_s<seed>.pt
Cache (once): baseline/sota_bench/sensexamp_cache/{esm1b_emb.h5, stc_desc.h5}

Usage: PYTHONPATH=. python baseline/sota_bench/retrain_sensexamp_neg.py \
          [--seeds 120-129] [--smoke] [--gpu 1] [--rebuild-cache]
"""
import argparse
import copy
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SOTA = Path(__file__).resolve().parent
REPO = SOTA / 'sensexamp_repo'
ROOT = SOTA.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tools'))  # structure_data_generate lives here
os.chdir(REPO)  # their code uses relative dataset paths

OUT = ROOT / 'runs_review/benchmark_restricted'
TRAIN_CSV = OUT / 'train_restricted4929_tiger.csv'
NEG_CSV = SOTA / 'data_restricted/uniprot_negatives.csv'
NEG_LOG10 = 4.0
PEP_CSV = ROOT / 'runs_review/final11_peptides.csv'
CACHE = SOTA / 'sensexamp_cache'

HP = dict(lr=1e-4, eps=1e-8, grad_clip=1.0, batch_size=64, epochs=80,
          val_batch=32)
MODEL_KW = dict(stc_size=676, emb_size=1280, d_inner=2048,
                n_slf_layers=2, n_cross_layers=1, n_head=4,
                d_k=1280, d_v=1280, dropout=0.2)


class AD(dict):
    """Minimal attribute-dict mimicking the mmcv Config surface used by
    build_loss_evaluator / the models (cfg.benchmark_name,
    cfg.model.losses.<name>.{weight,kwargs})."""

    def __getattr__(self, k):
        v = self.get(k)
        if isinstance(v, dict) and not isinstance(v, AD):
            v = AD(v)
        return v

    def copy(self):
        # keep the attribute-access surface after copy (build_loss_evaluator
        # calls cfg.model.losses.copy() then accesses .weight/.kwargs)
        return AD({k: (AD(v) if isinstance(v, dict) else v)
                   for k, v in self.items()})


def load_universe():
    tr = pd.read_csv(TRAIN_CSV)
    pos = tr.groupby('sequence', as_index=False).MIC_Escherichia_coli.mean()
    neg = pd.read_csv(NEG_CSV)
    # authors' preprocessing drops sequences < 6 aa (cal_pep_fromlist); a
    # handful of UniProt negatives are 5 aa - excluded, count logged
    neg = neg[neg.sequence.str.len() >= 6].reset_index(drop=True)
    print(f'SenseXAMP corpus: {len(pos)} pos + {len(neg)} neg '
          f'({7962 - len(neg)} negs <6 aa dropped)', flush=True)
    seqs = pos.sequence.tolist() + neg.sequence.tolist()
    mics = np.concatenate([pos.MIC_Escherichia_coli.values.astype(float),
                           np.full(len(neg), 10.0 ** NEG_LOG10)])
    labels = [1.0] * len(pos) + [0.0] * len(neg)
    return seqs, mics, labels


def load_ext():
    pep = pd.read_csv(PEP_CSV)
    return pep.sequence.tolist()


def build_cache(train_seqs, ext_seqs, device):
    import h5py
    import torch
    import esm  # fair-esm; loads ESM-1b from local torch-hub cache
    CACHE.mkdir(exist_ok=True)
    all_seqs = list(dict.fromkeys(train_seqs + ext_seqs))
    maxlen = max(len(s) for s in all_seqs)

    print(f'[cache] ESM-1b embeddings ({len(all_seqs)} seqs, '
          f'max_length={maxlen}) ...', flush=True)
    if (CACHE / 'esm1b_emb.h5').exists():
        print('  esm1b_emb.h5 exists, skip', flush=True)
    else:
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        model.eval().to(device)
        bc = alphabet.get_batch_converter()
        # the authors' bundled esm_project.BatchConverter accepts max_length
        # and pads to max(batch_max, max_length) with padding_idx; replicated
        # here with stock fair-esm (identical semantics)
        pad_to = maxlen + 2  # + BOS/EOS
        with h5py.File(CACHE / 'esm1b_emb.h5', 'w') as hf:
            for c in range(0, len(all_seqs), 32):
                chunk = all_seqs[c:c + 32]
                for s in chunk:
                    _, _, toks = bc([(s, s)])
                    toks = torch.nn.functional.pad(
                        toks, (0, pad_to - toks.shape[1]),
                        value=alphabet.padding_idx)
                    with torch.no_grad():
                        r = model(toks.to(device), repr_layers=[33],
                                  return_contacts=False)
                    emb = r['representations'][33].squeeze(0).cpu().numpy()
                    hf.create_dataset(s, data=emb.astype(np.float32))
                if (c // 32) % 100 == 0:
                    print(f'  {c + 32}/{len(all_seqs)}', flush=True)

    print('[cache] 676-dim descriptors ...', flush=True)
    from structure_data_generate.cal_pep_des import cal_pep
    rows = {}
    for s in all_seqs:
        assert len(s) >= 6
        rows[s] = cal_pep(s)
    desc = pd.DataFrame(rows).T  # (n_seq, ~676+Sequence col)
    feat_cols = [c for c in desc.columns if c != 'Sequence']
    desc = desc[feat_cols].astype(float)
    assert desc.shape[1] == 676, f'descriptor count {desc.shape[1]} != 676'
    # StcProcessor: z-score with train-corpus stats; std == 0 -> keep raw
    train_set = set(train_seqs)
    tr_mask = np.array([s in train_set for s in desc.index])
    mean = desc[tr_mask].mean(axis=0)
    std = desc[tr_mask].std(axis=0)
    nz = std != 0
    desc_n = desc.copy()
    desc_n.loc[:, nz] = (desc.loc[:, nz] - mean[nz]) / std[nz]
    with h5py.File(CACHE / 'stc_desc.h5', 'w') as hf:
        for s in desc_n.index:
            hf.create_dataset(s, data=desc_n.loc[s].values.astype(np.float32))
    print(f'[cache] done ({desc_n.shape})', flush=True)


EMB = {}
STC = {}


class DS:
    """In-RAM datasets (h5 preloaded once; the h5py-per-item random reads
    dominate epoch time otherwise)."""

    def __init__(self, seqs, mics, labels):
        self.seqs = seqs
        self.mics = mics
        self.labels = labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        s = self.seqs[i]
        return {'seq': s,
                'label': float(self.labels[i]),
                'mic': float(self.mics[i]),
                'emb': EMB[s],
                'stc': STC[s]}


def batchify(model, batch, device, train=False):
    inp = {'batch_tokens': batch['emb'].to(device).float(),
           'stc': batch['stc'].to(device).float(),
           'mic': batch['mic'].to(device).float(),
           'label': batch['label'].to(device).float()}
    out = model(inp)
    if train:  # authors' parse_losses: sum the loss dict values
        return sum(out.values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='120-129')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--gpu', default='1')
    ap.add_argument('--rebuild-cache', action='store_true')
    args = ap.parse_args()
    a, b = (int(x) for x in args.seeds.split('-'))
    seeds = list(range(a, b + 1))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    import torch
    from torch.utils.data import DataLoader
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_seqs, mics, labels = load_universe()
    ext_seqs = load_ext()

    # label universe: log10 MIC ug/mL, z-scored with train-corpus statistics
    # (the deep transformer stack does not converge on small-range raw log
    # labels with the authors' lr/epochs; standardization is the minimal
    # documented adaptation; inverted exactly at prediction)
    logmic = np.log10(mics)
    MU, SD = float(logmic.mean()), float(logmic.std())
    ys = (logmic - MU) / SD
    print(f'label z-score: mu={MU:.4f} sd={SD:.4f}', flush=True)

    if args.rebuild_cache or not (CACHE / 'stc_desc.h5').exists():
        build_cache(train_seqs, ext_seqs, device)

    import h5py
    print('preloading cache into RAM ...', flush=True)
    with h5py.File(CACHE / 'esm1b_emb.h5', 'r') as f:
        for s in set(train_seqs) | set(ext_seqs):
            EMB[s] = f[s][:]
    with h5py.File(CACHE / 'stc_desc.h5', 'r') as f:
        for s in set(train_seqs) | set(ext_seqs):
            STC[s] = f[s][:]

    epochs = 2 if args.smoke else HP['epochs']
    ckpt_dir = OUT / 'sota_models/sensexamp_neg'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    n = len(train_seqs)

    from Ampmm_base.models.TRMfusion import MultiModalFusionModel
    cfg = AD(benchmark_name='amp_reg',
             model={'losses': {'Ensemble_MSELoss': {
                 'weight': 1.0,
                 'kwargs': {'loss_weight': {'stc_pred': 0.25,
                                            'slf_pred': 0.25,
                                            'final_pred': 0.5}}}}})

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g).tolist()
        n_val = int(n * 0.1)
        val_i = perm[:n_val]
        tr_i = perm[n_val:]
        tr_ds = DS([train_seqs[i] for i in tr_i], [ys[i] for i in tr_i],
                   [labels[i] for i in tr_i])
        va_ds = DS([train_seqs[i] for i in val_i], [ys[i] for i in val_i],
                   [labels[i] for i in val_i])
        ex_ds = DS(ext_seqs, [0.0] * len(ext_seqs), [1.0] * len(ext_seqs))
        tl = DataLoader(tr_ds, batch_size=HP['batch_size'], shuffle=True)
        vl = DataLoader(va_ds, batch_size=HP['val_batch'], shuffle=False)
        el = DataLoader(ex_ds, batch_size=HP['val_batch'], shuffle=False)

        model = MultiModalFusionModel(cfg, **MODEL_KW).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=HP['lr'],
                               eps=HP['eps'])
        for ep in range(epochs):
            model.train()
            tot, seen = 0.0, 0
            for batch in tl:
                loss = batchify(model, batch, device, train=True)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               HP['grad_clip'])
                opt.step()
                tot += loss.item() * len(batch['mic'])
                seen += len(batch['mic'])
            if ep % 10 == 0 or ep == epochs - 1:
                model.eval()
                preds, gts = [], []
                with torch.no_grad():
                    for batch in vl:
                        r = batchify(model, batch, device)
                        preds.extend(r['model_outputs'].cpu().tolist())
                        gts.extend(r['labels'].cpu().tolist())
                rmse = float(np.sqrt(np.mean(
                    (np.array(preds) - np.array(gts)) ** 2)))
                print(f'seed {seed} ep {ep}: train {tot / seen:.5f} '
                      f'val_rmse {rmse:.5f}', flush=True)
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in el:
                r = batchify(model, batch, device)
                preds.extend(r['model_outputs'].cpu().tolist())
        log10mic = (np.array(preds) * SD + MU).tolist()
        out = pd.DataFrame({'log10mic': log10mic}, index=ext_seqs)
        out.index.name = ''
        fn = OUT / f'sota_sensexamp_neg_s{seed}_preds.csv'
        out.to_csv(fn)
        if not args.smoke:
            torch.save({'model': model.state_dict()},
                       ckpt_dir / f'sx_s{seed}.pt')
        print(f'seed {seed}: preds -> {fn.name} '
              f'(head {np.round(log10mic[:3], 3)})', flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Retrain HyperAMP (Dong et al., Adv Sci 2025, 10.1002/advs.202506332;
official code: ComputBiophys/AMPPainter, HyperAMP/ subdirectory) on the shared
restricted corpus under the unified +neg protocol, seeds 120-129.

Recipe (authors' HyperAMP/train.py, kept verbatim):
  - node features: frozen ankh-base per-residue embeddings (768-d)
  - hypergraph: consecutive k-gram windows, k in {2,3,4}, edge weight =
    TF(kgram in peptide, L2-normaled count) * IDF(corpus) + 1
  - HGNN: 2-layer hypergraph conv (768->640->320) per k, concat 960*3 -> FC
  - MSE on reward labels r = 1/(1+10^(v-2)); Adam lr 1e-4 weight_decay 5e-4,
    batch 128, 20 epochs, best-val-RMSE checkpoint (deepcopy, like authors)

Adaptations (documented for METHODS):
  - ankh-base loaded from local /data4T/ubuntu/wangyue/models/ankh-base
    (authors' own test_local pattern; zero network)
  - dhg 0.9.7 kwarg `e_weight` (authors' older dhg used `edge_weight`;
    identical semantics)
  - sequences longer than 40 aa are truncated to their first 40 residues,
    exactly like the authors' own preprocessing (ankh_embedding.py
    seq[:max_length=40]); 40 is an architecture constant (fc1 in_dim 960*3)
  - TF-IDF table stored sparse ({peptide: {kgram: value}}) instead of the
    authors' dense DataFrame - identical for every looked-up (nonzero) cell;
    the dense form would need ~tens of GB at our corpus size
  - label universe: v = log10 MIC ug/mL (project standard; negatives at
    pseudo MIC 10,000 ug/mL -> r ~= 0.0099); predictions inverted exactly:
    v = 2 - log10(1/r - 1)

Original author weights (hyp_best_model_V1) are NOT loaded.

Outputs (per seed):
  runs_review/benchmark_restricted/sota_hyperamp_neg_s<seed>_preds.csv
  runs_review/benchmark_restricted/sota_models/hyperamp_neg/hyperamp_s<seed>.pt
Cache (once): baseline/sota_bench/hyperamp_cache/{ankh_emb.npz, k_idf.pkl,
  kgram_tfidf.pkl}

Usage: PYTHONPATH=. python baseline/sota_bench/retrain_hyperamp_neg.py \
          [--seeds 120-129] [--smoke] [--gpu 1] [--rebuild-cache]
"""
import argparse
import copy
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SOTA = Path(__file__).resolve().parent
REPO = SOTA / 'ampainter_repo'
ROOT = SOTA.parent.parent
sys.path.insert(0, str(REPO / 'HyperAMP'))

OUT = ROOT / 'runs_review/benchmark_restricted'
TRAIN_CSV = OUT / 'train_restricted4929_tiger.csv'
NEG_CSV = SOTA / 'data_restricted/uniprot_negatives.csv'
NEG_LOG10 = 4.0  # pseudo MIC 10,000 ug/mL
PEP_CSV = ROOT / 'runs_review/final11_peptides.csv'
ANKH_DIR = Path('/data4T/ubuntu/wangyue/models/ankh-base')
CACHE = SOTA / 'hyperamp_cache'
CACHE_NONEG = SOTA / 'hyperamp_cache_noneg'   # [noneg] pos-only TF-IDF refit
MAXLEN = 40  # architecture constant: HGNN.fc1 in_dim 960*3 = (16*40 + 320)*3;
# authors' preprocessing truncates seq[:40] (ankh_embedding.py); sequences
# longer than 40 aa are truncated identically here (corpus: few; documented)

HP = dict(lr=1e-4, weight_decay=5e-4, batch_size=128, epochs=20)


def reward(value):
    return 1.0 / (1.0 + 10.0 ** (value - 2.0))


def reward_inv(r):
    """Exact inverse of r = 1/(1+10^(v-2)):  v = 2 + log10((1-r)/r)."""
    r = np.clip(r, 1e-7, 1.0 - 1e-7)
    return 2.0 + np.log10((1.0 - r) / r)


def load_universe(no_neg=False):
    tr = pd.read_csv(TRAIN_CSV)
    pos = tr.groupby('sequence', as_index=False).MIC_Escherichia_coli.mean()
    if no_neg:                      # [noneg campaign] positives-only training
        assert len(pos) == 4929, len(pos)
        return ([f'p{i}' for i in range(len(pos))], pos.sequence.tolist(),
                pos.MIC_Escherichia_coli.values.astype(float))
    neg = pd.read_csv(NEG_CSV)
    names = [f'p{i}' for i in range(len(pos))] + [f'n{i}' for i in range(len(neg))]
    seqs = pos.sequence.tolist() + neg.sequence.tolist()
    mics = np.concatenate([pos.MIC_Escherichia_coli.values.astype(float),
                           np.full(len(neg), 10.0 ** NEG_LOG10)])
    assert len(pos) == 4929 and len(neg) == 7962
    return names, seqs, mics


def build_noneg_cache(pos_names, pos_seqs, ext_names, ext_seqs):
    """[noneg] positives-only cache: reuse the ankh embeddings from the main
    cache (name-keyed; positives are p* entries — no GPU re-extraction) and
    REFIT the k-gram TF-IDF/IDF tables on the 4,929 positives only (the +neg
    tables' document frequencies include the negative pool and must not be
    reused)."""
    CACHE_NONEG.mkdir(exist_ok=True)
    src = dict(np.load(CACHE / 'ankh_emb.npz'))
    need = set(pos_names) | set(ext_names)
    np.savez(CACHE_NONEG / 'ankh_emb.npz',
             **{k: v for k, v in src.items() if k in need})
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    tfidf_lookup = {}
    k_idf_list = []
    for k in (2, 3, 4):
        vec = CountVectorizer(ngram_range=(k, k), analyzer='char')
        X = vec.fit_transform(pos_seqs)
        words = vec.get_feature_names_out()
        tf = TfidfTransformer(use_idf=False).fit_transform(X)
        idf = TfidfTransformer(use_idf=True, smooth_idf=False).fit(X).idf_
        ridf = {w: idf[j] for j, w in enumerate(words)}
        for r in range(X.shape[0]):
            row = X.getrow(r)
            tfr = tf.getrow(r).toarray().ravel()
            d = {}
            for j in row.indices:
                d[words[j]] = float(tfr[j] * idf[j] + 1.0)
            tfidf_lookup[pos_names[r]] = d
        k_idf_list.append({k: ridf})
        print(f'  [noneg] k={k}: vocab {len(words)}', flush=True)
    with open(CACHE_NONEG / 'kgram_tfidf.pkl', 'wb') as f:
        pickle.dump(tfidf_lookup, f)
    with open(CACHE_NONEG / 'k_idf.pkl', 'wb') as f:
        pickle.dump(k_idf_list, f)
    print('[noneg] cache done (ankh reused, TF-IDF refit on 4,929 positives)',
          flush=True)


def load_ext():
    pep = pd.read_csv(PEP_CSV)
    return [f'e{i}' for i in range(len(pep))], pep.sequence.tolist()


# ---------------- cache phase (once, seed-independent) ----------------

def build_cache(train_names, train_seqs, ext_names, ext_seqs, device):
    import torch
    from transformers import T5EncoderModel, AutoTokenizer
    CACHE.mkdir(exist_ok=True)
    print('[cache] ankh-base embeddings ...', flush=True)
    pm = T5EncoderModel.from_pretrained(str(ANKH_DIR))
    tk = AutoTokenizer.from_pretrained(str(ANKH_DIR))
    pm.eval().to(device)
    all_names = train_names + ext_names
    all_seqs = train_seqs + ext_seqs
    order = np.argsort([len(s) for s in all_seqs], kind='stable')
    embs = {}
    B = 48
    for c in range(0, len(order), B):
        idx = order[c:c + B]
        batch = [list(all_seqs[i]) for i in idx]
        ids = tk.batch_encode_plus(batch, add_special_tokens=True, padding=True,
                                   is_split_into_words=True, return_tensors='pt')
        with torch.no_grad():
            # [v2-13 patch] pass attention_mask: pad tokens must not contaminate
            # residue embeddings (the mask-less rule made features depend on
            # batch composition; timing/accuracy paths now share one definition)
            out = pm(input_ids=ids['input_ids'].to(device),
                     attention_mask=ids['attention_mask'].to(device))[0].cpu().numpy()
        for j, i in enumerate(idx):
            L = len(all_seqs[i])
            embs[all_names[i]] = out[j, :L, :].astype(np.float32)
        if (c // B) % 50 == 0:
            print(f'  {c + B}/{len(order)}', flush=True)
    np.savez(CACHE / 'ankh_emb.npz', **embs)

    print('[cache] TF-IDF tables on train corpus ...', flush=True)
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    tfidf_lookup = {}  # name -> {kgram_lower: tfidf}
    k_idf_list = []
    for k in (2, 3, 4):
        vec = CountVectorizer(ngram_range=(k, k), analyzer='char')
        X = vec.fit_transform(train_seqs)  # rows aligned with train_names
        words = vec.get_feature_names_out()
        tf = TfidfTransformer(use_idf=False).fit_transform(X)
        idf = TfidfTransformer(use_idf=True, smooth_idf=False).fit(X).idf_
        ridf = {w: idf[j] for j, w in enumerate(words)}
        for r in range(X.shape[0]):
            row = X.getrow(r)
            tfr = tf.getrow(r).toarray().ravel()
            d = {}
            for j in row.indices:
                d[words[j]] = float(tfr[j] * idf[j] + 1.0)
            tfidf_lookup[train_names[r]] = d
        k_idf_list.append({k: ridf})
        print(f'  k={k}: vocab {len(words)}', flush=True)
    with open(CACHE / 'kgram_tfidf.pkl', 'wb') as f:
        pickle.dump(tfidf_lookup, f)
    with open(CACHE / 'k_idf.pkl', 'wb') as f:
        pickle.dump(k_idf_list, f)
    print('[cache] done', flush=True)


# ---------------- hypergraph + data pipeline ----------------

class Featurizer:
    """Replicates HyperAMP/utils.py tf_idf/calc_tfidf/get_hypergraph."""

    def __init__(self, tfidf_lookup, k_idf_list, embs):
        from dhg.structure.hypergraphs import Hypergraph
        self.Hypergraph = Hypergraph
        self.lookup = tfidf_lookup
        # k_idf_list = [{2: {...}}, {3: {...}}, {4: {...}}] -> {k: {...}}
        self.idf = {k: d[k] for d in k_idf_list for k in d}
        self.embs = embs

    def _ew(self, name, seq, edge, k):
        edge = edge.lower()
        d = self.lookup.get(name)
        if d is not None and edge in d:
            return d[edge]
        # calc_tfidf fallback (external peptides): single-doc tf * corpus idf + 1
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        vec = CountVectorizer(ngram_range=(k, k), analyzer='char')
        X = vec.fit_transform([seq])
        words = vec.get_feature_names_out()
        tf = TfidfTransformer(use_idf=False).fit_transform(X).toarray().ravel()
        tfv = float(tf[list(words).index(edge)])
        idv = self.idf[k].get(edge, 1.0)
        return tfv * idv + 1.0

    def hypergraph_params(self, name, seq):
        num = len(seq)
        params = []
        for k in (2, 3, 4):
            elist, ew = [], []
            for i in range(num - k + 1):
                elist.append(list(range(i, i + k)))
                ew.append(self._ew(name, seq, seq[i:i + k], k))
            hg = self.Hypergraph(num, elist, e_weight=ew)
            params.append(hg.L_HGNN)
        return params


def pad_sparse(X, max_length):
    return X.sparse_resize_((max_length, max_length),
                            X.sparse_dim(), X.dense_dim())


class SparseLike:
    """Dense tensor wrapped to satisfy the model's `L_HGNN.to_dense()` call.
    The authors' torch/dhg versions collated sparse Laplacians directly;
    current torch cannot batch sparse tensors. Numerics identical: the model
    densifies L_HGNN in its very first op."""

    def __init__(self, dense):
        self._d = dense

    def to_dense(self):
        return self._d

    def to(self, device):
        return SparseLike(self._d.to(device))


def make_collate(maxlen):
    def collate(batch):
        import torch
        emb_list, lab_list, dense_list = [], [], []
        for emb, params, label in batch:
            emb = torch.tensor(np.asarray(emb))
            if emb.shape[0] < maxlen:
                pad = torch.zeros((maxlen - emb.shape[0], emb.shape[1]))
                emb = torch.cat((emb, pad), 0)
            emb_list.append(emb)
            lab_list.append(label)
            dense_list.append([pad_sparse(p, maxlen).to_dense()
                               for p in params])
        emb = torch.utils.data.dataloader.default_collate(emb_list)
        # (B, 3, maxlen, maxlen) -> list of 3 SparseLike (B, maxlen, maxlen)
        st = torch.stack([torch.stack([d[k] for d in dense_list])
                          for k in range(3)])
        params = [SparseLike(st[k]) for k in range(3)]
        lab = torch.utils.data.dataloader.default_collate(lab_list)
        return [emb, params, lab]
    return collate


class EmbDS:
    """Pre-featurized dataset: (emb, [L_k2, L_k3, L_k4], reward_label)."""

    def __init__(self, names, seqs, ys, featurizer):
        self.items = []
        for nm, sq, y in zip(names, seqs, ys):
            sq = sq[:MAXLEN]  # authors' truncation (seq[:40])
            self.items.append((featurizer.embs[nm][:MAXLEN],
                               lambda nm=nm, sq=sq: featurizer.hypergraph_params(nm, sq),
                               float(y)))
        self._cache_hg = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        emb, hg_fn, y = self.items[i]
        if i not in self._cache_hg:
            self._cache_hg[i] = hg_fn()
        return emb, self._cache_hg[i], y


def run_epoch(model, loader, device, opt=None):
    import torch
    torch_mod = torch
    model.train() if opt is not None else model.eval()
    preds, labels, tot = [], [], 0.0
    ctx = torch_mod.enable_grad() if opt is not None else torch_mod.no_grad()
    with ctx:
        for emb, params, label in loader:
            emb = emb.to(device).float()
            params = [p.to(device) for p in params]
            label = label.to(device).float()
            out = model(emb, params)
            loss = torch_mod.nn.functional.mse_loss(out, label)
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()
            else:
                preds.extend(out.cpu().tolist())
                labels.extend(label.cpu().tolist())
            tot += loss.item() * len(label)
    n = len(loader.dataset)
    return tot / n, preds, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='120-129')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--gpu', default='1')
    ap.add_argument('--rebuild-cache', action='store_true')
    ap.add_argument('--no-neg', action='store_true',
                    help='[noneg] train on the 4,929 positives only; TF-IDF tables '
                         'refit on positives into hyperamp_cache_noneg (ankh embeddings '
                         'reused from the main cache)')
    ap.add_argument('--predict-only', action='store_true',
                    help='load saved checkpoints and rewrite the per-seed '
                         'prediction CSVs (no training)')
    args = ap.parse_args()
    a, b = (int(x) for x in args.seeds.split('-'))
    seeds = list(range(a, b + 1))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    import torch
    from torch.utils.data import DataLoader
    from HyperAMP import HGNN  # authors' model, verbatim
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_names, train_seqs, mics = load_universe(args.no_neg)
    ext_names, ext_seqs = load_ext()
    ys = reward(np.log10(mics))

    cache = CACHE_NONEG if args.no_neg else CACHE
    if args.no_neg:
        if args.rebuild_cache or not (cache / 'kgram_tfidf.pkl').exists():
            build_noneg_cache(train_names, train_seqs, ext_names, ext_seqs)
    elif args.rebuild_cache or not (CACHE / 'ankh_emb.npz').exists():
        build_cache(train_names, train_seqs, ext_names, ext_seqs, device)
    print('loading cache ...', flush=True)
    embs = dict(np.load(cache / 'ankh_emb.npz'))
    with open(cache / 'kgram_tfidf.pkl', 'rb') as f:
        tfidf_lookup = pickle.load(f)
    with open(cache / 'k_idf.pkl', 'rb') as f:
        k_idf_list = pickle.load(f)
    feat = Featurizer(tfidf_lookup, k_idf_list, embs)

    print('featurizing datasets (hypergraphs) ...', flush=True)
    tr_ds = EmbDS(train_names, train_seqs, ys, feat)
    ex_ds = EmbDS(ext_names, ext_seqs, [0.0] * len(ext_names), feat)
    collate = make_collate(MAXLEN)
    el = DataLoader(ex_ds, batch_size=HP['batch_size'], shuffle=False,
                    collate_fn=collate)

    epochs = 2 if args.smoke else HP['epochs']
    ckpt_dir = OUT / ('sota_models/hyperamp_noneg' if args.no_neg
                      else 'sota_models/hyperamp_neg')
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    n_train = len(tr_ds)
    val_frac = 0.1

    if args.predict_only:
        for seed in seeds:
            model = HGNN().to(device)
            sd = torch.load(ckpt_dir / f'hyperamp_s{seed}.pt',
                            map_location='cpu')['model']
            model.load_state_dict(sd)
            _, ep_pred, _ = run_epoch(model, el, device)
            log10mic = reward_inv(np.array(ep_pred))
            out = pd.DataFrame({'log10mic': log10mic}, index=ext_seqs)
            out.index.name = ''
            fn = OUT / (f'sota_hyperamp_noneg_s{seed}_preds.csv' if args.no_neg
                        else f'sota_hyperamp_neg_s{seed}_preds.csv')
            out.to_csv(fn)
            print(f'seed {seed}: re-predicted -> {fn.name} '
                  f'(head {np.round(log10mic[:3], 3)})', flush=True)
        return

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_train, generator=g).tolist()
        n_val = int(n_train * val_frac)
        val_idx, train_idx = set(perm[:n_val]), [i for i in perm[n_val:]]
        sub_train = [tr_ds[i] for i in train_idx]
        sub_val = [tr_ds[i] for i in sorted(val_idx)]

        tl = DataLoader(sub_train, batch_size=HP['batch_size'], shuffle=True,
                        collate_fn=collate)
        vl = DataLoader(sub_val, batch_size=HP['batch_size'], shuffle=False,
                        collate_fn=collate)
        el = DataLoader(ex_ds, batch_size=HP['batch_size'], shuffle=False,
                        collate_fn=collate)

        model = HGNN().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=HP['lr'],
                               weight_decay=HP['weight_decay'])
        best_rmse, best_sd, best_ep = 1e9, None, -1
        for ep in range(epochs):
            tr_loss, _, _ = run_epoch(model, tl, device, opt)
            _, vp, vy = run_epoch(model, vl, device)
            rmse = float(np.sqrt(np.mean((np.array(vp) - np.array(vy)) ** 2)))
            if rmse < best_rmse:
                best_rmse, best_ep = rmse, ep
                best_sd = copy.deepcopy(model.state_dict())
            print(f'seed {seed} ep {ep}: train {tr_loss:.5f} val_rmse {rmse:.5f}',
                  flush=True)
        model.load_state_dict(best_sd)
        _, ep_pred, _ = run_epoch(model, el, device)
        log10mic = reward_inv(np.array(ep_pred))
        out = pd.DataFrame({'log10mic': log10mic}, index=ext_seqs)
        out.index.name = ''
        fn = OUT / (f'sota_hyperamp_noneg_s{seed}_preds.csv' if args.no_neg
                    else f'sota_hyperamp_neg_s{seed}_preds.csv')
        out.to_csv(fn)
        if not args.smoke:
            torch.save({'model': best_sd},
                       ckpt_dir / f'hyperamp_s{seed}.pt')
        print(f'seed {seed}: best ep {best_ep} (val_rmse {best_rmse:.5f}), '
              f'preds -> {fn.name} (head {np.round(log10mic[:3], 3)})', flush=True)


if __name__ == '__main__':
    main()

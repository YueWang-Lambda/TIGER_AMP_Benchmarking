#!/usr/bin/env python3
"""Cost timing for the 4 retrained benchmark additions (torch side: HyperAMP,
SenseXAMP-Ecoli) on the actual external workload (208 peptides).

Protocol identical to scripts/eval_cost_timing_external.py: reconstruct the
submitted model (seed-123 weights from runs_review/benchmark_restricted/
sota_models/<method>/), warm up once, time 3 repeats of the full 208-peptide
inference, report the median. Per-query feature computation (frozen PLM
forward, descriptors, hypergraph construction) IS included; one-time model
loading and train-corpus statistics fitting are NOT.

The TF/Keras additions (MBC-Attention, AMPlify) are timed by the sister
script eval_cost_timing_new4_tf.py (mbc_att env); both write disjoint method
keys into the same output json.

Usage: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/eval_cost_timing_new4_torch.py
"""
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ['TRANSFORMERS_OFFLINE'] = '1'

ROOT = Path(__file__).resolve().parents[1]
SOTA = ROOT / 'baseline/sota_bench'
OUT = ROOT / 'runs_review/benchmark_restricted/new4_cost_timing.json'
for p in (str(ROOT), str(SOTA), str(ROOT / 'scripts')):
    sys.path.insert(0, p)

PEP_CSV = ROOT / 'runs_review/final11_peptides.csv'


def pep():
    """Fresh read of the 208 query sequences inside every timed repeat."""
    return list(dict.fromkeys(pd.read_csv(PEP_CSV).sequence))
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CKPT = ROOT / 'runs_review/benchmark_restricted/sota_models'


def timed(fn, reps=3):
    fn()  # warm-up
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        ts.append(round(time.time() - t0, 4))
    return {'repeats_s': ts, 'median_s': sorted(ts)[1]}


# pair endpoint (v13b): the PRIMARY benchmark panel — 11 families, 2,211
# edges whose endpoints are exactly the 208 test peptides (full table spans
# 21 families / 2,748 edges; extra 537 non-benchmark-family edges NOT
# differenced). Canonical keys: pair_keys_primary_2211.csv; unit = LOG2.
PAIRS = None


def pair_delta(preds):
    """preds: {seq: log10-MIC}; returns the 2,211 ordered LOG2 deltas
    (evaluator semantics); non-finite => featuriser drop => loud assertion."""
    global PAIRS
    if PAIRS is None:
        K = pd.read_csv(ROOT / 'runs_review/benchmark_restricted/pair_keys_primary_2211.csv')
        PAIRS = list(zip(K.anchor_sequence, K.query_sequence))
        assert len(PAIRS) == 2211
    d = np.array([(preds.get(a, np.nan) - preds.get(b, np.nan)) / math.log10(2)
                  for a, b in PAIRS])
    assert np.isfinite(d).all(), 'primary-panel endpoint missing from preds'
    return d


def t_hyperamp():
    """Per-query: ankh-base forward (frozen, live) + TF-IDF hypergraph build
    (ext peptides take the authors' calc_tfidf fallback path) + HGNN forward.
    One-time: ankh/HGNN loads, TF-IDF tables."""
    import retrain_hyperamp_neg as rh
    from torch.utils.data import DataLoader
    from HyperAMP import HGNN

    model = HGNN().to(DEVICE)
    model.load_state_dict(torch.load(CKPT / 'hyperamp_neg/hyperamp_s123.pt',
                                     map_location='cpu')['model'])
    model.eval()

    from transformers import T5EncoderModel, AutoTokenizer
    pm = T5EncoderModel.from_pretrained(str(rh.ANKH_DIR))
    tk = AutoTokenizer.from_pretrained(str(rh.ANKH_DIR))
    pm.eval().to(DEVICE)

    with open(rh.CACHE / 'kgram_tfidf.pkl', 'rb') as f:
        tfidf_lookup = pickle.load(f)
    with open(rh.CACHE / 'k_idf.pkl', 'rb') as f:
        k_idf_list = pickle.load(f)
    feat = rh.Featurizer(tfidf_lookup, k_idf_list, embs={})  # live embeddings
    collate = rh.make_collate(rh.MAXLEN)

    def run():
        PEP = pep()                                          # input re-read
        preds_local = {}
        with torch.no_grad():
            items = []
            for i in range(0, len(PEP), 32):
                chunk = PEP[i:i + 32]
                ids = tk.batch_encode_plus([list(s) for s in chunk],
                                           add_special_tokens=True, padding=True,
                                           is_split_into_words=True,
                                           return_tensors='pt')
                # [v2-13] attention_mask — same definition as the rebuilt cache
                out = pm(input_ids=ids['input_ids'].to(DEVICE),
                         attention_mask=ids['attention_mask'].to(DEVICE))[0].cpu().numpy()
                for j, s in enumerate(chunk):
                    emb = out[j, :len(s), :]
                    items.append((emb[:rh.MAXLEN],
                                  feat.hypergraph_params(f'tmp{i + j}',
                                                         s[:rh.MAXLEN]),
                                  0.0))
            loader = DataLoader(items, batch_size=128, shuffle=False,
                                collate_fn=collate)
            k = 0                       # global row counter (loader order)
            for emb, params, _ in loader:
                y = model(emb.to(DEVICE).float(),
                          [p.to(DEVICE) for p in params])
                # reward_inv (retrain_hyperamp_neg, exact): v = 2 + log10((1-r)/r)
                rc = y.clamp(1e-7, 1.0 - 1e-7)
                v = 2.0 + torch.log10((1.0 - rc) / rc)
                for j in range(len(v)):
                    preds_local[PEP[k]] = float(v[j]); k += 1
        pair_delta(preds_local)          # 2,211-edge endpoint (v13)
    return timed(run)


def _desc_stats():
    """Descriptor (mean, std, columns) fitted on the train corpus, exactly as
    in the retraining cache; persisted to sexamp_cache/desc_stats.npz."""
    import retrain_sensexamp_neg as rs
    stats_fn = rs.CACHE / 'desc_stats.npz'
    if stats_fn.exists():
        z = np.load(stats_fn, allow_pickle=True)
        return z['mean'], z['std'], list(z['columns'])
    tr_seqs, _, _ = rs.load_universe()
    from structure_data_generate.cal_pep_des import cal_pep
    rows = {s: cal_pep(s) for s in tr_seqs}
    desc = pd.DataFrame(rows).T
    cols = [c for c in desc.columns if c != 'Sequence']
    desc = desc[cols].astype(float)
    mean, std = desc.mean(axis=0), desc.std(axis=0)
    np.savez(stats_fn, mean=mean.values, std=std.values, columns=np.array(cols))
    return mean, std, cols


def t_sensexamp():
    """Per-query: ESM-1b forward (frozen, authors' per-sequence loop) +
    676-dim descriptors (their cal_pep) + train-stat z-score + fusion forward.
    One-time: ESM/ckpt loads, z-score statistics."""
    sys.path.insert(0, str(SOTA / 'sensexamp_repo'))
    sys.path.insert(0, str(SOTA / 'sensexamp_repo/tools'))
    os.chdir(SOTA / 'sensexamp_repo')
    import esm
    import retrain_sensexamp_neg as rs
    from structure_data_generate.cal_pep_des import cal_pep

    from Ampmm_base.models.TRMfusion import MultiModalFusionModel
    cfg = rs.AD(benchmark_name='amp_reg',
                model={'losses': {'Ensemble_MSELoss': {
                    'weight': 1.0,
                    'kwargs': {'loss_weight': {'stc_pred': 0.25,
                                               'slf_pred': 0.25,
                                               'final_pred': 0.5}}}}})
    model = MultiModalFusionModel(cfg, **rs.MODEL_KW).to(DEVICE)
    model.load_state_dict(torch.load(CKPT / 'sensexamp_neg/sx_s123.pt',
                                     map_location='cpu')['model'])
    model.eval()

    esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    esm_model.eval().to(DEVICE)
    bc = alphabet.get_batch_converter()
    mean, std, cols = _desc_stats()
    nz = np.asarray(std != 0)
    SXPREDS = {}
    # one-time label calibration constants (driver convention: z over the
    # full train-corpus log-MIC before the seed split), applied per query
    _, mics, _ = rs.load_universe()
    logmic = np.log10(mics)
    MU, SD = float(logmic.mean()), float(logmic.std())

    def run():
        PEP = pep()                                          # input re-read
        with torch.no_grad():
            for i in range(0, len(PEP), 32):
                chunk = PEP[i:i + 32]
                embs = []
                for s in chunk:
                    _, _, toks = bc([(s, s)])
                    toks = torch.nn.functional.pad(
                        toks, (0, 49 + 2 - toks.shape[1]),
                        value=alphabet.padding_idx)
                    r = esm_model(toks.to(DEVICE), repr_layers=[33],
                                  return_contacts=False)
                    embs.append(r['representations'][33].squeeze(0))
                desc = pd.DataFrame([cal_pep(s) for s in chunk])[cols]\
                    .astype(float)
                desc.loc[:, nz] = (desc.loc[:, nz] - mean[nz]) / std[nz]
                X = torch.stack(embs).float().to(DEVICE)
                S = torch.from_numpy(desc.values.astype(np.float32)).to(DEVICE)
                out = model({'batch_tokens': X, 'stc': S,
                             'mic': torch.zeros(len(chunk)),
                             'label': torch.zeros(len(chunk))})
                y = out['model_outputs'] if isinstance(out, dict) else out
                # per-query label inversion: z-scored log-MIC -> log10 MIC
                mic = y * SD + MU
                for j, s in enumerate(chunk):
                    SXPREDS[s] = float(mic[j])
        pair_delta(SXPREDS)              # 2,211-edge endpoint (v13)
    return timed(run)


METHODS = {'HyperAMP+neg': t_hyperamp,
           'SenseXAMP-Ecoli+neg': t_sensexamp}


def main():
    result = {}
    if OUT.exists():
        result = json.loads(OUT.read_text())
    result["n_peptides"] = len(pep())
    result['device'] = (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else 'cpu')
    result['batch_note'] = ('deployment-style; per-query feature computation '
                            'included, one-time model loading excluded '
                            '(protocol of eval_cost_timing_external.py)')
    result.setdefault('methods', {})
    for name, fn in METHODS.items():
        try:
            r = fn()
            # per-method device record (v13b): torch methods run on GPU
            r['device'] = (torch.cuda.get_device_name(0)
                           if torch.cuda.is_available() else 'cpu')
            result['methods'][name] = r
            print(f'[{name}] median {r["median_s"]}s  {r["repeats_s"]}',
                  flush=True)
        except Exception as e:
            result['methods'][name] = {'error': f'{type(e).__name__}: {e}'}
            print(f'[{name}] ERROR {type(e).__name__}: {e}', flush=True)
        torch.cuda.empty_cache()
    OUT.write_text(json.dumps(result, indent=2))
    print(f'saved {OUT}', flush=True)


if __name__ == '__main__':
    main()

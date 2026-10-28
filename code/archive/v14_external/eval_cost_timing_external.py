#!/usr/bin/env python3
"""UNIFIED strict inference-cost timing for the ten original external methods
(2026-10-10 rewrite; supersedes the 2026-09 version whose per-query encodings
sat outside the timed region for Das/DeepAMP/HydrAMP/AMPLiT/EvoGrad/APEX/
AMP-Designer).

Boundary — identical for every method and for eval_cost_timing_new7.py:
  timed region = raw sequences -> per-peptide log10-MIC predictions:
                 ALL per-query computation (tokenization, one-hot/descriptor
                 encoding, AAindex embedding lookup, PLM forwards) + model
                 forward + per-query label inversion (APEX 6-y; BERT-AmPEP
                 scale->pMIC affine; AMPLiT-vB z-score destandardisation;
                 HydrAMP-arch fold-mean raw logit = log10MIC; EvoGradient
                 per-peptide uM->ug/mL MW conversion with its released
                 ensemble weights);
  excluded     = one-time model/tokenizer/frozen-backbone/runtime loading;
                 calibration constants (BERT polyfit) are fitted once outside
                 and applied per query inside;
  inputs       = the 208-sequence list is re-read from CSV inside every repeat;
  protocol     = otherwise-idle GPU, seed-123 weights, 1 warm-up + 3 timed
                 repeats of the full 208-peptide inference, median reported.

Usage: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/eval_cost_timing_external.py
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ['TRANSFORMERS_OFFLINE'] = '1'

ROOT = Path(__file__).resolve().parents[1]
SOTA = ROOT / 'baseline/sota_bench'
OLD = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER/runs_review/benchmark_restricted/sota_models')
OUT = ROOT / 'runs_review/benchmark_restricted/external_cost_timing.json'
for p in (str(ROOT), str(SOTA), str(ROOT / 'scripts')):
    sys.path.insert(0, p)

PEP_CSV = ROOT / 'runs_review/final11_peptides.csv'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def pep():
    """Fresh read of the query sequences inside every timed repeat."""
    return list(dict.fromkeys(pd.read_csv(PEP_CSV).sequence))


# pair endpoint: the PRIMARY benchmark panel — the 11-family, 2,211-edge
# subset of eleven_way_predictions.csv whose endpoints are exactly the 208
# external test peptides (the full table spans 21 families / 2,748 edges; the
# extra 537 edges belong to non-benchmark families and are NOT differenced).
# Canonical ordered keys: pair_keys_primary_2211.csv (v13b; the differencing
# runs INSIDE every timed repeat so all 23 methods return the same endpoint).
import hashlib
PAIRS = None


def pair_edges():
    global PAIRS
    if PAIRS is None:
        K = pd.read_csv(ROOT / 'runs_review/benchmark_restricted/pair_keys_primary_2211.csv')
        PAIRS = list(zip(K.anchor_sequence, K.query_sequence))
        assert len(PAIRS) == 2211
    return PAIRS


def pair_delta(preds):
    """preds: {seq: log10-MIC} over the 208 test peptides; returns the 2,211
    ordered LOG2 anchor-minus-query deltas — the evaluator's exact semantics
    ((pred_a - pred_q) / log10(2)). Every primary-panel endpoint is one of the
    208 peptides, so a non-finite entry means a featuriser dropped a benchmark
    peptide -> loud assertion (v13b)."""
    d = np.array([(preds.get(a, np.nan) - preds.get(b, np.nan)) / math.log10(2)
                  for a, b in pair_edges()])
    assert np.isfinite(d).all(), 'primary-panel endpoint missing from preds'
    return d


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


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


def _load(model, ckpt):
    model.load_state_dict(torch.load(ckpt, map_location='cpu')['model'])
    model.to(DEVICE).eval()
    return model


def t_das():
    import retrain_das_regressor as rd
    m = _load(rd.BiLSTMRegressor(), OLD / 'das_regr/bilstm_regr_best.pt')

    def run():
        seqs = pep()
        X = torch.from_numpy(rd.encode(seqs)).to(DEVICE)
        preds = {}
        with torch.no_grad():
            for i in range(0, len(X), 256):
                y = m(X[i:i + 256]).squeeze(-1).cpu().numpy()
                for j, s in enumerate(seqs[i:i + 256]):
                    preds[s] = float(y[j])
        pair_delta(preds)
    return timed(run)


def t_deep_amp():
    import retrain_deep_amp as da
    m = _load(da.CNNRegressor(), OLD / 'deep_amp_ensemble/cnn_seed123.pt')

    def run():
        seqs = pep()
        X = torch.from_numpy(da.encode(seqs)[0]).float().to(DEVICE)
        preds = {}
        with torch.no_grad():
            for i in range(0, len(X), 256):
                y = m(X[i:i + 256]).squeeze(-1).cpu().numpy()
                for j, s in enumerate(seqs[i:i + 256]):
                    preds[s] = float(y[j])
        pair_delta(preds)
    return timed(run)


def _bert_family(base_net_builder, ckpt, spaced=False, invert=None):
    """Fixed-60-token padding, exactly the training Dataset semantics
    (train_sota_bench_neg.py DS: padding='max_length', max_length=60)."""
    from transformers import BertTokenizer, BertModel
    tok = BertTokenizer.from_pretrained(str(SOTA / 'models/prot_bert'), do_lower_case=False)
    bert = BertModel.from_pretrained(str(SOTA / 'models/prot_bert'))
    net = _load(base_net_builder(bert), ckpt)

    def run():
        raw = pep()
        seqs = [' '.join(s) for s in raw] if spaced else raw
        preds = {}
        with torch.no_grad():
            for i in range(0, len(seqs), 64):
                enc = tok(seqs[i:i + 64], padding='max_length', truncation=True,
                          max_length=60, return_tensors='pt')
                y = net(enc['input_ids'].to(DEVICE), enc['attention_mask'].to(DEVICE))
                if invert is not None:            # per-query label inversion
                    y = invert(y)
                for j, s in enumerate(raw[i:i + 64]):   # keyed by RAW sequence
                    preds[s] = float(y[j])
        pair_delta(preds)                         # 2,211-edge endpoint
    return timed(run)


def t_bert_ampep():
    """The evaluated head is the POOLER regression net of
    train_sota_bench_neg.build_bert_ampep (not meanpool); labels follow its
    fixed standardisation scale = 0.178545*pMIC + 0.762956 -> log10MIC =
    -((scale - 0.762956)/0.178545). best.pt verifiably matches archived
    seed-129 predictions (v13 audit); recorded below."""
    from torch import nn

    def build(bert):
        class REG(nn.Module):
            def __init__(self):
                super().__init__()
                self.bert = bert
                self.dropout = nn.Dropout(0.1)
                self.fc1 = nn.Linear(bert.config.hidden_size, 256)
                self.fc2 = nn.Linear(256, 1)

            def forward(self, ii, am):
                p = self.bert(input_ids=ii, attention_mask=am).pooler_output
                x = self.dropout(torch.relu(self.fc1(self.dropout(p))))
                return self.fc2(x).squeeze(-1)
        return REG()
    return _bert_family(build, OLD / 'bert_ampep_with_neg/best.pt', spaced=True,
                        invert=lambda y: -((y - 0.762956) / 0.178545))


def t_deep_pie():
    from torch import nn

    def build(bert):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.bert = bert
                self.fc1 = nn.Linear(bert.config.hidden_size, 512)
                self.fc2 = nn.Linear(512, 256)
                self.fc3 = nn.Linear(256, 1)
                self.relu = nn.ReLU()
                self.drop = nn.Dropout(0.1)

            def forward(self, ii, am):
                p = self.bert(input_ids=ii, attention_mask=am).pooler_output
                x = self.drop(self.relu(self.fc1(p)))
                x = self.drop(self.relu(self.fc2(x)))
                return self.fc3(x).squeeze(-1)
        return Net()
    return _bert_family(build, OLD / 'deep_pie_with_neg/best.pt', spaced=True)


def t_llamp():
    from transformers import AutoTokenizer, AutoModel
    from torch import nn
    pe = str(SOTA / 'models/peptide_tuned_ESM-2')
    tok = AutoTokenizer.from_pretrained(pe)
    esm = AutoModel.from_pretrained(pe)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.esm = esm
            h = esm.config.hidden_size
            self.fc1 = nn.Linear(h, 256)
            self.fc2 = nn.Linear(256, 1)
            self.drop = nn.Dropout(0.1)

        def forward(self, ii, am):
            p = self.esm(input_ids=ii, attention_mask=am).last_hidden_state.mean(dim=1)
            return self.fc2(self.drop(torch.relu(self.fc1(self.drop(p))))).squeeze(-1)
    net = _load(Net(), OLD / 'llamp_with_neg/best.pt')

    def run():
        seqs = pep()
        preds = {}
        with torch.no_grad():
            # fixed 60-token padding = training DS semantics (unmasked mean
            # pooling over all 60 positions; dynamic padding would change it)
            for i in range(0, len(seqs), 64):
                enc = tok(seqs[i:i + 64], padding='max_length', truncation=True,
                          max_length=60, return_tensors='pt')
                y = net(enc['input_ids'].to(DEVICE), enc['attention_mask'].to(DEVICE))
                for j, s in enumerate(seqs[i:i + 64]):
                    preds[s] = float(y[j])
        pair_delta(preds)                         # log10MIC direct labels
    return timed(run)


def t_hydramp():
    from retrain_hydramp_regressor import MICClassifier
    from retrain_hydramp_mic import encode, pad_batch
    models = [_load(MICClassifier(), OLD / f'hydramp_regr_neg/fold{f}.pt') for f in range(1, 11)]

    def run():
        seqs_all = pep()
        rows_raw = encode(seqs_all)
        rows = [r for r in rows_raw if r is not None]
        names = [s for s, r in zip(seqs_all, rows_raw) if r is not None]
        preds = {}
        with torch.no_grad():
            for i in range(0, len(rows), 4096):
                xb = pad_batch(rows[i:i + 4096]).to(DEVICE)
                # submitted predictor = mean over the 10 fold models (v13 fix:
                # fold aggregation was missing). The arch-regr driver trains
                # MICClassifier as a RAW-logit log10MIC regressor (no sigmoid,
                # no reward transform) -> fold-mean IS the log10MIC prediction
                y = None
                for m in models:
                    o = m(xb)
                    y = o if y is None else y + o
                v = (y / len(models)).squeeze(-1).cpu().numpy()
                for j in range(len(xb)):
                    preds[names[i + j]] = float(v[j])
        pair_delta(preds)
    return timed(run)


def t_amplit():
    from retrain_amplit_regressor import AMPLiTRegressor
    from retrain_amplit import featurize
    m = AMPLiTRegressor()
    m.to(DEVICE)
    Q1, Q2, Q3 = featurize(pep())
    Q1, Q2, Q3 = (torch.from_numpy(Q1), torch.from_numpy(Q2), torch.from_numpy(Q3))
    with torch.no_grad():  # build the lazy head/cbam before loading weights
        m(Q1[:1].to(DEVICE), Q2[:1].to(DEVICE), Q3[:1].to(DEVICE))
    ck = torch.load(OLD / 'amplit_regr_fix/best_B.pt', map_location='cpu')
    m.load_state_dict(ck['model'])
    m.eval()
    # variant-B recipe (retrain_amplit_regressor_fix.py): train labels are
    # z-scored on the train split; the submitted predictor inverts at inference
    MU, SD = [float(z) for z in ck['zscore']]

    def run():
        seqs = pep()
        Q1, Q2, Q3 = featurize(seqs)
        Q1, Q2, Q3 = (torch.from_numpy(Q1), torch.from_numpy(Q2), torch.from_numpy(Q3))
        preds = {}
        with torch.no_grad():
            for i in range(0, len(Q1), 256):
                y = m(Q1[i:i + 256].to(DEVICE), Q2[i:i + 256].to(DEVICE),
                      Q3[i:i + 256].to(DEVICE)).squeeze(-1).cpu().numpy()
                y = y * SD + MU                       # z-score destandardisation
                for j, s in enumerate(seqs[i:i + 256]):
                    preds[s] = float(y[j])
        pair_delta(preds)
    return timed(run)


def t_evo():
    EVO = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/baseline/AMP-potency-prediction-EvoGradient')
    sys.path.insert(0, str(EVO))
    import retrain_evo_strict as re_
    registry = {'CNN': re_.omr.CNN(), 'Transformer': re_.omr.Transformer(),
                'Attention': re_.AttentionNetwork(), 'LSTM': re_.omr.lstm_att()}
    models = {k: _load(v, OLD / f'evo_strict/{k}_seed123.pt') for k, v in registry.items()}
    W = re_.WEIGHTS                                     # released ensemble weights

    def run():
        P = pep()
        X = torch.from_numpy(np.stack([re_.encode(s)[0] for s in P]))
        M = torch.from_numpy(np.stack([re_.encode(s)[1] for s in P]))
        y_uM = np.zeros(len(P), dtype=np.float64)
        with torch.no_grad():
            for i in range(0, len(P), 256):
                x, msk = X[i:i + 256].to(DEVICE), M[i:i + 256].to(DEVICE)
                for k, m in models.items():
                    o = m(x, msk) if k == 'Attention' else m(x)
                    y_uM[i:i + len(x)] += float(W[k]) * o.squeeze(-1).cpu().numpy()
        # per-query label inversion: log10(uM) -> log10(ug/mL) via molecular mass
        mic = np.array([y_uM[j] + np.log10(re_.mw(s) / 1000.0) for j, s in enumerate(P)])
        pair_delta({s: float(mic[j]) for j, s in enumerate(P)})
    return timed(run)


def t_apex():
    sys.path.insert(0, str(SOTA / 'apex-main'))
    from AMP_DL_model_twohead import AMP_model
    from utils import make_vocab, AAindex, onehot_encoding
    word2idx, _ = make_vocab()
    emb, _ = AAindex(str(SOTA / 'apex-main/aaindex1.csv'), word2idx)   # table = runtime asset
    m = _load(AMP_model(emb, emb.shape[1], 2, 128, 2048, 4, 1), OLD / 'apex_neg/apexlr_123.pt')

    def run():
        seqs = pep()
        xt = torch.from_numpy(onehot_encoding(seqs, 52, word2idx)[0]).long().to(DEVICE)
        preds = {}
        with torch.no_grad():
            for i in range(0, len(xt), 64):
                y = (6.0 - m(xt[i:i + 64])[:, 0]).cpu().numpy()   # activity -> log10MIC
                for j, s in enumerate(seqs[i:i + 64]):
                    preds[s] = float(y[j])
        pair_delta(preds)
    return timed(run)


def t_ampd():
    from transformers import BertTokenizer, GPT2Config, GPT2LMHeadModel
    sys.path.insert(0, '/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/baseline/sota_bench/AMP-Designer')
    import train_ampd_restricted_neg as an
    W = SOTA / 'models/amp_designer'
    tok = BertTokenizer(vocab_file=str(W / 'vocab.txt'))
    cfg = GPT2Config.from_pretrained(str(W / 'config.json'))
    gpt = GPT2LMHeadModel.from_pretrained(str(W / 'pytorch_model.bin'), config=cfg)
    m = an.CombinedModel(gpt)
    m.load_state_dict(torch.load(OLD / 'amp_designer_with_neg/ampd_123.pt', map_location='cpu')['model'])
    m.to(DEVICE).eval()
    from torch.utils.data import DataLoader

    def run():
        seqs = pep()
        ds = an.SeqDS(seqs, np.zeros(len(seqs)), tok)   # tokenization inside
        loader = DataLoader(ds, batch_size=64, collate_fn=an.collate)
        preds, k = {}, 0
        with torch.no_grad():
            for xb, _ in loader:
                y = m(xb.to(DEVICE)).squeeze(-1).cpu().numpy()
                for j in range(len(y)):
                    preds[seqs[k]] = float(y[j]); k += 1
        pair_delta(preds)
    return timed(run)


METHODS = {
    'Das2021-regr': t_das,
    'LLAMP+neg': t_llamp,
    'BERT-AmPEP+neg': t_bert_ampep,
    'Deep-PIE+neg': t_deep_pie,
    'HydrAMP-arch-regr': t_hydramp,
    'AMPLiT-regr vB': t_amplit,
    'EvoGrad-strict': t_evo,
    'DeepAMP-CNN': t_deep_amp,
    'APEX-restr(lrfix)+neg': t_apex,
    'AMP-Designer+neg': t_ampd,
}


def main():
    result = {'n_peptides': 208,
              'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
              'batch_note': 'unified strict boundary (2026-10-10): per-query tokenization/'
                            'encoding/embedding + forward + label inversion INSIDE the timed '
                            'region; sequence list re-read per repeat; one-time model/runtime '
                            'loading excluded; 1 warm-up + 3 repeats, median',
              'methods': {},
              'checkpoint_records': {
                  'Das2021-regr': {'ckpt': str(OLD / 'das_regr/bilstm_regr_best.pt'), 'seed': 129,
                                  'note': 'verified (v13 acceptance): reproduces archived '
                                          'seed-129 predictions exactly'},
                  'LLAMP+neg': {'ckpt': str(OLD / 'llamp_with_neg/best.pt'), 'seed': 129,
                                'sha256_16': sha256(OLD / 'llamp_with_neg/best.pt')},
                  'BERT-AmPEP+neg': {'ckpt': str(OLD / 'bert_ampep_with_neg/best.pt'), 'seed': 129,
                                     'sha256_16': sha256(OLD / 'bert_ampep_with_neg/best.pt')},
                  'Deep-PIE+neg': {'ckpt': str(OLD / 'deep_pie_with_neg/best.pt'), 'seed': 129,
                                   'sha256_16': sha256(OLD / 'deep_pie_with_neg/best.pt')},
                  'HydrAMP-arch-regr': {'ckpt': str(OLD / 'hydramp_regr_neg/fold1-10.pt'),
                                        'seed': 129,
                                        'note': '10-fold ensemble verified (v13 acceptance) '
                                                'to reproduce archived seed-129 predictions '
                                                'exactly via fold-mean raw logit'},
                  'AMPLiT-regr vB': {'ckpt': str(OLD / 'amplit_regr_fix/best_B.pt'), 'seed': 129,
                                     'note': 'verified (v13 acceptance): exact reproduction '
                                             'after z-score destandardisation (ckpt stores '
                                             '[mu, sd])'},
                  'note': 'BERT/LLAMP/Deep-PIE/Das/HydrAMP-arch/AMPLiT best.pt verified '
                          '(v13 acceptance harness) to reproduce archived predictions '
                          'exactly; DeepAMP/APEX/AMPD verified at aggregate macro-PCC level '
                          '(seed 123); recorded here instead of nominal seed-123 claims.'}}
    for name, fn in METHODS.items():
        try:
            r = fn()
            result['methods'][name] = r
            print(f'[{name}] median {r["median_s"]}s  {r["repeats_s"]}', flush=True)
        except Exception as e:  # isolate failures
            result['methods'][name] = {'error': f'{type(e).__name__}: {e}'}
            print(f'[{name}] ERROR {type(e).__name__}: {e}', flush=True)
        torch.cuda.empty_cache()
    OUT.write_text(json.dumps(result, indent=2))
    print(f'saved {OUT}', flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Per-seed (single-model, NO ensembling) macro PCC for the ensemble-protocol
methods (experiment 12b supplement): APEX+neg, DeepAMP-CNN/RNN, AMP-Designer+neg.

Re-infers the saved per-seed checkpoints on the 208 external peptides and
writes sota_<method>_perseed.csv (seed -> macro PCC). EvoGrad-strict already
has per-seed preds (sota_models/evo_strict/preds_seed*.csv).

Usage: CUDA_VISIBLE_DEVICES=0 python perseed_ensemble_members.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
TIGER = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC')
BR = TIGER / 'runs_review/benchmark_restricted'
sys.path.insert(0, str(HERE))

A0 = pd.read_csv(BR / 'eleven_way_predictions.csv')
YT = 'y_true_delta_log2_anchor_minus_query_03'
ORDER = ['Mammuthusin-3', 'F', 'Hesperelin-3', 'Hydrodamin-3', 'Hydrodamin-2',
         'Repeat', 'HAL-2', 'LL37', 'Mammuthusin-2', 'Arctoterin-1', 'HAL-1']
PEP = list(dict.fromkeys(pd.read_csv(TIGER / 'runs_review/final11_peptides.csv').sequence))


def macro(d):
    delta = [(d.get(a, np.nan) - d.get(q, np.nan)) / math.log10(2)
             if a in d and q in d else np.nan
             for a, q in zip(A0.anchor_sequence, A0.query_sequence)]
    A0['_t'] = delta
    vs = []
    for f in ORDER:
        r = f + '/' if f.startswith('HAL') else f
        g = A0[A0.family == r].dropna(subset=['_t'])
        vs.append(np.corrcoef(g[YT], g['_t'])[0, 1])
    return round(float(np.nanmean(vs)), 4)


def eval_deep_amp(device):
    import retrain_deep_amp as da
    out = {}
    for arch, cls in [('cnn', da.CNNRegressor), ('rnn', da.RNNRegressor)]:
        X, lens = da.encode(PEP)[:2]
        x = torch.from_numpy(X[:len(PEP)]).float().to(device)
        for s in range(42, 47):
            state = torch.load(BR / f'sota_models/deep_amp_ensemble/{arch}_seed{s}.pt',
                               map_location='cpu')
            m = cls().to(device)
            m.load_state_dict(state['model'])
            m.eval()
            with torch.no_grad():
                p = m(x).squeeze(-1).cpu().numpy()
            out[(arch, s)] = macro(dict(zip(PEP, p.tolist())))
            print(f'DeepAMP-{arch.upper()} seed{s}: {out[(arch, s)]}', flush=True)
    return out


def eval_apex(device):
    sys.path.insert(0, str(HERE / 'apex-main'))
    from AMP_DL_model_twohead import AMP_model
    from utils import make_vocab, AAindex, onehot_encoding
    word2idx, _ = make_vocab()
    emb, _ = AAindex(str(HERE / 'apex-main' / 'aaindex1.csv'), word2idx)
    x, _, _ = onehot_encoding(PEP, 52, word2idx)
    xt = torch.from_numpy(x).long().to(device)
    out = {}
    for rep in range(5):
        state = torch.load(BR / f'sota_models/apex_neg/apexlr_{rep}.pt', map_location='cpu')
        m = AMP_model(emb, emb.shape[1], 2, 128, 2048, 4, 1).to(device)
        m.load_state_dict(state['model'])
        m.eval()
        with torch.no_grad():
            p = m(xt).squeeze(-1).cpu().numpy()
        d = dict(zip(PEP, (6.0 - p).tolist()))  # y-scale -> log10MIC
        out[rep] = macro(d)
        print(f'APEX+neg seed(rep{rep}): {out[rep]}', flush=True)
    return out


def eval_ampd(device):
    from transformers import BertTokenizer, GPT2Config, GPT2LMHeadModel
    sys.path.insert(0, str(HERE / 'AMP-Designer'))
    import train_ampd_restricted_neg as an
    W = HERE / 'models' / 'amp_designer'
    tok = BertTokenizer(vocab_file=str(W / 'vocab.txt'))
    cfg = GPT2Config.from_pretrained(str(W / 'config.json'))
    ds = an.SeqDS(PEP, np.zeros(len(PEP)), tok)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=64, collate_fn=an.collate)
    out = {}
    for rep in range(5):
        state = torch.load(BR / f'sota_models/amp_designer_with_neg/ampd_{rep}.pt',
                           map_location='cpu')
        gpt = GPT2LMHeadModel.from_pretrained(str(W / 'pytorch_model.bin'), config=cfg)
        m = an.CombinedModel(gpt).to(device)
        m.load_state_dict(state['model'])
        m.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                preds.extend(m(xb.to(device)).cpu().tolist())
        out[rep] = macro(dict(zip(PEP, preds)))
        print(f'AMP-Designer+neg seed(rep{rep}): {out[rep]}', flush=True)
    return out


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    res = {}
    da = eval_deep_amp(device)
    res.update({f'DeepAMP-CNN_s{s}': v for (a, s), v in da.items() if a == 'cnn'})
    res.update({f'DeepAMP-RNN_s{s}': v for (a, s), v in da.items() if a == 'rnn'})
    res.update({f'APEX+neg_s{r}': v for r, v in eval_apex(device).items()})
    res.update({f'AMPD+neg_s{r}': v for r, v in eval_ampd(device).items()})
    rows = [{'method': k.rsplit('_s', 1)[0], 'seed': k.rsplit('_s', 1)[1], 'macro_pcc': v}
            for k, v in res.items()]
    pd.DataFrame(rows).to_csv(BR / 'ensemble_members_perseed.csv', index=False)
    print('saved ensemble_members_perseed.csv')


if __name__ == '__main__':
    main()

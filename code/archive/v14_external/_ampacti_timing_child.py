#!/usr/bin/env python3
"""Child process for eval_cost_timing_new7.py t_ampacti (runs in `ampacti` env).

Loads the deep-forest seed-123 model once (one-time cost, protocol-excluded),
prints MODEL_READY, then does warm-up + 3 timed repeats, each consisting of:
208-peptide feature construction + deep-forest prediction + the UNIFIED pair
endpoint (ordered 2,211 primary-panel edges from pair_keys_primary_2211.csv,
LOG2 anchor-minus-query deltas) — [v13b] the differencing is INSIDE the timed
region, in this child, not in the already-stopped parent. Prints per-repeat
seconds as JSON.

Optional argv: <dump_preds_csv> <dump_pair_npy> — after timing, dump the last
repeat's per-peptide predictions and pair vector for the acceptance harness.

Kept as a file (not a -c string) to dodge nested-quote breakage.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SB = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC/baseline/sota_bench')
BR = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC/runs_review/benchmark_restricted')
sys.path.insert(0, str(SB / 'AMPActiPred'))
os.chdir(SB / 'AMPActiPred')
from predict_lib import construct_features  # noqa: E402
import joblib  # noqa: E402

IN_CSV = SB / 'emb_cache/ampacti_timing_in.csv'
PAIR_KEYS = BR / 'pair_keys_primary_2211.csv'
CK = BR / 'sota_models/ampacti_neg/df_s123.joblib'
dump_preds = sys.argv[1] if len(sys.argv) > 1 else None
dump_pair = sys.argv[2] if len(sys.argv) > 2 else None

if not CK.exists():
    # seed-123 refit on the cached corpus features (same recipe as the driver)
    from deepforest import CascadeForestRegressor
    ftr = pd.read_csv(SB / 'emb_cache/ampacti_feat_corpus.csv')
    tr = pd.read_csv(BR / 'train_restricted4929_tiger.csv')
    pos = tr.groupby('sequence', as_index=False).MIC_Escherichia_coli.mean()
    neg = pd.read_csv(SB / 'data_restricted/uniprot_negatives.csv')[['sequence']]
    neg['MIC_Escherichia_coli'] = 10000.0
    corpus = pd.concat([pos, neg], ignore_index=True)
    assert len(ftr) == len(corpus), f'{len(ftr)} vs {len(corpus)}'
    xcols = [c for c in ftr.columns if c not in ('Sequence', 'Id')]
    y = np.log10(corpus.MIC_Escherichia_coli.values.astype(float))
    np.random.seed(123)
    m = CascadeForestRegressor(random_state=123)
    try:
        m.set_random_state(123)
    except Exception:
        pass
    m.fit(ftr[xcols].values, y)
    CK.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(m, CK)

m = joblib.load(CK)
xcols = [c for c in pd.read_csv(SB / 'emb_cache/ampacti_feat_corpus.csv', nrows=2).columns
         if c not in ('Sequence', 'Id')]
# one-time (protocol-excluded, same convention as TIGER's pair tables): the
# ordered primary-panel edge list; the differencing itself is timed, below
_edges = pd.read_csv(PAIR_KEYS)
E = list(zip(_edges.anchor_sequence, _edges.query_sequence))
assert len(E) == 2211
print('MODEL_READY', flush=True)

ts = []
preds = None
pair = None
for rep in range(4):
    t0 = time.perf_counter()
    d = pd.read_csv(IN_CSV)[['Id', 'Sequence']]   # 查询文件每轮现读，不预载
    feats = construct_features(d)
    y = np.asarray(m.predict(feats[xcols].values)).reshape(-1)
    preds = dict(zip([s.upper().strip() for s in d.Sequence], y))
    pair = np.array([(preds.get(a, np.nan) - preds.get(b, np.nan)) / np.log10(2)
                     for a, b in E])
    assert np.isfinite(pair).all(), 'primary-panel endpoint missing'
    if rep > 0:
        ts.append(round(time.perf_counter() - t0, 4))
print('TIMES ' + json.dumps(ts), flush=True)
if dump_preds:
    pd.Series(preds, name='log10mic').to_csv(dump_preds)
    np.save(dump_pair, pair)
    print('DUMPED', flush=True)

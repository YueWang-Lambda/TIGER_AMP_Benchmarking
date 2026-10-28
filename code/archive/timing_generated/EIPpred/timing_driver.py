#!/usr/bin/env python3
"""Inference-cost timing for the seven v2-9 retrained additions (NEW8 minus
cancelled AMPCliff): EIPpred / AMPActiPred / AMPGen / AMPpredictor / ANIA /
tsAMP-CS / GGN-ABPMIC.

2026-10-10 v12-audit rewrite. Unified strict boundary (user rule):
  timed region = per-query feature computation (frozen PLM forwards,
  descriptors, graph builds) + task-model forward + per-query label
  inversion, with the 208-sequence input re-read inside every repeat;
  excluded   = one-time model/runtime loading. Task networks now load the
  EVALUATED weights (seed-123 checkpoints), not fresh instances:
    AMPGen / tsAMP / AMPpredictor : sota_models/<m>_neg/task_s123.pt
    ([v2-12 patch] persisted by the drivers; AMPGen & tsAMP retrain
    reproduce the archived per-seed predictions byte-for-byte; the
    AMPpredictor retrain is same-recipe/same-architecture — its upstream
    run is not bit-reproducible, so the timing loads the retrained
    seed-123 model, latency-equivalent by architecture)
    GGN-ABPMIC : ggn_neg/s123/random_split1_best.pth (the exact
    accuracy-evaluated checkpoint)
    ANIA       : sota_ANIA+neg_s123.pt loaded through the repo's own
    ANIA constructor (inception + transformer attention), replacing the
    previous Conv1d stand-in; CGR encoding runs in a RESIDENT R
    subprocess (sentinel-synced) and the forward is timed together with
    it, not summed separately
    EIPpred    : descriptor computation in a RESIDENT feature subprocess
    (start-up excluded, per-query compute included) + pMIC->log10MIC
    sign inversion (labels are -log10 ug/mL)
  All GPU forwards are bracketed by torch.cuda.synchronize().

Output: runs_review/benchmark_restricted/new7_cost_timing.json
Usage: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/eval_cost_timing_new7.py [--only m1,m2]
"""
import argparse
import math
import os
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
SB = ROOT / 'baseline' / 'sota_bench'
sys.path.insert(0, str(SB))
BR = ROOT / 'runs_review/benchmark_restricted'
PEP_CSV = ROOT / 'runs_review/final11_peptides.csv'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def pep():
    """Fresh read of the 208 query sequences inside every timed repeat."""
    return pd.read_csv(PEP_CSV)[['sequence']].dropna().sequence.tolist()


import hashlib

# pair endpoint: the PRIMARY benchmark panel (v13b) — 11 families, 2,211
# edges whose endpoints are exactly the 208 test peptides (the full table
# spans 21 families / 2,748 edges; the extra 537 non-benchmark-family edges
# are NOT differenced). Canonical keys: pair_keys_primary_2211.csv.
# Differencing runs INSIDE the timed region; unit = LOG2, the evaluator's
# (pred_a - pred_q) / log10(2).
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
    ordered LOG2 deltas. All primary endpoints are covered, so a non-finite
    entry means a featuriser drop -> loud assertion."""
    d = np.array([(preds.get(a, np.nan) - preds.get(b, np.nan)) / math.log10(2)
                  for a, b in pair_edges()])
    assert np.isfinite(d).all(), 'primary-panel endpoint missing from preds'
    return d


def timed(fn):
    fn()  # warm-up
    ts = []
    for _ in range(3):
        if DEV == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if DEV == 'cuda':
            torch.cuda.synchronize()
        ts.append(round(time.perf_counter() - t0, 4))
    return {'repeats_s': ts, 'median_s': sorted(ts)[1]}


# ---------------- EIPpred (resident descriptor child + RF + inversion) ----
EIP_CHILD = r'''
import sys, os
sys.path.insert(0, r"{root}")
os.chdir(r"{root}")
# eippred.features_calulate redirects sys.stdout to descriptor output files
# and never restores it -> sentinel lines MUST bypass sys.stdout (file=REAL)
REAL = sys.stdout
import eippred
print("READY", file=REAL, flush=True)
while True:
    line = sys.stdin.readline()
    if not line or line.strip() == "QUIT":
        break
    if line.strip() == "GO":
        eippred.features_calulate(r"{fin}", r"{fout}")
        sys.stdout = REAL
        print("DONE", file=REAL, flush=True)
'''


def t_eippred():
    import joblib
    mrmr = pd.read_csv(SB / 'EIPpred/Data/selected_features_mrmr1000_new.csv')['SelectedFeatures'].tolist()
    ck = BR / 'sota_models/eippred_neg/model.joblib'
    if not ck.exists():
        from retrain_eippred_neg import corpus  # type: ignore
        ftr = pd.read_csv(SB / 'emb_cache/eippred_feat_corpus.csv')
        tr = corpus()
        from sklearn.ensemble import RandomForestRegressor
        clf = RandomForestRegressor(random_state=123, n_jobs=32).fit(
            ftr[mrmr].values, -np.log10(tr.MIC_Escherichia_coli.values.astype(float)))
        ck.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, ck)
    clf = joblib.load(ck)
    fin = SB / 'emb_cache/eippred_in_ext.txt'
    fout = SB / 'emb_cache/eippred_timing_ext.csv'
    fin.parent.mkdir(exist_ok=True)
    code = EIP_CHILD.format(root=str(SB / 'EIPpred'), fin=str(fin), fout=str(fout))
    child = subprocess.Popen([sys.executable, '-c', code], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             text=True, cwd=str(SB / 'EIPpred'))
    while child.stdout.readline().strip() != 'READY':
        pass                                            # drain eippred import chatter

    def drain_done():
        # eippred 打印进度行与哨兵交错 → 排水至标记（EOF 视为致命）
        while True:
            line = child.stdout.readline()
            if not line:
                raise RuntimeError('EIPpred child died')
            if line.strip() == 'DONE':
                return

    def run():
        fin.write_text('\n'.join(s.upper() for s in pep()) + '\n')   # input read
        child.stdin.write('GO\n'); child.stdin.flush()               # per-query features
        drain_done()
        f = pd.read_csv(fout)
        f = f.loc[:, ~f.columns.str.startswith('Unnamed')]
        y = clf.predict(f[mrmr].values)                              # pMIC = -log10
        mic = -y                                                     # -> log10 MIC
        pair_delta({s: float(mic[j]) for j, s in enumerate(pep())})
    try:
        return timed(run)
    finally:
        child.stdin.write('QUIT\n'); child.stdin.flush(); child.wait()


# ---------------- AMPActiPred (resident deep-forest child; v13b: pair stage
# ---------------- moved INSIDE the timed child) -----------------------------
def t_ampacti():
    env = '/home/ubuntu/anaconda3/envs/ampacti/bin/python'
    in_csv = SB / 'emb_cache/ampacti_timing_in.csv'
    # write the query file fresh (one-time, outside timing) so the child's
    # per-repeat reads see exactly the 208 external peptides
    pep_df = pd.read_csv(PEP_CSV)[['sequence']].dropna()
    in_csv.parent.mkdir(exist_ok=True)
    pd.DataFrame({'Id': [f'p{i}' for i in range(len(pep_df))],
                  'Sequence': [s.upper().strip() for s in pep_df.sequence]}
                 ).to_csv(in_csv, index=False)
    import subprocess as sp
    r = sp.run([env, str(ROOT / 'scripts/_ampacti_timing_child.py')],
               capture_output=True, text=True)
    assert 'MODEL_READY' in r.stdout, r.stderr[-300:]
    ts = json.loads([l for l in r.stdout.splitlines() if l.startswith('TIMES ')][-1][6:])
    # child (timed): input re-read per repeat + raw 553-dim descriptor
    # construction + deep-forest prediction + the unified 2,211-edge LOG2
    # pair endpoint (v13b; previously the pair stage was missing entirely)
    return {'repeats_s': ts, 'median_s': sorted(ts)[1]}


# ---------------- AMPGen (frozen esm2_3B + fitted scaler + ckpt LSTM) -----
def t_ampgen():
    import retrain_ampgen_neg as drv  # type: ignore
    import esm as esmlib
    ck = torch.load(BR / 'sota_models/ampgen_neg/task_s123.pt', map_location='cpu')
    model, _ = esmlib.pretrained.load_model_and_alphabet('esm2_t36_3B_UR50D')
    model = model.cuda().eval()
    bc = model.alphabet.get_batch_converter()
    lstm = drv.LSTMModel(2560, 128, 2, 1, 0.7).to(DEV)
    lstm.load_state_dict(ck['model']); lstm.eval()
    mean, scale = ck['sc_mean'], ck['sc_scale']

    def run():
        seqs = pep()
        outs = []
        with torch.no_grad():
            for i in range(0, len(seqs), 16):
                b = [(f'p{j}', s) for j, s in enumerate(seqs[i:i + 16])]
                _, _, toks = bc(b)
                r = model(toks.cuda(), repr_layers=[36], return_contacts=False)
                for j, s in enumerate(seqs[i:i + 16]):
                    reps = r['representations'][36][j, 1:len(s) + 1]   # residues only
                    outs.append(reps.mean(0).cpu().numpy())            # (v13: no BOS/EOS/pad)
        E = np.array(outs)
        X = (E - mean) / scale                                    # fitted feature scaler
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(DEV)
            y = lstm(xb).squeeze(-1).cpu().numpy()                 # log10 ug/mL direct
        pair_delta({s: float(y[j]) for j, s in enumerate(seqs)})
    return timed(run)


# ---------------- tsAMP (frozen esm1v + strain concat + ckpt MLP) ----------
def t_tsamp():
    import esm as esmlib
    import retrain_tsamp_neg as drv  # type: ignore
    model, alphabet = esmlib.pretrained.load_model_and_alphabet('esm1v_t33_650M_UR90S_1')
    model = model.cuda().eval()
    bc = alphabet.get_batch_converter()
    strain = torch.load(SB / 'tsAMP/data_strain/Ec_ATCC_25922.pt',
                        map_location='cpu')['mean_representations'][33].float()
    net = drv.TSAMPCS().to(DEV).eval()
    net.load_state_dict(torch.load(BR / 'sota_models/tsamp_neg/task_s123.pt',
                                   map_location='cpu')['model'])

    def run():
        seqs = pep()
        rows = []
        with torch.no_grad():
            for i in range(0, len(seqs), 16):
                b = [(f'p{j}', s) for j, s in enumerate(seqs[i:i + 16])]
                _, _, toks = bc(b)
                r = model(toks.cuda(), repr_layers=[33], return_contacts=False)
                for j, s in enumerate(seqs[i:i + 16]):
                    rows.append(r['representations'][33][j, 1:len(s) + 1].mean(0).cpu())
        embs = torch.stack(rows, 0)
        X = torch.cat((embs, strain.expand(len(embs), -1)), dim=1).to(DEV)
        with torch.no_grad():
            y = net(X).squeeze(-1).cpu().numpy()                  # log10 ug/mL direct
        pair_delta({s: float(y[j]) for j, s in enumerate(seqs)})
    return timed(run)


# ---------------- ANIA (resident R CGR child + real ANIA net) --------------
ANIA_CHILD = r'''
import sys, os
os.environ.setdefault("R_HOME", "{rhome}")
for p in ({paths}):
    sys.path.insert(0, p)
import pandas as pd
from src.inference.fasta_encoder import encode_fasta_to_cgr
REAL = sys.stdout                      # sentinel lines bypass any stdout redirect
print("READY", file=REAL, flush=True)
while True:
    line = sys.stdin.readline()
    if not line or line.strip() == "QUIT":
        break
    if line.strip() == "GO":
        seqs = pd.read_csv(r"{pepcsv}")[["sequence"]].dropna().sequence.tolist()
        with open("/tmp/ania_t.fasta", "w") as fh:
            for i, s in enumerate(seqs):
                fh.write(f">s{{i:05d}}\n{{s}}\n")
        encode_fasta_to_cgr(fasta_path="/tmp/ania_t.fasta",
                            output_csv_path="/tmp/ania_t.csv", resolution=16)
        sys.stdout = REAL
        print("DONE", file=REAL, flush=True)
'''


def t_ania():
    ANIA = SB / 'ANIA'
    child_code = ANIA_CHILD.format(
        rhome='/home/ubuntu/anaconda3/envs/ania/lib/R',
        pepcsv=str(PEP_CSV),
        paths=', '.join('r"%s"' % p for p in
                        (str(ANIA), str(ANIA / 'src/utils/logging_toolkit/src/python'),
                         str(ANIA / 'src'))))
    child = subprocess.Popen(['/home/ubuntu/anaconda3/envs/ania/bin/python', '-c', child_code],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, text=True)
    assert child.stdout.readline().strip() == 'READY'
    # real model: repo constructor + evaluated s123 weights (inception+attention)
    sys.path.insert(0, str(ANIA / 'src/utils/logging_toolkit/src/python'))
    sys.path.insert(0, str(ANIA / 'src'))
    sys.path.insert(0, str(ANIA))
    os.environ.setdefault('R_HOME', '/home/ubuntu/anaconda3/envs/ania/lib/R')
    from src.models.deep_learning.ANIA import ANIA  # noqa
    ck = torch.load(BR / 'sota_models/ania_neg/sota_ANIA+neg_s123.pt', map_location='cpu')
    hp = ck['hyperparams']
    i1 = sum(hp[k] for k in hp if k.startswith('inception1_') and k.endswith('_channels'))
    i2 = sum(hp[k] for k in hp if k.startswith('inception2_') and k.endswith('_channels'))
    net = ANIA(in_channels=11, inception1_out_channels=i1, inception2_out_channels=i2,
               **{k: v for k, v in hp.items()
                  if k in ANIA.__init__.__code__.co_varnames})
    net.load_state_dict(ck['state_dict'])
    net.to(DEV).eval()

    def run():
        child.stdin.write('GO\n'); child.stdin.flush()     # fresh CGR encode (R, resident)
        assert child.stdout.readline().strip() == 'DONE'
        df = pd.read_csv('/tmp/ania_t.csv')
        X = torch.tensor(df.iloc[:, 3:3 + 2816].to_numpy(dtype=np.float32)
                         ).reshape(len(df), 11, 16, 16).to(DEV)
        with torch.no_grad():
            y = net(X).squeeze(-1).cpu().numpy()           # log-MIC direct (driver domain)
        pair_delta({s: float(y[j]) for j, s in enumerate(pep())})
    try:
        return timed(run)
    finally:
        child.stdin.write('QUIT\n'); child.stdin.flush(); child.wait()


# ---------------- GGN (descriptors + masked ProtT5 + DSSP + build + GNN) -----
DESC_CHILD = r'''
import sys, os
sys.path.insert(0, r"{amprepo}")
os.chdir(r"{amprepo}")
from predict_lib import construct_features
import pandas as pd, json as _json, sys as _s
REAL = _s.stdout
print("READY", file=REAL, flush=True)
while True:
    line = _s.stdin.readline()
    if not line or line.strip() == "QUIT":
        break
    if line.strip() == "GO":
        d = pd.read_csv(r"{incsv}")
        feats = construct_features(d)
        feats.to_csv(r"{outcsv}", index=False)
        _s.stdout = REAL
        print("DONE", file=REAL, flush=True)
'''


def t_ggn():
    sys.path.insert(0, str(SB / 'GGN-ABPMIC'))
    os.environ['GGN_SEEDS'] = '1'
    os.environ.pop('GGN_GRAPH_CACHE', None)
    from myData import ProteinGraphDataset
    from model import AMPRegressor
    from my_features import _parse_dssp, _align_dssp, _transform_dssp
    from transformers import T5Tokenizer, T5EncoderModel
    import subprocess as sp
    WORK = SB / 'ggn_work'
    TMP = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC/artifacts/repair_v15_v14input/timing/EIPpred/ggn_live')                  # isolated (v13): never
    for sub in ('prottrans', 'dssp', 'dssp_tmp'):       # touch ggn_work caches
        (TMP / sub).mkdir(parents=True, exist_ok=True)
    DSSP_BIN = '/home/ubuntu/anaconda3/envs/dsspbio/bin/mkdssp'
    PT5 = '/data4T/ubuntu/wangyue/models/Prot-T5-XL-U50'
    tok = T5Tokenizer.from_pretrained(PT5, do_lower_case=False)
    enc = T5EncoderModel.from_pretrained(PT5).eval().cuda().half()
    net = AMPRegressor(device='cuda').cuda().eval()
    ck = torch.load(BR / 'sota_models/ggn_neg/s123/random_split1_best.pth',
                    map_location='cuda')
    net.load_state_dict(ck['model_state_dict'])              # evaluated weights
    # resident 553-dim descriptor child (AMPActiPred suite, ampacti env)
    in_csv = TMP / 'desc_in.csv'
    out_csv = TMP / 'desc_out.csv'
    child_code = DESC_CHILD.format(amprepo=str(SB / 'AMPActiPred'),
                                   incsv=str(in_csv), outcsv=str(out_csv))
    child = sp.Popen(['/home/ubuntu/anaconda3/envs/ampacti/bin/python', '-c', child_code],
                     stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.DEVNULL, text=True)
    assert child.stdout.readline().strip() == 'READY'
    desc_cols = [c for c in pd.read_csv(WORK / 'ext.csv', nrows=2).columns
                 if c not in ('sequence', 'value')]

    class A: pass
    a = A()
    a.train_csv = str(TMP / 'ext.csv'); a.val_csv = str(TMP / 'ext.csv')
    a.pdb_dir = str(WORK / 'pdb'); a.prottrans_dir = str(TMP / 'prottrans')
    a.dssp_dir = str(TMP / 'dssp'); a.device = 'cuda'
    from torch_geometric.loader import DataLoader

    def run():
        seqs = pep()
        # (1) 553-dim descriptors, live (resident child)
        pd.DataFrame({'Id': [f'p{i}' for i in range(len(seqs))],
                      'Sequence': [s.upper().strip() for s in seqs]}).to_csv(in_csv, index=False)
        child.stdin.write('GO\n'); child.stdin.flush()
        while child.stdout.readline().strip() != 'DONE':
            pass
        desc = pd.read_csv(out_csv)
        # (2) masked ProtT5 per-residue forward (v13 unified definition)
        with torch.no_grad():
            for i in range(0, len(seqs), 32):
                b = seqs[i:i + 32]
                to = tok([' '.join(s) for s in b], return_tensors='pt', padding=True)
                out = enc(input_ids=to.input_ids.cuda(),
                          attention_mask=to.attention_mask.cuda()).last_hidden_state
                for j, s in enumerate(b):
                    np.save(TMP / 'prottrans' / f'{s}_prottrans.npy',
                            out[j, :len(s)].cpu().numpy().astype(np.float32))
        # (3) DSSP serial + parse/align/transform
        for s in seqs:
            tp = TMP / 'dssp_tmp' / f'{s}.dssp'
            sp.run([DSSP_BIN, str(WORK / 'pdb' / f'{s}.pdb'), str(tp)],
                   check=True, capture_output=True)
            ds_, feat = _parse_dssp(str(tp))
            if s in ds_:
                feat = _align_dssp(ds_, feat, s)
            np.save(TMP / 'dssp' / f'{s}_dssp.npy', _transform_dssp(feat))
        # (4) temp ext csv (sequence+value+553 live descriptors) -> graphs -> GNN
        tab = pd.DataFrame({'sequence': seqs, 'value': [0.0] * len(seqs)})
        tab = tab.merge(desc.drop(columns=['Id', 'Sequence']), left_on='sequence',
                        right_on=None, how='left') if False else tab
        # merge by order: construct_features preserves input row order
        for c in desc_cols:
            tab[c] = desc[c].values
        tab.to_csv(TMP / 'ext.csv', index=False)
        ds__ = ProteinGraphDataset(a, mode='test')
        preds = {}
        k = 0
        with torch.no_grad():
            for batch in DataLoader(ds__, batch_size=16):
                y, _ = net(batch.to('cuda'))
                y = y.squeeze(-1).cpu().numpy()
                for j in range(len(y)):
                    preds[seqs[k]] = float(y[j]); k += 1
        pair_delta(preds)
    try:
        return timed(run)
    finally:
        child.stdin.write('QUIT\n'); child.stdin.flush(); child.wait()


# ---------------- AMPpredictor (frozen ESM-1b + contacts + fp + ckpt GCN) --
def t_apppred():
    import retrain_amppredictor_neg as drv  # type: ignore
    import esm as esmlib
    import json as _json
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from torch_geometric.data import Data, Batch
    SEQ_MAX = 65
    model, alphabet = esmlib.pretrained.load_model_and_alphabet('esm1b_t33_650M_UR50S')
    model = model.cuda().eval()
    bc = alphabet.get_batch_converter()
    net = drv.GNNPredictor().to(DEV).eval()
    net.load_state_dict(torch.load(BR / 'sota_models/amppredictor_neg/task_s123.pt',
                                   map_location='cpu')['model'])
    smiles = _json.load(open(drv.REPO / 'data/smiles_file.json'))

    def run():
        SEQS = pep()
        feats, conts = {}, {}
        with torch.no_grad():
            for i in range(0, len(SEQS), 16):
                b = [(f'p{j}', s) for j, s in enumerate(SEQS[i:i + 16])]
                _, _, toks = bc(b)
                r = model(toks.cuda(), repr_layers=[33], return_contacts=True)
                for j, s in enumerate(SEQS[i:i + 16]):
                    feats[s] = r['representations'][33][j, 1:len(s) + 1].cpu()
                    conts[s] = r['contacts'][j, :len(s), :len(s)].cpu()
        lookup = {k: np.array(rdMolDescriptors.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(v), radius=3, nBits=2048)) for k, v in smiles.items()}
        lookup[' '] = np.zeros(2048, dtype=int)
        lookup['0'] = np.zeros(2048, dtype=int)
        gs, fs = [], []
        for s in SEQS:
            feat = feats[s]; L = feat.shape[0]
            x = torch.zeros(SEQ_MAX, 1280); x[:L] = feat
            c = torch.zeros(SEQ_MAX, SEQ_MAX); c[:L, :L] = conts[s]
            c += torch.eye(SEQ_MAX)
            gs.append(Data(x=x, edge_index=torch.nonzero(c >= 0.5).t().contiguous()))
            fp = np.asarray([lookup[ch] for ch in s])
            fp = np.concatenate([fp, np.zeros((SEQ_MAX - len(s), 2048))], 0)
            fs.append(torch.tensor(fp.mean(axis=1), dtype=torch.float32))
        fT = torch.stack(fs)
        preds = {}
        with torch.no_grad():
            for i in range(0, len(gs), 128):
                bg = Batch.from_data_list(gs[i:i + 128]).to(DEV)
                y = net(bg, fT[i:i + 128].to(DEV)).squeeze(-1).cpu().numpy()
                for j in range(len(y)):
                    preds[SEQS[i + j]] = float(y[j])
        pair_delta(preds)
    return timed(run)


TIMERS = {'EIPpred+neg': t_eippred, 'AMPActiPred+neg': t_ampacti,
          'AMPGen+neg': t_ampgen, 'AMPpredictor+neg': t_apppred,
          'ANIA+neg': t_ania, 'tsAMP-CS+neg': t_tsamp, 'GGN-ABPMIC+neg': t_ggn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None)
    args = ap.parse_args()
    only = args.only.split(',') if args.only else list(TIMERS)
    out = {'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
           'n_peptides': 208,
           'batch_note': 'unified strict boundary (2026-10-10 v12 fix): evaluated seed-123 '
                         'task weights loaded (AMPGen/tsAMP/AMPpredictor task_s123.pt, GGN '
                         'random_split1_best.pth, ANIA s123 through the repo constructor, '
                         'EIPpred rf_s123); per-query features + forward + label inversion '
                         'inside; EIPpred/ANIA feature subprocesses RESIDENT (start-up '
                         'excluded, per-query compute inside); inputs re-read per repeat; '
                         'GPU forwards cuda.synchronize-bracketed; 1 warm-up + 3 repeats, '
                         'median. AMPpredictor note: upstream training is not bit-'
                         'reproducible; task_s123.pt is a same-recipe retrain '
                         '(latency-equivalent by architecture).',
           'methods': {}}
    for name in only:
        print(f'=== timing {name} ===', flush=True)
        try:
            out['methods'][name] = TIMERS[name]()
            print(f"[{name}] median {out['methods'][name]['median_s']}s", flush=True)
        except Exception as e:
            out['methods'][name] = {'error': str(e)[:300]}
            print(f'[{name}] ERROR {e}', flush=True)
    (BR / 'new7_cost_timing.json').write_text(json.dumps(out, indent=2))
    print('saved new7_cost_timing.json')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""ANIA (SilverGojo4/ANIA, Brief Bioinform 2026; CGR + Inception + Transformer
attention E. coli MIC regressor) retrained under the unified "+neg" protocol (v2-9).

Fidelity notes (from repo source read; all adaptations documented):
- Features = the repo's own CGR encoder: R kaos::cgr(seq_base="AMINO", res=16)
  FCGR counts (256, flattened order="F") + 10 AAindex property channels
  (ARGP820101, CHAM830107, FAUJ880103, GRAR740102, JANJ780101, KYTJ820101,
  NAKH920104, ROSM880102, WERD780104, ZIMJ680101; pixel value = mean over the
  unique 3-mers mapped to that pixel of the summed residue property, flattened
  order="C") = 11x16x16. The train-side encoder (src/features/cgr_encoding.py
  encode_fasta_to_cgr_multi) and the inference-side encoder
  (src/inference/fasta_encoder.py encode_fasta_to_cgr) are mathematically
  identical (same kaos call, same k=3 pixel mapping, same property list/order,
  same flatten orders); we run the inference-side encoder ONCE over corpus+ext
  (single rpy2/R session) and reuse the cache for both training and prediction
  -> no train/test feature skew, no per-sequence R subprocess overhead.
- Train CSV replicates the exact upstream column contract consumed by
  src/models/deep_learning/train.py (hardcoded start_idx=7, end_idx=2822):
  ID, Sequence, Targets, Sequence Length, Molecular Weight, Log MIC Value,
  MIC Group, then the 2816 CGR(resolution=16)|i-j / PROP(pid)|i-j columns in
  upstream order; features go back in through the repo's own
  extract_cgr_features_and_target_for_dl() (which asserts the 11x16x16 dim).
  Sequence Length/Molecular Weight/MIC Group are dummy values (never read by
  the DL stage; only ID/Sequence/Targets metadata + target + [7:2823] matter).
- Training = repo's train_ania() verbatim: GridSearch over the published
  configs/dl_hyperparameters.json (2x2x2x3 = 24 combos), AdamW, MSE, AMP
  autocast + GradScaler, ReduceLROnPlateau(factor .5, patience 5), early-stop
  patience 10, best combo by best val loss, final-state weights saved with
  hyperparams + loss curves in the checkpoint.
- [v2-9 patch] src/models/deep_learning/train.py (2 minimal patches, marked
  in-file): (1) the DL path had NO seeding anywhere (main.py --random_state
  only reaches the ML stage) -> python/np/torch RNGs seeded from env ANIA_SEED
  (this driver exports the protocol seed; default 42 keeps upstream behavior);
  (2) the val split took the first 80% / last 20% of rows in CSV row order ->
  rows shuffled by a seeded permutation before torch.split.
- Label domain: native 'Log MIC Value' = log10(uM); the unified protocol
  redefines the column as log10(ug/mL): positives = groupby(sequence).mean()
  of MIC_Escherichia_coli in train_restricted4929_tiger.csv (4,929), negatives
  = uniprot_negatives.csv at MIC=10000 ug/mL (7,962, log10=4.0), corpus rows
  12,891 (13 pos/neg overlaps kept, tsAMP precedent). Upstream DBAASP
  z-score filtering + min-aggregation are superseded by this corpus
  construction (labels = per-sequence MEAN, protocol definition). Ext-208 =
  final11_peptides.csv (MIC_E_coli_ug_mL, dropna).
- Released weights/ANIA_EC.pt (and SA/PA) never loaded; a fresh model is
  trained per seed 120-129.
- Env restoration (not code changes): the empty logging_toolkit submodule
  restored at its pinned commit a35c446; the kaos R package shipped in-repo
  (src/utils/kaos) installed into the ania env's R library; R_HOME must be
  set before rpy2 import (conda r-base lives inside the env).

Usage:
  python retrain_ania_neg.py --features [--limit N]    # one-time CGR cache (corpus+ext)
  python retrain_ania_neg.py --seed 120 [--smoke]      # train + infer -> preds CSV
Run with /home/ubuntu/anaconda3/envs/ania/bin/python; GPU via CUDA_VISIBLE_DEVICES.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ANIA = HERE / 'ANIA'
V2 = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC/artifacts/repair_v15/external_runs/ANIA/seed129/workspace')
OUT = V2 / 'runs_review/benchmark_restricted/sota_models/ania_neg'
OUT_NONEG = V2 / 'runs_review/benchmark_restricted/sota_models/ania_noneg'
CACHE = HERE / 'ania_cache'

RES = 16
NFEAT = 11 * RES * RES                      # 2816
START_IDX, END_IDX = 7, 2822                # upstream hard contract (dl/train.py)
PROPERTY_IDS = ['ARGP820101', 'CHAM830107', 'FAUJ880103', 'GRAR740102',
                'JANJ780101', 'KYTJ820101', 'NAKH920104', 'ROSM880102',
                'WERD780104', 'ZIMJ680101']


def setup_ania_paths():
    """Make the ANIA repo importable (repo root + src + logging toolkit)."""
    os.environ.setdefault('R_HOME', '/home/ubuntu/anaconda3/envs/ania/lib/R')
    for p in (str(ANIA),
              str(ANIA / 'src/utils/logging_toolkit/src/python'),
              str(ANIA / 'src')):
        if p not in sys.path:
            sys.path.insert(0, p)


def build_corpus():
    """Unified +neg corpus: 4,929 positives (per-sequence MEAN) + 7,962 negatives."""
    tr = pd.read_csv(V2 / 'runs_review/benchmark_restricted/train_restricted4929_tiger.csv')
    pos = tr.groupby('sequence', as_index=False).MIC_Escherichia_coli.mean()
    neg = pd.read_csv(HERE / 'data_restricted/uniprot_negatives.csv')[['sequence']]
    neg['MIC_Escherichia_coli'] = 10000.0
    corpus = pd.concat([pos, neg], ignore_index=True)
    assert corpus.sequence.map(lambda s: s == s.upper()).all(), 'lowercase sequence found'
    return corpus


def build_ext():
    """External test 208 = final11_peptides.csv (dropna on MIC, tsAMP precedent)."""
    ext = pd.read_csv(V2 / 'runs_review/final11_peptides.csv')[
        ['sequence', 'MIC_E_coli_ug_mL']].dropna()
    assert ext.sequence.map(lambda s: s == s.upper()).all(), 'lowercase sequence found'
    return ext


def feature_columns():
    """Exact upstream column names/order of the 2816 CGR+PROP block."""
    cols = [f'CGR(resolution={RES}) | {i + 1}-{j + 1}'
            for i in range(RES) for j in range(RES)]
    for pid in PROPERTY_IDS:
        cols += [f'PROP({pid}) | {i + 1}-{j + 1}'
                 for i in range(RES) for j in range(RES)]
    return cols


# ============================== --features stage ==============================
def run_features(limit=None):
    setup_ania_paths()
    from src.inference.fasta_encoder import encode_fasta_to_cgr

    CACHE.mkdir(parents=True, exist_ok=True)
    corpus, ext = build_corpus(), build_ext()
    seqs = list(dict.fromkeys(corpus.sequence.tolist() + ext.sequence.tolist()))
    if limit:
        seqs = seqs[:limit]
    print(f'[features] {len(seqs)} unique sequences '
          f'(corpus rows {len(corpus)}, ext {len(ext)})', flush=True)

    fasta = CACHE / 'cgr_all.fasta'
    with open(fasta, 'w') as fh:
        for i, s in enumerate(seqs):
            fh.write(f'>s{i:05d}\n{s}\n')

    t0 = time.time()
    encode_fasta_to_cgr(fasta_path=str(fasta),
                        output_csv_path=str(CACHE / 'cgr_all.csv'),
                        resolution=RES)
    dt = time.time() - t0
    print(f'[features] CGR encoding done in {dt:.1f}s '
          f'({dt / max(len(seqs), 1) * 1000:.1f} ms/seq)', flush=True)

    df = pd.read_csv(CACHE / 'cgr_all.csv')
    # encoder normalizes to upper + drops non-canonical; all inputs are canonical
    assert len(df) == len(seqs), f'row mismatch: {len(df)} vs {len(seqs)}'
    assert df['Sequence'].tolist() == seqs, 'sequence order mismatch after merge'
    feats = df.iloc[:, 3:3 + NFEAT].to_numpy(dtype=np.float32)
    assert np.isfinite(feats).all(), 'non-finite CGR features'
    np.savez(CACHE / 'cgr_features.npz',
             sequences=np.array(seqs), features=feats)
    print(f'[features] cache saved: {CACHE}/cgr_features.npz {feats.shape}', flush=True)

    if limit:  # subset path-validation run: stop before the big CSVs
        return

    # corpus CSV in the exact upstream layout consumed by dl/train.py
    feat_by_seq = dict(zip(seqs, feats))
    labels = np.log10(corpus.MIC_Escherichia_coli.values.astype(float))
    prefix = pd.DataFrame({
        'ID': [f'c{i:05d}' for i in range(len(corpus))],
        'Sequence': corpus.sequence.values,
        'Targets': 'Escherichia coli',
        'Sequence Length': corpus.sequence.str.len().values,
        'Molecular Weight': 0.0,          # dummy: never read by the DL stage
        'Log MIC Value': labels,          # log10(ug/mL), protocol redefinition
        'MIC Group': 'medium',            # dummy: never read by the DL stage
    })
    fdf = pd.DataFrame(np.stack([feat_by_seq[s] for s in corpus.sequence]),
                       columns=feature_columns())
    out = pd.concat([prefix, fdf], axis=1)
    assert list(out.columns[START_IDX:END_IDX + 1]) == feature_columns()
    out.to_csv(CACHE / 'corpus_ania.csv', index=False)
    print(f'[features] corpus CSV: {CACHE}/corpus_ania.csv '
          f'{out.shape}, labels [{labels.min():.3f},{labels.max():.3f}]', flush=True)

    # ext CSV in the exact layout consumed by inference/ania_predictor.py
    eprefix = pd.DataFrame({
        'ID': [f'e{i:05d}' for i in range(len(ext))],
        'Sequence': ext.sequence.values,
        'Sequence Length': ext.sequence.str.len().values,
    })
    edf = pd.DataFrame(np.stack([feat_by_seq[s] for s in ext.sequence]),
                       columns=feature_columns())
    pd.concat([eprefix, edf], axis=1).to_csv(CACHE / 'ext_ania.csv', index=False)
    print(f'[features] ext CSV: {CACHE}/ext_ania.csv ({len(ext)} rows)', flush=True)


# ============================== --seed stage ==============================
def run_seed(args):
    setup_ania_paths()
    import torch
    from setup_logging import setup_logging
    from src.models.deep_learning.train import train_ania
    from src.models.deep_learning.utils import extract_cgr_features_and_target_for_dl
    from src.inference.ania_predictor import predict_ania_from_csv

    seed = args.seed
    os.environ['ANIA_SEED'] = str(seed)   # consumed by the [v2-9 patch] in dl/train.py
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    for f in ('corpus_ania.csv', 'ext_ania.csv'):
        if not (CACHE / f).exists():
            raise FileNotFoundError(f'{CACHE}/{f} missing -- run --features first')

    out_dir = OUT_NONEG if args.no_neg else OUT
    corpus_csv = CACHE / 'corpus_ania.csv'
    if args.no_neg:
        # positives-only corpus CSV, filtered from the cached +neg corpus
        tr = pd.read_csv(V2 / 'runs_review/benchmark_restricted/train_restricted4929_tiger.csv')
        pos = set(tr.groupby('sequence').MIC_Escherichia_coli.mean().index)
        full = pd.read_csv(CACHE / 'corpus_ania.csv')
        sub = full[full.Sequence.isin(pos)].reset_index(drop=True)
        # 13 sequences appear in BOTH pools (corpus() concat kept both rows);
        # the positive-label row comes first -> keep='first' retains it
        sub = sub.drop_duplicates('Sequence', keep='first').reset_index(drop=True)
        assert len(sub) == 4929, len(sub)
        corpus_csv = CACHE / 'corpus_ania_noneg.csv'
        sub.to_csv(corpus_csv, index=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'logs').mkdir(exist_ok=True)
    logger = setup_logging(
        input_config_file=str(ANIA / 'configs/general_logging.json'),
        output_log_path=str(out_dir / f'logs/train_s{seed}.log'),
        logger_name='general_logger',
        handler_name='general',
    )

    # features -> (N, 11, 16, 16) via the repo's own extractor (asserts dims)
    _, X, y = extract_cgr_features_and_target_for_dl(
        file_path=str(corpus_csv),
        metadata_columns=['ID', 'Sequence', 'Targets'],
        target_column='Log MIC Value',
        feature_start_idx=START_IDX,
        feature_end_idx=END_IDX,
        height=RES,
        width=RES,
    )
    print(f'[seed {seed}] corpus tensors: X {tuple(X.shape)}, '
          f'y [{y.min().item():.3f},{y.max().item():.3f}] '
          f'({int((y == 4.0).sum())} negatives @ log10=4.0)', flush=True)

    # hyperparameter config: upstream grid verbatim, unless --smoke/--epochs
    hp_path = ANIA / 'configs/dl_hyperparameters.json'
    if args.smoke or args.epochs:
        cfg = json.loads(hp_path.read_text())['ania']
        if args.smoke:
            for k in ('batch_size', 'learning_rate', 'weight_decay', 'dropout_rate'):
                cfg[k] = [cfg[k][-1]]          # collapse grid to a single combo
            cfg['epochs'] = [6]
        if args.epochs:
            cfg['epochs'] = [args.epochs]
        hp_path = CACHE / ('smoke_dl_hyperparameters.json' if args.smoke
                           else f'dl_hp_ep{args.epochs}.json')
        hp_path.write_text(json.dumps({'ania': cfg}, indent=2))
    patience = 2 if args.smoke else args.patience

    model_out = out_dir / (f'sota_ANIA_noneg_s{seed}' if args.no_neg
                           else f'sota_ANIA+neg_s{seed}')
    train_ania(
        X_train_fcgr=X,
        y_train=y,
        hyperparams_config_path=str(hp_path),
        model_output_path=str(model_out),
        logger=logger,
        train_split=0.8,
        patience=patience,
        device=args.device,
        random_search=args.random_search,
        num_random_samples=args.num_random_samples,
    )

    ckpt = torch.load(str(model_out) + '.pt', map_location='cpu', weights_only=False)
    tl, vl = ckpt['train_stats']['train_losses'], ckpt['train_stats']['val_losses']
    curve = ', '.join(f'{v:.3f}' for v in vl)
    print(f'[seed {seed}] best val loss {ckpt["val_metrics"]["best_val_loss"]:.4f}; '
          f'val curve ({len(vl)} ep): {curve}', flush=True)
    print(f'[seed {seed}] best hyperparams: '
          f'{json.dumps(ckpt["hyperparams"])}', flush=True)

    # inference on ext-208 through the repo's own predictor
    tmp = out_dir / f'tmp_pred_s{seed}.csv'
    predict_ania_from_csv(
        input_csv_path=str(CACHE / 'ext_ania.csv'),
        model_ckpt_path=str(model_out) + '.pt',
        output_csv_path=str(tmp),
        resolution=RES,
        start_idx=3,
        end_idx=3 + NFEAT - 1,
        device=args.device,
    )
    pred = pd.read_csv(tmp)
    ext = build_ext()
    assert pred['Sequence'].tolist() == ext.sequence.tolist()
    o = pd.DataFrame({'log10mic': pred['Predicted Log MIC Value'].values})
    o.index = ext.sequence.values
    o.index.name = 'sequence'
    name = (f'sota_ANIA_noneg_s{seed}_preds.csv' if args.no_neg
            else f'sota_ANIA+neg_s{seed}_preds.csv')
    o.to_csv(out_dir / name)
    tmp.unlink()
    truth = np.log10(ext.MIC_E_coli_ug_mL.values.astype(float))
    err = o.log10mic.values - truth
    print(f'[seed {seed}] preds range [{o.log10mic.min():.2f},{o.log10mic.max():.2f}] '
          f'(truth [{truth.min():.2f},{truth.max():.2f}]), '
          f'ext RMSE {np.sqrt((err ** 2).mean()):.3f} -> '
          f'{out_dir}/{name}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--no-neg', action='store_true',
                    help='[noneg] train on the 4,929 positives only (filters the cached '
                         '+neg corpus CSV to positive sequences)')
    ap.add_argument('--features', action='store_true',
                    help='one-time CGR feature cache over corpus+ext')
    ap.add_argument('--limit', type=int, default=None,
                    help='subset size for --features path validation')
    ap.add_argument('--seed', type=int, default=None,
                    help='protocol seed (120-129): train + infer')
    ap.add_argument('--smoke', action='store_true',
                    help='single grid combo, 6 epochs, patience 2')
    ap.add_argument('--epochs', type=int, default=None,
                    help='override epochs in the (otherwise verbatim) grid')
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--random-search', action='store_true')
    ap.add_argument('--num-random-samples', type=int, default=50)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    if not args.features and args.seed is None:
        ap.error('need --features or --seed N')
    if args.features:
        run_features(limit=args.limit)
    if args.seed is not None:
        run_seed(args)


if __name__ == '__main__':
    main()

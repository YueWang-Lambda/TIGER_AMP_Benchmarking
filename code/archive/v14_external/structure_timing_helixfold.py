#!/usr/bin/env python3
"""HelixFold-Single per-sequence structure-prediction timing (30-peptide sample).

Model load is one-time and NOT counted; one warm-up sequence is run first.
Per-sequence timing covers: feature encoding -> model forward -> PDB write.
"""
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

HELIXFOLD_DIR = Path('/data4T/ubuntu/gonghaifan/POAP/3_helixfold')
MODEL = HELIXFOLD_DIR / 'helixfold-single.pdparams'
OUT = Path('/tmp/helixfold_timing')

sys.path.insert(0, str(HELIXFOLD_DIR))
import ml_collections  # noqa: E402
import numpy as np  # noqa: E402
import paddle  # noqa: E402
from paddle import nn  # noqa: E402

from alphafold_paddle.common import protein, residue_constants  # noqa: E402
from alphafold_paddle.data.data_utils import single_sequence_to_features  # noqa: E402
from alphafold_paddle.model import config, features, utils as model_utils  # noqa: E402
from alphafold_paddle.model.modules import Attention  # noqa: E402
from utils.model_tape import RunTapeModel  # noqa: E402
from utils.utils import get_model_parameter_size, tree_map  # noqa: E402

sys.path.insert(0, '/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/helixfold')
from run_helixfold import install_attention_patch  # noqa: E402

paddle.set_device('gpu:0')
install_attention_patch(paddle, nn, Attention)

model_config = ml_collections.ConfigDict(json.load(open(HELIXFOLD_DIR / 'model_configs/tape-lnw4.json')))
tape_config = ml_collections.ConfigDict(json.load(open(HELIXFOLD_DIR / 'tape/configs/deberta_1B_bs_cp.json')))
af2_config = config.model_config('seq512_pair64_l24_vio0')
t0 = time.time()
model = RunTapeModel(None, model_config, tape_config, af2_config)
model.load_params(str(MODEL))
model.eval()
af2_config.data.eval.delete_msa_block = False
print(f'model loaded in {time.time()-t0:.1f}s, size={get_model_parameter_size(model)}', flush=True)


def predict_one(sequence: str, pdb_path: Path):
    description = '>' + sequence
    raw_features = single_sequence_to_features(sequence, description)
    feature_data = features.np_example_to_features(np_example=raw_features, config=af2_config)
    batch = {'name': [description],
             'feat': tree_map(lambda v: paddle.to_tensor(v[None, ...]), feature_data),
             'label': {}}
    with paddle.no_grad():
        results = model(batch, compute_loss=False)
    numpy_features = tree_map(lambda v: v[0].numpy(), batch['feat'])
    numpy_results = tree_map(lambda v: v[0].numpy(), results)
    numpy_results.update(model_utils.get_confidence_metrics(numpy_results))
    b_factors = np.repeat(numpy_results['plddt'][:, None], residue_constants.atom_type_num, axis=-1)
    prot = protein.from_prediction(numpy_features, numpy_results, b_factors=b_factors)
    pdb_path.write_text(protein.to_pdb(prot), encoding='utf-8')


sample = pd.read_csv(OUT / 'sample30.csv')
seqs = sample.sequence.tolist()

# warm-up (not counted)
predict_one(seqs[0], OUT / 'pdb/_warmup.pdb')
print('warm-up done', flush=True)

records = []
for i, seq in enumerate(seqs):
    t0 = time.time()
    predict_one(seq, OUT / 'pdb' / f'{seq}.pdb')
    dt = time.time() - t0
    records.append({'sequence': seq, 'length': int(sample.sequence_length.iloc[i]),
                    'family': sample.family.iloc[i], 'helixfold_seconds': round(dt, 3)})
    print(f'[{i+1}/{len(seqs)}] len={len(seq)} {dt:.2f}s', flush=True)

pd.DataFrame(records).to_csv(OUT / 'helixfold_times.csv', index=False)
times = [r['helixfold_seconds'] for r in records]
print(json.dumps({'n': len(times), 'median_s': float(np.median(times)),
                  'mean_s': float(np.mean(times)), 'min_s': min(times), 'max_s': max(times)}), flush=True)

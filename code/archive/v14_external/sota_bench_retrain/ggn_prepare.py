#!/usr/bin/env python3
"""GGN-ABPMIC feature preparation for the unified protocol (run once).

Produces under baseline/sota_bench/ggn_work/:
  train.csv / ext.csv     sequence,value + 553 descriptor columns (the
                          descriptor block reuses AMPActiPred's
                          construct_features — the two repos share the same
                          553-dim descriptor set by construction)
  prottrans/{seq}_prottrans.npy   per-residue 1024d from the cached
                          ProtT5-XL-U50 extraction (emb_cache/
                          prott5_perresidue.npz), exported in the repo's format
  pdb/ -> symlinks to ../ggn_pdb (Rosetta structures, {seq}.pdb)
  dssp/{seq}_dssp.npy     mkdssp (dsspbio env) + the repo's parse/align/
                          transform functions (vendored)

Structures WITHOUT a Rosetta PDB (39/13,086) are dropped from the corpus and
logged — GGN cannot build graphs without coordinates.

Usage: python ggn_prepare.py [--dssp_jobs 8]
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V2 = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC')
GGN = HERE / 'GGN-ABPMIC'
WORK = HERE / 'ggn_work'
DSSP_BIN = '/home/ubuntu/anaconda3/envs/dsspbio/bin/mkdssp'


def tables():
    tr = pd.read_csv(V2 / 'runs_review/benchmark_restricted/train_restricted4929_tiger.csv')
    pos = tr.groupby('sequence', as_index=False).MIC_Escherichia_coli.mean()
    neg = pd.read_csv(HERE / 'data_restricted/uniprot_negatives.csv')[['sequence']]
    neg['MIC_Escherichia_coli'] = 10000.0
    corpus = pd.concat([pos, neg], ignore_index=True)
    corpus['value'] = np.log10(corpus.MIC_Escherichia_coli.astype(float))
    ext = pd.read_csv(V2 / 'runs_review/final11_peptides.csv')[
        ['sequence', 'MIC_E_coli_ug_mL']].dropna().rename(
            columns={'MIC_E_coli_ug_mL': 'value'})
    ext['value'] = np.log10(ext['value'].astype(float))
    return corpus, ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dssp_jobs', type=int, default=8)
    args = ap.parse_args()
    (WORK / 'prottrans').mkdir(parents=True, exist_ok=True)
    (WORK / 'dssp').mkdir(parents=True, exist_ok=True)
    if not (WORK / 'pdb').exists():
        (WORK / 'pdb').symlink_to(HERE / 'ggn_pdb')

    corpus, ext = tables()
    # ---- 553-dim descriptors (shared with AMPActiPred) ----
    for tag, df in (('train', corpus), ('ext', ext)):
        f = HERE / 'emb_cache' / f'ampacti_feat_{"corpus" if tag == "train" else tag}.csv'
        assert f.exists(), f'先跑 retrain_ampacti_neg.py --features 生成 {f}'
        feat = pd.read_csv(f)
        feat = feat.rename(columns={'Sequence': 'sequence'})
        merged = df[['sequence', 'value']].merge(feat.drop_duplicates('sequence'),
                                                 on='sequence', how='left')
        merged = merged.drop(columns=[c for c in ('Id', 'Sequence') if c in merged.columns])
        # drop sequences without structure (GGN needs coordinates)
        have = merged.sequence.apply(lambda s: (WORK / 'pdb' / f'{s}.pdb').exists())
        dropped = (~have).sum()
        merged = merged[have].reset_index(drop=True)
        assert not merged.isna().any().any(), '特征或结构缺失未清零'
        merged.to_csv(WORK / f'{tag}.csv', index=False)
        print(f'{tag}.csv: {len(merged)} rows ({dropped} dropped, no structure), '
              f'{merged.shape[1] - 2} descriptor cols')

    # ---- ProtTrans per-seq npy export ----
    d = np.load(HERE / 'emb_cache/prott5_perresidue.npz')
    flat, lens, keys = d['flat'], d['lens'], d['keys']
    off = 0
    n_new = 0
    for k, L in zip(keys, lens):
        k = str(k)
        L = int(L)
        out = WORK / 'prottrans' / f'{k}_prottrans.npy'
        if not out.exists():
            np.save(out, flat[off:off + L].astype(np.float32))
            n_new += 1
        off += L
    print(f'prottrans npy: {n_new} new exports')

    # ---- DSSP: parallel mkdssp, then batched parse ----
    seqs = list(pd.concat([corpus, ext]).sequence)
    seqs = [s for s in seqs if (WORK / 'pdb' / f'{s}.pdb').exists()
            and not (WORK / 'dssp' / f'{s}_dssp.npy').exists()]
    if seqs:
        (WORK / 'dssp_tmp').mkdir(exist_ok=True)
        cmdfile = WORK / 'dssp_cmds.txt'
        with open(cmdfile, 'w') as f:
            for s in seqs:
                # mkdssp 4.x: positional input output (classic format by .dssp ext)
                f.write(f'{DSSP_BIN} {WORK}/pdb/{s}.pdb {WORK}/dssp_tmp/{s}.dssp\n')
        subprocess.run(['bash', '-c',
                        f'xargs -P {args.dssp_jobs} -I CMD bash -c CMD < {cmdfile}'],
                       check=False)
        sys.path.insert(0, str(GGN))
        from my_features import _parse_dssp, _align_dssp, _transform_dssp
        ok = 0
        for s in seqs:
            df_ = WORK / 'dssp_tmp' / f'{s}.dssp'
            if not df_.exists():
                continue
            try:
                ds, feat = _parse_dssp(str(df_))
                if ds != s:
                    feat = _align_dssp(ds, feat, s)
                np.save(WORK / 'dssp' / f'{s}_dssp.npy', _transform_dssp(feat))
                ok += 1
            except Exception as e:
                print(f'DSSP 失败 {s}: {e}')
        print(f'dssp npy: {ok}/{len(seqs)}')
    print('DONE')


if __name__ == '__main__':
    main()

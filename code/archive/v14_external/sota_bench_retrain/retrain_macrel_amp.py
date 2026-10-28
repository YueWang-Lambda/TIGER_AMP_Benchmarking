#!/usr/bin/env python3
"""Retrain the Macrel AMP random forest (the classifier behind AMPSphere,
Santos-Junior et al., Cell 187:3761, 2024; Macrel v1.0.0, MIT licence) on the
TIGER restricted 4,929-peptide set.

Faithful reimplementation of BigDataBiology/macrel v1.0.0 train/ code:

  - feature table (macrel/AMP_features.py, 22 columns):
      * per-sequence normalisation: strip trailing '*', strip leading 'M'
      * 9 AA-composition fractions (tiny/small/aliphatic/aromatic/nonpolar/
        polar/charged/basic/acidic)
      * 7 R Peptides features via rpy2 in the original:
        charge(pH=7, pKscale="EMBOSS"), pI(pKscale="EMBOSS"), aIndex,
        instaIndex, boman, hydrophobicity(scale="Eisenberg"),
        hmoment(angle=100, window=11)   -> computed here by an equivalent
        Rscript call (R package Peptides, Rconv env)
      * 6 CTDD first-occurrence positions: SA groups [ALFCGIVW, RKQEND,
        MSPTHY] and HB groups [ILVWAMGT, FYSQCN, PHKEDR], as % of length
  - model: sklearn RandomForestClassifier(n_estimators=101,
        random_state=12345) (train/train-models.py); AMP probability from
        predict_proba
  - training data: our 4,929 AMP positives + the shared UniProt non-AMP pool
    (the original model was trained on the AmPEP positive/negative sets; the
    benchmark protocol retrains every method on the same restricted TIGER set)

Output: runs_review/benchmark_restricted/sota_models/macrel_amp/rf.pkl +
        sota_macrel_preds.csv (P(AMP), 208 peptides)
"""
import argparse
import gzip
import os
import pickle
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SOTA = Path(os.path.dirname(os.path.abspath(__file__)))
TIGER = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC')
TRAIN_CSV = TIGER / 'runs_review/benchmark_restricted/train_restricted4929_tiger.csv'
NEG_CSV = SOTA / 'data_restricted/uniprot_negatives.csv'
PEP_CSV = TIGER / 'runs_review/final11_peptides.csv'
OUT_DIR = TIGER / 'runs_review/benchmark_restricted'
CKPT_DIR = TIGER / 'runs_review/benchmark_restricted/sota_models/macrel_amp'
RSCRIPT = '/home/ubuntu/anaconda3/envs/Rconv/bin/Rscript'

# macrel v1.0.0 macrel/AMP_features.py, verbatim
GROUPS_SA = ['ALFCGIVW', 'RKQEND', 'MSPTHY']
GROUPS_HB = ['ILVWAMGT', 'FYSQCN', 'PHKEDR']
_AA_GROUPS = [
    set('ACGST'), set('ABCDGNPSTV'), set('AILV'), set('FHWY'),
    set('ACFGILMPVWY'), set('DEHKNQRSTZ'), set('BDEHKRZ'), set('HKR'), set('BDEZ'),
]
R_FEATURES = '''
.libPaths(c(path.expand('~/R/x86_64-pc-linux-gnu-library/4.3'), .libPaths()))
library(Peptides)
d <- read.table('%s', sep='\\t', header=FALSE, col.names=c('i','seq'), quote='')
s <- as.character(d$seq)
ch <- charge(seq=s, pH=7, pKscale='EMBOSS')
pI <- pI(seq=s, pKscale='EMBOSS')
aIndex <- aIndex(seq=s)
instaIndex <- instaIndex(seq=s)
boman <- boman(seq=s)
hydrophobicity <- hydrophobicity(seq=s, scale='Eisenberg')
hmoment <- hmoment(seq=s, angle=100, window=11)
out <- data.frame(ch, pI, aIndex, instaIndex, boman, hydrophobicity, hmoment)
write.table(out, '%s', sep='\\t', row.names=FALSE, quote=FALSE)
'''


def normalize(seq: str) -> str:
    """macrel v1.0.0: strip trailing '*', strip leading 'M'."""
    if seq.endswith('*'):
        seq = seq[:-1]
    if seq.startswith('M'):
        seq = seq[1:]
    return seq


def amino_acid_composition(seq):
    return np.array([sum(map(g.__contains__, seq)) for g in _AA_GROUPS],
                    dtype=float) / len(seq)


def ctdd(sequence, groups):
    code = []
    for group in groups:
        for i, aa in enumerate(sequence):
            if aa in group:
                code.append((i + 1) / len(sequence) * 100)
                break
        else:
            code.append(0)
    return code


def r_features(seqs):
    with tempfile.TemporaryDirectory() as td:
        inp, outp = Path(td) / 'in.tsv', Path(td) / 'out.tsv'
        with open(inp, 'w') as f:
            for i, s in enumerate(seqs):
                f.write(f'{i}\t{s}\n')
        rcode = R_FEATURES % (inp, outp)
        rfile = Path(td) / 'feat.R'
        rfile.write_text(rcode)
        res = subprocess.run([RSCRIPT, str(rfile)], capture_output=True, text=True)
        if res.returncode != 0 or not outp.exists():
            raise RuntimeError(f'R feature step failed:\n{res.stderr[-2000:]}')
        out = pd.read_table(outp)
        assert len(out) == len(seqs), (len(out), len(seqs))
        return out.values.astype(float)


def feature_table(seqs):
    seqs = [normalize(s) for s in seqs]
    if any(len(s) == 0 for s in seqs):
        raise ValueError('sequence became empty after M/* normalisation')
    aa = np.array([amino_acid_composition(s) for s in seqs])
    ct = np.array([ctdd(s, GROUPS_SA + GROUPS_HB) for s in seqs])
    rf = r_features(seqs)
    X = np.hstack([aa, rf, ct])
    cols = ['tinyAA', 'smallAA', 'aliphaticAA', 'aromaticAA', 'nonpolarAA',
            'polarAA', 'chargedAA', 'basicAA', 'acidicAA', 'charge', 'pI',
            'aindex', 'instaindex', 'boman', 'hydrophobicity', 'hmoment',
            'SA.Group1.residue0', 'SA.Group2.residue0', 'SA.Group3.residue0',
            'HB.Group1.residue0', 'HB.Group2.residue0', 'HB.Group3.residue0']
    return pd.DataFrame(X, columns=cols)


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    tr = pd.read_csv(TRAIN_CSV)
    pos_seqs = sorted(set(tr.sequence.astype(str).str.upper()))
    neg_seqs = pd.read_csv(NEG_CSV).sequence.astype(str).str.upper().tolist()
    pep = pd.read_csv(PEP_CSV)
    test_seqs = list(dict.fromkeys(pep.sequence))

    print(f'computing features: {len(pos_seqs)} pos + {len(neg_seqs)} neg + '
          f'{len(test_seqs)} test', flush=True)
    Xpos, Xneg, Xtest = feature_table(pos_seqs), feature_table(neg_seqs), feature_table(test_seqs)

    X = pd.concat([Xpos, Xneg], ignore_index=True)
    y = np.array(['AMP'] * len(Xpos) + ['NAMP'] * len(Xneg))

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=101, random_state=12345, n_jobs=16)
    rf.fit(X.values, y)
    oob_note = '(no OOB — saved model trained without oob_score, as released)'

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(CKPT_DIR / 'rf.pkl.gz', 'wb') as f:
        pickle.dump(rf, f)

    train_p = rf.predict_proba(Xpos.values)
    amp_col = list(rf.classes_).index('AMP')
    print(f'train AMP-prob: mean={train_p[:, amp_col].mean():.3f} '
          f'(fitted, sanity only) {oob_note}', flush=True)

    tp = rf.predict_proba(Xtest.values)[:, amp_col]
    pd.Series(dict(zip(test_seqs, tp.tolist())), name='p_amp').to_csv(
        OUT_DIR / 'sota_macrel_preds.csv')
    print(f'test P(AMP): mean={tp.mean():.3f} range [{tp.min():.3f},{tp.max():.3f}] '
          f'-> {OUT_DIR / "sota_macrel_preds.csv"}', flush=True)


if __name__ == '__main__':
    main()

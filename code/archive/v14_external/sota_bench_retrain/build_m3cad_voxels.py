#!/usr/bin/env python3
"""Voxelize the benchmark universes with M3-CAD's 3D voxel coloring method
(experiment 12 supplement: M3CAD-MIC arm).

Faithful replication of the official code (github.com/haifangong/M3CAD,
identification/dataset.py::pdb_parser + demo_voxelization/demo.py channel
dicts): each peptide is placed in a 64^3 grid (coordinates pre-centered on the
atom centroid, as the paper prescribes — their parser assumes pre-centered
PDBs), atoms occupy cubes by van der Waals radius, and three channels are
written per voxel:
  ch0 atomic mass (H1 C12 N14 O16 S30)
  ch1 solubility  (hydrophobic 255 / mid 155 / hydrophilic 55, per residue)
  ch2 charge      (acidic 55 / neutral 155 / basic 255, per residue)
Channel codes follow the CODE (55/155/255), which differs from the paper text
(86/128/168/255) — disclosed.

Universes (structure-bearing only, like the TIGER arms — 102 pool entries
lacking structures are excluded and disclosed):
  train: 4,929 restricted positives (data/3D_data_train_eva_Rosetta)
       + 7,860 negatives      (data/3D_data_negatives_rosetta)
  test : the 208 external peptides (data/test_external/test_activity_*/pdb)

Outputs (baseline/sota_bench/m3cad_cache/):
  vox_train.npy  (12789, 3, 64, 64, 64) uint8 memmap  (~10.0 GB)
  vox_test.npy   (208,   3, 64, 64, 64) uint8
  meta_train.csv / meta_test.csv  (sequence, y_log10mic, src, pdb)

Usage: python build_m3cad_voxels.py [--workers 16]
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TIGER = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC')
CACHE = HERE / 'm3cad_cache'
TRAIN_CSV = TIGER / 'runs_review/benchmark_restricted/train_restricted4929_tiger.csv'
NEG_CSV = TIGER / 'metadata/uniprot_neg_full_pool.csv'  # structure-bearing 7,860
TEST_PEP = TIGER / 'runs_review/final11_peptides.csv'
POS_PDB_DIR = TIGER / 'data/3D_data_train_eva_Rosetta'
NEG_PDB_DIR = TIGER / 'data/3D_data_negatives_rosetta'
TEST_PDB_DIRS = [TIGER / 'data/test_external/test_activity_apexd/pdb',
                 TIGER / 'data/test_external/test_activity_ll37/pdb',
                 TIGER / 'data/test_external/test_activity_apexgo/pdb']
NEG_LOG10 = 4.0

# ---- official channel dicts (dataset.py / demo.py, verbatim) ----
ATOMS = {'H': 1, 'C': 12, 'N': 14, 'O': 16, 'S': 30}
ATOMS_R = {'H': 1, 'C': 1.5, 'N': 1.5, 'O': 1.5, 'S': 2}


def clamp(n, smallest, largest):
    return sorted([smallest, n, largest])[1]


def make_dicts():
    from Bio.PDB.Polypeptide import protein_letters_3to1
    return protein_letters_3to1


def voxelize_pdb(pdb_path: str) -> np.ndarray | None:
    """Official pdb_parser logic + centroid pre-centering (paper: 'centroid
    as the origin'; their released PDBs are pre-centered, ours are not)."""
    from Bio.PDB import PDBParser, is_aa
    three2one = make_dicts()
    voxel = np.zeros((3, 64, 64, 64), dtype=np.uint8)
    try:
        structure = PDBParser(QUIET=True).get_structure('x', pdb_path)
        chain = next(structure[0].get_chains())
        # solubility / charge codes per residue (demo.py AMINO_ACID_* dicts)
        WATER = {'A': 255, 'V': 255, 'P': 255, 'F': 255, 'W': 255, 'I': 255,
                 'L': 255, 'G': 155, 'M': 155, 'Y': 55, 'S': 55, 'T': 55,
                 'C': 55, 'N': 55, 'Q': 55, 'D': 55, 'E': 55, 'K': 55, 'R': 55, 'H': 55}
        CHARGE = {'D': 55, 'E': 55, 'A': 155, 'V': 155, 'P': 155, 'F': 155,
                  'W': 155, 'I': 155, 'L': 155, 'G': 155, 'M': 155, 'Y': 155,
                  'S': 155, 'T': 155, 'C': 155, 'N': 155, 'Q': 155, 'K': 255,
                  'R': 255, 'H': 255}
        atoms = [(a.get_coord(), a.id, three2one.get(r.get_resname(), 'X'))
                 for r in chain if is_aa(r, standard=True) for a in r if a.id in ATOMS]
        if not atoms:
            return None
        coords = np.asarray([c for c, _, _ in atoms], dtype=np.float64)
        center = coords.mean(axis=0)
        for (coord, atom_id, amino), c in zip(atoms, coords - center):
            x_i = int(clamp(c[0], -31, 31) + 32)
            y_i = int(clamp(c[1], -31, 31) + 32)
            z_i = int(clamp(c[2], -31, 31) + 32)
            radius = ATOMS_R[atom_id]
            r_range = 1 if radius <= 1.5 else int(radius)
            z_s = slice(max(0, z_i - r_range), min(64, z_i + r_range))
            y_s = slice(max(0, y_i - r_range), min(64, y_i + r_range))
            x_s = slice(max(0, x_i - r_range), min(64, x_i + r_range))
            voxel[0, x_s, y_s, z_s] = ATOMS[atom_id]
            voxel[1, x_s, y_s, z_s] = WATER.get(amino, 0)
            voxel[2, x_s, y_s, z_s] = CHARGE.get(amino, 0)
        return voxel
    except Exception:
        return None


def _work(job):
    i, seq, pdb = job
    v = voxelize_pdb(pdb)
    return i, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)

    # ---- assemble universes ----
    tr = pd.read_csv(TRAIN_CSV)
    pos = tr.groupby('sequence', as_index=False).MIC_Escherichia_coli.mean()
    rows = [(s, np.log10(m), 'pos', str(POS_PDB_DIR / f'{s}.pdb'))
            for s, m in zip(pos.sequence, pos.MIC_Escherichia_coli.values.astype(float))]
    neg = pd.read_csv(NEG_CSV)
    rows += [(s, NEG_LOG10, 'neg', str(NEG_PDB_DIR / f'{s}.pdb'))
             for s in neg.sequence]
    test_map = {}
    for d in TEST_PDB_DIRS:
        for p in d.glob('*.pdb'):
            test_map.setdefault(p.stem, str(p))
    pep = pd.read_csv(TEST_PEP)
    test_rows = [(s, np.nan, 'test', test_map[s]) for s in
                 dict.fromkeys(pep.sequence) if s in test_map]
    print(f'train rows={len(rows)}  test rows={len(test_rows)} '
          f'(missing test pdbs: {len(set(dict.fromkeys(pep.sequence)) - set(test_map))})',
          flush=True)

    for tag, rws in [('train', rows), ('test', test_rows)]:
        n = len(rws)
        arr = np.lib.format.open_memmap(CACHE / f'vox_{tag}.npy', mode='w+',
                                        dtype=np.uint8, shape=(n, 3, 64, 64, 64))
        jobs = [(i, s, p) for i, (s, y, src, p) in enumerate(rws)]
        failed = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, v in ex.map(_work, jobs, chunksize=32):
                if v is None:
                    failed += 1
                    continue
                arr[i] = v
        arr.flush()
        pd.DataFrame(rws, columns=['sequence', 'y_log10mic', 'src', 'pdb']).to_csv(
            CACHE / f'meta_{tag}.csv', index=False)
        occ = np.asarray(arr[:64]).astype(float)
        print(f'{tag}: {n} voxels written, failed={failed}, '
              f'nonzero frac(first64)={float((occ > 0).mean()):.4f}', flush=True)


if __name__ == '__main__':
    main()

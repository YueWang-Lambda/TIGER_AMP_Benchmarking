#!/usr/bin/env python3
"""FastRelax per-PDB timing on the 30 HelixFold-predicted structures.

Single process (production used ProcessPoolExecutor parallelism; per-structure
serial time is the fair per-unit cost). PyRosetta init is one-time, not counted.
One warm-up PDB first. Timing: pose load -> FastRelax -> dump.
"""
import time
from pathlib import Path

import csv
import pyrosetta

OUT = Path('/tmp/helixfold_timing')

t0 = time.time()
pyrosetta.init(extra_options='-mute all')
print(f'pyrosetta init {time.time()-t0:.1f}s', flush=True)

scorefxn = pyrosetta.get_fa_scorefxn()
relax = pyrosetta.rosetta.protocols.relax.FastRelax()
relax.set_scorefxn(scorefxn)


def relax_one(pdb: Path, out: Path):
    pose = pyrosetta.pose_from_pdb(str(pdb))
    relax.apply(pose)
    pose.dump_pdb(str(out))


# warm-up
relax_one(OUT / 'pdb' / '_warmup.pdb', OUT / 'relaxed' / '_warmup.pdb')
print('warm-up done', flush=True)

rows = list(csv.DictReader(open(OUT / 'sample30.csv')))
records = []
for i, row in enumerate(rows):
    seq = row['sequence']
    src = OUT / 'pdb' / f'{seq}.pdb'
    dst = OUT / 'relaxed' / f'{seq}.pdb'
    t0 = time.time()
    relax_one(src, dst)
    dt = time.time() - t0
    records.append({'sequence': seq, 'length': int(row['sequence_length']),
                    'relax_seconds': round(dt, 3)})
    print(f'[{i+1}/30] len={len(seq)} {dt:.2f}s', flush=True)

with open(OUT / 'relax_times.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['sequence', 'length', 'relax_seconds']); w.writeheader(); w.writerows(records)
ts = [r['relax_seconds'] for r in records]
print({'n': 30, 'median_s': sorted(ts)[15], 'mean_s': sum(ts)/30, 'min_s': min(ts), 'max_s': max(ts)})

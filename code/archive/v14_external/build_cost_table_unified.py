#!/usr/bin/env python3
"""Rebuild the 23-method accuracy-cost table from the FOUR unified strict
timing jsons (tiger / external / new4 / new7, all under the 2026-10-10
boundary: per-query features + forward + label inversion inside; one-time
loading outside; inputs re-read per repeat; 1 warm-up + 3 repeats, median).

Writes:
  runs_review/benchmark_restricted/benchmark_cost_pareto.json   (canonical)
  reports/methods_package_S4_v1/4_S4_benchmark/results/cost_timing_table.csv
  reports/methods_package_S4_v1/4_S4_benchmark/results/benchmark_cost_pareto.json
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BR = ROOT / 'runs_review/benchmark_restricted'
PKG = ROOT / 'reports/methods_package_S4_v1/4_S4_benchmark'

bench = json.loads((BR / 'benchmark_fullpool_seed10.json').read_text())
tiger = json.loads((BR / 'tiger_cost_timing.json').read_text())
ext = json.loads((BR / 'external_cost_timing.json').read_text())
n4 = json.loads((BR / 'new4_cost_timing.json').read_text())
n7 = json.loads((BR / 'new7_cost_timing.json').read_text())

# structure-generation increment (folding+relax, serial extrapolation) — unchanged
STRUCT_X208 = 328.1

NAME = {  # timing-json key -> benchmark key
    'S4': 'TIGER-pair (S4)', 's4_single_n1n2': 'TIGER-single',
    'Das2021-regr': 'Das2021-regr (ours)',
    'HydrAMP-arch-regr': 'HydrAMP-arch-regr (ours)',
    'AMPLiT-regr vB': 'AMPLiT-regr vB (ours)',
    'AMPlify-regr': 'AMPlify-regr (ours)',
}
rows = []
for k, v in tiger['inference'].items():
    t = v['median_s']
    b = NAME[k]
    rows.append({'method': b, 'macro_PCC': bench[b]['mean'],
                 'time_s_no_struct': t, 'time_s_with_struct': t + STRUCT_X208,
                 'cohort': 'TIGER'})
for src in (ext, n4, n7):
    for k, v in src['methods'].items():
        assert 'error' not in v, f'{k}: {v.get("error")}'
        t = v['median_s']
        b = NAME.get(k, k)
        ts = t + STRUCT_X208 if b == 'GGN-ABPMIC+neg' else t
        rows.append({'method': b, 'macro_PCC': bench[b]['mean'],
                     'time_s_no_struct': t, 'time_s_with_struct': ts,
                     'cohort': 'external'})
assert len(rows) == 23, len(rows)

pareto = {
    'device': 'NVIDIA GeForce RTX 4090 D',
    'timing_protocol_unified': (
        'ALL 23 methods timed 2026-10-10 under one strict boundary: per-query '
        'feature computation (tokenization/encoding/embedding/PLM forwards/'
        'descriptors/graph builds), model forward and per-query label inversion '
        'INSIDE the timed region; one-time model/runtime loading EXCLUDED '
        '(resident subprocesses sentinel-synced); the 208-sequence input re-read '
        'inside every repeat; otherwise-idle GPU, seed-123 weights, 1 warm-up + '
        '3 repeats, median. TIGER numbers include fresh per-panel graph '
        'construction, fold z-score replay, 5-fold forward, fold-mean '
        'aggregation and OOF calibration; checkpoint loading excluded.'),
    'structure_timing': {
        'x208_seconds': STRUCT_X208,
        'note': 'HelixFold-Single + PyRosetta FastRelax, serial extrapolation '
                'from 30 length-stratified peptides (median 1.5775 s/peptide); '
                'added to TIGER-pair/TIGER-single/GGN-ABPMIC in the '
                'with-structure scenario only; accuracy identical between '
                'scenarios'},
    'rows': rows,
}

df = pd.DataFrame(rows)


def pareto_members(d, time_col):
    sel = []
    for i, r in d.iterrows():
        dominated = ((d[time_col] <= r[time_col]) &
                     (d.macro_PCC >= r.macro_PCC) &
                     ((d[time_col] < r[time_col]) | (d.macro_PCC > r.macro_PCC))
                     ).any()
        if not dominated:
            sel.append(i)
    return d.loc[sel].sort_values([time_col, 'macro_PCC'], ascending=[True, False])


for col, flag in (('time_s_with_struct', 'pareto_new_structure_scenario'),
                  ('time_s_no_struct', 'pareto_structures_ready_scenario')):
    front = pareto_members(df, col)
    df[flag] = df.method.isin(front.method)
    pareto['frontier_' + ('new_structures' if 'with' in col else 'structures_ready')] = front.method.tolist()

df = df.sort_values('macro_PCC', ascending=False)
csv = df[['method', 'cohort', 'macro_PCC', 'time_s_no_struct', 'time_s_with_struct',
          'pareto_new_structure_scenario', 'pareto_structures_ready_scenario']]
PKG.mkdir(parents=True, exist_ok=True)
csv.to_csv(PKG / 'results/cost_timing_table.csv', index=False)
(BR / 'benchmark_cost_pareto.json').write_text(json.dumps(pareto, indent=2, ensure_ascii=False))
(PKG / 'results/benchmark_cost_pareto.json').write_text(json.dumps(pareto, indent=2, ensure_ascii=False))
print(csv.to_string(index=False))
print('frontier (ready):', pareto['frontier_structures_ready'])
print('frontier (with struct):', pareto['frontier_new_structures'])

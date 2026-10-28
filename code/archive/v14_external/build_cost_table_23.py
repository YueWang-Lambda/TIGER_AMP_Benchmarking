#!/usr/bin/env python3
"""Extend the accuracy-cost tables to the 23-method benchmark (2026-10-08).

Reads:
  runs_review/benchmark_restricted/benchmark_cost_pareto.json (16 rows)
  runs_review/benchmark_restricted/new7_cost_timing.json      (7 timings)
  runs_review/benchmark_restricted/benchmark_fullpool_seed10.json (means)
Writes:
  benchmark_cost_pareto.json  (canonical, 23 rows, Pareto flags recomputed)
  methods_package_S4_v1/4_S4_benchmark/results/cost_timing_table.csv
  methods_package_S4_v1/4_S4_benchmark/results/benchmark_cost_pareto.json

Fair-timing note (user audit 2026-10-08): per-query feature computation fully
included for every method — frozen PLM forwards (esm2-3B / esm1v / esm1b /
ProtT5), descriptor suites, CGR encoding, Morgan fingerprints, DSSP runs and
graph builds all run inside the timed region; test-set inputs are read fresh
per repeat (no preload); only one-time model/runtime loading is excluded.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BR = ROOT / 'runs_review/benchmark_restricted'
PKG = ROOT / 'reports/methods_package_S4_v1/4_S4_benchmark'

pareto = json.loads((BR / 'benchmark_cost_pareto.json').read_text())
timing = json.loads((BR / 'new7_cost_timing.json').read_text())
bench = json.loads((BR / 'benchmark_fullpool_seed10.json').read_text())

existing = {r['method'] for r in pareto['rows']}
assert len(existing) == 16, len(existing)
STRUCT_X208 = pareto['structure_timing']['x208_seconds']   # fold+relax, 208 seqs

for name, m in timing['methods'].items():
    assert 'error' not in m, f'{name}: {m.get("error")}'
    assert name not in existing and name in bench, name
    t = m['median_s']
    # GGN 是唯一需三维结构的非 TIGER 方法：with_struct 场景加 fold+relax
    ts = t + STRUCT_X208 if name == 'GGN-ABPMIC+neg' else t
    pareto['rows'].append({'method': name, 'macro_PCC': bench[name]['mean'],
                           'time_s_no_struct': t,
                           'time_s_with_struct': ts, 'cohort': 'external'})
pareto['n_methods'] = 23
pareto['timing_note_new7'] = (
    'the seven v2-9 retrained additions were timed 2026-10-08 on an '
    'otherwise-idle RTX 4090 D (1 warm-up + 3 repeats, median). Fairness '
    'audit: ALL per-query feature computation inside the timed region '
    '(frozen PLM forwards, descriptors, CGR/R encode, Morgan fps, DSSP, '
    'graph builds); inputs read fresh per repeat; only one-time model '
    'loading excluded. AMPActiPred deep-forest runs in its resident py3.9 '
    'subprocess; ANIA = CGR encode (resident ania-env R) + CNN forward. '
    'GGN with_struct adds the 328.1 s fold+relax pipeline; see '
    'new7_cost_timing.json')

df = pd.DataFrame(pareto['rows'])


def pareto_members(d, time_col):
    sel = []
    for i, r in d.iterrows():
        dominated = ((d[time_col] <= r[time_col]) &
                     (d.macro_PCC >= r.macro_PCC) &
                     ((d[time_col] < r[time_col]) | (d.macro_PCC > r.macro_PCC))
                     ).any()
        if not dominated:
            sel.append(i)
    return d.loc[sel].sort_values([time_col, 'macro_PCC'],
                                  ascending=[True, False])


for col, flag in (('time_s_with_struct', 'pareto_new_structure_scenario'),
                  ('time_s_no_struct', 'pareto_structures_ready_scenario')):
    front = pareto_members(df, col)
    df[flag] = df.method.isin(front.method)
    pareto['frontier_' + ('new_structures' if 'with' in col
                          else 'structures_ready')] = front.method.tolist()

df = df.sort_values('macro_PCC', ascending=False)
csv = df[['method', 'cohort', 'macro_PCC', 'time_s_no_struct',
          'time_s_with_struct', 'pareto_new_structure_scenario',
          'pareto_structures_ready_scenario']]
PKG.mkdir(parents=True, exist_ok=True)
csv.to_csv(PKG / 'results/cost_timing_table.csv', index=False)
(BR / 'benchmark_cost_pareto.json').write_text(
    json.dumps(pareto, indent=2, ensure_ascii=False))
(PKG / 'results/benchmark_cost_pareto.json').write_text(
    json.dumps(pareto, indent=2, ensure_ascii=False))
print(csv.to_string(index=False))
print('frontier (with struct):', pareto['frontier_new_structures'])
print('frontier (structures ready):', pareto['frontier_structures_ready'])

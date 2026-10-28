#!/usr/bin/env python3
"""Extend the accuracy-cost tables to the 16-method benchmark (2026-10-05).

Reads:
  runs_review/benchmark_restricted/benchmark_cost_pareto.json (12 rows)
  runs_review/benchmark_restricted/new4_cost_timing.json       (4 timings)
  runs_review/benchmark_restricted/benchmark_fullpool_seed10.json (means)
Writes:
  benchmark_cost_pareto.json  (canonical, 16 rows, Pareto flags recomputed)
  methods_package_S4_v1/4_S4_benchmark/results/cost_timing_table.csv
  methods_package_S4_v1/4_S4_benchmark/results/benchmark_cost_pareto.json
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BR = ROOT / 'runs_review/benchmark_restricted'
PKG = ROOT / 'reports/methods_package_S4_v1/4_S4_benchmark'

pareto = json.loads((BR / 'benchmark_cost_pareto.json').read_text())
timing = json.loads((BR / 'new4_cost_timing.json').read_text())
bench = json.loads((BR / 'benchmark_fullpool_seed10.json').read_text())

NEW = {'MBC-Attention+neg': 0.1723, 'HyperAMP+neg': 2.9301,
       'SenseXAMP-Ecoli+neg': 6.8726, 'AMPlify-regr (ours)': 0.0382}
# verify against the measured timing file and the benchmark means
for name, t in NEW.items():
    assert abs(timing['methods'][name]['median_s'] - t) < 1e-9, name
    assert name in bench, name

existing = {r['method'] for r in pareto['rows']}
assert len(existing) == 12
for name, t in NEW.items():
    m = bench[name]['mean']
    pareto['rows'].append({'method': name, 'macro_PCC': m,
                           'time_s_no_struct': t,
                           'time_s_with_struct': t, 'cohort': 'external'})
pareto['n_methods'] = 16
pareto['timing_note_new4'] = (
    'the four retrained additions were timed 2026-10-05 on an otherwise-idle '
    'RTX 4090 D (seed-123 weights; per-query feature computation incl. frozen '
    'ankh-base / ESM-1b forwards, descriptors and hypergraph construction '
    'included, one-time model loading excluded, 1 warm-up + 3 repeats, '
    'median); see new4_cost_timing.json')

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
print('frontier (with struct):',
      pareto['frontier_new_structures'])
print('frontier (structures ready):', pareto['frontier_structures_ready'])

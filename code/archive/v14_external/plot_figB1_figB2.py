#!/usr/bin/env python3
"""Regenerate FigB1 (23-method macro-PCC bars) and FigB2 (accuracy-cost
Pareto) for the S4 methods package; 2026-10-08 edition (16 -> 23 methods).

Inputs (package results/):
  benchmark_fullpool_seed10.json  per-method mean/std (external 14)
  s4_panels_seed10.json           TIGER-pair(S4) / TIGER-single rows
  benchmark_cost_pareto.json      16 rows incl. Pareto flags
Outputs: figures/FigB1_benchmark_macroPCC.{png,pdf}, figures/FigB2_benchmark_pareto.{png,pdf}
House style: Arial, red #e42326 (TIGER) / blue #4258a8 (external),
error bars = across-seed SD, Nature ticks; FigB2 log-x, grey step frontier,
method abbreviations beside points; TIGER hollow = structures ready (2.1 s),
filled = with structure generation (~330 s).
Usage: conda activate ccseg && python plot_figB1_figB2.py
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

for f in ('Arial.ttf', 'Arial_Bold.ttf', 'Arial_Italic.ttf',
          'Arial_Bold_Italic.ttf'):
    font_manager.fontManager.addfont(
        '/usr/share/fonts/truetype/msttcorefonts/' + f)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7.5})

HERE = Path(__file__).resolve().parent          # .../4_S4_benchmark/code
MOD = HERE.parent                                # .../4_S4_benchmark
ROOT = HERE.parents[3]                           # project root (deltaMIC)
BR = ROOT / 'runs_review/benchmark_restricted'

RED, BLUE, GREY = '#e42326', '#4258a8', '#9a9a9a'
bench = json.loads((MOD / 'results/benchmark_fullpool_seed10.json').read_text())
panels = json.loads((BR / 's4_panels_seed10.json').read_text())
pareto = json.loads((MOD / 'results/benchmark_cost_pareto.json').read_text())

TIGER = {'TIGER-pair (S4)': panels['模态']['S4(gps)'],
         'TIGER-single': panels['头类型']['single(N2预训练+N1行注入)']}

# ---------------- FigB1: 16-method macro PCC bars ----------------
rows = []
for m, ent in bench.items():
    if m in TIGER:
        continue
    rows.append((m, ent['mean'], ent['std'], 'external'))
for m, ent in TIGER.items():
    rows.append((m, ent['mean'], ent['std'], 'TIGER'))
rows.sort(key=lambda r: -r[1])

fig, ax = plt.subplots(figsize=(3.5, 5.7))
y = np.arange(len(rows))
for i, (m, mean, sd, coh) in enumerate(rows):
    ax.barh(i, mean, xerr=sd, height=0.62, color=RED if coh == 'TIGER' else BLUE,
            error_kw=dict(elinewidth=0.7, capsize=1.4, ecolor='0.25'))
    ax.text(mean + sd + 0.012, i, f'{mean:.3f}', va='center', fontsize=6.6)
ax.set_yticks(y)
ax.set_yticklabels([r[0].replace(' (ours)', '\n(ours)') for r in rows],
                   fontsize=6.6)
ax.invert_yaxis()
ax.set_xlabel('External macro PCC\n(11 families, 10 seeds)')
ax.set_xlim(0, 0.80)
ax.tick_params(axis='x', direction='out', top=False)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(MOD / 'figures/FigB1_benchmark_macroPCC.png', dpi=450)
fig.savefig(MOD / 'figures/FigB1_benchmark_macroPCC.pdf')
plt.close(fig)

# ---------------- FigB2: accuracy-cost Pareto ----------------
df = pd.DataFrame(pareto['rows'])
ABBR = {'TIGER-pair (S4)': 'TIGER-pair', 'TIGER-single': 'TIGER-single',
        'APEX-restr(lrfix)+neg': 'APEX', 'Das2021-regr (ours)': 'Das2021',
        'LLAMP+neg': 'LLAMP', 'BERT-AmPEP+neg': 'BERT-AmPEP',
        'Deep-PIE+neg': 'Deep-PIE', 'HydrAMP-arch-regr': 'HydrAMP',
        'AMPLiT-regr vB': 'AMPLiT', 'EvoGrad-strict': 'EvoGrad',
        'DeepAMP-CNN': 'DeepAMP', 'AMP-Designer+neg': 'AMP-Designer',
        'AMPLiT-regr vB (ours)': 'AMPLiT', 'HydrAMP-arch-regr (ours)': 'HydrAMP',
        'EvoGrad-strict': 'EvoGrad', 'AMPlify-regr (ours)': 'AMPlify',
        'MBC-Attention+neg': 'MBC-Attn', 'HyperAMP+neg': 'HyperAMP',
        'SenseXAMP-Ecoli+neg': 'SenseXAMP', 'AMPlify-regr (ours)': 'AMPlify',
        'AMPActiPred+neg': 'AMPActiPred', 'AMPGen+neg': 'AMPGen',
        'EIPpred+neg': 'EIPpred', 'AMPpredictor+neg': 'AMPpredictor',
        'ANIA+neg': 'ANIA', 'tsAMP-CS+neg': 'tsAMP-CS',
        'GGN-ABPMIC+neg': 'GGN-ABPMIC'}
front = pareto['frontier_structures_ready']

fig, ax = plt.subplots(figsize=(4.4, 3.4))
ext = df[df.cohort == 'external']
ax.scatter(ext.time_s_with_struct, ext.macro_PCC, s=22, color=BLUE,
           zorder=3, label='External methods (21)')
# per-method label offsets to de-crowd the close pairs:
# LLAMP/AMPlify nearly coincide; Das2021 label would cross the APEX point;
# EvoGrad sits ~8 pt under HydrAMP-arch; EIPpred/HyperAMP nearly coincide
# on the 3 s vertical; AMPGen/tsAMP/AMPpredictor crowd the 0.3-1.1 s band
OFF = {'LLAMP+neg': ((-5, 5), 'right'),
       'AMPlify-regr (ours)': ((5, -8), 'left'),
       'Das2021-regr (ours)': ((0, 6), 'center'),
       'EvoGrad-strict': ((5, -8), 'left'),
       'HydrAMP-arch-regr (ours)': ((5, 7), 'left'),
       'AMPLiT-regr vB (ours)': ((0, 7), 'center'),
       'EvoGrad-strict': ((5, -7), 'left'),
       'EIPpred+neg': ((0, 7), 'center'),
       'HyperAMP+neg': ((5, -8), 'left'),
       'AMPGen+neg': ((0, 7), 'center'),
       'tsAMP-CS+neg': ((6, -7), 'left'),
       'AMPpredictor+neg': ((-6, 5), 'right'),
       'ANIA+neg': ((0, -9), 'center'),
       'GGN-ABPMIC+neg': ((0, 7), 'center')}
for _, r in ext.iterrows():
    xy, ha = OFF.get(r.method, ((4, 3), 'left'))
    ax.annotate(ABBR[r.method], (r.time_s_with_struct, r.macro_PCC),
                textcoords='offset points', xytext=xy, fontsize=6.2,
                color='#31508f', ha=ha)
tig = df[df.cohort == 'TIGER']
ax.scatter(tig.time_s_no_struct, tig.macro_PCC, s=26, facecolors='none',
           edgecolors=RED, linewidths=1.2, zorder=4,
           label='TIGER (structures ready)')
ax.scatter(tig.time_s_with_struct, tig.macro_PCC, s=26, color=RED, zorder=4,
           label='TIGER (+ structure generation)')
for _, r in tig.iterrows():
    ax.annotate(ABBR[r.method] + ' (2.1 s)', (r.time_s_no_struct, r.macro_PCC),
                textcoords='offset points', xytext=(6, -1), fontsize=6.2,
                color=RED)
    ax.annotate(ABBR[r.method], (r.time_s_with_struct, r.macro_PCC),
                textcoords='offset points', xytext=(4, 3), fontsize=6.2,
                color=RED)

fr = df[df.method.isin(front)].sort_values('time_s_with_struct')
ax.step(fr.time_s_with_struct, fr.macro_PCC, where='post', color=GREY,
        linewidth=0.9, zorder=2)

ax.set_xscale('log')
ax.set_xlabel('Inference time, s (log) — 208 peptides / 2,211 edges')
ax.set_ylabel('External macro PCC')
ax.set_ylim(0.2, 0.70)
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(direction='out', top=False)
ax.legend(frameon=False, fontsize=6.4, loc='lower right')
fig.tight_layout()
fig.savefig(MOD / 'figures/FigB2_benchmark_pareto.png', dpi=450)
fig.savefig(MOD / 'figures/FigB2_benchmark_pareto.pdf')
plt.close(fig)
print('FigB1/FigB2 written to', MOD / 'figures')

#!/usr/bin/env python3
"""Evaluation of the s4_pair_neg arm (TIGER-pair + negative-pair injection,
joint mode = AMP-AMP protocol verbatim on negatives: NW >= 0.7, +-5 length
window, directed both ways, no neg-neg pairs; everything else identical to
S4_baseline). Seeds 120-129.

Endpoint: the 2,211-edge primary panel, 5-fold mean + OOF calibrator — the
run_arm flow of eval_cost_timing_tiger. Metrics per seed: family-equal-weight
MACRO PCC and MACRO RSE (code.utils.metrics.regression_metrics per family,
log2 domain, >=4-edge rule) — matching the s4_panels_fourmetrics pipeline.

Paired baseline: s4_panels_fourmetrics.json['S4']['per_seed'] (same seeds,
both metrics); deltas + Wilcoxon signed-rank.

Usage: PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python scripts/eval_s4pairneg_seed10.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
BR = ROOT / 'runs_review/benchmark_restricted'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SEEDS = tuple(range(120, 130))


def evaluate_arm(seeds=SEEDS):
    from eval_neg_twostage_arms import PANELS, KEEP_RAW
    from code.dataloader import build_graphs, preprocess_dataset
    from code.evaluation import eval_explicit_pairs, load_checkpoint_model
    from code.utils.config import load_config
    from code.utils.constants import CFU_COL, TARGET_COL
    from code.utils.scaling import apply_p_f_stats, apply_node_x_stats
    from code.utils.metrics import apply_calibrator, regression_metrics

    def build_panel(cfg, panel, pepcsv, pdbdir, keep):
        tmp = Path(f'/tmp/s4pairneg_eval_{panel}.csv')
        df = pd.read_csv(pepcsv)
        df = df[df.sequence.astype(str).str.upper().str.strip().isin(keep)]
        o = pd.DataFrame({'sequence': df.sequence.astype(str).str.upper().str.strip()})
        mic = df.get('MIC_Escherichia_coli', df.get('MIC_E_coli_ug_mL', df.get('MIC_ug_per_mL')))
        o[TARGET_COL] = pd.to_numeric(mic, errors='coerce')
        o[CFU_COL] = '1E5 - 1E6'
        for c in ('n_terminus', 'c_terminus'):
            o[c] = df[c] if c in df.columns else ''
        o.to_csv(tmp, index=False)
        filt, _ = preprocess_dataset(str(tmp), str(pdbdir), cfg, f's4pn_eval_{panel}')
        graphs, rows, _ = build_graphs(filt, cfg)
        return graphs, rows

    yt_csv = pd.read_csv(BR / 'eleven_way_predictions.csv')
    truth = {(str(f).strip('/'), a, q): t for f, a, q, t in
             zip(yt_csv.family, yt_csv.anchor_sequence, yt_csv.query_sequence,
                 yt_csv.y_true_delta_log2_anchor_minus_query_03)}

    per = {}
    for s in seeds:
        run = ROOT / f'runs_ablation/s4_pair_neg/s4_pair_neg_seed{s}'
        if not run.exists():
            per[s] = None
            continue
        cfg = load_config(run / 'config.json')
        models = [load_checkpoint_model(run / 'checkpoints' / f'fold{f}_best.pt', cfg, device)[0]
                  for f in range(1, 6)]
        zstats = []
        for f in range(1, 6):
            zs = []
            for nm, fn in ((f'p_feature_zscore_fold{f}.json', apply_p_f_stats),
                           (f'node_feature_zscore_fold{f}.json', apply_node_x_stats)):
                sp = run / 'intermediate' / nm
                zs.append((json.loads(sp.read_text()), fn) if sp.exists() else (None, fn))
            zstats.append(zs)
        cal = json.loads((run / 'results' / 'calibrator.json').read_text())
        frames = []
        for panel, (pepcsv, pdbdir, cfu) in PANELS.items():
            pairs = pd.read_csv(ROOT / f'runs_review/moe_random_predictions_{panel}.csv')
            pairs = pairs[pairs.family.isin(KEEP_RAW)][
                ['anchor_sequence', 'query_sequence', 'family',
                 'y_true_delta_log2_anchor_minus_query_03']]
            keep = set(pairs.anchor_sequence) | set(pairs.query_sequence)
            frames.append((build_panel(cfg, panel, pepcsv, pdbdir, keep), pairs))
        YT = 'y_true_delta_log2_anchor_minus_query'
        YP = 'y_pred_delta_log2_anchor_minus_query'
        fold_frames = []
        for fi, model in enumerate(models):
            for (graphs, rows), pairs in frames:
                for st, fn in zstats[fi]:
                    if st is not None:
                        fn(graphs, st)
                pred, _ = eval_explicit_pairs(model, graphs, rows, pairs, cfg, device)
                pred['fold'] = fi + 1
                fold_frames.append(pred)
        allf = pd.concat(fold_frames, ignore_index=True)
        ens = allf.groupby(['family', 'anchor_sequence', 'query_sequence'], as_index=False).agg(
            **{YT: (YT, 'first'), YP: (YP, 'mean')})
        ens = apply_calibrator(ens, cal)
        K = pd.read_csv(BR / 'pair_keys_primary_2211.csv')
        m = ens.assign(_fam=ens.family.astype(str).str.rstrip('/')).merge(
            K.assign(_fam=K.family.astype(str).str.rstrip('/'))[
                ['_fam', 'anchor_sequence', 'query_sequence']],
            on=['_fam', 'anchor_sequence', 'query_sequence'], how='inner')
        assert len(m) == 2211, len(m)
        pccs, rses = [], []
        for fam, g in m.groupby('_fam'):
            if len(g) < 4:
                continue
            y_true = [truth[(fam, a, q)] for a, q in zip(g.anchor_sequence, g.query_sequence)]
            met = regression_metrics(y_true, g[YP].tolist(), input_log_base=2)
            pccs.append(met['PCC'])
            rses.append(met['RSE'])
        per[s] = {'PCC': float(np.mean(pccs)), 'RSE': float(np.mean(rses))}
        print(f'  s4_pair_neg seed {s}: PCC {per[s]["PCC"]:.4f} RSE {per[s]["RSE"]:.4f}',
              flush=True)
    return per


def main():
    per = evaluate_arm()
    base = json.loads((BR / 's4_panels_fourmetrics.json').read_text())['S4']['per_seed']
    from scipy.stats import wilcoxon
    rows = {}
    for met in ('PCC', 'RSE'):
        v = np.array([per[s][met] for s in SEEDS if per.get(s)])
        b = np.array([base[str(s)][met] for s in SEEDS if per.get(s)])
        d = v - b                      # s4_pair_neg minus S4(noneg)
        try:
            w_p = float(wilcoxon(v, b).pvalue)
        except Exception:
            w_p = float('nan')
        rows[met] = {'s4_pair_neg_mean': round(float(v.mean()), 4),
                     's4_pair_neg_std': round(float(v.std()), 4),
                     'S4_baseline_mean': round(float(b.mean()), 4),
                     'delta': round(float(d.mean()), 4),
                     'wilcoxon_p': round(w_p, 5),
                     'per_seed': {str(s): round(per[s][met], 4) for s in SEEDS if per.get(s)},
                     'baseline_per_seed': {str(s): base[str(s)][met] for s in SEEDS}}
    out = {'protocol': 's4_pair_neg = S4_baseline + neg_train_inject (joint mode: AMP-AMP '
                       'pairing protocol applied verbatim to negatives, NW>=0.7, +-5 window, '
                       'directed, no neg-neg; smoke-verified injected ~106-160 neg pairs per '
                       'fold); 2,211-edge primary panel, macro family-equal-weight, log2; '
                       'baseline = s4_panels_fourmetrics S4 (same seeds)',
           'metrics': rows}
    (BR / 'benchmark_s4pairneg_seed10.json').write_text(json.dumps(out, indent=2))
    for met in ('PCC', 'RSE'):
        r = rows[met]
        print(f"{met}: s4_pair_neg {r['s4_pair_neg_mean']:.4f}±{r['s4_pair_neg_std']:.4f} "
              f"vs S4 {r['S4_baseline_mean']:.4f} Δ {r['delta']:+.4f} (p={r['wilcoxon_p']})",
              flush=True)
    print('saved benchmark_s4pairneg_seed10.json', flush=True)


if __name__ == '__main__':
    main()

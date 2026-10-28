#!/usr/bin/env python3
"""UNIFIED strict inference-cost timing for the TIGER arms (S4 pair and
s4_single_n1n2 single, seed 123) on the actual external workload
(208 peptides, 2,211 edges). 2026-10-10 rewrite to match the boundary of
eval_cost_timing_external.py / eval_cost_timing_new7.py:

  timed region = raw per-panel peptide CSVs -> calibrated per-pair predictions:
                 fresh graph construction per panel (PDB parse + kNN build +
                 raw feature caching; no /tmp reuse) + fold z-score replay +
                 5-fold forward over the panel pair tables + fold-mean
                 aggregation + OOF calibrator application;
  excluded     = one-time loading (5 fold checkpoints per arm, pair tables,
                 fold z-stats, calibrator coefficients, configs);
  inputs       = panel peptide CSVs are re-read and re-built every repeat;
  protocol     = otherwise-idle GPU, 1 warm-up + 3 timed repeats, median.

Usage: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/eval_cost_timing_tiger.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from eval_neg_twostage_arms import PANELS, KEEP_RAW, YT  # noqa: E402
from code.dataloader import build_graphs, preprocess_dataset  # noqa: E402
from code.evaluation import eval_explicit_pairs, load_checkpoint_model  # noqa: E402
from code.utils.config import load_config  # noqa: E402
from code.utils.constants import CFU_COL, TARGET_COL  # noqa: E402
from code.utils.scaling import apply_p_f_stats, apply_node_x_stats, cache_raw_p_f, cache_raw_node_x  # noqa: E402
from code.utils.metrics import apply_calibrator  # noqa: E402

OUT = ROOT / 'runs_review/benchmark_restricted/tiger_cost_timing.json'
ARMS = {'S4': ROOT / 'runs_ablation/S4/S4_seed123',
        's4_single_n1n2': ROOT / 'runs_ablation/s4_single_n1n2/s4_single_n1n2_seed123'}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def build_panel(cfg, panel, pepcsv, pdbdir, keep_seqs=None):
    tmp = Path(f'/tmp/cost_timing_{panel}.csv')
    df = pd.read_csv(pepcsv)
    if keep_seqs is not None:  # only the peptides the edge tables actually reference
        df = df[df.sequence.astype(str).str.upper().str.strip().isin(keep_seqs)]
    out = pd.DataFrame({'sequence': df.sequence.astype(str).str.upper().str.strip()})
    mic = df.get('MIC_Escherichia_coli', df.get('MIC_E_coli_ug_mL', df.get('MIC_ug_per_mL')))
    out[TARGET_COL] = pd.to_numeric(mic, errors='coerce')
    out[CFU_COL] = '1E5 - 1E6'
    for c in ('n_terminus', 'c_terminus'):
        out[c] = df[c] if c in df.columns else ''
    out.to_csv(tmp, index=False)
    filt, _ = preprocess_dataset(str(tmp), str(pdbdir), cfg, f'cost_{panel}')
    graphs, rows, _ = build_graphs(filt, cfg)
    cache_raw_p_f(graphs)
    cache_raw_node_x(graphs)
    return graphs, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acceptance', action='store_true',
                    help='single untimed pass per arm; verify the CALIBRATED pair output '
                         '(edge identity + macro PCC + elementwise values vs the archived '
                         'seed-123 predictions); exit 1 on failure')
    ap.add_argument('--selftest-negative', action='store_true',
                    help='acceptance + fault-injection: uncalibrated and wrong-unit '
                         'corruptions of the real frame must be rejected')
    args = ap.parse_args()

    cfg0 = load_config(ARMS['S4'] / 'config.json')

    # ---- one-time (excluded): pair tables, per-arm models / z-stats / calibrators
    panels = {}
    for panel, (pepcsv, pdbdir, cfu) in PANELS.items():
        pairs = pd.read_csv(ROOT / f'runs_review/moe_random_predictions_{panel}.csv')
        pairs = pairs[pairs.family.isin(KEEP_RAW)][['anchor_sequence', 'query_sequence', 'family', YT]]
        keep = set(pairs.anchor_sequence) | set(pairs.query_sequence)
        panels[panel] = (pepcsv, pdbdir, keep, pairs)

    arms_ready = {}
    for arm, run in ARMS.items():
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
        arms_ready[arm] = (cfg, models, zstats, cal)
        print(f'[{arm}] preloaded 5 folds (excluded from timing)', flush=True)

    def run_arm(arm):
        cfg, models, zstats, cal = arms_ready[arm]
        frames = []
        for panel, (pepcsv, pdbdir, keep, pairs) in panels.items():
            graphs, rows = build_panel(cfg0, panel, pepcsv, pdbdir, keep)  # fresh graphs
            for fi, model in enumerate(models):
                for st, fn in zstats[fi]:
                    if st is not None:
                        fn(graphs, st)
                pred, _ = eval_explicit_pairs(model, graphs, rows, pairs, cfg, device)
                pred['fold'] = fi + 1
                pred['panel'] = panel
                frames.append(pred)
            del graphs
        allf = pd.concat(frames, ignore_index=True)
        YT_COL = 'y_true_delta_log2_anchor_minus_query'
        YP_COL = 'y_pred_delta_log2_anchor_minus_query'
        ens = allf.groupby(['panel', 'family', 'anchor_sequence', 'query_sequence'], as_index=False).agg(
            **{YT_COL: (YT_COL, 'first'), YP_COL: (YP_COL, 'mean')})
        ens = apply_calibrator(ens, cal)   # [v13b fix] apply_calibrator RETURNS a new
        return ens                         # frame (df.copy()); the old code discarded it
                                             # and returned UNCALIBRATED predictions

    if args.acceptance or args.selftest_negative:
        sys.exit(run_acceptance(run_arm, panels, selftest_negative=args.selftest_negative))

    result = {'n_peptides': 208, 'n_edges': int(sum(len(p[3]) for p in panels.values())),
              'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
              'batch_note': 'unified strict boundary (2026-10-10): fresh per-panel graph '
                            'construction + fold z-score replay + 5-fold forward + fold-mean '
                            'aggregation + OOF calibrator INSIDE the timed region (v13b: the '
                            'calibrated frame is the returned output); checkpoint '
                            'loading and pair tables excluded (one-time); 1 warm-up + 3 '
                            'repeats, median',
              'inference': {}}
    for arm in ARMS:
        run_arm(arm)                       # warm-up
        reps = []
        for rep in range(3):
            torch.cuda.synchronize()
            t0 = time.time()
            run_arm(arm)
            torch.cuda.synchronize()
            reps.append(round(time.time() - t0, 3))
            print(f'[{arm} rep{rep+1}] {reps[-1]}s', flush=True)
        result['inference'][arm] = {'repeats_s': reps, 'median_s': sorted(reps)[1]}

    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def run_acceptance(run_arm, panels, selftest_negative=False):
    """Execute the REAL timed path once per arm (untimed) and verify its final
    output: (i) canonical 2,211-edge identity, (ii) macro PCC vs
    s4_panels_seed10.json seed 123, and — [v13c] — (iii) ELEMENTWISE value
    agreement of the calibrated prediction column against the independently
    archived seed-123 calibrated predictions (v12 rse_prediction_cache), plus
    log2MAE agreement. PCC alone is invariant to positive affine transforms
    and cannot detect a discarded calibrator or a unit error; the value-level
    comparison can (real-path agreement ~6.3e-7; tolerance 1e-5 covers device
    fp noise). --selftest-negative injects the two historical corruptions
    (uncalibrated raw column; wrong log10 unit) and requires rejection."""
    YT_COL = 'y_true_delta_log2_anchor_minus_query'
    YP_COL = 'y_pred_delta_log2_anchor_minus_query'
    BR = ROOT / 'runs_review/benchmark_restricted'
    CACHE = ROOT / 'reports/S4_study_nature_v12/rse_prediction_cache'
    REF_FILE = {'S4': CACHE / 'TIGER-pair_seed123.csv',
                's4_single_n1n2': CACHE / 'TIGER-single_seed123.csv'}
    VALUE_TOL = 1e-5
    MAE_TOL = 1e-4
    K = pd.read_csv(BR / 'pair_keys_primary_2211.csv')
    canon = sorted(zip(K.anchor_sequence, K.query_sequence))
    p = json.loads((BR / 's4_panels_seed10.json').read_text())
    panels_ref = {'S4': p['模态']['S4(gps)']['per_seed']['123'],
                  's4_single_n1n2': p['头类型']['single(N2预训练+N1行注入)']['per_seed']['123']}

    def value_check(ens, ref):
        # join-key normalisation: the pair tables write HAL families with a
        # trailing '/' ('HAL-2/'), the v12 archive without ('HAL-2')
        e = ens.assign(_fam=ens.family.astype(str).str.rstrip('/'))
        r = ref.assign(_fam=ref.family.astype(str).str.rstrip('/'))
        key = ['_fam', 'anchor_sequence', 'query_sequence']
        m = e[key + [YP_COL]].merge(r[key + ['y_true', 'y_pred']], on=key, how='inner')
        if len(m) != 2211:
            return False, {'n_aligned': len(m)}
        dd = (m[YP_COL] - m['y_pred']).abs()
        mae_t = float(np.mean(np.abs(m[YP_COL] - m['y_true'])))
        mae_r = float(np.mean(np.abs(m['y_pred'] - m['y_true'])))
        ok = bool(dd.max() <= VALUE_TOL and abs(mae_t - mae_r) <= MAE_TOL)
        return ok, {'n_aligned': len(m), 'max_abs_pair_diff': float(dd.max()),
                    'log2MAE_timing': round(mae_t, 6), 'log2MAE_archived': round(mae_r, 6)}

    out = {'mode': 'tiger endpoint acceptance (v13c): real single pass; canonical edge '
                   'identity + macro PCC vs panels + ELEMENTWISE calibrated-value '
                   'agreement vs the archived seed-123 predictions (v12 cache)',
           'arms': {}}
    ok = True
    for arm in ARMS:
        ens = run_arm(arm)
        keys = sorted(zip(ens.anchor_sequence, ens.query_sequence))
        edge_ok = keys == canon
        vs = []
        for _, g in ens.groupby('family'):
            if len(g) < 4:
                continue
            yt, yp = g[YT_COL].to_numpy(float), g[YP_COL].to_numpy(float)
            vs.append(float(np.corrcoef(yt, yp)[0, 1]))
        macro = float(np.mean(vs))
        diff = abs(macro - panels_ref[arm])
        ref = pd.read_csv(REF_FILE[arm])
        vals_ok, vinfo = value_check(ens, ref)
        verdict = 'PASS' if (edge_ok and diff <= 1e-3 and vals_ok) else 'FAIL'
        ok &= verdict == 'PASS'
        out['arms'][arm] = {'verdict': verdict, 'edges_match_canonical_2211': edge_ok,
                            'n_edges': len(ens), 'macro_PCC': round(macro, 4),
                            'panels_seed123': panels_ref[arm], 'abs_diff': round(diff, 6),
                            'value_check': vinfo}
        print(arm, out['arms'][arm], flush=True)
        if selftest_negative:
            # fault-injection controls on the REAL frame: both historical
            # corruptions must be REJECTED by value_check
            corrupted = ens.copy()
            corrupted[YP_COL] = corrupted['y_pred_delta_raw']      # uncalibrated
            c1, _ = value_check(corrupted, ref)
            corrupted2 = ens.copy()
            corrupted2[YP_COL] = ens[YP_COL] * np.log10(2)         # wrong unit
            c2, _ = value_check(corrupted2, ref)
            out['arms'][arm]['negative_controls_rejected'] = {'uncalibrated': (not c1),
                                                              'wrong_unit': (not c2)}
            ok &= (not c1) and (not c2)
            print(f'  negative controls [{arm}]: uncalibrated rejected={not c1}, '
                  f'wrong-unit rejected={not c2}', flush=True)
    (BR / 'timing_acceptance_v13_tiger.json').write_text(json.dumps(out, indent=2))
    return 0 if ok else 1


if __name__ == '__main__':
    main()

"""Measure the actual matched-seed P11/S11 ensemble with the reused V16 S11 and V17 unsigned-20000 P11 under the retained timing protocol.

Models are resident; each arm rebuilds live graph features, runs its five folds,
aligns and calibrates all 2,211 outputs. The timed endpoint averages the two arms.
Acceptance is outside the clock. No training or cached query predictions in run().
"""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from pathlib import Path
import sys, json, hashlib, time, subprocess, argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V15 = Path(os.environ.get('V17_DELIVERY_DIR',str(ROOT/'reports/S4_study_nature_v17')))
PAIR_ONLY = '--pair' in sys.argv
OUT = ROOT / 'artifacts/ablation_v17_u20000/timing' / ('TIGER-pair' if PAIR_ONLY else 'TIGER-ensemble')
WORK = ROOT / 'artifacts/ablation_v17_u20000'
KEYS = ['family', 'anchor_sequence', 'query_sequence']
sys.path.insert(0, str(ROOT / 'scripts/ablation_v17'))
sys.path.insert(0, str(WORK / 'runtime'))
if 'code' in sys.modules and not hasattr(sys.modules['code'], '__path__'):
    del sys.modules['code']
import tiger_external as te
import torch
torch.set_num_threads(2)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description='V17 resident-model timing: one warm-up, 20 value-checked repeats.')
    parser.add_argument('--pair',action='store_true',help='Time only the unsigned-20,000 pair model; default: P11/S11 ensemble')
    args=parser.parse_args();assert args.pair==PAIR_ONLY
    assert json.loads((WORK/'training_and_inference_complete.json').read_text())['status']=='PASS'
    def check_exclusive_gpu():
        pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
        foreign=[int(p.strip()) for p in pids if p.strip() and int(p.strip())!=os.getpid()]
        assert not foreign, ('Competing GPU work invalidates timing',foreign)
    check_exclusive_gpu()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'result.json').exists():
        old=json.loads((OUT/'result.json').read_text());assert old['status']=='PASS'
        for file,h in old['checkpoint_files'].items():assert sha(file)==h,file
        print('Retaining accepted timing',OUT,flush=True);return
    edges = pd.read_csv(V15 / 'source_snapshot/canonical_pairs.csv')
    assert len(edges) == 2211 and not edges.duplicated(KEYS).any()
    reg = json.loads((WORK / 'run_registry.json').read_text())
    device = torch.device('cuda:0')
    arms, expected, files = {}, {}, {}
    for arm, label, method in [('P11','factor_P11','TIGER-pair'), ('S11','s4_single_n1n2','TIGER-single')]:
        path = Path(next(r['run_dir'] for r in reg if r['label'] == label and r['seed'] == 123))
        if PAIR_ONLY and arm == 'S11': continue
        assert json.loads((path / 'DONE.json').read_text())['status'] == 'PASS'
        cfg = te.load_config(path / 'config.json')
        models = [te.load_checkpoint_model(path / 'checkpoints' / f'fold{f}_best.pt', cfg, device)[0] for f in range(1,6)]
        pf = [json.loads((path / 'intermediate' / f'p_feature_zscore_fold{f}.json').read_text()) for f in range(1,6)]
        nf = [json.loads((path / 'intermediate' / f'node_feature_zscore_fold{f}.json').read_text()) for f in range(1,6)] if cfg.node_feature_scaling == 'zscore' else [None]*5
        cal = json.loads((path / 'results/calibrator.json').read_text())
        arms[arm] = (cfg, models, pf, nf, cal)
        refpath = V15 / ('source_snapshot/predictions/tiger/s4_single_n1n2_seed123.csv' if arm=='S11' else 'source_snapshot/predictions/tiger/factor_P11_seed123.csv')
        ref = pd.read_csv(refpath, float_precision='round_trip')
        expected[arm] = edges[KEYS].merge(ref, on=KEYS, validate='one_to_one').y_pred.to_numpy(float)
        for p in [path/'config.json', path/'results/calibrator.json', refpath] + list((path/'checkpoints').glob('fold*_best.pt')) + list((path/'intermediate').glob('*zscore_fold*.json')):
            files[str(p)] = sha(p)
    if not PAIR_ONLY: expected['TIGER-ensemble'] = (expected['P11'] + expected['S11']) / 2

    def run_arm(arm):
        cfg, models, pf, nf, cal = arms[arm]
        roster = pd.read_csv(V15 / 'source_snapshot/external_peptides.csv')
        rows = roster[['sequence','n_terminus','c_terminus','panel']].copy()
        rows['cfu_group'] = '1E5 - 1E6'
        rows[te.TARGET_COL] = roster.MIC_E_coli_ug_mL
        rows['pdb_path'] = [str(ROOT / te.PDB_DIRS[r.panel] / f'{r.sequence}.pdb') for r in rows.itertuples()]
        graphs, kept, _ = te.build_graphs(rows, cfg)
        assert len(graphs) == 208
        predictions = []
        for model, scaler, node_scaler in zip(models, pf, nf):
            te.apply_p_f_stats(graphs, scaler)
            if node_scaler is not None:
                te.apply_node_x_stats(graphs, node_scaler)
            with torch.inference_mode():
                pred, _ = te.eval_explicit_pairs(model, graphs, kept, edges, cfg, device)
            pred.family = pred.family.str.rstrip('/')
            aligned = edges[KEYS].merge(pred, on=KEYS, validate='one_to_one')
            predictions.append(aligned.y_pred_delta_log2_anchor_minus_query.to_numpy(float))
        return np.mean(predictions, axis=0) * cal['slope'] + cal['intercept']

    def run():
        if PAIR_ONLY: return {'P11': run_arm('P11')}
        pair = run_arm('P11')
        single = run_arm('S11')
        return {'P11': pair, 'S11': single, 'TIGER-ensemble': (pair + single) / 2}

    def validate(actual):
        check = {}
        for key, values in actual.items():
            assert values.shape == (2211,) and np.isfinite(values).all()
            error = float(np.max(np.abs(values - expected[key])))
            assert error <= 1e-5, (key, error)
            check[key] = {'max_abs_pair_diff': error, 'n_pair': 2211, 'all_finite': True}
        return check

    checks = [validate(run())]
    check_exclusive_gpu()
    torch.cuda.synchronize()
    print('WARMUP PASS', checks[-1], flush=True)
    times = []
    for i in range(20):
        check_exclusive_gpu()
        torch.cuda.synchronize()
        start = time.perf_counter()
        actual = run()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        checks.append(validate(actual))
        check_exclusive_gpu()
        print('REPEAT', i+1, times[-1], flush=True)
    pd.DataFrame({'repeat': range(1,21), 'seconds': times}).to_csv(OUT/'repeats.csv', index=False)
    result = {'status': 'PASS', 'method': 'TIGER-pair' if PAIR_ONLY else 'TIGER-ensemble', 'seed': 123, 'condition': 'with_negatives',
              'median_s': float(np.median(times)), 'min_s': min(times), 'max_s': max(times), 'repeats_s': times,
              'checks': checks, 'checkpoint_files': files, 'script_sha256': sha(__file__),
              'device': 'GPU:0 ('+torch.cuda.get_device_name(0)+')', 'python': sys.executable,
              'gpu_exclusive_checks': 'No competing compute process before/after every repetition',
              'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'cpu_threads': 2,
              'structure_increment_s': 328.12, 'structure_increment_count': 1,
              'single_lineage': 'V16 s4_single_n1n2 / alias factor_S11',
              'pair_lineage': 'V16 s4_bal_unsigned_bal20000 / V17 main factor_P11',
              'protocol': ('Pair P11; ' if PAIR_ONLY else 'P11/S11 ensemble; ') + 'Retained V15 timing protocol: one warm-up + 20 repeats; resident seed123 five-fold models for both arms; sequential full live features + 2211 calibrated log2 predictions per arm + equal averaging; loading and acceptance excluded; shared structures generated once (328.12 s serial estimate outside measured inference).'}
    if PAIR_ONLY:
        result['checks'] = [dict(n_pair=c['P11']['n_pair'], finite=c['P11']['all_finite'], independent_pair_algebra=True, max_abs_pair_diff=c['P11']['max_abs_pair_diff'], calibrated_reference_match=True) for c in checks]
        result['protocol'] = 'Resident V17 unsigned-quota-20000 seed123 five-fold pair model; 1 warmup + 20 repeats; fresh live graph features, 2211 log2 differences, fold averaging and calibration included; model loading and value-level reference acceptance excluded.'
    (OUT/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print('ACCEPTED', result['median_s'], flush=True)

if __name__ == '__main__':
    main()

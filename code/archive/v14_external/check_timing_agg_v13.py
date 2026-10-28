#!/usr/bin/env python3
"""Second-stage (aggregate) acceptance for the v13 timing paths — [v13d
rework] with UPSTREAM-FAILURE INHERITANCE, hard input-integrity guards and
DECLARED-seed binding.

The aggregate stage recomputes macro PCC from the per-peptide predictions
captured by check_timing_acceptance_v13.py and reconciles it against
benchmark_fullpool_seed10.json at each method's DECLARED checkpoint seed.
[v13d fault-injection fix] the recomputation can no longer override a
first-stage failure: before any aggregate verdict, every method's UPSTREAM
verdict and endpoint evidence (n_pair == 2211, shape_ok, all_finite,
values_match_independent_recompute) are validated. PASS is the only
acceptable upstream verdict for normal methods, PASS-AGG only for the three
registered aggregate-referenced methods, NOTE only for the registered
same-recipe proxy — and never at the expense of the endpoint algebra check.
Any upstream MISMATCH / ERROR / FAIL / missing status is appended to the
error list and forces a non-zero exit, even when the recomputed PCC matches
(a matching PCC is scale/shift-tolerant and can mask endpoint corruption).

Structural hard fails (exit 1) also cover: missing suite files, missing or
unknown method entries, entries without preds, missing/non-PASS tiger or
AMPActiPred acceptance records, and a wrong count of preds-carrying methods.

--selftest injects the three prescribed negative controls into a TEMP copy of
the real acceptance records (normal-method endpoint MISMATCH; ERROR with
preds retained; same-recipe proxy endpoint MISMATCH) and requires the gate to
fail on every one; the untouched real records must pass.

Usage: PYTHONPATH=. python scripts/check_timing_agg_v13.py [--selftest]
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import eval_seed10_benchmark as eb  # noqa: E402  (module-level: reads csvs only)

BR = ROOT / 'runs_review/benchmark_restricted'
TOL = 6e-4  # benchmark stores 4-decimal values; max rounding error 5e-4

# acceptance-suite method name -> (benchmark key, declared seed | 'NOTE' | 'CHILD')
EXPECTED = {
    'external': {
        'Das2021-regr': ('Das2021-regr (ours)', 129),
        'LLAMP+neg': ('LLAMP+neg', 129),
        'BERT-AmPEP+neg': ('BERT-AmPEP+neg', 129),
        'Deep-PIE+neg': ('Deep-PIE+neg', 129),
        'HydrAMP-arch-regr': ('HydrAMP-arch-regr (ours)', 129),
        'AMPLiT-regr vB': ('AMPLiT-regr vB (ours)', 129),
        'EvoGrad-strict': ('EvoGrad-strict', 123),
        'DeepAMP-CNN': ('DeepAMP-CNN', 123),
        'APEX-restr(lrfix)+neg': ('APEX-restr(lrfix)+neg', 123),
        'AMP-Designer+neg': ('AMP-Designer+neg', 123),
    },
    'new7': {
        'EIPpred+neg': ('EIPpred+neg', 123),
        'AMPActiPred+neg': ('AMPActiPred+neg', 'CHILD'),
        'AMPGen+neg': ('AMPGen+neg', 123),
        'AMPpredictor+neg': ('AMPpredictor+neg', 'NOTE'),
        'ANIA+neg': ('ANIA+neg', 123),
        'tsAMP-CS+neg': ('tsAMP-CS+neg', 123),
        'GGN-ABPMIC+neg': ('GGN-ABPMIC+neg', 123),
    },
    'new4_torch': {
        'HyperAMP+neg': ('HyperAMP+neg', 123),
        'SenseXAMP-Ecoli+neg': ('SenseXAMP-Ecoli+neg', 123),
    },
    'new4_tf': {
        'MBC-Attention+neg': ('MBC-Attention+neg', 123),
        'AMPlify-regr (ours)': ('AMPlify-regr (ours)', 123),
    },
}
# methods whose registered upstream verdict class is PASS-AGG (aggregate-level
# reference; their single archived csvs are stale artifacts)
AGG_REF = {'DeepAMP-CNN', 'APEX-restr(lrfix)+neg', 'AMP-Designer+neg'}


def endpoint_evidence_ok(entry):
    ep = entry.get('endpoint', {})
    return (ep.get('n_pair') == 2211 and ep.get('shape_ok', False)
            and ep.get('all_finite', False)
            and ep.get('values_match_independent_recompute', False))


def run_checks(br, bench):
    """Full gate over the acceptance records in directory `br`. Returns
    (errors, out_dict). Never raises on bad inputs — every problem becomes an
    entry of `errors` so the caller can force a non-zero exit."""
    errs, out = [], {'tolerance': TOL, 'methods': {}}
    acc = {}
    # ---- structural guards: suite files present, method sets exact ----
    for suite, exp in EXPECTED.items():
        f = br / f'timing_acceptance_v13_{suite}.json'
        if not f.exists():
            errs.append(f'missing suite file: {f.name}')
            continue
        d = json.loads(f.read_text()).get('methods', {})
        acc[suite] = d
        missing = sorted(set(exp) - set(d))
        unknown = sorted(set(d) - set(exp))
        if missing:
            errs.append(f'{suite}: missing method entries {missing}')
        if unknown:
            errs.append(f'{suite}: unknown method entries {unknown}')
    tf = br / 'timing_acceptance_v13_tiger.json'
    if not tf.exists():
        errs.append('missing timing_acceptance_v13_tiger.json')
    else:
        t = json.loads(tf.read_text()).get('arms', {})
        for arm in ('S4', 's4_single_n1n2'):
            if t.get(arm, {}).get('verdict') != 'PASS':
                errs.append(f'tiger {arm} acceptance not PASS')

    for suite, exp in EXPECTED.items():
        for m, (_bk, seed) in exp.items():
            entry = acc.get(suite, {}).get(m, {})
            if seed == 'CHILD':
                if entry.get('verdict') != 'PASS':
                    errs.append(f'{suite}/{m}: child-path acceptance not PASS')
                continue
            # ---- [v13d] upstream-verdict inheritance + endpoint evidence ----
            v = entry.get('verdict')
            ep_ok = endpoint_evidence_ok(entry)
            if seed == 'NOTE':
                allowed = (v == 'NOTE')
            elif m in AGG_REF:
                allowed = (v == 'PASS-AGG')
            else:
                allowed = (v == 'PASS')
            if not (allowed and ep_ok):
                errs.append(f'{m}: upstream verdict={v!r}, endpoint_evidence_ok={ep_ok} '
                            f'(aggregate stage may not override a first-stage failure)')
                out['methods'][m] = {'suite': suite, 'verdict': 'UPSTREAM-FAIL',
                                     'upstream_verdict': v,
                                     'endpoint_evidence_ok': ep_ok}
                continue
            if 'preds' not in entry:
                errs.append(f'{suite}/{m}: entry lacks preds')
                continue
            # ---- declared-seed aggregate reconciliation (diagnostic must
            #      still pass for PASS/NOTE entries) ----
            mac = eb.macro({s: float(v) for s, v in entry['preds'].items()})
            per = bench.get(_bk, {}).get('per_seed', {})
            if not per:
                errs.append(f'{m}: no benchmark entry for {_bk}')
                continue
            best = min(per, key=lambda s: abs(per[s] - mac))
            if seed == 'NOTE':
                out['methods'][m] = {'suite': suite, 'macro': round(mac, 4),
                                     'verdict': 'NOTE',
                                     'endpoint_evidence_ok': ep_ok,
                                     'diagnostic_best_seed': best,
                                     'note': 'same-recipe retrain: closest-seed diagnostic '
                                             'only; endpoint algebra verified upstream'}
                print(f"{m:24s} NOTE      macro={mac:.4f} (diag best {best})", flush=True)
                continue
            sv = per.get(str(seed))
            if sv is None:
                errs.append(f'{m}: declared seed {seed} absent from benchmark per_seed')
                continue
            diff = abs(sv - mac)
            verdict = 'PASS' if diff <= TOL else 'MISMATCH'
            if verdict == 'MISMATCH':
                errs.append(f'{m}: declared-seed {seed} aggregate mismatch')
            out['methods'][m] = {'suite': suite, 'macro': round(mac, 4),
                                 'declared_seed': seed, 'bench_value': sv,
                                 'abs_diff': round(diff, 6), 'verdict': verdict,
                                 'endpoint_evidence_ok': ep_ok,
                                 'diagnostic_best_seed': best}
            print(f"{m:24s} {verdict:9s} macro={mac:.4f} seed={seed} bench={sv} "
                  f"diff={diff:.6f}", flush=True)

    n_pred = sum(1 for r in out['methods'].values() if 'macro' in r)
    out['n_methods_with_preds_checked'] = n_pred
    out['n_expected_with_preds'] = 20
    if n_pred != 20:
        errs.append(f'expected 20 preds-carrying methods checked, got {n_pred}')
    return errs, out


def selftest(bench):
    """Fault-injection: the three prescribed upstream-failure records must
    each flip the gate to failure even though their recomputed PCC would
    pass; the untouched real records must pass."""
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        for s in EXPECTED:
            shutil.copy(BR / f'timing_acceptance_v13_{s}.json', T)
        shutil.copy(BR / 'timing_acceptance_v13_tiger.json', T)

        def edit(suite, method, **fields):
            f = T / f'timing_acceptance_v13_{suite}.json'
            d = json.loads(f.read_text())
            d['methods'][method].update(fields)
            f.write_text(json.dumps(d))

        ep_bad = {'n_pair': 2211, 'shape_ok': True, 'all_finite': True,
                  'values_match_independent_recompute': False}
        edit('external', 'Das2021-regr', verdict='MISMATCH', endpoint=ep_bad)
        edit('external', 'LLAMP+neg', verdict='ERROR', endpoint=ep_bad)  # ERROR, preds kept
        edit('new7', 'AMPpredictor+neg', verdict='MISMATCH', endpoint=ep_bad)
        errs, _ = run_checks(T, bench)
        caught = {k: any(k in e for e in errs)
                  for k in ('Das2021-regr', 'LLAMP+neg', 'AMPpredictor+neg')}
        print('SELFTEST injected upstream failures caught:', caught, flush=True)
        if not all(caught.values()):
            print('SELFTEST FAIL', flush=True)
            return 1
        # the untouched real records must pass
        errs_real, _ = run_checks(BR, bench)
        if errs_real:
            print('SELFTEST FAIL: real records unexpectedly failed', errs_real, flush=True)
            return 1
        print('SELFTEST PASS (upstream failures propagate; real records pass)', flush=True)
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    A = ap.parse_args()
    bench = json.loads((BR / 'benchmark_fullpool_seed10.json').read_text())
    if A.selftest:
        sys.exit(selftest(bench))
    errs, out = run_checks(BR, bench)
    if errs:
        out['structural_errors'] = errs
    (BR / 'timing_acceptance_v13_agg.json').write_text(json.dumps(out, indent=2))
    print('saved timing_acceptance_v13_agg.json', flush=True)
    if errs:
        print('AGG HARD FAIL:', len(errs), 'error(s)', flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

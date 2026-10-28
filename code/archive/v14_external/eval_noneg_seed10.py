#!/usr/bin/env python3
"""Benchmark evaluation for the noneg campaign (all methods trained on the
4,929 positives only; 10 seeds 120-129) — SAME endpoint as the main benchmark:
family-equal-weighted macro PCC over the 2,211-edge primary panel, log2 units,
EvoGrad harmonised per-peptide uM->ug/mL.

Sources (noneg artefacts only; +neg archive never touched):
  15 csv methods   BR/sota_<base>_noneg_s<seed>_preds.csv
                   (the pos4929 outputs of the experiment-12 trio are RENAMED
                   to this contract by the campaign chain)
  EvoGrad-A        BR/sota_evo_strict_noneg_s<seed>_preds.csv (uM; MW-converted)
  DeepAMP/APEX/AMPD  live re-inference from the noneg ckpt dirs
                   (sota_models/deep_amp_noneg, apex_pos4929, amp_designer_pos4929)
  TIGER-single-n2-noneg  per-seed pair evaluation of
                   runs_ablation/s4_single_n2_noneg/s4_single_n2_noneg_seed<seed>
                   (5-fold mean + calibrator, mirroring eval_cost_timing_tiger)

Also emits the paired comparison vs the +neg benchmark on the SAME seeds
(per-seed macro deltas, Wilcoxon signed-rank + paired t).

Usage: PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python scripts/eval_noneg_seed10.py
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
sys.path.insert(0, str(ROOT / 'baseline/sota_bench'))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SEEDS = tuple(range(120, 130))

# noneg method name -> (csv base in BR, benchmark key for the +neg pairing)
CSV_METHODS = {
    'Das2021-regr (ours)': ('sota_das_regr_noneg', 'Das2021-regr (ours)'),
    'LLAMP+neg': ('sota_llamp_noneg', 'LLAMP+neg'),
    'BERT-AmPEP+neg': ('sota_bert_ampep_noneg', 'BERT-AmPEP+neg'),
    'Deep-PIE+neg': ('sota_deep_pie_noneg', 'Deep-PIE+neg'),
    'HydrAMP-arch-regr (ours)': ('sota_hydramp_regr_noneg', 'HydrAMP-arch-regr (ours)'),
    'AMPLiT-regr vB (ours)': ('sota_amplit_regr_vB_noneg', 'AMPLiT-regr vB (ours)'),
    'MBC-Attention+neg': ('sota_mbc_attention_noneg', 'MBC-Attention+neg'),
    'AMPlify-regr (ours)': ('sota_amplify_regr_noneg', 'AMPlify-regr (ours)'),
    'EIPpred+neg': ('sota_EIPpred_noneg', 'EIPpred+neg'),
    'AMPActiPred+neg': ('sota_AMPActiPred_noneg', 'AMPActiPred+neg'),
    'AMPGen+neg': ('sota_AMPgen_noneg', 'AMPGen+neg'),
    'AMPpredictor+neg': ('sota_AMPpredictor_noneg', 'AMPpredictor+neg'),
    'ANIA+neg': ('sota_ANIA_noneg', 'ANIA+neg'),
    'tsAMP-CS+neg': ('sota_tsAMP-CS_noneg', 'tsAMP-CS+neg'),
    'GGN-ABPMIC+neg': ('sota_GGN-ABPMIC_noneg', 'GGN-ABPMIC+neg'),
    'HyperAMP+neg': ('sota_hyperamp_noneg', 'HyperAMP+neg'),
    'SenseXAMP-Ecoli+neg': ('sota_sensexamp_noneg', 'SenseXAMP-Ecoli+neg'),
}

import eval_seed10_benchmark as eb   # macro(): family-equal-weighted pair PCC


def macro_csv(base, s):
    f = BR / f'{base}_s{s}_preds.csv'
    d = pd.read_csv(f, index_col=0).iloc[:, 0].to_dict()
    return float(eb.macro(d))


def evo_macro(s):
    sys.path.insert(0, '/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/baseline/'
                       'AMP-potency-prediction-EvoGradient')
    from retrain_evo_strict import mw
    d = pd.read_csv(BR / f'sota_evo_strict_noneg_s{s}_preds.csv',
                    index_col=0).iloc[:, 0].to_dict()
    d = {q: v + float(np.log10(mw(q) / 1000.0)) for q, v in d.items()}
    return float(eb.macro(d))


def ckpt_trio(seeds=SEEDS):
    """Verbatim re-inference of the evaluator blocks against the NONEG ckpt
    dirs; per-seed macro for DeepAMP-CNN / APEX / AMP-Designer."""
    out = {}
    import retrain_deep_amp as da
    pep = list(dict.fromkeys(pd.read_csv(ROOT / 'runs_review/final11_peptides.csv').sequence))
    X = torch.from_numpy(da.encode(pep)[0]).float().to(device)
    per = {}
    for s in seeds:
        m_ = da.CNNRegressor().to(device)
        m_.load_state_dict(torch.load(BR / f'sota_models/deep_amp_noneg/cnn_seed{s}.pt',
                                      map_location='cpu')['model'])
        m_.eval()
        with torch.no_grad():
            per[s] = float(eb.macro(dict(zip(pep, m_(X).squeeze(-1).cpu().numpy().tolist()))))
    out['DeepAMP-CNN'] = per

    sys.path.insert(0, str(ROOT / 'baseline/sota_bench/apex-main'))
    from AMP_DL_model_twohead import AMP_model
    from utils import make_vocab, AAindex, onehot_encoding
    word2idx, _ = make_vocab()
    emb, _ = AAindex(str(ROOT / 'baseline/sota_bench/apex-main/aaindex1.csv'), word2idx)
    xt = torch.from_numpy(onehot_encoding(pep, 52, word2idx)[0]).long().to(device)
    per = {}
    for s in seeds:
        m_ = AMP_model(emb, emb.shape[1], 2, 128, 2048, 4, 1).to(device)
        m_.load_state_dict(torch.load(BR / f'sota_models/apex_pos4929/apexlr_{s}.pt',
                                      map_location='cpu')['model'])
        m_.eval()
        with torch.no_grad():
            p = (6.0 - m_(xt).cpu().numpy()[:, 0]).tolist()
        per[s] = float(eb.macro(dict(zip(pep, p))))
    out['APEX-restr(lrfix)+neg'] = per

    from transformers import BertTokenizer, GPT2Config, GPT2LMHeadModel
    sys.path.insert(0, str(ROOT / 'baseline/sota_bench/AMP-Designer'))
    import train_ampd_restricted_neg as an
    from torch.utils.data import DataLoader
    W = ROOT / 'baseline/sota_bench/models/amp_designer'
    tok = BertTokenizer(vocab_file=str(W / 'vocab.txt'))
    cfg = GPT2Config.from_pretrained(str(W / 'config.json'))
    ds = an.SeqDS(pep, np.zeros(len(pep)), tok)
    loader = DataLoader(ds, batch_size=64, collate_fn=an.collate)
    per = {}
    for s in seeds:
        gpt = GPT2LMHeadModel.from_pretrained(str(W / 'pytorch_model.bin'), config=cfg)
        m_ = an.CombinedModel(gpt).to(device)
        m_.load_state_dict(torch.load(BR / f'sota_models/amp_designer_pos4929/ampd_{s}.pt',
                                      map_location='cpu')['model'])
        m_.eval()
        preds, k = [], 0
        with torch.no_grad():
            for xb, _ in loader:
                y = m_(xb.to(device)).cpu().numpy().tolist()
                preds.extend(y)
        per[s] = float(eb.macro(dict(zip(pep, preds))))
    out['AMP-Designer+neg'] = per
    return out


def tiger_single_noneg(seeds=SEEDS):
    """Pair-benchmark macro for the s4_single_n2_noneg arm, per seed —
    the run_arm flow of eval_cost_timing_tiger (fresh graphs, per-fold
    z-replay, 5-fold mean, OOF calibrator)."""
    from eval_neg_twostage_arms import PANELS, KEEP_RAW
    from code.dataloader import build_graphs, preprocess_dataset
    from code.evaluation import eval_explicit_pairs, load_checkpoint_model
    from code.utils.config import load_config
    from code.utils.constants import CFU_COL, TARGET_COL
    from code.utils.scaling import apply_p_f_stats, apply_node_x_stats

    def build_panel(cfg, panel, pepcsv, pdbdir, keep):
        tmp = Path(f'/tmp/noneg_eval_{panel}.csv')
        df = pd.read_csv(pepcsv)
        df = df[df.sequence.astype(str).str.upper().str.strip().isin(keep)]
        o = pd.DataFrame({'sequence': df.sequence.astype(str).str.upper().str.strip()})
        mic = df.get('MIC_Escherichia_coli', df.get('MIC_E_coli_ug_mL', df.get('MIC_ug_per_mL')))
        o[TARGET_COL] = pd.to_numeric(mic, errors='coerce')
        o[CFU_COL] = '1E5 - 1E6'
        for c in ('n_terminus', 'c_terminus'):
            o[c] = df[c] if c in df.columns else ''
        o.to_csv(tmp, index=False)
        filt, _ = preprocess_dataset(str(tmp), str(pdbdir), cfg, f'noneg_eval_{panel}')
        graphs, rows, _ = build_graphs(filt, cfg)
        return graphs, rows

    per = {}
    for s in seeds:
        run = ROOT / f'runs_ablation/s4_single_n2_noneg/s4_single_n2_noneg_seed{s}'
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
        from code.utils.metrics import apply_calibrator
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
        key = ['family', 'anchor_sequence', 'query_sequence']
        m = ens[key + [YP]].assign(_fam=ens.family.astype(str).str.rstrip('/')).merge(
            K.assign(_fam=K.family.astype(str).str.rstrip('/'))[['_fam', 'anchor_sequence',
                                                                 'query_sequence']],
            on=['_fam', 'anchor_sequence', 'query_sequence'], how='inner')
        assert len(m) == 2211, len(m)
        yt = pd.read_csv(BR / 'eleven_way_predictions.csv')
        truth = {(f.strip('/'), a, q): t for f, a, q, t in
                 zip(yt.family, yt.anchor_sequence, yt.query_sequence,
                     yt.y_true_delta_log2_anchor_minus_query_03)}
        from scipy.stats import pearsonr
        ps = []
        for fam, g in m.groupby('_fam'):
            y_true = [truth[(fam, a, q)] for a, q in zip(g.anchor_sequence, g.query_sequence)]
            ps.append(pearsonr(y_true, g[YP].tolist())[0])
        per[s] = float(np.mean(ps))
        print(f'  TIGER-single-n2-noneg seed {s}: macro {per[s]:.4f}', flush=True)
    return {'TIGER-single-n2-noneg': per}


def main():
    results = {}
    for name, (base, _bk) in CSV_METHODS.items():
        results[name] = {s: macro_csv(base, s) for s in SEEDS}
        print(f'{name}: done', flush=True)
    results['EvoGrad-strict-A'] = {s: evo_macro(s) for s in SEEDS}
    results.update(ckpt_trio())
    print('[ckpt trio done]', flush=True)
    results.update(tiger_single_noneg())
    print('[tiger single noneg done]', flush=True)

    # ---- paired comparison vs +neg on the SAME seeds ----
    bench = json.loads((BR / 'benchmark_fullpool_seed10.json').read_text())
    from scipy.stats import wilcoxon, ttest_rel
    rows = []
    for name, per in results.items():
        v = np.array([per[s] for s in SEEDS if per[s] is not None], dtype=float)
        bk = ('TIGER-single' if name == 'TIGER-single-n2-noneg' else
              'EvoGrad-strict' if name == 'EvoGrad-strict-A' else name)
        neg = bench.get(bk, {}).get('per_seed', {})
        if neg:
            nv = np.array([neg[str(s)] for s in SEEDS if per.get(s) is not None])
            d = nv - v                      # +neg minus noneg
            try:
                w_p = float(wilcoxon(nv, v).pvalue)
            except Exception:
                w_p = float('nan')
            t_p = float(ttest_rel(nv, v).pvalue) if len(v) > 1 else float('nan')
        else:
            nv, d, w_p, t_p = None, None, float('nan'), float('nan')
        rows.append({'method': name,
                     'noneg_mean': round(float(v.mean()), 4),
                     'noneg_std': round(float(v.std()), 4),
                     'posneg_same_seeds': round(float(nv.mean()), 4) if nv is not None else None,
                     'delta_posneg_minus_noneg': round(float(d.mean()), 4) if d is not None else None,
                     'wilcoxon_p': round(w_p, 5), 'paired_t_p': round(t_p, 5),
                     'per_seed': {str(s): (round(per[s], 4) if per[s] is not None else None)
                                  for s in SEEDS}})
    rows.sort(key=lambda r: -r['noneg_mean'])
    out = {'protocol': 'noneg campaign: 4,929 positives only, seeds 120-129; main '
                       'benchmark endpoint (family-equal-weight macro PCC, 2,211-edge '
                       'primary panel, log2; Evo uM->ug/mL converted)',
           'pairing': '+neg values from benchmark_fullpool_seed10.json on the SAME seeds',
           'methods': rows}
    (BR / 'benchmark_noneg_seed10.json').write_text(json.dumps(out, indent=2))
    for r in rows:
        print(f"{r['method']:28s} noneg {r['noneg_mean']:.4f}±{r['noneg_std']:.4f} "
              f"+neg(same seeds) {r['posneg_same_seeds']} "
              f"Δ {r['delta_posneg_minus_noneg']} (p={r['wilcoxon_p']})", flush=True)
    print('saved benchmark_noneg_seed10.json', flush=True)


if __name__ == '__main__':
    main()

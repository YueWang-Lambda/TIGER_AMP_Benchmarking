"""Unified benchmark runner for published hemolysis predictors.

Protocol (identical to the in-house toxin runs):
  - labels: T=512 inequality-aware HC50 table, n=1596
    (runs_ablation_toxin/both/toxicity_labeled_dataset.csv)
  - CV: StratifiedKFold(5, shuffle, random_state=42) — same folds as every
    in-house model; decision threshold re-calibrated per fold on train
    probabilities (max F1 over 0.05..0.95)
  - external: 88-peptide internal_toxin_cohort panel; per-fold probabilities
    averaged, majority vote over per-fold calibrated hard predictions
  - metrics: Acc / P / R / F1 / MCC / AUC-ROC / AUC-PR (+ FN/FP externally)

Adapters live in ``code/toxin_bench/methods/`` and expose::

    class Adapter:
        def __init__(self, seed: int, gpu: int, fold: int, **kwargs) -> None
        def fit(self, df_train: pd.DataFrame) -> None    # columns incl. sequence, label
        def predict_proba(self, df: pd.DataFrame) -> np.ndarray

Run one method end-to-end (CV + external) in a single process so fold models
stay in memory (no heterogeneous serialization):

  PYTHONPATH=. python -m code.toxin_bench.run_bench --method hemopred --gpu 0
  PYTHONPATH=. python -m code.toxin_bench.run_bench --method lysepred --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
TIGER_ROOT = HERE.parent.parent
if str(TIGER_ROOT) not in sys.path:
    sys.path.insert(0, str(TIGER_ROOT))

from code.toxin_filter.metrics import (  # noqa: E402
    classification_metrics,
    metrics_to_row,
    summarize_cv,
)

DEFAULT_LABELS = TIGER_ROOT / "runs_ablation_toxin" / "both" / "toxicity_labeled_dataset.csv"
DEFAULT_PANEL = (
    TIGER_ROOT / "data" / "test_external" / "test_toxin_internal_toxin_cohort"
    / "internal_toxin_cohort_toxicity_labeled.csv"
)


def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        m = classification_metrics(y_true, (y_prob >= t).astype(int), y_prob)
        if m["f1"] > best_f1:
            best_f1, best_t = m["f1"], float(t)
    return best_t


def external_metrics(y: np.ndarray, yp: np.ndarray, pr: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    fn = int(((yp == 0) & (y == 1)).sum())
    fp = int(((yp == 1) & (y == 0)).sum())
    tn = int(((yp == 0) & (y == 0)).sum())
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, yp)),
        "precision": float(precision_score(y, yp, zero_division=0)),
        "recall": float(recall_score(y, yp, zero_division=0)),
        "specificity": tn / n_neg if n_neg else None,
        "f1": float(f1_score(y, yp, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, yp)),
        "auc_roc": float(roc_auc_score(y, pr)),
        "auc_pr": float(average_precision_score(y, pr)),
        "false_safe_rate": fn / n_pos if n_pos else None,
        "fn": fn,
        "fp": fp,
    }


def main():
    from code.toxin_bench.methods import METHODS, method_meta

    p = argparse.ArgumentParser(description="Unified hemolysis-predictor benchmark")
    p.add_argument("--method", required=True, choices=sorted(METHODS))
    p.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--panel-csv", type=Path, default=DEFAULT_PANEL)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out-root", type=Path, default=TIGER_ROOT / "runs_toxin_bench")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--skip-external", action="store_true")
    args = p.parse_args()

    n_splits = 2 if args.smoke else args.n_splits
    out_dir = args.out_root / (args.method + ("_smoke" if args.smoke else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.labels_csv)
    df["sequence"] = df["sequence"].astype(str).str.upper().str.strip()
    y = df["label"].to_numpy(dtype=int)
    panel = pd.read_csv(args.panel_csv)
    panel["sequence"] = panel["sequence"].astype(str).str.upper().str.strip()
    y_ext = panel["label"].to_numpy(dtype=int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    fold_rows = []
    ext_fold_probs, ext_fold_preds = [], []
    thresholds = []
    for fold_i, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        print(f"[{args.method}] Fold {fold_i}/{n_splits} train={len(tr)} val={len(va)} val_pos={int(y[va].sum())}", flush=True)
        adapter = METHODS[args.method](seed=args.seed + fold_i, gpu=args.gpu, fold=fold_i)
        train_df = df.iloc[tr].reset_index(drop=True)
        adapter.fit(train_df)

        va_prob = np.asarray(adapter.predict_proba(df.iloc[va].reset_index(drop=True)), dtype=float)
        thr = _best_threshold(y[tr], np.asarray(adapter.predict_proba(train_df), dtype=float))
        thresholds.append(thr)
        metrics = classification_metrics(y[va], (va_prob >= thr).astype(int), va_prob)
        fold_rows.append(metrics_to_row("cv", args.method, fold_i, metrics))
        print(f"  thr={thr:.2f} Acc={metrics['accuracy']:.4f} MCC={metrics['mcc']:.4f} "
              f"AUC-ROC={metrics['auc_roc']:.4f} AUC-PR={metrics['auc_pr']:.4f}", flush=True)

        if not args.skip_external:
            ext_prob = np.asarray(adapter.predict_proba(panel), dtype=float)
            ext_fold_probs.append(ext_prob)
            ext_fold_preds.append((ext_prob >= thr).astype(int))

    cv_df = pd.DataFrame(fold_rows)
    summary_df = summarize_cv(fold_rows)
    cv_df.to_csv(out_dir / "cv_fold_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)
    means = summary_df[summary_df["split"] == "cv_mean"].set_index("model")
    stds = summary_df[summary_df["split"] == "cv_std"].set_index("model")
    cols = ["accuracy", "precision", "recall", "f1", "mcc", "auc_roc", "auc_pr"]
    pretty = pd.DataFrame(
        [{"model": m, **{c: f"{means.loc[m, c]:.4f}±{stds.loc[m, c]:.4f}" for c in cols}} for m in means.index]
    )
    pretty.to_csv(out_dir / "summary_mean_std_pretty.csv", index=False)

    result = {
        "method": args.method,
        "meta": method_meta(args.method),
        "labels_csv": str(args.labels_csv),
        "panel_csv": str(args.panel_csv),
        "n": int(len(y)),
        "n_toxin": int(y.sum()),
        "folds": n_splits,
        "seed": args.seed,
        "thresholds": thresholds,
        "smoke": bool(args.smoke),
        "cv": pretty.to_dict(orient="records")[0] if len(pretty) else None,
    }

    if not args.skip_external and ext_fold_probs:
        ext_prob_mean = np.mean(ext_fold_probs, axis=0)
        ext_pred_maj = (np.mean(ext_fold_preds, axis=0) >= 0.5).astype(int)
        result["external"] = external_metrics(y_ext, ext_pred_maj, ext_prob_mean)
        pred_out = panel[["sequence"]].copy()
        for k, (pr, pd_) in enumerate(zip(ext_fold_probs, ext_fold_preds), start=1):
            pred_out[f"prob_fold{k}"] = pr
            pred_out[f"pred_fold{k}"] = pd_
        pred_out["prob"] = ext_prob_mean
        pred_out["pred"] = ext_pred_maj
        pred_out.to_csv(out_dir / "external_predictions.csv", index=False)
        print(f"[{args.method}] external: {json.dumps(result['external'], indent=None)}", flush=True)

    (out_dir / "run_result.json").write_text(json.dumps(result, indent=2))
    print(f"[{args.method}] wrote {out_dir}")


if __name__ == "__main__":
    main()

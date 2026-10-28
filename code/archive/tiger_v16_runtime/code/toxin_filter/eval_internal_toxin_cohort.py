#!/usr/bin/env python3
"""Evaluate published toxin checkpoints on the 88-peptide internal_toxin_cohort panel.

Reads the clean labels pack under
``data/test_external/test_toxin_internal_toxin_cohort/`` and scores the five-fold
``both`` (or chosen mode) classical-ML checkpoints under
``checkpoints/ablation_toxin/<mode>/checkpoints/``.

Optionally also scores deep-learning checkpoints (``--dl-ckpt-dir``:
``fusion_seq_glob`` / ``tiger_seq_glob`` / ``metric_learning`` .pt folds) and
TIGER-single GNN checkpoints (``--gnn-ckpt-dir``: ``tiger_single_gnn`` .pt
folds; PDBs resolved via ``--gnn-pdb-dirs``, placeholder graphs allowed only
for graph-free modality arms).

Writes predictions + metrics next to the labels pack (or ``--out-dir``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.toxin_filter.features import (  # noqa: E402
    ToxinDataset,
    apply_zscore,
    build_tabular_features,
    calculate_global_properties,
)

DEFAULT_LABELS = (
    ROOT
    / "data"
    / "test_external"
    / "test_toxin_internal_toxin_cohort"
    / "internal_toxin_cohort_toxicity_labeled.csv"
)
DEFAULT_PANEL = (
    ROOT
    / "data"
    / "test_external"
    / "test_toxin_internal_toxin_cohort"
    / "internal_toxin_cohort_hemolysis_active_micmin_le128.csv"
)
DEFAULT_CKPT = ROOT / "checkpoints" / "ablation_toxin" / "both" / "checkpoints"
DEFAULT_PREDICTED_PDB = (
    ROOT / "data" / "test_external" / "test_toxin_internal_toxin_cohort" / "pdb_predicted"
)
DEFAULT_TEST_PDB = ROOT / "data" / "3D_data_test_Rosetta"


def _load_panel(labels_csv: Path, panel_csv: Path | None) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    # Normalize column names used across packs.
    colmap = {c.lower(): c for c in labels.columns}
    seq_col = colmap.get("sequence") or colmap.get("seq")
    if seq_col is None:
        raise SystemExit(f"No sequence column in {labels_csv}")
    out = pd.DataFrame({"sequence": labels[seq_col].astype(str).str.upper().str.strip()})
    if "label" in labels.columns:
        out["label"] = labels["label"].astype(int)
    elif "gt_toxin_le512" in labels.columns:
        out["label"] = labels["gt_toxin_le512"].astype(int)
    else:
        raise SystemExit(f"No binary label column in {labels_csv}")
    if "hc50" in labels.columns:
        out["hc50"] = labels["hc50"]
    elif "HC50" in labels.columns:
        out["hc50"] = labels["HC50"]
    if panel_csv is not None and panel_csv.exists():
        panel = pd.read_csv(panel_csv)
        # Attach mic_min / richer columns when available (by sequence).
        pseq = "sequence" if "sequence" in panel.columns else ("Seq" if "Seq" in panel.columns else None)
        if pseq:
            panel = panel.copy()
            panel["_seq"] = panel[pseq].astype(str).str.upper().str.strip()
            keep = [
                c
                for c in panel.columns
                if c
                not in {
                    pseq,
                    "_seq",
                    "label",
                    "hc50",
                    "HC50",
                    "sequence",
                    "Seq",
                    "row_id",
                    "ID",
                    "label_rule",
                    "label_deterministic",
                }
            ]
            merge = panel[["_seq"] + keep].drop_duplicates("_seq")
            out = out.merge(merge, left_on="sequence", right_on="_seq", how="left").drop(columns=["_seq"])
    out = out.drop(columns=["row_id"], errors="ignore")
    out.insert(0, "row_id", np.arange(1, len(out) + 1))
    return out


def predict(df: pd.DataFrame, ckpt_dir: Path, feature_mode: str = "both", label_threshold: float = 512) -> pd.DataFrame:
    feat_df = pd.DataFrame(
        {
            "sequence": df["sequence"],
            "n_terminus": [""] * len(df),
            "c_terminus": [""] * len(df),
        }
    )
    X_raw = build_tabular_features(feat_df, feature_mode=feature_mode)
    model_names = sorted({p.name.split("_fold")[0] for p in ckpt_dir.glob("*_fold*.joblib")})
    if not model_names:
        raise SystemExit(f"No *_fold*.joblib under {ckpt_dir}")

    out = df.copy()
    all_probs: dict[str, np.ndarray] = {}
    load_skipped: dict[str, str] = {}
    for name in model_names:
        fold_probs = []
        fold_preds = []
        try:
            for fold in range(1, 6):
                path = ckpt_dir / f"{name}_fold{fold}.joblib"
                if not path.exists():
                    raise SystemExit(f"Missing checkpoint {path}")
                payload = joblib.load(path)
                stats = {k: np.asarray(v) for k, v in payload["zscore_stats"].items()}
                X = apply_zscore(X_raw, stats)
                model = payload["model"]
                thr = float(payload.get("decision_threshold", 0.5))
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(X)[:, 1]
                else:
                    scores = model.decision_function(X)
                    prob = 1.0 / (1.0 + np.exp(-scores))
                pred = (prob >= thr).astype(int)
                fold_probs.append(prob)
                fold_preds.append(pred)
                out[f"prob_{name}_fold{fold}"] = prob
                out[f"pred_{name}_fold{fold}"] = pred
        except SystemExit:
            raise
        except Exception as exc:  # unpickling is env-version sensitive (e.g. sklearn GB)
            load_skipped[name] = f"{type(exc).__name__}: {exc}"
            for col in list(out.columns):
                if col.startswith((f"prob_{name}_fold", f"pred_{name}_fold")):
                    out = out.drop(columns=col)
            print(f"[classical] skipping {name}: {load_skipped[name]}")
            continue
        mean_prob = np.mean(fold_probs, axis=0)
        maj_pred = (np.mean(fold_preds, axis=0) >= 0.5).astype(int)
        out[f"prob_{name}"] = mean_prob
        out[f"pred_{name}"] = maj_pred
        all_probs[name] = mean_prob
    out.attrs["load_skipped"] = load_skipped

    ens = np.mean(np.stack(list(all_probs.values()), axis=0), axis=0)
    out["prob_ensemble"] = ens
    out["pred_ensemble"] = (ens >= 0.5).astype(int)
    # Prefer CatBoost as the primary reported model when present.
    primary = "CatBoost" if "CatBoost" in all_probs else sorted(all_probs)[0]
    out["prob_toxin"] = out[f"prob_{primary}"]
    out["pred_toxin"] = out[f"pred_{primary}"]
    out["primary_model"] = primary
    out["label_threshold"] = label_threshold
    return out


def predict_dl(df: pd.DataFrame, ckpt_dir: Path) -> pd.DataFrame:
    """Score the 3 toxin_filter DL models (seq+global, no structures)."""
    import torch

    from code.toxin_filter.models_dl import FusionSeqGlob, MetricLearningNet, TIGERSeqGlob

    ckpt_dir = Path(ckpt_dir)
    payloads = {}
    for path in sorted(ckpt_dir.glob("*_fold*.pt")):
        name = path.name.split("_fold")[0]
        payloads.setdefault(name, {})[int(path.stem.split("fold")[-1])] = torch.load(path, map_location="cpu", weights_only=False)
    if not payloads:
        raise SystemExit(f"No *_fold*.pt under {ckpt_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequences = df["sequence"].astype(str).tolist()
    global_raw = np.stack([calculate_global_properties(s, "", "") for s in sequences])
    dummy = np.zeros(len(sequences), dtype=int)

    out = df.copy()
    for name, folds in payloads.items():
        fold_probs, fold_preds = [], []
        for fold in sorted(folds):
            payload = folds[fold]
            stats = {k: np.asarray(v) for k, v in payload["zscore_stats"].items()}
            g = apply_zscore(global_raw, stats).astype(np.float32)
            seq_style = "poap" if payload["kind"] == "fusion" else "integer"
            ds = ToxinDataset(sequences, g, dummy, max_len=payload["max_len"], seq_style=seq_style)
            loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)
            if payload["kind"] == "fusion":
                model = FusionSeqGlob(q_encoder="gru", glob_dim=payload["glob_dim"])
            elif payload["kind"] == "tiger":
                model = TIGERSeqGlob(max_len=payload["max_len"], emb_dim=128, glob_dim=payload["glob_dim"])
            else:
                model = MetricLearningNet(max_len=payload["max_len"], emb_dim=128, glob_dim=payload["glob_dim"])
            model.load_state_dict(payload["state_dict"])
            model.to(device).eval()
            probs = []
            with torch.no_grad():
                for seq_t, globf, _, _ in loader:
                    seq_t, globf = seq_t.to(device), globf.to(device)
                    logits = model(seq_t, globf)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    probs.append(torch.sigmoid(logits.view(-1)).cpu().numpy())
            prob = np.concatenate(probs).astype(float)
            pred = (prob >= float(payload["decision_threshold"])).astype(int)
            fold_probs.append(prob)
            fold_preds.append(pred)
            out[f"prob_{name}_fold{fold}"] = prob
            out[f"pred_{name}_fold{fold}"] = pred
        out[f"prob_{name}"] = np.mean(fold_probs, axis=0)
        out[f"pred_{name}"] = (np.mean(fold_preds, axis=0) >= 0.5).astype(int)
    return out


def predict_gnn(
    df: pd.DataFrame,
    ckpt_dir: Path,
    pdb_dirs: list[Path],
    *,
    feature_modalities: str,
    config: dict,
    label: str = "tiger_single_gnn",
) -> tuple[pd.DataFrame, dict]:
    """Score TIGER-single GNN folds; graphs z-scored per fold from checkpoint stats."""
    import torch

    from code.dataloader import SingleDataset, single_collate
    from code.toxin_filter.graphs_toxin import build_panel_graphs
    from code.toxin_filter.models_gnn import TIGERSingleToxin
    from code.utils.config import Config
    from code.utils.scaling import apply_p_f_stats, apply_node_x_stats

    ckpt_dir = Path(ckpt_dir)
    fold_files = sorted(ckpt_dir.glob("tiger_single_gnn_fold*.pt"), key=lambda p: int(p.stem.split("fold")[-1]))
    if not fold_files:
        raise SystemExit(f"No tiger_single_gnn_fold*.pt under {ckpt_dir}")
    if label == "tiger_single_gnn":
        raise SystemExit("predict_gnn requires a distinct label (e.g. tiger_single_gnn_gsh)")

    allow_placeholder = "g" not in feature_modalities.lower()
    graphs, kept_df, gstats = build_panel_graphs(
        df[["sequence", "label"]].copy(), pdb_dirs, Config(**config), allow_placeholder=allow_placeholder
    )
    if len(graphs) != len(df):
        raise SystemExit(f"Panel graphs lost rows: {len(graphs)}/{len(df)} (stats={gstats})")
    kept_df = kept_df.reset_index(drop=True)
    assert kept_df["sequence"].tolist() == df["sequence"].astype(str).tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = df.copy()
    name = label
    fold_probs, fold_preds = [], []
    for path in fold_files:
        fold = int(path.stem.split("fold")[-1])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        apply_p_f_stats(graphs, payload["p_stats"], None)
        apply_node_x_stats(graphs, payload["node_stats"], None)
        model = TIGERSingleToxin(Config(**config))
        model.load_state_dict(payload["state_dict"])
        model.to(device).eval()
        ds = SingleDataset(graphs, np.zeros(len(graphs)), list(range(len(graphs))))
        loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, collate_fn=single_collate)
        probs = []
        with torch.no_grad():
            for batch, _ in loader:
                probs.append(torch.sigmoid(model.logits(batch.to(device))).cpu().numpy())
        prob = np.concatenate(probs).astype(float)
        pred = (prob >= float(payload["decision_threshold"])).astype(int)
        fold_probs.append(prob)
        fold_preds.append(pred)
        out[f"prob_{name}_fold{fold}"] = prob
        out[f"pred_{name}_fold{fold}"] = pred
        # restore raw features before the next fold's stats are applied
        for g in graphs:
            if hasattr(g, "_p_f_raw"):
                g.p_f = g._p_f_raw.clone()
            if hasattr(g, "_x_s_raw"):
                g.x_s = g._x_s_raw.clone()
    out[f"prob_{name}"] = np.mean(fold_probs, axis=0)
    out[f"pred_{name}"] = (np.mean(fold_preds, axis=0) >= 0.5).astype(int)
    return out, gstats


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())
    n_pos = int(y_true.sum())
    n_neg = int((y_true == 0).sum())
    # Safety-oriented rates (toxic = positive class)
    recall = float(recall_score(y_true, y_pred, zero_division=0))  # sensitivity / toxic recall
    false_safe_rate = float(fn / n_pos) if n_pos else None  # FN among true toxics (= 1 - recall)
    specificity = float(tn / n_neg) if n_neg else None
    out = {
        "n": int(len(y_true)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else None,
        "auc_roc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "auc_pr": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "false_negative": fn,
        "false_positive": fp,
        "true_positive": tp,
        "true_negative": tn,
        "false_safe_rate": false_safe_rate,
        "false_safe_count": fn,
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": [[tn, fp], [fn, tp]],
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }
    return out


def _sequence_overlap_audit(panel_sequences: list[str], train_csv: Path) -> dict:
    """Compare external panel sequences against the toxin training label table."""
    if not train_csv.exists():
        return {"skipped": True, "reason": f"missing {train_csv}"}
    train = pd.read_csv(train_csv)
    seq_col = next((c for c in train.columns if c.lower() in {"sequence", "seq"}), None)
    if seq_col is None:
        return {"skipped": True, "reason": f"no sequence column in {train_csv}"}
    train_set = {str(s).upper().strip() for s in train[seq_col].tolist()}
    panel_set = {str(s).upper().strip() for s in panel_sequences}
    overlap = sorted(panel_set & train_set)
    return {
        "train_csv": str(train_csv),
        "n_train_sequences": len(train_set),
        "n_panel_sequences": len(panel_set),
        "n_overlap": len(overlap),
        "frac_panel_in_train": float(len(overlap) / len(panel_set)) if panel_set else None,
        "overlap_sequences": overlap,
        "note": (
            "n_overlap=0 means the 88-peptide external panel shares no exact sequences "
            "with the published toxin training table."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--panel-csv", type=Path, default=DEFAULT_PANEL)
    p.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--feature-mode", default="both", choices=["both", "global", "sequence"])
    p.add_argument("--label-threshold", type=float, default=512, help="HC50 toxin boundary used for the labels (ug/mL)")
    p.add_argument(
        "--dl-ckpt-dir",
        type=Path,
        default=None,
        action="append",
        help="Dir with fusion/tiger/metric *_fold*.pt checkpoints (repeatable)",
    )
    p.add_argument(
        "--gnn-ckpt-dir",
        type=Path,
        default=None,
        action="append",
        help="Dir with tiger_single_gnn_fold*.pt checkpoints; arm name = parent dir (repeatable)",
    )
    p.add_argument(
        "--gnn-pdb-dirs",
        type=Path,
        default=None,
        action="append",
        help="PDB pools for panel graph building, resolution order (default: pdb_predicted, 3D_data_test_Rosetta)",
    )
    p.add_argument(
        "--train-toxin-csv",
        type=Path,
        default=ROOT / "data" / "trainval_dbassp" / "toxin" / "toxicity_labeled_dataset.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "test_external" / "test_toxin_internal_toxin_cohort",
    )
    args = p.parse_args(argv)

    if not args.labels_csv.exists():
        raise SystemExit(f"Missing labels: {args.labels_csv}")
    if not args.ckpt_dir.exists():
        raise SystemExit(f"Missing checkpoints: {args.ckpt_dir}")

    df = _load_panel(args.labels_csv, args.panel_csv if args.panel_csv.exists() else None)
    pred = predict(df, args.ckpt_dir, feature_mode=args.feature_mode, label_threshold=args.label_threshold)

    extra_models: list[str] = []
    gnn_summaries: dict[str, dict] = {}
    if args.dl_ckpt_dir:
        for dl_dir in args.dl_ckpt_dir:
            pred = predict_dl(pred, dl_dir)
        extra_models += [m for m in ("fusion_seq_glob", "tiger_seq_glob", "metric_learning") if f"prob_{m}" in pred.columns]
    if args.gnn_ckpt_dir:
        pdb_dirs = args.gnn_pdb_dirs or [DEFAULT_PREDICTED_PDB, DEFAULT_TEST_PDB]
        for gnn_dir in args.gnn_ckpt_dir:
            arm = gnn_dir.parent.name
            meta_path = gnn_dir.parent / "run_meta.json"
            if not meta_path.exists():
                raise SystemExit(f"Missing {meta_path} (needed to rebuild the GNN config)")
            meta = json.loads(meta_path.read_text())
            label = f"tiger_single_gnn_{arm}"
            pred, gstats = predict_gnn(
                pred,
                gnn_dir,
                pdb_dirs,
                feature_modalities=meta["config"]["feature_modalities"],
                config=meta["config"],
                label=label,
            )
            extra_models.append(label)
            gnn_summaries[label] = {"ckpt_dir": str(gnn_dir), "feature_modalities": meta["config"]["feature_modalities"], "graph_stats": gstats}

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "internal_toxin_cohort_hemolysis_predictions_both.csv"
    keep = [
        c
        for c in [
            "row_id",
            "sequence",
            "hc50",
            "label",
            "mic_min",
            "prob_toxin",
            "pred_toxin",
            "primary_model",
            "prob_CatBoost",
            "pred_CatBoost",
            "prob_XGB",
            "pred_XGB",
            "prob_LGBM",
            "pred_LGBM",
            "prob_RF",
            "pred_RF",
            "prob_SVM",
            "pred_SVM",
            "prob_GB",
            "pred_GB",
            "prob_MLP",
            "pred_MLP",
            "prob_Adaboost",
            "pred_Adaboost",
            "prob_LR",
            "pred_LR",
            "prob_ensemble",
            "pred_ensemble",
            "label_threshold",
        ]
        + [c for c in pred.columns if c.startswith(("prob_fusion", "pred_fusion", "prob_tiger", "pred_tiger", "prob_metric", "pred_metric")) and "_fold" not in c]
        if c in pred.columns
    ]
    pred[keep].to_csv(pred_path, index=False)
    pred.to_csv(out_dir / "internal_toxin_cohort_hemolysis_predictions_both_full.csv", index=False)

    y = pred["label"].to_numpy(dtype=int)
    summary = {
        "n": int(len(pred)),
        "feature_mode": args.feature_mode,
        "checkpoints": str(args.ckpt_dir),
        "labels_csv": str(args.labels_csv),
        "primary_model": str(pred["primary_model"].iloc[0]),
        "metrics_reported": [
            "accuracy",
            "precision",
            "recall",
            "sensitivity",
            "specificity",
            "f1",
            "mcc",
            "auc_roc",
            "auc_pr",
            "false_safe_rate",
            "confusion_matrix",
        ],
        "sequence_overlap_audit": _sequence_overlap_audit(
            pred["sequence"].astype(str).tolist(), args.train_toxin_csv
        ),
    }
    if gnn_summaries:
        summary["gnn_runs"] = gnn_summaries
    classical_skipped = dict(pred.attrs.get("load_skipped", {}))
    if classical_skipped:
        summary["classical_load_skipped"] = classical_skipped
        summary["classical_load_skipped_note"] = (
            "Models whose published joblib folds could not be unpickled in this "
            "environment (native-module version mismatch); use the published "
            "predictions CSV for their external numbers."
        )
    reported = ["toxin", "CatBoost", "XGB", "ensemble", *extra_models]
    for name in reported:
        pcol = f"pred_{name}"
        probcol = f"prob_{name}"
        if pcol not in pred.columns:
            continue
        summary[name] = _metrics(y, pred[pcol].to_numpy(dtype=int), pred[probcol].to_numpy(dtype=float))

    (out_dir / "internal_toxin_cohort_prediction_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Compact safety-focused CSV for primary models
    rows = []
    for name in reported:
        if name not in summary:
            continue
        m = summary[name]
        rows.append(
            {
                "model": name,
                "n": m["n"],
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall_toxic": m["recall"],
                "false_safe_rate": m["false_safe_rate"],
                "specificity": m["specificity"],
                "f1": m["f1"],
                "mcc": m["mcc"],
                "auc_roc": m["auc_roc"],
                "auc_pr": m["auc_pr"],
                "tp": m["true_positive"],
                "fp": m["false_positive"],
                "fn": m["false_negative"],
                "tn": m["true_negative"],
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "internal_toxin_cohort_safety_metrics.csv", index=False)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {pred_path}")
    print(f"Wrote {out_dir / 'internal_toxin_cohort_safety_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

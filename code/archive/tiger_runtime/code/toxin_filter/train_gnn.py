"""5-fold CV training of the TIGER-single encoder on HC50 toxicity labels.

Protocol matches ``code/toxin_filter.run`` (the classical-ML / DL toxin runs):
StratifiedKFold(n_splits, shuffle, seed) — default seed 42 aligns with the
archived ``runs_ablation_toxin`` folds so CV rows are directly comparable —
BCEWithLogitsLoss(pos_weight=n_neg/n_pos) + WeightedRandomSampler balancing,
AdamW, early stop on val AUC-ROC, hard-decision threshold re-calibrated on
train probabilities (max F1 over 0.05..0.95).

Model: ``TIGERSingleToxin`` = full ``PeptideEncoder`` (graph/seq/global,
``feature_modalities`` gating) + classification head. Per-fold z-scoring of
``p_f`` / node ``x_s`` fits on the training rows only
(``code/utils/scaling.py``), mirroring the MIC training pipeline.

Run from TIGER root:
  PYTHONPATH=. python -m code.toxin_filter.train_gnn --feature-modalities gsh --gpu 0
  PYTHONPATH=. python -m code.toxin_filter.train_gnn --feature-modalities gs  --gpu 0
  PYTHONPATH=. python -m code.toxin_filter.train_gnn --feature-modalities gsh --smoke
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler

HERE = Path(__file__).resolve().parent
TIGER_ROOT = HERE.parent.parent
if str(TIGER_ROOT) not in sys.path:
    sys.path.insert(0, str(TIGER_ROOT))

from code.dataloader import SingleDataset, make_batch, single_collate  # noqa: E402
from code.toxin_filter.graphs_toxin import build_toxin_graphs  # noqa: E402
from code.toxin_filter.metrics import (  # noqa: E402
    classification_metrics,
    format_metrics_table,
    metrics_to_row,
    summarize_cv,
)
from code.toxin_filter.models_gnn import TIGERSingleToxin  # noqa: E402
from code.utils.config import Config  # noqa: E402
from code.utils.scaling import (  # noqa: E402
    apply_p_f_stats,
    apply_node_x_stats,
    fit_p_f_stats,
    fit_node_x_stats,
)

MODEL_NAME = "tiger_single_gnn"
DEFAULT_LABELS = TIGER_ROOT / "runs_ablation_toxin" / "both" / "toxicity_labeled_dataset.csv"


def parse_args():
    p = argparse.ArgumentParser(description="TIGER-single encoder, HC50 toxicity 5-fold CV")
    p.add_argument("--label-csv", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--pdb-dir", type=Path, default=TIGER_ROOT / "data" / "3D_data_train_eva_Rosetta")
    p.add_argument("--cache-dir", type=Path, default=TIGER_ROOT / "artifacts" / "cache_toxin")
    p.add_argument("--feature-modalities", default="gps", choices=["gps", "ps", "p", "s", "g", "pg", "sg"])
    p.add_argument("--threshold", type=float, default=512.0)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42, help="fold split seed; 42 matches archived toxin runs")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--gpu", type=int, default=0, help="GPU index; -1 for CPU")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--force-rebuild-cache", action="store_true")
    p.add_argument("--smoke", action="store_true", help="2 folds x 3 epochs sanity run")
    return p.parse_args()


def make_config(args) -> Config:
    # Encoder settings mirror the released TIGER-single (panel 07 / benchmark_single)
    # plus Config defaults: gatv2 x2, attention pooling, integer seq, cross_qs fusion.
    cfg = Config(
        model_kind="single",
        feature_modalities=args.feature_modalities,
        structure_features="se",
        seq_encoding="integer",
        fusion="attention",
        fusion_attn_mode="cross_qs",
        gnn_type="gatv2",
        graph_pooling="attention",
        emb_dim=90,
        num_layer=2,
        dropout_ratio=0.35,
        max_len=50,
        min_len=6,
        p_feature_scaling="zscore",
        node_feature_scaling="zscore",
        include_node_coords=False,
        use_cfu_feature=False,
        use_cfu_protocol=False,
        feature_path=str(TIGER_ROOT / "data" / "metadata" / "features.txt"),
        train_pdb_dir=str(args.pdb_dir),
    )
    return cfg


def _class_weights(y: np.ndarray) -> torch.Tensor:
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)
    return torch.tensor([n_neg / n_pos], dtype=torch.float32)


def _balanced_loader(ds: SingleDataset, y: np.ndarray, batch_size: int) -> DataLoader:
    counts = np.bincount(y.astype(int), minlength=2).astype(float)
    counts[counts == 0] = 1.0
    weights = 1.0 / counts[y.astype(int)]
    sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(weights), replacement=True)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, collate_fn=single_collate)


@torch.no_grad()
def _predict_probs(model, graphs, indices, device, batch_size=256):
    model.eval()
    ds = SingleDataset(graphs, np.zeros(len(graphs)), indices)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=single_collate)
    probs = []
    for batch, _ in loader:
        batch = batch.to(device)
        probs.append(torch.sigmoid(model.logits(batch)).cpu().numpy())
    return np.concatenate(probs).astype(float)


def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        m = classification_metrics(y_true, (y_prob >= t).astype(int), y_prob)
        if m["f1"] > best_f1:
            best_f1, best_t = m["f1"], float(t)
    return best_t


def train_fold(
    model,
    graphs,
    labels,
    tr_idx,
    va_idx,
    *,
    device,
    cfg: Config,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    seed: int,
    eval_every: int = 1,
):
    """Train one fold; returns (fold_metrics, checkpoint_payload)."""
    torch.manual_seed(seed)

    # Per-fold z-score fit on train rows only (mirrors MIC train.py).
    p_stats = fit_p_f_stats(graphs, tr_idx)
    node_stats = fit_node_x_stats(graphs, tr_idx)
    apply_p_f_stats(graphs, p_stats, list(range(len(graphs))))
    apply_node_x_stats(graphs, node_stats, list(range(len(graphs))))

    y_tr = labels[tr_idx]
    tr_ds = SingleDataset(graphs, labels, tr_idx)
    tr_loader = _balanced_loader(tr_ds, y_tr, batch_size)
    pos_weight = _class_weights(y_tr).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state, best_auc, stale = None, -1.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch, y in tr_loader:
            batch, y = batch.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model.logits(batch), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        if epoch % eval_every and epoch != epochs:
            continue
        va_prob = _predict_probs(model, graphs, va_idx, device)
        auc = classification_metrics(labels[va_idx], (va_prob >= 0.5).astype(int), va_prob)["auc_roc"]
        if np.isfinite(auc) and auc > best_auc:
            best_auc, stale = auc, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += eval_every
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    tr_prob = _predict_probs(model, graphs, tr_idx, device)
    thr = _best_threshold(y_tr, tr_prob)
    va_prob = _predict_probs(model, graphs, va_idx, device)
    metrics = classification_metrics(labels[va_idx], (va_prob >= thr).astype(int), va_prob)

    # Restore raw features so the next fold re-fits cleanly.
    for g in graphs:
        if hasattr(g, "_p_f_raw"):
            g.p_f = g._p_f_raw.clone()
        if hasattr(g, "_x_s_raw"):
            g.x_s = g._x_s_raw.clone()

    payload = {
        "model_name": MODEL_NAME,
        "fold": None,
        "feature_modalities": cfg.feature_modalities,
        "state_dict": model.state_dict(),
        "decision_threshold": thr,
        "p_stats": p_stats,
        "node_stats": node_stats,
        "train_idx": np.asarray(tr_idx).tolist(),
        "val_idx": np.asarray(va_idx).tolist(),
        "best_val_auc": best_auc,
        "metrics": metrics,
    }
    return metrics, payload


def main():
    args = parse_args()
    suffix = "_smoke" if args.smoke else ""
    n_splits = 2 if args.smoke else args.n_splits
    epochs = 3 if args.smoke else args.epochs
    out_dir = args.out_dir or (TIGER_ROOT / "runs_toxin_single" / f"{args.feature_modalities}{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    cfg = make_config(args)

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    graphs, kept_df, gstats = build_toxin_graphs(
        args.label_csv, args.pdb_dir, cfg, cache_dir=args.cache_dir, force_rebuild=args.force_rebuild_cache
    )
    sequences = kept_df["sequence"].tolist()
    labels = kept_df["label"].to_numpy(dtype=int)
    print(f"graphs={len(graphs)} failed={gstats['graph_failed_rows']} toxin={int(labels.sum())} non_toxin={int((labels==0).sum())}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    fold_rows = []
    for fold_i, (tr, va) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        print(f"======== Fold {fold_i}/{n_splits} train={len(tr)} val={len(va)} val_pos={int(labels[va].sum())}")
        model = TIGERSingleToxin(cfg).to(device)
        metrics, payload = train_fold(
            model, graphs, labels, tr, va,
            device=device, cfg=cfg, epochs=epochs, batch_size=args.batch_size,
            lr=args.lr, weight_decay=args.weight_decay, patience=args.patience,
            seed=args.seed + fold_i,
        )
        payload["fold"] = fold_i
        torch.save(payload, ckpt_dir / f"{MODEL_NAME}_fold{fold_i}.pt")
        fold_rows.append(metrics_to_row("cv", MODEL_NAME, fold_i, metrics))
        print(f"  Acc={metrics['accuracy']:.4f} P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
              f"F1={metrics['f1']:.4f} MCC={metrics['mcc']:.4f} AUC-ROC={metrics['auc_roc']:.4f} AUC-PR={metrics['auc_pr']:.4f}")

    cv_df = pd.DataFrame(fold_rows)
    summary_df = summarize_cv(fold_rows)
    cv_df.to_csv(out_dir / "cv_fold_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)
    format_metrics_table(summary_df).to_csv(out_dir / "summary_metrics_pretty.csv", index=False)

    means = summary_df[summary_df["split"] == "cv_mean"].set_index("model")
    stds = summary_df[summary_df["split"] == "cv_std"].set_index("model")
    cols = ["accuracy", "precision", "recall", "f1", "mcc", "auc_roc", "auc_pr"]
    rows = [{"model": m, **{c: f"{means.loc[m, c]:.4f}±{stds.loc[m, c]:.4f}" for c in cols}} for m in means.index]
    pd.DataFrame(rows).to_csv(out_dir / "summary_mean_std_pretty.csv", index=False)

    meta = {
        "model": MODEL_NAME,
        "label_csv": str(args.label_csv),
        "pdb_dir": str(args.pdb_dir),
        "graph_stats": gstats,
        "n": int(len(labels)),
        "n_toxin": int(labels.sum()),
        "n_non_toxin": int((labels == 0).sum()),
        "threshold": args.threshold,
        "n_splits": n_splits,
        "seed": args.seed,
        "epochs": epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "smoke": bool(args.smoke),
        "config": cfg.to_dict(),
        "note": (
            "TIGER-single encoder (PeptideEncoder g/s/h modalities, cross_qs attention fusion) "
            "with binary BCE head on HC50<=512 labels; fold split seed=42 aligns with archived "
            "runs_ablation_toxin folds; per-fold p_f/node_x z-score fit on train rows only."
        ),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    cfg.save(out_dir / "config.json")
    print(f"Wrote metrics + checkpoints to {out_dir}")


if __name__ == "__main__":
    main()

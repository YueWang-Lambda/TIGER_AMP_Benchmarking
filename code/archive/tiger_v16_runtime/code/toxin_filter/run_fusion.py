"""Fusion experiment:理化特征(XGB) × 结构模态(gsh) (runs_toxin_fusion/).

L1  probability ensemble : fixed 0.5/0.5 logit-mean + OOF-MCC-weighted variant
L2  stacking             : LR meta-learner over [logit(p_xgb), logit(p_gsh), diff]
                           with clean per-fold OOF (meta trained on other folds)
L3  XGB + OOF embeddings : XGB over [31-D tabular | 90-D gsh OOF embedding]
                           (train peptides use the checkpoint of their own val
                           fold -> leakage-free; external uses 5-fold mean)

Protocol: identical labels/folds/threshold calibration as every in-house run;
baselines (retrained XGB, gsh) recomputed fresh here so all numbers share one
environment (ccseg) and one feature pipeline.

Run: PYTHONPATH=. python -m code.toxin_filter.run_fusion --gpu 0
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
TIGER_ROOT = HERE.parent.parent
if str(TIGER_ROOT) not in sys.path:
    sys.path.insert(0, str(TIGER_ROOT))

from code.dataloader import SingleDataset, single_collate  # noqa: E402
from code.toxin_filter.features import apply_zscore, build_tabular_features  # noqa: E402
from code.toxin_filter.graphs_toxin import build_panel_graphs, build_toxin_graphs  # noqa: E402
from code.toxin_filter.metrics import classification_metrics  # noqa: E402
from code.toxin_filter.models_gnn import TIGERSingleToxin  # noqa: E402
from code.toxin_bench.run_bench import external_metrics  # noqa: E402
from code.utils.config import Config  # noqa: E402
from code.utils.scaling import apply_p_f_stats, apply_node_x_stats  # noqa: E402

LABELS = TIGER_ROOT / "runs_ablation_toxin" / "both" / "toxicity_labeled_dataset.csv"
PDB_DIR = TIGER_ROOT / "data" / "3D_data_train_eva_Rosetta"
XGB_CKPT = TIGER_ROOT / "runs_toxin_single" / "dl_baselines" / "checkpoints"
GSH_DIR = TIGER_ROOT / "runs_toxin_single" / "gsh"
PANEL_88 = TIGER_ROOT / "data" / "test_external" / "test_toxin_internal_toxin_cohort" / "internal_toxin_cohort_toxicity_labeled.csv"
PANEL_200 = TIGER_ROOT / "runs_toxin_bench" / "external_testset_merged_200.csv"
PANEL_PDB_DIRS = [
    TIGER_ROOT / "data" / "test_external" / "test_toxin_internal_toxin_cohort" / "pdb_predicted",
    TIGER_ROOT / "data" / "3D_data_test_Rosetta",
]
OUT = TIGER_ROOT / "runs_toxin_fusion"

import joblib  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def best_threshold(y, prob):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        m = classification_metrics(y, (prob >= t).astype(int), prob)
        if m["f1"] > best_f1:
            best_f1, best_t = m["f1"], float(t)
    return best_t


def xgb_features(df: pd.DataFrame) -> np.ndarray:
    feat = pd.DataFrame({
        "sequence": df["sequence"].astype(str).str.upper().str.strip(),
        "n_terminus": df.get("n_terminus", pd.Series([""] * len(df))),
        "c_terminus": df.get("c_terminus", pd.Series([""] * len(df))),
    })
    return build_tabular_features(feat, feature_mode="both")


class GshRunner:
    """Loads the 5 gsh fold checkpoints; serves OOF probs/embeddings + external."""

    def __init__(self, graphs, labels, device):
        import json as _json

        meta = _json.loads((GSH_DIR / "run_meta.json").read_text())
        self.cfg_dict = meta["config"]
        self.cfg = Config(**self.cfg_dict)
        self.graphs = graphs
        self.labels = labels
        self.device = device
        self.folds = []
        for k in range(1, 6):
            p = torch.load(GSH_DIR / "checkpoints" / f"tiger_single_gnn_fold{k}.pt", map_location="cpu", weights_only=False)
            self.folds.append(p)

    def _apply_stats(self, payload):
        apply_p_f_stats(self.graphs, payload["p_stats"], None)
        apply_node_x_stats(self.graphs, payload["node_stats"], None)

    def _restore(self):
        for g in self.graphs:
            if hasattr(g, "_p_f_raw"):
                g.p_f = g._p_f_raw.clone()
            if hasattr(g, "_x_s_raw"):
                g.x_s = g._x_s_raw.clone()

    @torch.no_grad()
    def forward_fold(self, k: int):
        """Returns (probs, embeddings, representation) for ALL graphs under fold-k stats."""
        payload = self.folds[k - 1]
        self._apply_stats(payload)
        model = TIGERSingleToxin(self.cfg).to(self.device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        ds = SingleDataset(self.graphs, np.zeros(len(self.graphs)), list(range(len(self.graphs))))
        loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=single_collate)
        probs, embs = [], []
        for batch, _ in loader:
            batch = batch.to(self.device)
            e = model.encoder(batch)
            probs.append(torch.sigmoid(model.head(e).squeeze(-1)).cpu().numpy())
            embs.append(e.cpu().numpy())
        self._restore()
        return np.concatenate(probs), np.concatenate(embs)


def cv_metrics_oof(y, prob):
    thr = best_threshold(y, prob)
    m = classification_metrics(y, (prob >= thr).astype(int), prob)
    return thr, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    OUT.mkdir(exist_ok=True)

    df = pd.read_csv(LABELS)
    df["sequence"] = df["sequence"].astype(str).str.upper().str.strip()
    y = df["label"].to_numpy(int)
    n = len(df)
    X_raw = xgb_features(df)

    # ---- XGB retrained folds: OOF probs + stored fold indices ----
    xgb_folds = []
    for k in range(1, 6):
        xgb_folds.append(joblib.load(XGB_CKPT / f"XGB_fold{k}.joblib"))
    p_xgb_oof = np.zeros(n)
    for p in xgb_folds:
        va = np.asarray(p["val_idx"])
        X = apply_zscore(X_raw, {k2: np.asarray(v) for k2, v in p["zscore_stats"].items()})
        p_xgb_oof[va] = p["model"].predict_proba(X[va])[:, 1]

    # ---- gsh folds: OOF probs + OOF embeddings ----
    graphs, kept_df, _ = build_toxin_graphs(LABELS, PDB_DIR, Config(**json.loads((GSH_DIR / "run_meta.json").read_text())["config"]))
    assert kept_df["sequence"].tolist() == df["sequence"].tolist(), "graph cache order mismatch"
    gsh = GshRunner(graphs, y, device)
    p_gsh_oof = np.zeros(n)
    E_oof = np.zeros((n, 90), dtype=np.float32)
    for k, payload in enumerate(gsh.folds, start=1):
        va = np.asarray(payload["val_idx"])
        probs, embs = gsh.forward_fold(k)
        p_gsh_oof[va] = probs[va]
        E_oof[va] = embs[va]

    # ---- baselines (fresh, single environment, ORIGINAL protocol:
    #      per-fold train-calibrated thresholds for CV; majority vote externally) ----
    def fold_preds_majority(fold_prob_list, fold_thr_list):
        return (np.mean([(p >= t).astype(int) for p, t in zip(fold_prob_list, fold_thr_list)], axis=0) >= 0.5).astype(int)

    xgb_thr = [float(p["decision_threshold"]) for p in xgb_folds]
    gsh_thr = [float(p["decision_threshold"]) for p in gsh.folds]
    results = {"baselines": {}}
    for name, prob in [("XGB", p_xgb_oof), ("gsh", p_gsh_oof)]:
        # CV with each fold's own train-calibrated threshold on its val rows
        preds = np.zeros(n, dtype=int)
        thrs = xgb_thr if name == "XGB" else gsh_thr
        folds = xgb_folds if name == "XGB" else gsh.folds
        for f, t in zip(folds, thrs):
            va = np.asarray(f["val_idx"])
            preds[va] = (prob[va] >= t).astype(int)
        m = classification_metrics(y, preds, prob)
        results["baselines"][name] = {"cv": m, "fold_thresholds": thrs}

    # ================= L1: logit-mean ensemble =================
    lx, lg = logit(p_xgb_oof), logit(p_gsh_oof)
    tr_all = np.asarray(xgb_folds[0]["train_idx"])  # placeholder, per-fold below

    def cv_eval_traincal(prob, name):
        """CV with the fusion's threshold calibrated on TRAIN rows' OOF probs."""
        preds = np.zeros(n, dtype=int)
        for f in xgb_folds:
            tr, va = np.asarray(f["train_idx"]), np.asarray(f["val_idx"])
            t = best_threshold(y[tr], prob[tr])
            preds[va] = (prob[va] >= t).astype(int)
        return classification_metrics(y, preds, prob)

    l1 = {}
    for tag, w in [("fixed", 0.5)]:
        prob = 1 / (1 + np.exp(-(w * lx + (1 - w) * lg)))
        l1[tag] = {"w": w, "cv": cv_eval_traincal(prob, tag), "oof_prob": prob}
    best_w, best_mcc = 0.5, -1
    for w in np.linspace(0.1, 0.9, 17):
        mcc = matthews_corrcoef(y, (1 / (1 + np.exp(-(w * lx + (1 - w) * lg)))) >= 0.5)
        if mcc > best_mcc:
            best_mcc, best_w = mcc, float(w)
    prob = 1 / (1 + np.exp(-(best_w * lx + (1 - best_w) * lg)))
    l1["oof_weighted"] = {"w": best_w, "cv": cv_eval_traincal(prob, "w"), "oof_prob": prob}

    # ================= L2: stacking (clean per-fold OOF) =================
    Z = np.stack([lx, lg, lx - lg], axis=1)
    p_l2_oof = np.zeros(n)
    for p in xgb_folds:  # reuse the same fold indices
        va, tr = np.asarray(p["val_idx"]), np.asarray(p["train_idx"])
        meta = LogisticRegression(max_iter=1000)
        meta.fit(Z[tr], y[tr])
        p_l2_oof[va] = meta.predict_proba(Z[va])[:, 1]
    l2 = {"cv": cv_eval_traincal(p_l2_oof, "l2"), "oof_prob": p_l2_oof}

    # ================= L3: XGB over [tabular | OOF embedding] =================
    p_l3_oof = np.zeros(n)
    xgb2_folds = []
    for p in xgb_folds:
        va, tr = np.asarray(p["val_idx"]), np.asarray(p["train_idx"])
        stats = {k2: np.asarray(v) for k2, v in p["zscore_stats"].items()}
        Xn = apply_zscore(X_raw, stats)
        F = np.concatenate([Xn, E_oof], axis=1)
        model = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                              colsample_bytree=0.8, scale_pos_weight=float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)),
                              objective="binary:logistic", eval_metric="auc", random_state=42, n_jobs=-1, tree_method="hist")
        model.fit(F[tr], y[tr])
        p_l3_oof[va] = model.predict_proba(F[va])[:, 1]
        xgb2_folds.append({"model": model, "stats": stats})
    l3 = {"cv": cv_eval_traincal(p_l3_oof, "l3"), "oof_prob": p_l3_oof}

    results.update({"L1": {k: {kk: vv for kk, vv in v.items() if kk != "oof_prob"} for k, v in l1.items()},
                    "L2": {k: v for k, v in l2.items() if k != "oof_prob"},
                    "L3": {k: v for k, v in l3.items() if k != "oof_prob"}})

    # ================= External evaluation (both panels) =================
    ext_out = {}
    for panel_name, panel_path in [("panel88", PANEL_88), ("panel200", PANEL_200)]:
        panel = pd.read_csv(panel_path)
        panel["sequence"] = panel["sequence"].astype(str).str.upper().str.strip()
        yext = panel["label"].to_numpy(int)
        Xe_raw = xgb_features(panel)
        graphs_e, kept_e, gst = build_panel_graphs(panel[["sequence", "label"]].copy(), PANEL_PDB_DIRS, gsh.cfg, allow_placeholder=False)
        assert kept_e["sequence"].tolist() == panel["sequence"].tolist()
        ge = GshRunner(graphs_e, yext, device)
        p_xgb_ext = np.mean([pf["model"].predict_proba(apply_zscore(Xe_raw, {k2: np.asarray(v) for k2, v in pf["zscore_stats"].items()}))[:, 1]
                             for pf in xgb_folds], axis=0)
        fold_probs, fold_embs = [], []
        for k in range(1, 6):
            probs, embs = ge.forward_fold(k)
            fold_probs.append(probs)
            fold_embs.append(embs)
        p_gsh_ext = np.mean(fold_probs, axis=0)
        E_ext = np.mean(fold_embs, axis=0)

        row = {}
        # baselines: majority vote over per-fold train-calibrated thresholds (original protocol)
        row["XGB"] = external_metrics(
            yext,
            fold_preds_majority(
                [pf["model"].predict_proba(apply_zscore(Xe_raw, {k2: np.asarray(v) for k2, v in pf["zscore_stats"].items()}))[:, 1] for pf in xgb_folds],
                xgb_thr,
            ),
            p_xgb_ext,
        )
        row["gsh"] = external_metrics(yext, fold_preds_majority(fold_probs, gsh_thr), p_gsh_ext)
        lx_e, lg_e = logit(p_xgb_ext), logit(p_gsh_ext)
        for tag in ("fixed", "oof_weighted"):
            w = l1[tag]["w"]
            t = best_threshold(y, l1[tag]["oof_prob"])  # calibrated on all OOF (train side)
            prob = 1 / (1 + np.exp(-(w * lx_e + (1 - w) * lg_e)))
            row[f"L1_{tag}"] = external_metrics(yext, (prob >= t).astype(int), prob)
        # L2 external: refit meta on ALL OOF (external peptides unseen)
        meta_full = LogisticRegression(max_iter=1000).fit(Z, y)
        t = best_threshold(y, p_l2_oof)
        prob = meta_full.predict_proba(np.stack([lx_e, lg_e, lx_e - lg_e], axis=1))[:, 1]
        row["L2"] = external_metrics(yext, (prob >= t).astype(int), prob)
        # L3 external: 5 fold models over [tabular | mean embedding], majority vote
        t3 = [best_threshold(y[np.asarray(f_idx["train_idx"])], p_l3_oof[np.asarray(f_idx["train_idx"])]) for f_idx in xgb_folds]
        probs3 = [f["model"].predict_proba(np.concatenate([apply_zscore(Xe_raw, f["stats"]), E_ext], axis=1))[:, 1] for f in xgb2_folds]
        prob = np.mean(probs3, axis=0)
        row["L3"] = external_metrics(yext, fold_preds_majority(probs3, t3), prob)
        ext_out[panel_name] = row

    # ---- save ----
    np.savez(OUT / "oof_probs.npz", p_xgb=p_xgb_oof, p_gsh=p_gsh_oof, p_l2=p_l2_oof, p_l3=p_l3_oof)
    results["external"] = ext_out
    (OUT / "fusion_results.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps({"cv": {k: (v["cv"]["mcc"] if isinstance(v, dict) and "cv" in v else v) for k, v in
                             {"XGB": results["baselines"]["XGB"], "gsh": results["baselines"]["gsh"],
                              "L1_fixed": l1["fixed"], "L1_weighted": l1["oof_weighted"], "L2": l2, "L3": l3}.items()},
                       "external": ext_out}, indent=2, default=float))


if __name__ == "__main__":
    main()

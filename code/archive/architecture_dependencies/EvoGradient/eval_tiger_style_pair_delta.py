#!/usr/bin/env python3
"""Evaluate EvoGradient AMP regression in TIGER's pair-delta protocol.

Protocol (single-activity then delta):
  1) Predict absolute log10(MIC) for every unique peptide independently
  2) Convert to log2: pred_log2 = pred_log10 / log10(2)
  3) Pair delta: y_pred = pred_log2(anchor) - pred_log2(query)
  4) Report TIGER metrics: log10MAE / RSE / PCC / KCC / spearman (+ sign_accuracy)

Datasets:
  - test_ll37: neighbor pairs (query=LL37 panel, anchor=train-like neighbors)
  - test_apexgo: template-centric pairs (all 10 families, both directions)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Required so torch.load can unpickle regression checkpoints.
import otherModelsRegression  # noqa: F401

ROOT = Path(__file__).resolve().parent
TIGER_ROOT = ROOT.parents[1] / "TIGER"
LOG10_OF_2 = math.log10(2.0)


class PositionalEncoding(nn.Module):
    def __init__(self, len, d_model=20, dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(len, d_model)
        position = torch.arange(0, len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe


class AttentionNetwork(nn.Module):
    """Must match the class pickled in model/regression/Attention.pth."""

    def __init__(self, batch_size=128, embedding_size=20, num_tokens=100, num_classes=1, num_heads=4):
        super().__init__()
        self.pe = PositionalEncoding(len=num_tokens, d_model=embedding_size)
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_tokens = num_tokens
        self.num_classes = num_classes
        self.hidden1 = 20
        self.hidden2 = 60
        self.hidden3 = 20
        self.dropout = 0.2
        self.relu = nn.ReLU()
        self.LN = nn.LayerNorm(normalized_shape=self.hidden1)
        self.fc1 = nn.Linear(self.embedding_size, self.hidden1)
        self.multihead_att = nn.MultiheadAttention(
            embed_dim=self.hidden1, num_heads=num_heads, batch_first=1, dropout=self.dropout
        )
        self.flatten = nn.Flatten()
        self.fc2 = nn.Linear(self.hidden1 * self.num_tokens, self.hidden2)
        self.fc3 = nn.Linear(self.hidden2, self.hidden3)
        self.new_fc4 = nn.Linear(self.hidden3, self.num_classes)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x, mask):
        x = self.pe(x)
        x = self.fc1(x)
        mask = mask.to(torch.bool)
        x, _w1 = self.multihead_att.forward(x, x, x, key_padding_mask=mask)
        x = self.flatten(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.new_fc4(x)
        return x


# Attention.pth was saved from AMP_regression.py (__main__.AttentionNetwork)
import __main__ as _main_mod

_main_mod.AttentionNetwork = AttentionNetwork
_main_mod.PositionalEncoding = PositionalEncoding

MODEL_LIST = {
    "CNN": ROOT / "model/regression/CNN.pth",
    "Transformer": ROOT / "model/regression/Transformer.pth",
    "Attention": ROOT / "model/regression/Attention.pth",
    "LSTM": ROOT / "model/regression/LSTM.pth",
}
ENSEMBLE_WEIGHT = {
    "CNN": 0.25000594,
    "Transformer": 0.2500046,
    "Attention": 0.25000825,
    "LSTM": 0.24998219,
}
AA_DICT = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
}


def regression_metrics(y_true, y_pred, *, input_log_base: float = 2) -> dict[str, float]:
    """Mirror TIGER/code/utils/metrics.py::regression_metrics."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    out: dict[str, float] = {"n": int(len(yt))}
    if len(yt) == 0:
        return out

    if float(input_log_base) == 2.0:
        yt2, yp2 = yt, yp
    elif float(input_log_base) == 10.0:
        yt2, yp2 = yt / LOG10_OF_2, yp / LOG10_OF_2
    else:
        scale = math.log(2.0) / math.log(float(input_log_base))
        yt2, yp2 = yt * scale, yp * scale

    mae2 = float(mean_absolute_error(yt2, yp2))
    mae10 = float(mae2 * LOG10_OF_2)
    mse = float(mean_squared_error(yt2, yp2))
    rmse = float(math.sqrt(mse))
    ss_res = float(np.sum((yt2 - yp2) ** 2))
    ss_tot = float(np.sum((yt2 - yt2.mean()) ** 2))
    rse = float(ss_res / ss_tot) if ss_tot > 0 else float("nan")
    r2 = float(r2_score(yt2, yp2)) if len(yt2) > 1 else float("nan")
    can_corr = len(yt2) > 2 and np.std(yt2) > 0 and np.std(yp2) > 0
    pcc = float(pearsonr(yt2, yp2)[0]) if can_corr else float("nan")
    kcc = float(kendalltau(yt2, yp2)[0]) if can_corr else float("nan")
    spear = float(spearmanr(yt2, yp2)[0]) if can_corr else float("nan")
    out.update(
        {
            "log2MAE": mae2,
            "log10MAE": mae10,
            "RSE": rse,
            "PCC": pcc,
            "KCC": kcc,
            "mae": mae2,
            "rmse": rmse,
            "mse": mse,
            "r2": r2,
            "rse": rse,
            "pearson": pcc,
            "spearman": spear,
            "kendall": kcc,
        }
    )
    vals = [mae2, rse, pcc, kcc]
    out["selection_score"] = (
        float("inf") if any(not np.isfinite(v) for v in vals) else float(mae2 + rse - pcc - kcc)
    )
    # Sign accuracy on non-ties (positive delta => query more active)
    nontie = np.abs(yt2) > 1e-12
    if nontie.any():
        out["sign_accuracy"] = float(np.mean(np.sign(yt2[nontie]) == np.sign(yp2[nontie])))
    else:
        out["sign_accuracy"] = float("nan")
    return out


def encode_seq(seq: str):
    num = [AA_DICT[c.upper()] for c in seq]
    torch_seq = torch.tensor(np.array(num, dtype=int))
    onehot = torch.nn.functional.one_hot(torch_seq, num_classes=20)
    pad = torch.nn.ZeroPad2d(padding=(0, 0, 0, 100 - len(num)))
    mask = torch.zeros(100, dtype=torch.int)
    mask[len(num) :] = 1
    return pad(onehot), mask


class SeqDataset(Dataset):
    def __init__(self, sequences: list[str]):
        self.seqs = sequences

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        x, mask = encode_seq(seq)
        return x, mask, seq


@torch.no_grad()
def predict_log10_mic(sequences: list[str], device: torch.device, batch_size: int = 256) -> pd.DataFrame:
    """Predict absolute log10(MIC) with the 4-model EvoGradient ensemble."""
    uniq = list(dict.fromkeys(sequences))
    loader = DataLoader(SeqDataset(uniq), batch_size=batch_size, shuffle=False)
    model_out: dict[str, list[float]] = {name: [] for name in MODEL_LIST}

    for name, path in MODEL_LIST.items():
        model = torch.load(path, map_location="cpu")
        model.to(device)
        model.eval()
        preds: list[float] = []
        for inputs, masks, _seqs in loader:
            inputs = inputs.float().to(device)
            masks = masks.float().to(device)
            if name != "Attention":
                out = model(inputs)
            else:
                out = model(inputs, masks)
            out = torch.squeeze(out).detach().cpu().numpy()
            if np.ndim(out) == 0:
                preds.append(float(out))
            else:
                preds.extend(float(v) for v in out)
        model_out[name] = preds
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame({"sequence": uniq})
    for name, vals in model_out.items():
        df[name] = vals
    df["pred_log10_mic"] = sum(df[n] * ENSEMBLE_WEIGHT[n] for n in MODEL_LIST)
    df["pred_log2_mic"] = df["pred_log10_mic"] / LOG10_OF_2
    return df


def absolute_metrics(true_log2: np.ndarray, pred_log2: np.ndarray) -> dict[str, float]:
    yt = np.asarray(true_log2, dtype=float)
    yp = np.asarray(pred_log2, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if len(yt) == 0:
        return {"n": 0}
    mae2 = float(mean_absolute_error(yt, yp))
    can_corr = len(yt) > 2 and np.std(yt) > 0 and np.std(yp) > 0
    return {
        "n": int(len(yt)),
        "log2MAE": mae2,
        "log10MAE": float(mae2 * LOG10_OF_2),
        "rmse_log2": float(math.sqrt(mean_squared_error(yt, yp))),
        "pearson": float(pearsonr(yt, yp)[0]) if can_corr else float("nan"),
        "spearman": float(spearmanr(yt, yp)[0]) if can_corr else float("nan"),
        "kendall": float(kendalltau(yt, yp)[0]) if can_corr else float("nan"),
    }


def score_pairs(
    pairs: pd.DataFrame,
    pred_map: dict[str, float],
    *,
    anchor_col: str,
    query_col: str,
    true_delta_col: str,
) -> pd.DataFrame:
    rows = []
    for rec in pairs.itertuples(index=False):
        a = getattr(rec, anchor_col)
        q = getattr(rec, query_col)
        if a not in pred_map or q not in pred_map:
            continue
        pred_a = pred_map[a]
        pred_q = pred_map[q]
        y_true = float(getattr(rec, true_delta_col))
        y_pred = float(pred_a - pred_q)
        row = {
            "anchor_sequence": a,
            "query_sequence": q,
            "pred_log2_anchor": pred_a,
            "pred_log2_query": pred_q,
            "y_true_delta_log2_anchor_minus_query": y_true,
            "y_pred_delta_log2_anchor_minus_query": y_pred,
            "y_true_delta_log10_anchor_minus_query": y_true * LOG10_OF_2,
            "y_pred_delta_log10_anchor_minus_query": y_pred * LOG10_OF_2,
        }
        for col in (
            "family",
            "similarity",
            "anchor_mic_ug_mL",
            "query_mic_ug_mL",
            "anchor_MIC_ug_per_mL",
            "query_MIC_ug_per_mL",
            "query_name",
            "pair_type",
            "anchor_id",
            "query_id",
        ):
            if hasattr(rec, col):
                row[col] = getattr(rec, col)
        rows.append(row)
    return pd.DataFrame(rows)


def metrics_payload(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {"n": 0}
    return regression_metrics(
        df["y_true_delta_log2_anchor_minus_query"],
        df["y_pred_delta_log2_anchor_minus_query"],
    )


def eval_ll37(out_dir: Path, device: torch.device) -> dict:
    seq_csv = TIGER_ROOT / "data/test_ll37/ll37_sequences_mic.csv"
    pair_csv = TIGER_ROOT / "data/test_ll37/ll37_pairs_mic_delta.csv"
    seqs = pd.read_csv(seq_csv)
    pairs = pd.read_csv(pair_csv)

    all_seqs = sorted(set(pairs["query_sequence"]) | set(pairs["anchor_sequence"]))
    pred_df = predict_log10_mic(all_seqs, device)
    pred_map = dict(zip(pred_df["sequence"], pred_df["pred_log2_mic"]))

    # Absolute activity on LL37 panel sequences
    seq_pred = seqs.merge(pred_df, left_on="sequence", right_on="sequence", how="left")
    abs_m = absolute_metrics(seq_pred["MIC_log2"].to_numpy(), seq_pred["pred_log2_mic"].to_numpy())

    pair_pred = score_pairs(
        pairs,
        pred_map,
        anchor_col="anchor_sequence",
        query_col="query_sequence",
        true_delta_col="delta_log2_anchor_minus_query",
    )
    pair_m = metrics_payload(pair_pred)

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / "sequence_predictions.csv", index=False)
    seq_pred.to_csv(out_dir / "ll37_panel_absolute_predictions.csv", index=False)
    pair_pred.to_csv(out_dir / "pair_predictions.csv", index=False)

    summary = {
        "dataset": "test_ll37",
        "protocol": "single_activity_then_delta",
        "model": "EvoGradient AMP_regression ensemble (CNN+Transformer+Attention+LSTM)",
        "n_unique_sequences_scored": int(len(pred_df)),
        "n_pairs": int(len(pair_pred)),
        "absolute_activity_on_ll37_panel": abs_m,
        "pair_delta_metrics": pair_m,
        "note": (
            "EvoGradient predicts log10(MIC); converted to log2 before delta. "
            "y_pred_delta = pred_log2(anchor) - pred_log2(query)."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def eval_apexgo(label: str, out_dir: Path, device: torch.device) -> dict:
    pep_csv = TIGER_ROOT / f"data/test_apexgo/labels/apexgo_peptides_{label}.csv"
    pair_csv = TIGER_ROOT / f"data/test_apexgo/pairs/pairs_template_centric_alldelta_{label}.csv"
    peptides = pd.read_csv(pep_csv)
    pairs = pd.read_csv(pair_csv)

    all_seqs = sorted(set(peptides["sequence"]))
    pred_df = predict_log10_mic(all_seqs, device)
    pred_map = dict(zip(pred_df["sequence"], pred_df["pred_log2_mic"]))

    pep_pred = peptides.merge(pred_df, on="sequence", how="left")
    true_log2 = np.log2(pep_pred["MIC_Escherichia_coli"].astype(float).to_numpy())
    abs_m = absolute_metrics(true_log2, pep_pred["pred_log2_mic"].to_numpy())

    pair_pred = score_pairs(
        pairs,
        pred_map,
        anchor_col="anchor_sequence",
        query_col="query_sequence",
        true_delta_col="y_true_delta_log2_anchor_minus_query",
    )
    pair_m = metrics_payload(pair_pred)
    per_family = {}
    if len(pair_pred) and "family" in pair_pred.columns:
        for fam, g in pair_pred.groupby("family", dropna=True):
            per_family[str(fam)] = metrics_payload(g)

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / "sequence_predictions.csv", index=False)
    pep_pred.to_csv(out_dir / "peptide_absolute_predictions.csv", index=False)
    pair_pred.to_csv(out_dir / "pair_predictions.csv", index=False)
    if per_family:
        pd.DataFrame(
            [{"family": k, **v} for k, v in per_family.items()]
        ).sort_values("family").to_csv(out_dir / "per_family_metrics.csv", index=False)

    summary = {
        "dataset": f"test_apexgo/{label}",
        "protocol": "single_activity_then_delta",
        "pairing": "template_centric_alldelta",
        "model": "EvoGradient AMP_regression ensemble (CNN+Transformer+Attention+LSTM)",
        "n_peptides": int(len(pep_pred)),
        "n_pairs": int(len(pair_pred)),
        "absolute_activity_metrics": abs_m,
        "pair_delta_metrics": {k: v for k, v in pair_m.items()},
        "per_family": per_family,
        "note": (
            "EvoGradient predicts log10(MIC); converted to log2 before delta. "
            "y_pred_delta = pred_log2(anchor) - pred_log2(query)."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="EvoGradient → TIGER-style pair-delta eval")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "eval_tiger_style",
        help="Output directory",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ll37", "apexgo_geo3", "apexgo_AIC222"],
        choices=["ll37", "apexgo_geo3", "apexgo_AIC222", "apexgo_AIC221", "apexgo_ATCC11775"],
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for name in args.datasets:
        if name == "ll37":
            print("\n=== Evaluating test_ll37 ===")
            s = eval_ll37(args.out_dir / "ll37", device)
        else:
            label = name.replace("apexgo_", "")
            print(f"\n=== Evaluating test_apexgo ({label}) ===")
            s = eval_apexgo(label, args.out_dir / f"apexgo_{label}", device)
        summaries.append(s)
        m = s.get("pair_delta_metrics", {})
        print(
            f"  pairs={m.get('n')}  log10MAE={m.get('log10MAE'):.4f}  "
            f"RSE={m.get('RSE'):.4f}  PCC={m.get('PCC'):.4f}  "
            f"KCC={m.get('KCC'):.4f}  spearman={m.get('spearman'):.4f}  "
            f"sign_acc={m.get('sign_accuracy'):.4f}"
        )

    # Compact leaderboard
    rows = []
    for s in summaries:
        m = s.get("pair_delta_metrics", {})
        rows.append(
            {
                "dataset": s["dataset"],
                "n_pairs": m.get("n"),
                "log10MAE": m.get("log10MAE"),
                "log2MAE": m.get("log2MAE"),
                "RSE": m.get("RSE"),
                "PCC": m.get("PCC"),
                "KCC": m.get("KCC"),
                "spearman": m.get("spearman"),
                "sign_accuracy": m.get("sign_accuracy"),
                "selection_score": m.get("selection_score"),
            }
        )
    board = pd.DataFrame(rows)
    board.to_csv(args.out_dir / "leaderboard.csv", index=False)
    (args.out_dir / "leaderboard.json").write_text(json.dumps(summaries, indent=2))
    print("\n=== Leaderboard ===")
    print(board.to_string(index=False))
    print(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    # torch.load of full models may need weights_only=False on newer torch; keep default.
    main()

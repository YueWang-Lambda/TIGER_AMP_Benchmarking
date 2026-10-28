#!/usr/bin/env python3
"""Nature-style figure: TIGER vs EvoGradient on LL37 + APEX-GO (geo3).

Core metrics: log10MAE, RSE, PCC, KCC.
APEX-GO panel uses distinct markers per family.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from scipy.stats import kendalltau, pearsonr
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parent
TIGER = ROOT.parents[1] / "TIGER"
OUT = ROOT / "eval_tiger_style" / "figures"
LOG10_OF_2 = math.log10(2.0)

CORE = ["log10MAE", "RSE", "PCC", "KCC"]
METRIC_LABELS = {
    "log10MAE": r"log$_{10}$MAE" + "\n(lower better)",
    "RSE": "RSE\n(lower better)",
    "PCC": "PCC\n(higher better)",
    "KCC": "KCC\n(higher better)",
}
# Short labels for axis titles in family panels
METRIC_SHORT = {
    "log10MAE": r"log$_{10}$MAE",
    "RSE": "RSE",
    "PCC": "PCC",
    "KCC": "KCC",
}

COLOR_TIGER = "#0072B2"
COLOR_EVO = "#D55E00"

FAMILIES = [
    "Arctoterin-1",
    "Equusin-4",
    "Hesperelin-3",
    "Hydrodamin-2",
    "Hydrodamin-3",
    "Lophiosin-1",
    "Mammuthusin-2",
    "Mammuthusin-3",
    "Mylodonin-2",
    "Mylodonin-3",
]
FAMILY_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p"]
FAMILY_COLORS = [
    "#4C72B0",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#CCB974",
    "#64B5CD",
    "#8C8C8C",
    "#E17C05",
    "#2A9D8F",
    "#9B2226",
]


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if len(yt) == 0:
        return {k: float("nan") for k in CORE} | {"n": 0}
    mae2 = float(mean_absolute_error(yt, yp))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    rse = float(ss_res / ss_tot) if ss_tot > 0 else float("nan")
    can = len(yt) > 2 and np.std(yt) > 0 and np.std(yp) > 0
    return {
        "n": int(len(yt)),
        "log10MAE": mae2 * LOG10_OF_2,
        "RSE": rse,
        "PCC": float(pearsonr(yt, yp)[0]) if can else float("nan"),
        "KCC": float(kendalltau(yt, yp)[0]) if can else float("nan"),
    }


def ensure_evo_preds_for_seqs(sequences: list[str], cache_csv: Path) -> dict[str, float]:
    """Return sequence -> pred_log2_mic, predicting any missing sequences."""
    from eval_tiger_style_pair_delta import predict_log10_mic

    if cache_csv.exists():
        df = pd.read_csv(cache_csv)
        pmap = dict(zip(df["sequence"], df["pred_log2_mic"]))
    else:
        pmap = {}

    missing = [s for s in dict.fromkeys(sequences) if s not in pmap]
    if missing:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Predicting {len(missing)} missing sequences on {device}...")
        new = predict_log10_mic(missing, device)
        for r in new.itertuples(index=False):
            pmap[r.sequence] = float(r.pred_log2_mic)
        # merge & save
        out = pd.DataFrame({"sequence": list(pmap.keys()), "pred_log2_mic": list(pmap.values())})
        # keep richer columns if available
        if cache_csv.exists():
            old = pd.read_csv(cache_csv)
            out = pd.concat([old, new], ignore_index=True).drop_duplicates("sequence", keep="last")
        else:
            out = new
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache_csv, index=False)
    return pmap


def collect_metrics():
    """Protocol-matched TIGER vs EvoGradient.

    - LL37: TIGER trained with sim≥0.3 (neighbor-delta setting)
    - APEX-GO: TIGER trained with sim≥0.7 (high-similarity / local SAR setting)
    """
    tiger_ll37_exp = "fusion_attention_grid__fusion_attn_sim0p3_bal400"
    tiger_apex_exp = "fusion_attention_grid__fusion_attn_sim0p7_bal400"

    apex_sum = json.loads(
        (
            TIGER
            / "outputs_ll37_fusion_attn_grid/apexgo_geo3_sim0p7_eval"
            / tiger_apex_exp
            / "summary.json"
        ).read_text()
    )
    evo_sum = json.loads((ROOT / "eval_tiger_style/apexgo_geo3/summary.json").read_text())

    # --- Panel a: APEX-GO bidirectional (sim0.7 TIGER) ---
    overall_rows = []
    for method, m in [
        ("TIGER", apex_sum["ensemble_bidirectional"]),
        ("EvoGradient", evo_sum["pair_delta_metrics"]),
    ]:
        overall_rows.append(
            {
                "dataset": "APEX-GO (geo3)",
                "method": method,
                "n": m["n"],
                **{k: float(m[k]) for k in CORE},
            }
        )

    # --- Panel b: template→variant (TIGER sim0.7 + Evo) ---
    tiger_tmpl_pairs = pd.read_csv(
        TIGER
        / "outputs_ll37_fusion_attn_grid/apexgo_geo3_sim0p7_eval"
        / tiger_apex_exp
        / "pair_predictions_ensemble_template_to_variant.csv"
    )
    pair_meta = pd.read_csv(TIGER / "data/test_apexgo/pairs/pairs_template_centric_alldelta_geo3.csv")
    evo_pairs = pd.read_csv(ROOT / "eval_tiger_style/apexgo_geo3/pair_predictions.csv")
    key = ["anchor_sequence", "query_sequence"]
    evo_pairs = evo_pairs.drop(columns=["family", "pair_type"], errors="ignore").merge(
        pair_meta[key + ["pair_type", "family"]], on=key, how="left"
    )
    evo_tmpl = evo_pairs[evo_pairs["pair_type"] == "template_to_variant"].copy()

    tmpl_overall_rows = []
    tmpl_fam_rows = []
    for method, df in [("TIGER", tiger_tmpl_pairs), ("EvoGradient", evo_tmpl)]:
        m = regression_metrics(
            df["y_true_delta_log2_anchor_minus_query"],
            df["y_pred_delta_log2_anchor_minus_query"],
        )
        tmpl_overall_rows.append(
            {"dataset": "APEX-GO (geo3)", "method": method, **{k: m[k] for k in CORE}, "n": m["n"]}
        )
        for fam in FAMILIES:
            g = df[df["family"] == fam]
            mf = regression_metrics(
                g["y_true_delta_log2_anchor_minus_query"],
                g["y_pred_delta_log2_anchor_minus_query"],
            )
            tmpl_fam_rows.append({"family": fam, "method": method, **{k: mf[k] for k in CORE}, "n": mf["n"]})

    tmpl_overall = pd.DataFrame(tmpl_overall_rows)
    tmpl_overall["method"] = pd.Categorical(tmpl_overall["method"], ["TIGER", "EvoGradient"], ordered=True)
    tmpl_family = pd.DataFrame(tmpl_fam_rows)
    tmpl_family["family"] = pd.Categorical(tmpl_family["family"], FAMILIES, ordered=True)
    tmpl_family["method"] = pd.Categorical(tmpl_family["method"], ["TIGER", "EvoGradient"], ordered=True)
    family = tmpl_family.copy()

    # --- LL37: sim0.3 TIGER (ens best_val) + Evo on matched pairs ---
    ll37_dir = TIGER / "outputs_ll37_eval_mic128_30" / tiger_ll37_exp
    frames = [pd.read_csv(ll37_dir / f"ll37_pairs_best_val_fold{f}.csv") for f in range(1, 6)]
    allf = pd.concat(frames, ignore_index=True)
    tiger_ll37 = (
        allf.groupby(["anchor_sequence", "query_sequence"], as_index=False)
        .agg(
            y_true_delta_log2_anchor_minus_query=("y_true_delta_log2_anchor_minus_query", "first"),
            y_pred_delta_log2_anchor_minus_query=("y_pred_delta_log2_anchor_minus_query", "mean"),
        )
    )
    seqs = sorted(set(tiger_ll37["anchor_sequence"]) | set(tiger_ll37["query_sequence"]))
    cache = ROOT / "eval_tiger_style/ll37/sequence_predictions_matched.csv"
    base_cache = ROOT / "eval_tiger_style/ll37/sequence_predictions.csv"
    if not cache.exists() and base_cache.exists():
        pd.read_csv(base_cache).to_csv(cache, index=False)
    pmap = ensure_evo_preds_for_seqs(seqs, cache)

    m_tiger = regression_metrics(
        tiger_ll37["y_true_delta_log2_anchor_minus_query"],
        tiger_ll37["y_pred_delta_log2_anchor_minus_query"],
    )
    yt_e, yp_e = [], []
    for r in tiger_ll37.itertuples(index=False):
        yt_e.append(r.y_true_delta_log2_anchor_minus_query)
        yp_e.append(pmap[r.anchor_sequence] - pmap[r.query_sequence])
    m_evo = regression_metrics(yt_e, yp_e)

    for method, m in [("TIGER", m_tiger), ("EvoGradient", m_evo)]:
        overall_rows.append(
            {"dataset": "LL37", "method": method, "n": m["n"], **{k: m[k] for k in CORE}}
        )

    overall = pd.DataFrame(overall_rows)
    overall["dataset"] = pd.Categorical(overall["dataset"], ["LL37", "APEX-GO (geo3)"], ordered=True)
    overall["method"] = pd.Categorical(overall["method"], ["TIGER", "EvoGradient"], ordered=True)
    overall = overall.sort_values(["dataset", "method"]).reset_index(drop=True)

    meta = {
        "protocol": "dataset-matched TIGER checkpoints",
        "tiger_ll37_model": tiger_ll37_exp,
        "tiger_ll37_similarity_threshold": 0.3,
        "tiger_apexgo_model": tiger_apex_exp,
        "tiger_apexgo_similarity_threshold": 0.7,
        "ll37_protocol": "neighbor pair-delta; 5-fold best_val ensemble; Evo matched on same pairs",
        "ll37_n_pairs": int(m_tiger["n"]),
        "apexgo_label": "geo3",
        "apexgo_n_pairs_panel_a": int(apex_sum["ensemble_bidirectional"]["n"]),
        "apexgo_n_pairs_panel_b": 100,
        "apexgo_panel_b_protocol": "template_to_variant only",
        "evogradient_model": "AMP_regression ensemble (single-activity → delta)",
        "core_metrics": CORE,
        "note": (
            "LL37 is a cross-peptide neighbor task (use sim0.3). "
            "APEX-GO is high-similarity local SAR (use sim0.7). "
            "Hard-filtering LL37 pairs to sim>=0.3 leaves few pairs; eval uses train-like neighbor top-k with fallback."
        ),
    }
    return overall, family, tmpl_overall, tmpl_family, meta


def _setup_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_figure(
    overall: pd.DataFrame,
    family: pd.DataFrame,
    tmpl_overall: pd.DataFrame,
    tmpl_family: pd.DataFrame,
    out_dir: Path,
):
    _setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.2, 6.8), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.0, 1.4],
        hspace=0.48,
        wspace=0.38,
        left=0.09,
        right=0.98,
        top=0.90,
        bottom=0.14,
    )

    datasets = ["LL37", "APEX-GO (geo3)"]
    methods = ["TIGER", "EvoGradient"]
    method_colors = {"TIGER": COLOR_TIGER, "EvoGradient": COLOR_EVO}
    x_base = np.array([0.0, 1.0])
    width = 0.32

    # ---- Panel a: overall (LL37 + APEX-GO bidirectional) ----
    for i, metric in enumerate(CORE):
        ax = fig.add_subplot(gs[0, i])
        for j, method in enumerate(methods):
            vals = [
                overall[(overall["dataset"] == ds) & (overall["method"] == method)].iloc[0][metric]
                for ds in datasets
            ]
            xpos = x_base + (j - 0.5) * width
            ax.bar(
                xpos,
                vals,
                width=width,
                color=method_colors[method],
                edgecolor="black",
                linewidth=0.45,
                zorder=3,
            )
        ax.set_xticks(x_base)
        ax.set_xticklabels(["LL37", "APEX-GO\n(geo3)"], fontsize=7)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.axhline(0, color="#888888", lw=0.5, zorder=1)
        if metric in ("PCC", "KCC"):
            ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
        if i == 0:
            ax.text(-0.32, 1.15, "a", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")

    method_handles_a = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=COLOR_TIGER,
            edgecolor="black",
            lw=0.4,
            label=r"TIGER (LL37: sim$\geq$0.3; APEX-GO: sim$\geq$0.7)",
        ),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_EVO, edgecolor="black", lw=0.4, label="EvoGradient"),
    ]
    fig.legend(
        handles=method_handles_a,
        loc="upper right",
        frameon=False,
        fontsize=7.5,
        bbox_to_anchor=(0.98, 0.935),
        ncol=2,
        handlelength=1.1,
        columnspacing=1.2,
    )

    # ---- Panel b: APEX-GO template→variant (100 pairs) — two bars + family points ----
    fam_short = {
        "Arctoterin-1": "Arc-1",
        "Equusin-4": "Equ-4",
        "Hesperelin-3": "Hes-3",
        "Hydrodamin-2": "Hyd-2",
        "Hydrodamin-3": "Hyd-3",
        "Lophiosin-1": "Lop-1",
        "Mammuthusin-2": "Mam-2",
        "Mammuthusin-3": "Mam-3",
        "Mylodonin-2": "Myl-2",
        "Mylodonin-3": "Myl-3",
    }
    rng = np.random.default_rng(0)
    fam_offsets = rng.uniform(-0.14, 0.14, size=len(FAMILIES))

    for i, metric in enumerate(CORE):
        ax = fig.add_subplot(gs[1, i])
        bar_vals = []
        for j, method in enumerate(methods):
            v = float(tmpl_overall[tmpl_overall["method"] == method].iloc[0][metric])
            bar_vals.append(v)
            ax.bar(
                j,
                v,
                width=0.55,
                color=method_colors[method],
                edgecolor="black",
                linewidth=0.45,
                alpha=0.35,
                zorder=2,
            )

        for fi, fam in enumerate(FAMILIES):
            for j, method in enumerate(methods):
                y = float(
                    tmpl_family[
                        (tmpl_family["family"] == fam) & (tmpl_family["method"] == method)
                    ].iloc[0][metric]
                )
                ax.scatter(
                    j + fam_offsets[fi],
                    y,
                    s=36,
                    marker=FAMILY_MARKERS[fi],
                    facecolors=FAMILY_COLORS[fi],
                    edgecolors="black",
                    linewidths=0.45,
                    zorder=4,
                )

        ax.set_xticks([0, 1])
        ax.set_xticklabels([r"TIGER" + "\n" + r"(sim$\geq$0.7)", "EvoGradient"], fontsize=7.0)
        ax.set_ylabel(METRIC_SHORT[metric])
        if metric in ("PCC", "KCC"):
            ax.set_ylim(-1.05, 1.05)
            ax.axhline(0, color="#888888", lw=0.5, zorder=1)
        elif metric == "RSE":
            point_vals = [
                float(
                    tmpl_family[
                        (tmpl_family["family"] == fam) & (tmpl_family["method"] == method)
                    ].iloc[0][metric]
                )
                for fam in FAMILIES
                for method in methods
            ]
            ymax = max(bar_vals + point_vals)
            ax.set_ylim(0, ymax * 1.08)
        else:
            ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
            ymin = min(0, min(bar_vals))
            ax.set_ylim(bottom=ymin)

        if i == 0:
            ax.text(-0.32, 1.10, "b", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
            ax.set_title(
                r"APEX-GO (geo3): template$\rightarrow$variant only (n=100)",
                loc="left",
                fontsize=8,
                pad=3,
            )

    fam_handles = [
        Line2D(
            [0],
            [0],
            marker=FAMILY_MARKERS[i],
            color="none",
            markerfacecolor=FAMILY_COLORS[i],
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=6.5,
            label=fam_short[FAMILIES[i]],
        )
        for i in range(len(FAMILIES))
    ]
    fig.legend(
        handles=fam_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=6.5,
        bbox_to_anchor=(0.5, 0.01),
        columnspacing=1.1,
        handletextpad=0.35,
        title="APEX-GO family",
        title_fontsize=7,
    )

    fig.suptitle(
        r"Pair-delta MIC prediction: TIGER (sim$\geq$0.7) vs EvoGradient",
        fontsize=10,
        fontweight="bold",
        y=0.965,
    )

    for ext in ("pdf", "png"):
        path = out_dir / f"tiger_vs_evogradient_core_metrics.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


def main():
    overall, family, tmpl_overall, tmpl_family, meta = collect_metrics()
    OUT.mkdir(parents=True, exist_ok=True)
    overall.to_csv(OUT / "overall_metrics_for_figure.csv", index=False)
    family.to_csv(OUT / "apexgo_family_metrics_for_figure.csv", index=False)
    tmpl_overall.to_csv(OUT / "apexgo_template_to_variant_overall.csv", index=False)
    tmpl_family.to_csv(OUT / "apexgo_template_to_variant_family.csv", index=False)
    (OUT / "figure_meta.json").write_text(json.dumps(meta, indent=2))
    print("Panel a (bidirectional APEX-GO):")
    print(overall.to_string(index=False))
    print("\nPanel b (template→variant):")
    print(tmpl_overall.to_string(index=False))
    print(tmpl_family.pivot_table(index="family", columns="method", values=CORE).round(3).to_string())
    plot_figure(overall, family, tmpl_overall, tmpl_family, OUT)


if __name__ == "__main__":
    main()

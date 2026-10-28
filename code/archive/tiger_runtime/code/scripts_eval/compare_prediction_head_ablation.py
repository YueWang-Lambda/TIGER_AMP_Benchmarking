#!/usr/bin/env python3
"""Internal ablation: siamese pair-Δ head vs direct peptide-MIC head.

This is **not** an external baseline. Same encoder / GroupKFold(sequence).
The single arm is trained and checkpointed on per-sequence log2(MIC).
Reported pair-Δ is score(template)-score(mutated) on the neighbor protocol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _first_existing(*cands: Path) -> Path:
    for p in cands:
        if (p / "results" / "summary.json").exists():
            return p
    return cands[0]


def _load_summary(run: Path) -> dict:
    p = run / "results" / "summary.json"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    return json.loads(p.read_text())


def _metrics_row(tag: str, m: dict | None) -> dict:
    if not m:
        return {"ablation_arm": tag}
    return {
        "ablation_arm": tag,
        "n": m.get("n"),
        "PCC": m.get("PCC"),
        "KCC": m.get("KCC"),
        "log10MAE": m.get("log10MAE"),
        "log2MAE": m.get("log2MAE"),
        "RSE": m.get("RSE"),
        "selection_score": m.get("selection_score"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-run",
        type=Path,
        default=None,
        help="Locked siamese pair-Δ arm (panel 02).",
    )
    parser.add_argument(
        "--single-run",
        type=Path,
        default=None,
        help="Direct log2(MIC) head arm (panel 07).",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    pair_run = args.pair_run
    single_run = args.single_run
    if pair_run is None:
        pair_run = _first_existing(
            ROOT / "runs_ablation/02_pair_balance_bin1/unsigned_bal10000",
            ROOT / "checkpoints/ablation/02_pair_balance_bin1/unsigned_bal10000",
        )
    elif not pair_run.is_absolute():
        pair_run = ROOT / pair_run
    if single_run is None:
        single_run = _first_existing(
            ROOT / "runs_ablation_repro/07_prediction_head/single_mic",
            ROOT / "runs_ablation/07_prediction_head/single_mic",
            ROOT / "checkpoints/ablation/07_prediction_head/single_mic",
        )
    elif not single_run.is_absolute():
        single_run = ROOT / single_run

    pair_sum = _load_summary(pair_run)
    single_sum = _load_summary(single_run)
    rows = [
        _metrics_row("pair_head (siamese Δ; train+eval pair-Δ)", pair_sum.get("cv")),
        _metrics_row("single_head eval pair-Δ score(t)-score(m)", single_sum.get("cv")),
        _metrics_row("single_head train/save peptide log2(MIC)", single_sum.get("cv_peptide")),
    ]
    table = pd.DataFrame(rows)
    out = args.out
    if out is None:
        out = single_run / "results" / "prediction_head_ablation.csv"
    elif not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    payload = {
        "panel": "07_prediction_head",
        "kind": "internal_ablation",
        "pair_run": str(pair_run),
        "single_run": str(single_run),
        "pair_model_kind": pair_sum.get("model_kind"),
        "single_model_kind": single_sum.get("model_kind"),
        "single_selection_target": single_sum.get("selection_target"),
        "rows": rows,
        "note": (
            "Internal TIGER ablation. Same GroupKFold(sequence) and locked encoder. "
            "Siamese arm trains and evaluates on pair Δ. Single arm trains and saves "
            "on per-sequence log2(MIC); reported pair-Δ is score(template)-score(mutated) "
            "on the same neighbor protocol. Not an external model comparison."
        ),
        "wrote": str(out),
    }
    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2))
    print(table.to_string(index=False))
    print(f"wrote {out}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()

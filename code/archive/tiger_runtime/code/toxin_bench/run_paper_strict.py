"""Paper-strict upper-bound benchmark: each published method retrained under its
OWN paper's training protocol, with hard no-test-leakage rules.

Leakage rules (uniform):
  - tuning / model selection / early stopping / thresholds happen ONLY on the
    1596-peptide training table (its internal CV or a train-carved validation);
  - the two test panels (88 / merged-200) are touched exactly once, at the end;
  - where the original protocol itself selects on test (LysePred), the
    selection is switched to validation and logged as a leakage-guard deviation.

Model convention: after protocol-internal selection, the final model is refit
on the FULL training table (standard practice for deployed predictors), then
evaluated on both panels with the paper's own decision rule.

Run: PYTHONPATH=. python -m code.toxin_bench.run_paper_strict --method hemopi1 --gpu 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TIGER_ROOT = HERE.parent.parent
if str(TIGER_ROOT) not in sys.path:
    sys.path.insert(0, str(TIGER_ROOT))

from code.toxin_bench.run_bench import external_metrics  # noqa: E402

TRAIN_CSV = TIGER_ROOT / "runs_ablation_toxin" / "both" / "toxicity_labeled_dataset.csv"
PANEL_88 = TIGER_ROOT / "data" / "test_external" / "test_toxin_internal_toxin_cohort" / "internal_toxin_cohort_toxicity_labeled.csv"
PANEL_200 = TIGER_ROOT / "runs_toxin_bench" / "external_testset_merged_200.csv"
OUT = TIGER_ROOT / "runs_toxin_bench" / "paper_strict"


def load(path):
    df = pd.read_csv(path)
    df["sequence"] = df["sequence"].astype(str).str.upper().str.strip()
    return df


def main():
    from code.toxin_bench.paper_strict_methods import METHODS
    from code.toxin_bench.paper_strict_methods2 import METHODS as M2
    from code.toxin_bench.paper_strict_methods3 import METHODS as M3
    METHODS = {**METHODS, **M2, **M3}

    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(METHODS))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42,
                    help="base seed; propagated to CV splits, estimator seeds and NN init (paper-pinned elements shift by seed-42)")
    ap.add_argument("--out-root", type=Path, default=OUT)
    args = ap.parse_args()

    out_dir = args.out_root / f"{args.method}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    train = load(TRAIN_CSV)
    panels = {"panel88": load(PANEL_88), "panel200": load(PANEL_200)}

    print(f"[{args.method}] paper-strict training on n={len(train)} (seed {args.seed}) ...", flush=True)
    model = METHODS[args.method]()
    info = model.train(train, gpu=args.gpu, seed=args.seed)  # {'threshold': float, 'meta': {...}}

    result = {"method": args.method, "seed": args.seed, "n_train": len(train), "info": info, "external": {}}
    for pname, panel in panels.items():
        prob = np.asarray(model.predict_proba(panel), dtype=float)
        thr = info["threshold"]
        yp = (prob >= thr).astype(int)
        y = panel["label"].to_numpy(int)
        result["external"][pname] = external_metrics(y, yp, prob)
        pred = panel[["sequence"]].copy()
        pred["prob"] = prob
        pred["pred"] = yp
        pred.to_csv(out_dir / f"external_predictions_{pname}.csv", index=False)
        print(f"[{args.method}] {pname}: MCC {result['external'][pname]['mcc']:.3f} "
              f"recall {result['external'][pname]['recall']:.3f} spec {result['external'][pname]['specificity']:.3f} "
              f"FN {result['external'][pname]['fn']}", flush=True)

    (out_dir / "run_result.json").write_text(json.dumps(result, indent=2, default=float))
    print(f"[{args.method}] wrote {out_dir}")


if __name__ == "__main__":
    main()

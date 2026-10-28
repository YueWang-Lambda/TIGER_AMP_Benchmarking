"""HemoPI2 (Rathore et al., Commun Biol 2025) — RF over 1190 composition features.

The upstream feature code ships in the PyPI wheel (not the GitHub repo); it is
vendored at baseline/toxin_hemo_bench/hemopi2_model/ (composition_calculate.py +
property tables, retrieved from the hemopi2-1.3 wheel). The classifier is the
pickle's exact spec: RandomForestClassifier(n_estimators=200, max_features=
'sqrt', random_state=42). Upstream preprocessing reproduced: sequences
truncated to the first 40 residues (their lenchk) before feature computation.
No feature scaling/selection at train or inference (raw 1190-column vector).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

HEMOPI2_MODEL_DIR = Path(
    "/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/baseline/toxin_hemo_bench/hemopi2_model"
)
FEATURE_SCRIPT = HEMOPI2_MODEL_DIR / "Model" / "composition_calculate.py"
PYTHON = sys.executable
MAX_LEN = 40  # upstream lenchk truncation


_FEATURE_CACHE: dict[tuple[str, ...], pd.DataFrame] = {}


def composition_features(seqs: list[str]) -> pd.DataFrame:
    key = tuple(seqs)
    if key in _FEATURE_CACHE:
        return _FEATURE_CACHE[key]
    with tempfile.TemporaryDirectory() as td:
        seq_file = Path(td) / "seqs.txt"
        out_dir = Path(td) / "wd"
        out_csv = Path(td) / "features.csv"
        out_dir.mkdir()
        seq_file.write_text("\n".join(s.upper()[:MAX_LEN] for s in seqs) + "\n")
        subprocess.run(
            [PYTHON, str(FEATURE_SCRIPT), str(seq_file), str(out_dir), str(out_csv)],
            check=True,
            capture_output=True,
            cwd=str(HEMOPI2_MODEL_DIR),
        )
        df = pd.read_csv(out_csv).loc[:, ~pd.read_csv(out_csv, nrows=0).columns.str.startswith("Unnamed:")]
    _FEATURE_CACHE[key] = df
    return df


def adapter(seed: int, gpu: int = -1, fold: int = 1, **kwargs):
    state: dict = {}

    class Adapter:
        def fit(self, df_train: pd.DataFrame):
            X = composition_features(df_train["sequence"].tolist())
            state["cols"] = [c for c in X.columns]
            state["model"] = RandomForestClassifier(
                n_estimators=200,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            )
            state["model"].fit(X.to_numpy(dtype=float), df_train["label"].to_numpy(int))
            return self

        def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
            X = composition_features(df["sequence"].tolist())
            X = X.reindex(columns=state["cols"], fill_value=0.0)
            return state["model"].predict_proba(X.to_numpy(dtype=float))[:, 1]

    return Adapter()


META = {
    "citation": "Rathore et al., Commun Biol 8:176 (2025), HemoPI2 (github.com/raghavagps/hemopi2)",
    "implementation": "vendored wheel feature script (1190 composition features) + upstream RF spec "
                      "read from their released pickle",
    "hyperparams": "RandomForest(n_estimators=200, max_features='sqrt', random_state=42); "
                   "sequences truncated to 40 aa (upstream lenchk); no scaling/selection",
    "weights": "features: baseline/toxin_hemo_bench/hemopi2_model (from PyPI hemopi2-1.3 wheel); "
               "classifier retrained from scratch",
}

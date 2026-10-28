"""HemoPI (Chaudhary et al., Sci Rep 2016) — per-paper reimplementation.

The original web-server SVM pipeline uses amino-acid composition (AAC, 20),
dipeptide composition (DPC, 400) and N/C-terminal 5-residue binary profiles.
Reimplemented here as an RBF-SVM over AAC+DPC (sklearn), the paper's best
single-feature-family composite, with probability calibration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA_ORDER)}


def aac_dpc(sequences: list[str]) -> np.ndarray:
    rows = []
    for seq in sequences:
        n = max(len(seq), 1)
        aac = np.zeros(20)
        for ch in seq:
            if ch in AA_IDX:
                aac[AA_IDX[ch]] += 1
        aac /= n
        dpc = np.zeros(400)
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            if a in AA_IDX and b in AA_IDX:
                dpc[AA_IDX[a] * 20 + AA_IDX[b]] += 1
        dpc /= max(len(seq) - 1, 1)
        rows.append(np.concatenate([aac, dpc]))
    return np.asarray(rows)


def build(seed: int):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    C=16.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def adapter(seed: int, gpu: int = -1, fold: int = 1, **kwargs):
    model = build(seed)

    class Adapter:
        def fit(self, df_train: pd.DataFrame):
            model.fit(aac_dpc(df_train["sequence"].tolist()), df_train["label"].to_numpy(int))
            return self

        def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
            return model.predict_proba(aac_dpc(df["sequence"].tolist()))[:, 1]

    return Adapter()


META = {
    "citation": "Chaudhary et al., Sci Rep 6:22843 (2016), HemoPI",
    "implementation": "per-paper reimplementation (sklearn SVC-RBF over AAC+DPC)",
    "hyperparams": "C=16, gamma=scale, class_weight=balanced, probability=True",
}

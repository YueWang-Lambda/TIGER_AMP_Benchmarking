"""HemoPred (Winogradoff et al., 2018) — per-paper reimplementation.

HemoPred trains random-forest ensembles over amino-acid composition features
of hemolytic/non-hemolytic peptides (HemoPI-1 era data). Reimplemented as
sklearn RandomForest over AAC+DPC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from code.toxin_bench.methods.hemopi1 import aac_dpc


def adapter(seed: int, gpu: int = -1, fold: int = 1, **kwargs):
    model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )

    class Adapter:
        def fit(self, df_train: pd.DataFrame):
            model.fit(aac_dpc(df_train["sequence"].tolist()), df_train["label"].to_numpy(int))
            return self

        def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
            return model.predict_proba(aac_dpc(df["sequence"].tolist()))[:, 1]

    return Adapter()


META = {
    "citation": "Winogradoff et al. (2018), HemoPred web server",
    "implementation": "per-paper reimplementation (sklearn RandomForest over AAC+DPC)",
    "hyperparams": "n_estimators=500, balanced_subsample",
}

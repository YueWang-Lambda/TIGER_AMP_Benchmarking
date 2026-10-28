"""Plisson non-hemolytic design baseline (plissonf/ML-guided-...) — ported.

56 modlamp descriptors (47 PeptideDescriptor via the exact upstream scale call
sequence + 9 GlobalDescriptor), min-max normalized with train-fold statistics,
then their HemoPI-1 model-3 spec:
GradientBoostingClassifier(n_estimators=208, max_depth=4, min_samples_leaf=10,
max_features='sqrt', random_state=42).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

PEP_SCALES: list[tuple[str, bool, bool]] = [
    # (scale, calculate_global, calculate_moment)  — order follows upstream notebook 2
    ("eisenberg", True, True),
    ("gravy", True, True),
    ("z3", False, False),   # autocorr(1)
    ("z5", False, False),   # autocorr(1)
    ("AASI", True, True),
    ("ABHPRK", True, False),
    ("argos", True, True),
    ("bulkiness", True, True),
    ("charge_phys", True, False),
    ("charge_acid", True, False),
    ("Ez", True, False),
    ("flexibility", True, True),
    ("grantham", True, False),
    ("hopp-woods", True, True),
    ("ISAECI", True, False),
    ("janin", True, True),
    ("kytedoolittle", True, True),
    ("levitt_alpha", True, True),
    ("MSS", True, True),
    ("MSW", True, False),
    ("pepArc", True, False),
    ("pepcats", True, False),
    ("polarity", True, True),
    ("PPCALI", True, False),
    ("refractivity", True, True),
    ("t_scale", True, False),
    ("TM_tend", True, True),
]


def modlamp_features(seqs: list[str]) -> np.ndarray:
    from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor

    pep = PeptideDescriptor(list(seqs), "eisenberg")
    first = True
    for scale, use_global, use_moment in PEP_SCALES:
        pep.load_scale(scale)
        if scale in ("z3", "z5"):
            pep.calculate_autocorr(1, append=not first)
        else:
            if use_global:
                pep.calculate_global(append=not first)
            if use_moment:
                pep.calculate_moment(append=True)
        first = False
    glo = GlobalDescriptor(list(seqs))
    glo.length()
    glo.boman_index(append=True)
    glo.aromaticity(append=True)
    glo.aliphatic_index(append=True)
    glo.instability_index(append=True)
    glo.calculate_charge(ph=7.4, amide=True, append=True)
    glo.calculate_MW(amide=False, append=True)
    glo.isoelectric_point(amide=False, append=True)
    glo.hydrophobic_ratio(append=True)
    return np.concatenate([np.asarray(pep.descriptor, dtype=float), np.asarray(glo.descriptor, dtype=float)], axis=1)


def adapter(seed: int, gpu: int = -1, fold: int = 1, **kwargs):
    state = {}

    class Adapter:
        def fit(self, df_train: pd.DataFrame):
            X = modlamp_features(df_train["sequence"].tolist())
            state["min"] = X.min(axis=0)
            state["max"] = X.max(axis=0)
            rng = state["max"] - state["min"]
            rng[rng < 1e-12] = 1.0
            Xn = (X - state["min"]) / rng
            state["model"] = GradientBoostingClassifier(
                n_estimators=208,
                max_depth=4,
                min_samples_leaf=10,
                max_features="sqrt",
                random_state=42,
            )
            state["model"].fit(Xn, df_train["label"].to_numpy(int))
            return self

        def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
            X = modlamp_features(df["sequence"].tolist())
            rng = state["max"] - state["min"]
            rng[rng < 1e-12] = 1.0
            Xn = (X - state["min"]) / rng
            return state["model"].predict_proba(Xn)[:, 1]

    return Adapter()


META = {
    "citation": "Plisson et al. (github.com/plissonf/ML-guided-discovery-and-design-of-non-hemolytic-peptides)",
    "implementation": "56 modlamp descriptors (upstream scale call order) + their HemoPI-1 model-3 GBC",
    "hyperparams": "GBC(n_estimators=208, max_depth=4, min_samples_leaf=10, max_features='sqrt', rs=42)",
    "deviations": "min-max stats from the training fold only (upstream normalized each split by "
                  "its own statistics); single model-3 spec instead of their 3-model ensemble",
}

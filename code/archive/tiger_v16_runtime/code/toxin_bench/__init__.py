"""Published hemolysis-predictor benchmark: unified retraining harness.

Each method under ``methods/`` is an adapter with ``fit(df_train)`` /
``predict_proba(df)``; ``run_bench.py`` drives the shared protocol:
StratifiedKFold(5, shuffle, seed=42) on the T=512 toxin labels
(n=1596, identical folds to all in-house runs), train-fold F1 threshold
calibration, and the 88-peptide external panel scored per fold
(mean probability + majority vote) — mirroring ``code/toxin_filter``.
"""

# Data dictionary

## External endpoint

`data/external_peptides.csv` contains 208 unique sequences and their family, peptide identifier, measured MIC in µg/mL, terminal metadata, inoculum metadata and source DOI. `data/canonical_pairs.csv` contains 2,211 rows with `family`, `anchor_sequence`, `query_sequence`, and `y_true` in log2-difference units. The three identity fields together form the primary key. A single recorded orientation is retained for each unordered pair. Family labels in the canonical endpoint are normalized (for example, `HAL-2`, without a trailing slash).

## Predictions

`provenance/prediction_index.csv` is the exhaustive list of 23 × 2 × 10 predictors. Each row gives the method, condition, seed, package-relative pair-prediction path and SHA-256, original source identifier, and optional peptide-prediction path and unit.

Each pair CSV repeats the four endpoint columns and adds `y_pred`. It is aligned by identity during evaluation, not by row position. All 2,211 predictions must be finite. The 420 external-method peptide files retain their source headers; exactly one column contains `log10mic`. These are log10 MIC in µg/mL except the explicitly indexed EvoGradient no-negative exports in µM. Conversion uses `log10 MIC(µg/mL) = log10 MIC(µM) + log10(MW/1000)`, then pair subtraction divided by log10(2). The residue masses and water term are in `data/evo_residue_masses.json`.

## Numerical tables

| File in `results/reference/` | Rows | Unit of observation |
|---|---:|---|
| `benchmark_per_seed.csv` | 460 | Method × condition × seed |
| `benchmark_family_per_seed.csv` | 5,060 | Method × condition × seed × family |
| `benchmark_summary.csv` | 184 | Method × condition × metric |
| `benchmark_paired_tests.csv` | 92 | Method × metric; with-minus-without contrast |
| `cost_timing_v17.csv` | 23 | Backbone, accepted with-negative seed-123 time |
| `cost_best_PCC.csv`, `cost_best_RSE.csv` | 23 each | One PCC-selected condition per backbone |
| `cost_conditions_PCC.csv`, `cost_conditions_RSE.csv` | 46 each | Both conditions |

`n=10` is the required seed count; `n_defined` counts finite values without changing the denominator. Missing correlation values are represented by empty CSV cells (pandas NaN). `sd` is sample SD (`ddof=1`), and `ci_low`/`ci_high` are two-sided 95% Student t intervals. The zero-contribution sensitivity columns retained in historical per-seed files do not enter primary results.

## Training inputs

The external-predictor file named `train_restricted4929_tiger.csv` contains 5,737 measured records; the name refers to 4,929 unique sequences after the specified arithmetic-mean MIC aggregation. `external_negatives_positive_precedence.csv` contains 7,949 background sequences after removal of 13 exact measured-pool conflicts. Background pseudo-labels are training conventions, not measured MIC values.

The TIGER raw training snapshot has 11,516 rows; the recorded cleaning and sequence–inoculum aggregation yield 7,217 observations / 6,847 sequences. TIGER's negative resource has 7,860 sequences. The historical terminal-feature encoding and external CFU code are retained. The frozen source, detailed Methods and saved configurations define the input transformations. Training structures, precomputed embedding/graph binary caches and checkpoints are external assets, excluded from this release.

## Timing

Each `results/timing/<method>.json` retains individual `repeats_s`, median/min/max, endpoint checks and recorded device/source information. Checkpoint paths and hashes in these JSON files are provenance identifiers; the corresponding binaries are not included. The structure-time source data are in `results/reference/structure_timing_paired_source_data.csv`.

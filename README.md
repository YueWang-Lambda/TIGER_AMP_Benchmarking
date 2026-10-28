# TIGER AMP benchmarking: within-family ΔMIC prediction

[Benchmark Methods](docs/methods_benchmark_EN.md) · [Reproduction guide](docs/REPRODUCTION_GUIDE.md) · [Data dictionary](docs/DATA_DICTIONARY.md)

This repository packages the **frozen V17 benchmark** of TIGER-pair, TIGER-single and 21 external antimicrobial-potency predictors. It contains the training and evaluation source record, input tables, archived predictions, numerical results, original benchmark figures and a portable CPU-only reproduction workflow. **Model weights are excluded.**

The endpoint is the activity difference between related peptides within a family:

`ΔMIC = log2 MIC(anchor) − log2 MIC(query)`

MIC is expressed in µg/mL. Positive differences indicate a more potent query peptide. The external panel contains **208 peptides, 11 families and 2,211 oriented within-family pairs**. Every backbone has two training protocols and ten seeds (120–129), giving **460 prediction sets**. TIGER-ensemble is a separately reported, matched-seed average of calibrated TIGER-pair and TIGER-single predictions.

## Reproduce the results

From this directory, using Python 3.10:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python code/verify_release.py
python -m unittest discover -s tests -v
python code/evaluate.py --check-reference
python code/plot_results.py
```

The commands need no GPU, checkpoint, external server, or parent project directory. They write to `reproduced/`, leaving frozen results intact. Evaluation checks pair identities and labels, reconstructs all four metrics from the 460 prediction files, independently reconstructs the 420 external-method pair vectors from peptide predictions, rebuilds the ensemble, and verifies the archived timing medians. The validated release produces **23,660 numerical reference checks**, 5,060 family-metric rows and 460 seed-metric rows. A missing method, seed, pair, nonfinite prediction, unit mismatch or reference mismatch causes a nonzero exit.

To inspect and edit figures interactively:

```bash
python -m pip install -r requirements-notebook.txt
jupyter lab Benchmark_reproduction.ipynb
```

Each figure has its own notebook cell. Plotting uses Arial when installed, with an explicitly recorded DejaVu Sans fallback. Original manuscript PDF exports are in [figures/reference](figures/reference); the portable plotting script produces editable PDF, SVG and PNG figures with the same numerical endpoints.

## Result conventions

| View | Training-condition rule | Files |
|---|---|---|
| Formal Figure 4 benchmark | With negatives for every method | `results/reference/benchmark_summary.csv`, filter `condition=with_negatives` |
| Main PCC and RSE Pareto views | Highest defined ten-seed **PCC** selects one complete condition per backbone; RSE uses the same condition | `results/reference/cost_best_PCC.csv`, `cost_best_RSE.csv` |
| Supplementary Pareto views | Both conditions, connected within each backbone; no structures-ready points | `results/reference/cost_conditions_*.csv` |
| Ensemble supplement | Same-seed, 0.5/0.5 average of the two calibrated TIGER predictions | `results/reference/ensemble/` |

Condition selection in the Pareto views uses the observed external panel and is descriptive. The reference TIGER-pair model uses **unsigned-bin quota 20,000 and neighbour-based validation**. TIGER-single is the retained ten-seed single-peptide-head batch. The frozen reference has not been replaced by subsequent HydrAMP/EvoGradient sensitivity experiments.

Within each seed, PCC, Kendall τb (KCC), log10MAE and RSE are computed separately for each family, then averaged with equal family weights. Reported means and **sample SD** summarize ten seed-level values. Thus, PCC here is family-macro pair-difference correlation. For log2 labels `y` and predictions `p`, `log10MAE = mean(abs(p-y))*log10(2)` and `RSE = sum((p-y)^2)/sum((y-mean(y))^2)` within each family.

Undefined correlations propagate through the family macro and ten-seed summary. AMPLiT with-negative seed 122 is constant within Hydrodamin-3: its primary PCC/KCC summary is NA, and its error metrics remain defined. No family or seed is dropped. Paired condition tests use matched seed IDs, Wilcoxon signed-rank tests and Holm correction across 23 methods per metric; the seed is the statistical replicate.

## Result preview

| Method | Condition selected by PCC | PCC (mean ± SD) |
|---|---|---|
| TIGER-pair | with_negatives | 0.6543 ± 0.0168 |
| TIGER-single | with_negatives | 0.6273 ± 0.0165 |
| AMPActiPred | with_negatives | 0.6038 ± 0.0122 |
| HyperAMP | without_negatives | 0.5426 ± 0.0342 |
| APEX | with_negatives | 0.5166 ± 0.0669 |

All 23 backbones and both conditions are available in [benchmark_results.json](results/reference/benchmark_results.json), the CSV tables, and the [complete Excel workbook](docs/DeltaMIC_V17_complete_benchmark.xlsx). TIGER-ensemble PCC is 0.6801 ± 0.0106 and is reported separately.

## Repository map

| Directory / file | Contents |
|---|---|
| [code/evaluate.py](code/evaluate.py) | Portable metric, uncertainty, paired-test, condition-selection and Pareto reconstruction |
| [code/plot_results.py](code/plot_results.py) | Four-metric benchmark and PCC/RSE Pareto figures, including the ensemble |
| [code/verify_release.py](code/verify_release.py) | Inventory, SHA-256, exclusion and documentation-link checks |
| [code/training](code/training) | Readable representative external training drivers, preserved as historical source |
| [code/archive](code/archive) | Executed source snapshots, TIGER runtimes, auxiliary architectures, evaluation/timing and original plotting code |
| [configs](configs) | Forty TIGER method–condition–seed configurations; 200 actual fold-index files in `provenance/tiger_fold_indices/` |
| [data](data) | Canonical endpoint, external peptide metadata and training inputs |
| [results/predictions](results/predictions) | 460 complete pair-level prediction CSVs in log2-difference units |
| [results/peptide_predictions](results/peptide_predictions) | 420 external-method peptide prediction CSVs, retaining native export units |
| [results/reference](results/reference) | Frozen numerical results and ensemble tables |
| [results/timing](results/timing) | Per-repetition times and acceptance evidence; 23 backbones plus ensemble |
| [provenance](provenance) | Prediction/source indices, source hashes, launch records and release validation |
| [environments](environments) | Historical launch evidence and dated environment metadata |
| [docs](docs) | English/Chinese Methods, complete result report and review guides |
| [tests](tests) | Endpoint, undefined-correlation, scale, multiplicity and Pareto tests |
| [MANIFEST.json](MANIFEST.json) | SHA-256 and size inventory of released files |

## Training and timing provenance

Training-code review starts with [provenance/method_registry.csv](provenance/method_registry.csv), which identifies a representative driver for all 23 backbones. The executed per-seed source index is [provenance/executed_source_index.csv](provenance/executed_source_index.csv). Source snapshots retain historical paths and are not invoked by the portable reproduction commands. Re-running optimization or live GPU inference requires the documented framework environments, structure assets and externally supplied pretrained encoders/checkpoints; see the [reproduction guide](docs/REPRODUCTION_GUIDE.md). The release validates prediction replay; it does not represent a fresh retraining or latency measurement.

Latency is the median of 20 full-workload measurements after warm-up, with models resident and pair construction included. Both training conditions share the measured with-negative time coordinate because architecture and parameter count agree. The learned numerical weights differ. A **328.12 s** structure-generation increment is a serial estimate for 208 peptides from 30 measured peptides, added once to structural pipelines. Device differences and timing-noise limits remain documented in Methods.

The formal HydrAMP stopping-checkpoint/padding behavior and EvoGradient attention-mask convention are preserved and disclosed in [Methods](docs/methods_benchmark_EN.md). Their later corrective experiments are separate from this frozen benchmark. The [complete Chinese result report](docs/DeltaMIC_V17_complete_benchmark_ZH.docx) retains those separate experiments in a labelled appendix.

## Attribution and release scope

Original method citations appear in the bilingual Methods. Third-party architecture sources retain available license files; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No new blanket license is assigned to third-party code or data. This package was prepared for source and result review; it does not publish a repository or transfer model checkpoints. Original V17 files remain unchanged.

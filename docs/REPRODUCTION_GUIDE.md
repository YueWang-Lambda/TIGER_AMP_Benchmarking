# Reproduction and source-review guide

## 1. Reproduce the published numerical endpoint

Use the commands in the [README](../README.md). The portable entry points resolve paths from their own location and may be called from a different working directory. They read only this release. The Python environment in `requirements.txt` is sufficient for all endpoint checks and figures. Notebook dependencies are optional.

The evaluator checks the full Cartesian product of methods, conditions and seeds; verifies prediction hashes; checks the 208-peptide / 2,211-pair identities; verifies measured labels; independently converts all external-method peptide predictions; recomputes family and seed metrics, sample SD and 95% intervals; reproduces paired tests and Holm adjustment; and rebuilds the same-seed ensemble. Numerical comparisons use absolute tolerance 1e-9, with equal NaN masks. Pair labels and raw peptide-to-pair algebra use tighter stated tolerances in the source. Correlation alone is never used to validate prediction scale.

`reproduced/validation.json` records the outcome. Input failures propagate to a nonzero exit. `--no-reference-check` is intended only for evaluating a deliberately changed prediction collection; it is not the release-verification command.

## 2. Read the training implementation

Start with [method_registry.csv](../provenance/method_registry.csv), [English Methods](methods_benchmark_EN.md) and the representative scripts under [code/training](../code/training). The method registry covers all 23 backbones. The two TIGER heads also have ten configurations for each condition under `configs/`.

`provenance/executed_source_index.csv` maps with-negative external method/seed combinations to exact generated driver source and locally imported helpers. The source store is content-addressed by SHA-256. `provenance/training/<method>/seed*/` retains actual invocation, input and completion evidence where archived. `provenance/source_files.json` records source and packaged hashes for copied files.

The readable representative driver is seed 123; the index preserves the other seed-specific generated source. Original source snapshots are kept unchanged, including absolute training-machine paths. They are audit material. They are not the portable evaluation entry points and should not be launched against those historical paths.

The earlier packaged external-code snapshot is under `code/archive/v14_external`. The later no-negative-capable drivers available in the project are separately captured under `code/archive/noneg_capable_drivers`; their source hashes identify this packaging-time snapshot. This distinction matters because the older package can predate the no-negative command-line patches. The current snapshot alone is not evidence that a file was byte-identical at historical training time. Archived predictions remain independently checkable for every condition.

TIGER's frozen runtime and campaign sources are under `code/archive/tiger_runtime`, `tiger_v16_runtime`, `tiger_campaign` and `tiger_v16_campaign`. Repair/adapter source and auxiliary architectures are also retained. These directories contain historical utilities beyond the one selected driver; numerical benchmark membership is defined by the method registry and prediction index, not by the presence of an auxiliary script.

## 3. Re-running training or live inference

Training is method-specific. Before invoking a historical driver:

1. Select the registered model/condition and inspect its saved arguments/configuration, source index and method-specific target transform.
2. Recreate the recorded framework environment. Historical interpreter and device evidence is in `environments/training_environment_records.csv`, `method_environment_table.csv` and `environment_audit.json`. The separate `*_conda_packages_20261028.json` files describe the environment at packaging time; they are not historical complete lockfiles and omit pip-only packages.
3. Supply required assets outside this repository: pretrained ESM/ProtBERT/ProtT5/ankh/generative-language-model encoders as applicable, the registered structure and DSSP resources for structural methods, and method-specific descriptor resources. Weights and binary graph/embedding caches are intentionally excluded.
4. Relocate input/output/model-resource paths into a new workspace while preserving the recorded data cleaning, negative-pool precedence, split rule, seed, stopping rule and preprocessing. Retain the original source snapshot and record any relocation diff.
5. Train and export one complete 208-peptide vector or one complete 2,211-pair vector per registered seed. Apply the documented inverse target and unit conversions exactly once. For TIGER, retain fold-specific preprocessing and the actual calibration output.
6. Evaluate the export with the same canonical endpoint. New training under different hardware/software may follow different stochastic trajectories; keep new results distinct from the frozen V17 reference.

This release's CPU workflow was tested from an isolated copy. Full model retraining and live GPU inference were not rerun during packaging, and the training directories are not presented as a one-command fresh-install training system. The documented separation makes the executable numerical reproduction and historical training evidence explicit.

## 4. Timing and Pareto reconstruction

Accepted timing records are under `results/timing`. The historical timing drivers and orchestration are under `code/archive/timing_generated`, `timing_orchestration`, `analysis_v17` and `ensemble`. Models were resident, and 20 repetitions followed warm-up. Live feature extraction, prediction transforms and canonical pair construction are included as recorded. Method-specific devices are retained.

The portable evaluator checks the medians against all 23 published backbone coordinates; it does not perform a new stopwatch experiment. A structural pipeline receives the documented serial estimate of 328.12 s once. The ensemble shares the structure library. Both training conditions use the with-negative time coordinate because the architectures and parameter counts agree. Main RSE uses the PCC-selected condition, while the supplementary views display both conditions. Frontiers are computed on the displayed full-workload points, with strict dominance in at least one dimension.

## 5. Interpretation of the frozen comparison

The primary metric is an equal-family macro over within-family differences, followed by a mean over seeds. It differs from pooling all pairs or evaluating absolute MIC across families. Paired seeds are computational replicates; shared peptide endpoints are not independent biological samples.

The main TIGER-pair reference and PCC-selected training conditions were selected using this external panel. The resulting comparisons are exploratory. Methods differ in their registered training corpora and method-specific filters. HydrAMP's archived stopping-state/padding behavior, EvoGradient's formal attention-mask behavior, SenseXAMP's pre-split target normalization, and the AMPLiT undefined correlation are explicitly described in Methods. Preserve those descriptions when using this package to review the reported result.

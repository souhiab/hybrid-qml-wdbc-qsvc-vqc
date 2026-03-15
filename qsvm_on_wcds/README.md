# qsvm_on_wcds

This directory contains the working source for the WDBC QSVC/VQC project.

## Tracked Source Files

- `qsvm.ipynb`: main notebook for the hybrid QML study.
- `REAL_QPU.ipynb`: notebook for the real-hardware QSVC workflow.
- `explo.ipynb`: exploratory notebook.
- `quantum_no_scaling_sweep.py`: batch sweep for QSVC without feature scaling.
- `best_classical_table.md`: compact markdown summary of the best classical baseline settings.

## Tracked Reproducibility Inputs

- `classical_rfe_fixed_counts.csv`
- `classical_rfe_configs_with_auc_checked.csv`
- `CLASSICAL_svm_on_rfe_configs.csv`
- `splits/split_idx.json`
- `tables/table3_best_configs.csv`

## Ignored Outputs

The following are treated as generated artifacts and are intentionally not committed:

- `outputs/`
- `kernels_noisy_aer_brisbane/`
- `saved_models/`
- `saved_sphere_assets/`
- `section_451_outputs/`
- `wdbc_spheres/`
- `real QPU/`
- top-level generated CSV, PNG, PDF, DB, and screenshot files

Do not commit API keys, backend snapshots, model binaries, or per-run kernel dumps from local experiments.

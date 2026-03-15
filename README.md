# hybrid-qml-wdbc-qsvc-vqc

Hybrid quantum/classical machine-learning experiments on the Wisconsin Diagnostic Breast Cancer dataset, centered on QSVC/VQC studies, noisy-kernel simulations, and real-QPU follow-up work.

## Repository Layout

- `qsvm_on_wcds/qsvm.ipynb`: main experiment notebook for classical baselines, QSVC/VQC sweeps, and analysis.
- `qsvm_on_wcds/REAL_QPU.ipynb`: hardware-oriented notebook for real-QPU workflow and summaries.
- `qsvm_on_wcds/explo.ipynb`: exploratory notebook.
- `qsvm_on_wcds/quantum_no_scaling_sweep.py`: script for QSVC sweeps without feature scaling.
- `qsvm_on_wcds/splits/split_idx.json`: fixed split indices for reproducible comparisons.
- `qsvm_on_wcds/tables/table3_best_configs.csv`: compact manifest of selected best configurations.

## Tracking Policy

The repository keeps source notebooks, scripts, small reproducibility inputs, and lightweight documentation under version control.

Large generated artifacts are intentionally excluded, including:

- backend dumps and QPU run folders
- kernel matrices and checkpoint CSVs
- generated figures, PDFs, tables, and screenshots
- saved model binaries and NumPy/joblib artifacts
- vendored dependency checkouts

This keeps the repo usable on GitHub while preserving the files needed to continue development.

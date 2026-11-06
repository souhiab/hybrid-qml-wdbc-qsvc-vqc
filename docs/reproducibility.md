# Reproducibility Guide

This document describes how to reproduce the experiments reported in the
manuscript *"Hybrid Quantum-Classical Breast Cancer Classification: A
Systematic Study of Feature Maps, Entanglement, and Scaling on WDBC"*.

## 1. Environment

Create a fresh environment and install the dependencies:

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

The table lists the historical repository pins in `requirements.txt`.
The original execution-time package lock is unavailable. The corrected QSVC
check used Python 3.11.5, Qiskit 1.4.2 and Qiskit Machine Learning 0.8.4.

| Package | Version |
|---------|---------|
| qiskit | 2.2.1 |
| qiskit-aer | 0.17.2 |
| qiskit-machine-learning | 0.8.2 |
| qiskit-ibm-runtime | 0.42.0 (real-QPU only) |

The classical scientific stack uses lower bounds. Record the resolved versions
with `pip freeze` for any new evaluation.

## 2. Dataset

All experiments use the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset
(569 samples, 30 features, binary malignant/benign target). It is loaded
directly from scikit-learn and requires no manual download:

```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X, y = data.data, data.target
```

The same data is publicly available from the UCI Machine Learning Repository.
**No private or personally identifiable clinical data are used.**

## 3. Preprocessing and feature selection

- **Train/test split:** `train_test_split(..., test_size=0.2,
  random_state=42, stratify=y)`. The fixed seed (`42`) makes the split
  deterministic. The exact indices are also stored in
  `qsvm_on_wdbc/splits/split_idx.json` for cross-checking.
- **Scaling:** classical baselines are evaluated under Standardization
  (`StandardScaler`), Normalization (`MinMaxScaler`/normalization), and the
  no-scaling condition. Simulator fits estimate scaling and RFE on the 455
  training rows. The no-scaling QSVC uses scaling only for RFE and passes raw
  selected columns to the kernel.
- **Feature selection:** Recursive Feature Elimination (RFE) is used to select
  feature subsets of sizes **k = 4, 7, 11, 17, 18**. The following tables
  contain archived feature sets and scores:
  - `qsvm_on_wdbc/classical_rfe_fixed_counts.csv`
  - `qsvm_on_wdbc/classical_rfe_configs_with_auc_checked.csv`
  - `qsvm_on_wdbc/CLASSICAL_svm_on_rfe_configs.csv`

## 4. Classical baselines

The best classical configuration per feature count is summarized in
`qsvm_on_wdbc/best_classical_table.md` (Logistic Regression with
Standardization is the strongest baseline, reaching ~0.991 test accuracy at
k = 17/18). The notebook writes training-only evaluations to separate files.

## 5. Quantum experiments (QSVC / VQC)

The quantum study sweeps the following configuration grid:

- **Feature maps:** `ZFeatureMap`, `ZZFeatureMap`, `PauliFeatureMap`
- **Entanglement:** `linear`, `full`, `circular`
- **Repetitions (reps):** `1, 2, 3, 4`
- **Feature counts (qubits):** `k = 4, 7, 11, 17, 18`
- **Kernel:** `FidelityStatevectorKernel` (statevector simulation) for QSVC
- **Classifier:** `QSVC(quantum_kernel=..., probability=True)`; VQC results are
  produced in the main notebook.

### 5.1 No-scaling QSVC sweep (script)

```bash
cd qsvm_on_wdbc
python quantum_no_scaling_sweep.py
```

The script reads the first archived RFE configuration key per width, refits
its feature selector on the training cases and uses raw selected features for
QSVC. Results are appended to `quantum_no_scaling_qsvc_results_train_only.csv`.
The resume key is (k, feature map, entanglement, repetitions).

### 5.2 Main study (notebook)

```bash
cd qsvm_on_wdbc
jupyter notebook qsvm.ipynb
```

The scaled QSVC/VQC fit cell uses widths 4, 7, 11 and 17. It refits scaling
and RFE for each preprocessing configuration and writes
`quantum_each_results_train_only.csv`. Resume from that file in the same
environment. The later historical analysis cells read the original archive
filenames. Keep those inputs separate from new evaluations.

### 5.3 Real-QPU follow-up (optional)

`qsvm_on_wdbc/REAL_QPU.ipynb` records historical device and noisy-Aer workflows.
Some archived hardware workflows fitted preprocessing on the full dataset.
Treat their outputs as historical feasibility evidence. It requires an IBM Quantum account.
**Provide your token via an environment variable — never hard-code it:**

```bash
# Windows (PowerShell):  $env:IBM_QUANTUM_TOKEN = "<your token>"
# Linux/macOS:           export IBM_QUANTUM_TOKEN="<your token>"
```

The notebook reads `os.environ["IBM_QUANTUM_TOKEN"]` in the account-setup cell.

## 6. Metrics

Reported metrics are accuracy (train/test), the train−test gap (overfit
score), ROC-AUC, F1, and the confusion-matrix breakdown (TP/FP/TN/FN), all
computed with scikit-learn.

## 7. Figures and tables

Figures and result tables are generated inside `qsvm_on_wdbc/qsvm.ipynb` and
`qsvm_on_wdbc/explo.ipynb` from the result CSVs. Generated image/PDF/CSV
artifacts are intentionally **not** version-controlled (see `.gitignore`);
re-running the notebooks regenerates them. The compact best-configuration
manifest is tracked at `qsvm_on_wdbc/tables/table3_best_configs.csv`.

## 8. Random seeds and reproducibility notes

- The classical/quantum train-test split uses `random_state=42`.
- Statevector QSVC results are deterministic given fixed inputs and package
  versions.
- **Hardware/backend variability:** real-QPU runs (`REAL_QPU.ipynb`) and
  shot-based noisy simulations are subject to device calibration drift and
  shot noise, so exact numbers will vary between executions. The qualitative
  conclusions are robust to this variability.

## 9. Quick validation (smoke test)

A lightweight smoke test verifies that the dataset loads, preprocessing runs,
and a small classical baseline trains:

```bash
python scripts/smoke_test.py
```

It completes in well under a minute and does not require quantum hardware. If
the quantum stack is installed, it also runs a minimal QSVC sanity check.

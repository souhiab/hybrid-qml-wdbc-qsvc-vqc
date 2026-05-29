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

The quantum stack is pinned in `requirements.txt` to the versions used to
produce the reported results:

| Package | Version |
|---------|---------|
| qiskit | 2.2.1 |
| qiskit-aer | 0.17.2 |
| qiskit-machine-learning | 0.8.2 |
| qiskit-ibm-runtime | 0.42.0 (real-QPU only) |

The classical scientific stack (numpy, pandas, scipy, scikit-learn,
matplotlib, seaborn) uses conservative lower bounds. Exact classical versions
are not pinned because the classical results are numerically stable across
recent releases; if you need bit-for-bit reproduction, pin them to the
versions installed in your environment (`pip freeze`).

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
  `qsvm_on_wcds/splits/split_idx.json` for cross-checking.
- **Scaling:** classical baselines are evaluated under Standardization
  (`StandardScaler`), Normalization (`MinMaxScaler`/normalization), and the
  no-scaling condition. The quantum no-scaling sweep
  (`quantum_no_scaling_sweep.py`) deliberately uses raw features.
- **Feature selection:** Recursive Feature Elimination (RFE) is used to select
  feature subsets of sizes **k = 4, 7, 11, 17, 18**. The selected feature sets
  are recorded in:
  - `qsvm_on_wcds/classical_rfe_fixed_counts.csv`
  - `qsvm_on_wcds/classical_rfe_configs_with_auc_checked.csv`
  - `qsvm_on_wcds/CLASSICAL_svm_on_rfe_configs.csv`

## 4. Classical baselines

The best classical configuration per feature count is summarized in
`qsvm_on_wcds/best_classical_table.md` (Logistic Regression with
Standardization is the strongest baseline, reaching ~0.991 test accuracy at
k = 17/18). These are reproduced inside `qsvm_on_wcds/qsvm.ipynb`.

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
cd qsvm_on_wcds
python quantum_no_scaling_sweep.py
```

This reads `classical_rfe_fixed_counts.csv`, runs the full QSVC grid on raw
(unscaled) RFE features, and appends rows to
`quantum_no_scaling_qsvc_results.csv`. The script is resumable: already
computed (k, feature_map, entanglement, reps) combinations are skipped.

### 5.2 Main study (notebook)

```bash
jupyter notebook qsvm_on_wcds/qsvm.ipynb
```

Run the cells top-to-bottom to reproduce the classical baselines, the
QSVC/VQC sweeps, the scaling comparison, and the analysis/plots.

### 5.3 Real-QPU follow-up (optional)

`qsvm_on_wcds/REAL_QPU.ipynb` runs the selected configuration on IBM Quantum
hardware and with noisy Aer simulation. It requires an IBM Quantum account.
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

Figures and result tables are generated inside `qsvm_on_wcds/qsvm.ipynb` and
`qsvm_on_wcds/explo.ipynb` from the result CSVs. Generated image/PDF/CSV
artifacts are intentionally **not** version-controlled (see `.gitignore`);
re-running the notebooks regenerates them. The compact best-configuration
manifest is tracked at `qsvm_on_wcds/tables/table3_best_configs.csv`.

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

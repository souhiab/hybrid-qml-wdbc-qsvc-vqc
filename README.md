# WDBC QSVC, classical and VQC experiments

Verified completed studies, corrected training-only preprocessing, original execution snapshots and two matched IBM hardware checks. No authoritative manuscript is included or used in this update.

## Completed evidence

|Study|Verified coverage|Type|
|---|---:|---|
|qsvc_fixed_split|256/256|QSVC fixed split|
|shared_qsvc_classical|255/255|QSVC/classical shared3x5folds|
|vqc_k04_cap100|192/192|VQC simulator|
|vqc_k04_budget_split1|144/144|VQC simulator|
|vqc_k04_budget_split2|96/96|VQC simulator|
|vqc_k04_cap200|192/192|VQC simulator|
|vqc_k07_budget|144/144|VQC simulator|
|vqc_k11_budget|144/144|VQC simulator|
|vqc_k07_cap300|192/192|VQC simulator|
|vqc_k11_cap400|192/192|VQC simulator|
|vqc_shared_k04_k07|180/180|VQC simulator|
|vqc_shared_k11|90/90|VQC simulator|
|vqc_k17_budget_100_300|96/96|VQC simulator|
|hardware_q9|2/2|QSVC hardware matched exact/finite-shot|

The 17-qubit cap-600 pilot is incomplete and excluded from completed results. The cap-100 and cap-300 stages are separate completed evidence; no final 17-qubit budget decision is claimed.

## Verify without training

```powershell
python tools/verify_results.py
```

Requires NumPy. This verifies saved files and recomputes metrics without fitting WDBC models or contacting IBM. See [reproducibility](docs/reproducibility.md) and [known issues](docs/KNOWN_ISSUES.md).

## Structure

```text
results/                 Verified predictions, metrics, workbooks, per-fit evidence
  STUDIES.json           Coverage and completion index
  qsvc_fixed_split/      256 corrected configurations
  shared_qsvc_classical/ 150 QSVC + 105 classical outer evaluations
  vqc_*/                 Completed grids, shared folds and budget pilots
  hardware_q9/           Two disjoint 72-case matched hardware cohorts
implementations/         Original validated code, configuration and split snapshots
tools/                   Portable read-only evidence verification
qsvm_on_wdbc/budgeted_qpu/ Guarded hardware implementation
scripts/                 Synthetic regression tests
docs/                    Corrections, limitations and reproduction instructions
legacy/                  Historical notebook guidance
```

The 255 shared QSVC/classical evaluations include 150 QSVC, 75 matched RBF-SVM and 30 verified strong-classical results. All use the same three-repeat, five-fold assignments. Later VQC shared-fold studies retain three training seed outcomes per candidate and partition. Do not count repeated patient predictions as independent patients.

Malignant is original source label 0. Training-only preprocessing repairs leakage, but historical selection on WDBC remains: these are internal exploratory comparisons, not independent validation. Hardware cohorts each contain 48 training and 24 held-out cases; matched exact, finite-shot and hardware results are reported separately regardless of performance.

Original notebooks are retained for historical compatibility. Their displayed outputs are not the default corrected results. Execution snapshots can contain original machine paths and must be relocated into new output directories before rerunning. No new training or QPU submissions were performed for this publication update.

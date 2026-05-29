# Hybrid Quantum-Classical Breast Cancer Classification (WDBC)

Source code and reproducibility materials for the study **"Hybrid
Quantum-Classical Breast Cancer Classification: A Systematic Study of Feature
Maps, Entanglement, and Scaling on WDBC."**

The project systematically compares classical baselines against quantum
support vector classifiers (QSVC) and variational quantum classifiers (VQC) on
the Wisconsin Diagnostic Breast Cancer (WDBC) dataset, studying the effect of
quantum feature maps, entanglement structure, circuit repetitions, feature
scaling, and the number of selected features. It also includes noisy-simulation
and real-QPU follow-up experiments on IBM Quantum hardware.

## Dataset

This study uses the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset
(569 samples, 30 features, binary malignant/benign target). It is loaded
directly from scikit-learn:

```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
```

The same dataset is publicly available from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
No manual download is required and **no private or personally identifiable
clinical data are included in this repository.**

## Repository contents

```
.
├── README.md
├── LICENSE                      # MIT
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── docs/
│   ├── reproducibility.md       # full step-by-step reproduction guide
│   └── manuscript_availability.md
├── scripts/
│   └── smoke_test.py            # quick environment / pipeline sanity check
└── qsvm_on_wcds/
    ├── qsvm.ipynb               # main study: classical baselines + QSVC/VQC sweeps + analysis
    ├── REAL_QPU.ipynb           # real-QPU + noisy-simulation follow-up (IBM Quantum)
    ├── explo.ipynb              # exploratory analysis and figures
    ├── quantum_no_scaling_sweep.py   # batch QSVC sweep on raw (unscaled) RFE features
    ├── best_classical_table.md  # best classical baseline per feature count
    ├── classical_rfe_fixed_counts.csv          # RFE-selected feature sets (k = 4,7,11,17,18)
    ├── classical_rfe_configs_with_auc_checked.csv
    ├── CLASSICAL_svm_on_rfe_configs.csv
    ├── splits/split_idx.json    # fixed train/test split indices
    └── tables/table3_best_configs.csv          # compact best-configuration manifest
```

Large generated artifacts (figures, PDFs, kernel matrices, saved models,
per-run dumps) are intentionally not version-controlled; they are regenerated
by running the notebooks. See `.gitignore` for the full list.

## Installation

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

The quantum stack is pinned to the versions used for the reported experiments
(`qiskit==2.2.1`, `qiskit-aer==0.17.2`, `qiskit-machine-learning==0.8.2`).

## Reproducing the experiments

A quick sanity check of the environment and pipeline:

```bash
python scripts/smoke_test.py
```

The no-scaling QSVC sweep (command-line, resumable):

```bash
cd qsvm_on_wcds
python quantum_no_scaling_sweep.py
```

The full study, including classical baselines, the QSVC/VQC configuration
sweeps, the scaling comparison, and all figures/tables, is reproduced by
running the notebook top-to-bottom:

```bash
jupyter notebook qsvm_on_wcds/qsvm.ipynb
```

See **[docs/reproducibility.md](docs/reproducibility.md)** for the detailed
methodology: preprocessing, RFE feature selection, feature maps, entanglement
settings, repetitions, metrics, random seeds, and how figures/tables are
generated.

### Expected outputs

- `qsvm_on_wcds/quantum_no_scaling_qsvc_results.csv` — one row per
  (feature count, feature map, entanglement, reps) with accuracy, AUC, F1,
  the overfit gap, and the confusion-matrix breakdown.
- Notebook runs regenerate the comparison tables and figures referenced in the
  manuscript.

### Real-QPU runs (optional)

`qsvm_on_wcds/REAL_QPU.ipynb` targets IBM Quantum hardware and noisy Aer
simulation. Provide your IBM Quantum API token through an environment variable
(never hard-code it):

```bash
# Windows (PowerShell):  $env:IBM_QUANTUM_TOKEN = "<your token>"
# Linux/macOS:           export IBM_QUANTUM_TOKEN="<your token>"
```

## Data availability

This study uses the Wisconsin Diagnostic Breast Cancer dataset, which is
publicly available from the UCI Machine Learning Repository and through the
`sklearn.datasets.load_breast_cancer` utility. No private or personally
identifiable clinical data are included in this repository.

## Code availability

The source code and reproducibility materials for the experiments reported in
the manuscript are publicly available in this repository:

https://github.com/souhiab/hybrid-qml-wdbc-qsvc-vqc

## Citation

If you use this code, please cite the associated manuscript. Citation metadata
is provided in [CITATION.cff](CITATION.cff).

## License

Released under the [MIT License](LICENSE).

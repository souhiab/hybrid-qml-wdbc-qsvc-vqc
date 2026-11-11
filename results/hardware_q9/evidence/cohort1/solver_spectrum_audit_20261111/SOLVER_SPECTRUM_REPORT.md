# Saved SVC, kernel spectrum and permutation audit

Generated: 2026-11-11T01:33:10.525420+00:00. sklearn 1.6.1. No QPU submission or new quantum computation. Original models, kernels, predictions and metrics are unchanged.

## Saved fitted models

|Kernel|fit_status_|n_iter_|Support vectors (class0 / class1)|
|---|---:|---:|---|
|local_exact|0|[17]|[8, 9]|
|local_shots|0|[24]|[9, 10]|
|hardware|0|[25]|[10, 9]|

fit_status_=0 records successful solver termination, not proof of a global optimum or generalization. Structured warnings were not captured at the original fits, so their historical absence cannot be inferred from the model files. No ConvergenceWarning occurrence was found in the saved original hardware console. All identity replays and permutation refits explicitly capture warnings; see per-fit JSON. n_iter_ describes the binary decision SVM fit, not every internal probability-calibration optimization.

## Spectrum magnitude

Matrices: saved raw 48x48 training kernels, symmetrized only for eigensolver round-off; no PSD correction or diagonal adjustment. A negative eigenvalue is counted when lambda < -1e-10. M- = sum(abs(lambda)) over these negative eigenvalues; S = sum(abs(lambda)) over the entire spectrum. Reported negative mass percentage =100*M-/S. The denominator is the absolute full spectrum, not the signed trace. Unthresholded mass, signed-trace ratio and squared-eigenvalue energy fraction are also saved.

|Kernel|Minimum eigenvalue|Negative count|M-|Full absolute spectrum S|100 M-/S|
|---|---:|---:|---:|---:|---:|
|local_exact|2.82880581819e-09|0|-0|48|-0.00000000%|
|local_shots|-0.258079623234|19|2.51920824926|53.0384164985|4.74978028%|
|hardware|-0.266371862602|20|2.33809179092|52.6761835818|4.43861273%|

The ideal exact kernel provides the numerical reference; finite-shot and hardware matrices are compared with it. Negative mass measures the magnitude of indefiniteness but has no universal acceptability threshold. It does not establish that a downstream solver is stable or unstable on its own.

## Exploratory permutation check

Predeclared identity replay, reverse order and10 fixed random permutations (seed20261111) for each kernel:36 classical fits in total. All reuse the same saved kernels and original SVC settings. Each permutation uses K[P,P], y[P] and K_test[:,P]; training outputs are inverse-permuted. No test-label feedback changes a fit or selects a permutation.

|Kernel|Permuted refits|Refits with any changed test prediction|Max changed /24|Max absolute malignant margin change|Max malignant probability change|Iteration range|Warnings|
|---|---:|---:|---:|---:|---:|---|---:|
|local_exact|11|0|0|0.000509832995|0.0344029336|16–23|0|
|local_shots|11|0|0|0.12663883|0.0713619417|20–35|0|
|hardware|11|0|0|0.000802127252|0.0318581072|20–28|0|

Identity replays reproduced the saved held-out classes, decision scores and probabilities. Row order can change probability=True internal calibration folds even with a fixed random seed; probability changes therefore do not isolate the decision solver. Native class predictions and decision margins are reported separately. This finite set of permutations is exploratory evidence only: unchanged predictions do not prove uniqueness, convexity, general stability or external validity. The original production model and score are retained regardless of the audit outcomes.

Files: diagnostic_plan.json, input_hashes.json, saved_svc_diagnostics.json, Spectrum_Comparison.csv, Eigenvalues.csv, Permutation_Stability.csv, stability_summary.json, per-fit warning/status JSON and permutation_predictions_scores.npz.

# Eleven-qubit VQC budget review

All 144 internal-validation fits completed: 16 fixed balanced configurations, three initialization/sampler seed pairs, budgets 100/200/300. Preprocessing fits only 364 original-training cases; evaluation uses the same 91 internal-validation cases. The original 114-case test split is not used.

| Cap | Mean validation accuracy | Mean malignant AUC | Converged | Budget exhausted | Mean actual nfev | Sum fit runtime (worker-minutes) |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 61.2179% | 0.673031 | 0/48 | 48/48 | 100.00 | 24.09 |
| 200 | 61.9277% | 0.686189 | 2/48 | 46/48 | 199.48 | 43.65 |
| 300 | 61.9505% | 0.686651 | 24/48 | 24/48 | 267.69 | 60.66 |

Sum fit runtimes add concurrent worker durations and are not elapsed wall time. Recorded elapsed stages at caps 200 and 300 were approximately 14.77 and 20.54 minutes respectively.

| Comparison | Mean accuracy change (percentage points) | Descriptive paired 95% patient-bootstrap interval |
|---|---:|---|
| 200-100 | +0.7097 | [-0.1145, +1.5110] |
| 300-200 | +0.0229 | [-0.5037, +0.5495] |
| 300-100 | +0.7326 | [-0.1145, +1.5568] |

Intervals use 10,000 paired resamples, seed 20261108, with the same patient indices across the fixed model panel. They are conditional on this split and these fitted models; repeated predictions are not extra patients. The 300-200 interval remains inconclusive relative to the proposed +0.5-point margin.

## Decision

No tested cap meets the predefined 90% optimizer-completion target (at least 44/48 fits). Therefore the frozen automatic recommendation remains unavailable. Cap 300 has the highest observed mean accuracy and substantially more convergence, but only half of fits converge; selecting it requires an explicit documented exception. Cap 200 offers nearly the same observed predictive metrics at lower runtime, with substantially less optimizer completion. Neither should be described as a converged full panel.

If optimizer completion is the priority, 300 is the provisional choice among tested caps, retaining and labeling all 24 valid budget-exhausted outcomes. If the 90% criterion remains binding, further budget investigation is needed before choosing a qualifying cap. Do not automatically extend or launch a complete grid.

Before choosing another cap, inspect the saved training-only histories of the 24 budget-limited fits and plan the next comparison in advance. Validation accuracy alone does not identify how much extra training is needed. Any future comparison should retain the matched configuration/seed panel and use fresh models from the same saved initialization rather than unsupported warm starts. No budgets above 300 have been launched.

## Configuration-specific changes and training diagnostics

{
  "budget_exhausted_at_300": 24,
  "mean_last20_best_training_loss_improvement_exhausted": 0.00043264685844285794,
  "median_last20_best_training_loss_improvement_exhausted": 0.00013414777295978064,
  "fit_pairs_improved_300_over_200": 13,
  "fit_pairs_unchanged": 19,
  "fit_pairs_declined": 16,
  "configuration_mean_change_min_pp": -2.930402930402931,
  "configuration_mean_change_max_pp": 1.8315018315018305
}

Detailed exhausted-fit training diagnostics: analysis/budget_review_training_diagnostics.json. Larger-width full grids remain unlaunched.

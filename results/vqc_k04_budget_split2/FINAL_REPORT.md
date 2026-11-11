# Second-split VQC budget confirmation

Descriptive results only. Budget selection and substantive interpretation are deferred to a later user review.
16 fixed balanced configurations, three seed pairs, two independent fresh fits at budgets 100 and 200: 96 expected fits.
Split seed 20261109: 364 training and 91 validation cases from the original 455 training cases only. Scaling and RFE fitted on the 364 cases only. The original test partition is not used.
The two internal splits overlap and must not be pooled as independent patient samples.

## Budget summaries

| Budget | Completed | Failed | Mean accuracy | Mean malignant AUC | Mean nfev | Runtime seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 48 | 0 | 0.722527 | 0.788141 | 98.854167 | 99.207191 |
| 200 | 48 | 0 | 0.722985 | 0.788307 | 139.812500 | 106.903434 |

## Paired descriptive uncertainty

[
  {
    "comparison": "200-100",
    "priority": "primary",
    "mean_change_pp": 0.04578754578754575,
    "ci95_low_pp": -0.11446886446886431,
    "ci95_high_pp": 0.2060439560439559,
    "proposed_margin_pp": 0.5,
    "margin_interpretation": "deferred_to_later_user_review",
    "bootstrap_resamples": 10000,
    "bootstrap_seed": 20261109
  }
]
10,000 paired patient resamples, seed 20261109; the same patient draws apply to every configuration, training seed and budget. The model panel remains fixed.
Intervals are conditional on this training split, fitted models, configurations and seeds. They exclude retraining and pilot-selection uncertainty. The proposed 0.5 percentage-point margin is recorded for later review; no automatic decision is made.
Budget-exhausted fits are valid budget-limited outcomes, not automatically converged. Failures and interrupted attempts remain recorded.
COBYLA restoration is unsupported: interrupted fits restart from their original vector and seeds under a new attempt ID.
## Execution

{
  "selected_workers": 3,
  "threads_per_worker": 1,
  "cpu_only": true,
  "physical_cores": 4,
  "logical_processors": 8,
  "system_ram_reserve_bytes": 1014017802,
  "available_ram_at_freeze": 7219097600,
  "prior_measured_worker_rss_bytes": 200000000,
  "memory_margin_factor_per_worker": 2,
  "reason": "Reuse validated three-worker first-pilot execution, reduced only for current CPU/RAM headroom. No extra tuning fits or metric-driven changes.",
  "optimizations": [
    "unchanged cached encoder states",
    "vectorized exact seeded shot counting",
    "invariant preprocessing reuse",
    "incremental sole-writer Excel"
  ],
  "frozen": "2026-11-08T02:56:32+00:00"
}

The unchanged verified numerical engine caches invariant encoder states and vectorizes exact seeded shot counting. BLAS/OpenMP threads are bounded to one per worker. Excel is written by one coordinator after authoritative artifact receipts.
## Resume

Run resume.ps1 in D:\PHYSIQUE MATHEMATIQUE\doctora\4 EME ANNE\articles\breast canser\wdbc_vqc_budget_confirmation_split2_20261108. Verified receipts are skipped.

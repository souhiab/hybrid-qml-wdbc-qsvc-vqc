# Four-qubit VQC internal-validation budget pilot

Final verification passed for all 144 fits and all ten workbook sheets. All scientific source files match the pre-execution freeze, and the numerical engine is byte-identical to the previously verified implementation. The attempt-log Excel roundtrip shifted some timestamps by one millisecond; the final readback allows that date-serial conversion error only for timestamps recorded at one-second precision. All non-timestamp values are compared exactly. No model, prediction or metric was changed by this export recovery.

An independent recalculation reproduced all 10,000 bootstrap patient-index draws and interval endpoints (maximum floating-point difference 2.22e-16). All 48 validation class-prediction arrays are identical between budgets 200 and 300, while three probability arrays differ slightly. This explains the degenerate [0,0] accuracy interval; it does not establish general equivalence.

Production used three workers with one BLAS/OpenMP thread each. Mean system CPU utilization was 53.2%, 53.7% and 55.7% for the remaining 100-stage fits, the 200 stage and the 300 stage, respectively. Minimum available RAM during those batches remained above 5.31 GB; observed per-worker RSS stayed below 199 MB. The six-fit resource benchmarks measured training-span throughput of 0.857, 1.200 and 1.500 fits/s for one, two and three workers. Including Excel synchronization, the corresponding short-benchmark throughputs were 0.530, 0.622 and 0.535 fits/s. Three workers were selected for training throughput; these short timing panels use different declared seeds and are descriptive. Production wall times including Excel were 35.4 seconds for the 30 post-benchmark 100-stage fits, 59.3 seconds for all 48 at 200, and 78.4 seconds for all 48 at 300. Runtime differences also reflect scheduling, cold-cache and I/O effects; the observed timing ratios are not pure objective-evaluation cost ratios.

The 100-evaluation recommendation concerns the proposed accuracy/runtime balance. It still leaves 38/48 fits budget-limited. If optimizer termination itself is required, 200 evaluations achieved convergence for 45/48 fits and 300 for 48/48 in this pilot. Those stopping-status gains did not translate into an accuracy gain exceeding the proposed margin. Training failures, interrupted fits and technical training retries were all zero. The only execution intervention concerned final Excel verification.

Execution: 144/144 valid fits, 0 failed, 0 pending; 16 balanced configurations, three fixed seed pairs, all three mandatory budgets.

This balanced pilot is not a full factorial grid. Feature-map index is (preprocessing index + repetition-pair index) modulo 4. It uses 364 pilot-training and 91 validation cases drawn only from the original 455 training cases. The original 114-case test partition was not inspected or scored.

## Existing training-history audit

Inspected 192 training traces and optimizer records only: 27 converged and 165 budget-exhausted. 179 traces meet the predefined finite-shot plateau-ambiguity flag. The flag compares late best-loss progress with the tail spread and upward steps; histories cannot isolate sampling variation from optimizer moves. With the installed integer sampler seed, repeated identical parameter vectors are deterministic.

## Complete budget comparison

| Budget | Valid / expected | Mean validation accuracy | Mean malignant AUC | Mean sensitivity | Mean specificity | Mean F1 | Mean nfev | Converged | Budget exhausted | Sum fit runtime (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 48/48 | 0.706502 | 0.787377 | 0.436275 | 0.867690 | 0.515127 | 98.90 | 10 | 38 | 86.664 |
| 200 | 48/48 | 0.707875 | 0.788103 | 0.439338 | 0.868056 | 0.518598 | 137.15 | 45 | 3 | 108.911 |
| 300 | 48/48 | 0.707875 | 0.788108 | 0.439338 | 0.868056 | 0.518598 | 138.60 | 48 | 0 | 144.524 |

Accuracy and secondary clinical metrics are averaged equally across the fixed 16 configurations and three seed pairs. Each fit uses the same 91 validation patients; predictions are never pooled as extra patients. Budget exhaustion is retained as a valid budget-limited result. Per-configuration three-seed means, sample SDs, minima and maxima are in Configuration_Summary.csv.

## Paired accuracy differences and uncertainty

| Comparison | Priority | Mean change (pp) | Descriptive 95% patient-bootstrap interval (pp) | Interpretation relative to proposed +0.5 pp |
|---|---|---:|---|---|
| 200-100 | primary | 0.137363 | [-0.068681, 0.343407] | incremental_gain_below_proposed_margin |
| 300-200 | primary | 0.000000 | [0.000000, 0.000000] | incremental_gain_below_proposed_margin |
| 300-100 | secondary | 0.137363 | [-0.068681, 0.343407] | incremental_gain_below_proposed_margin |

The bootstrap uses 10,000 ordinary resamples of the 91 validation patients, seed 20261108. The same sampled patient indices apply simultaneously to every budget, configuration and seed. The fitted-model panel stays fixed. Percentile endpoints are 2.5% and 97.5% using NumPy linear interpolation.

Intervals are conditional on this pilot-training split, these fitted models, these 16 configurations and these three seed pairs. They do not capture uncertainty from retraining on different training samples or selecting the pilot. They are descriptive: no formal multiple-comparison, equivalence or nonsignificance-as-equivalence claim is made. The 0.5-percentage-point margin is proposed for this comparison, not a universal scientific standard.

## Training progress and exceptions

- Budget 100: mean best training loss 0.74506379; mean fitted loss 0.74507651; mean best-loss improvement in the final 20 evaluations 0.00210080.
- Budget 200: mean best training loss 0.74397153; mean fitted loss 0.74402959; mean best-loss improvement in the final 20 evaluations 0.00005023.
- Budget 300: mean best training loss 0.74396678; mean fitted loss 0.74402739; mean best-loss improvement in the final 20 evaluations 0.00004869.
- 200-100: 3/48 seed-fit pairs regressed, 41 were unchanged, 4 improved. Configuration-level three-seed mean changes ranged from -0.366300 to 1.831502 pp.
  Regressing configurations: pilot-p0-r1-m1 (-0.366300 pp), pilot-p0-r2-m2 (-0.366300 pp), pilot-p3-r0-m3 (-0.366300 pp).
- 300-200: 0/48 seed-fit pairs regressed, 48 were unchanged, 0 improved. Configuration-level three-seed mean changes ranged from 0.000000 to 0.000000 pp.
- 300-100: 3/48 seed-fit pairs regressed, 41 were unchanged, 4 improved. Configuration-level three-seed mean changes ranged from -0.366300 to 1.831502 pp.
  Regressing configurations: pilot-p0-r1-m1 (-0.366300 pp), pilot-p0-r2-m2 (-0.366300 pp), pilot-p3-r0-m3 (-0.366300 pp).

| Grouping | Level | Comparison | Mean accuracy change (pp) | Negative / zero / positive seed-fit pairs |
|---|---:|---|---:|---|
| feature_map | 0 | 200-100 | 0.091575 | 0/11/1 |
| feature_map | 0 | 300-200 | 0.000000 | 0/12/0 |
| feature_map | 0 | 300-100 | 0.091575 | 0/11/1 |
| feature_map | 1 | 200-100 | -0.000000 | 1/10/1 |
| feature_map | 1 | 300-200 | 0.000000 | 0/12/0 |
| feature_map | 1 | 300-100 | -0.000000 | 1/10/1 |
| feature_map | 2 | 200-100 | -0.091575 | 1/11/0 |
| feature_map | 2 | 300-200 | 0.000000 | 0/12/0 |
| feature_map | 2 | 300-100 | -0.091575 | 1/11/0 |
| feature_map | 3 | 200-100 | 0.549451 | 1/9/2 |
| feature_map | 3 | 300-200 | 0.000000 | 0/12/0 |
| feature_map | 3 | 300-100 | 0.549451 | 1/9/2 |
| repetition_pair | 0 | 200-100 | -0.091575 | 1/11/0 |
| repetition_pair | 0 | 300-200 | 0.000000 | 0/12/0 |
| repetition_pair | 0 | 300-100 | -0.091575 | 1/11/0 |
| repetition_pair | 1 | 200-100 | 0.183150 | 1/9/2 |
| repetition_pair | 1 | 300-200 | 0.000000 | 0/12/0 |
| repetition_pair | 1 | 300-100 | 0.183150 | 1/9/2 |
| repetition_pair | 2 | 200-100 | -0.091575 | 1/11/0 |
| repetition_pair | 2 | 300-200 | 0.000000 | 0/12/0 |
| repetition_pair | 2 | 300-100 | -0.091575 | 1/11/0 |
| repetition_pair | 3 | 200-100 | 0.549451 | 0/10/2 |
| repetition_pair | 3 | 300-200 | 0.000000 | 0/12/0 |
| repetition_pair | 3 | 300-100 | 0.549451 | 0/10/2 |
| preprocessing | 0 | 200-100 | 0.274725 | 2/9/1 |
| preprocessing | 0 | 300-200 | 0.000000 | 0/12/0 |
| preprocessing | 0 | 300-100 | 0.274725 | 2/9/1 |
| preprocessing | 1 | 200-100 | 0.000000 | 0/12/0 |
| preprocessing | 1 | 300-200 | 0.000000 | 0/12/0 |
| preprocessing | 1 | 300-100 | 0.000000 | 0/12/0 |
| preprocessing | 2 | 200-100 | 0.274725 | 0/10/2 |
| preprocessing | 2 | 300-200 | 0.000000 | 0/12/0 |
| preprocessing | 2 | 300-100 | 0.274725 | 0/10/2 |
| preprocessing | 3 | 200-100 | -0.000000 | 1/10/1 |
| preprocessing | 3 | 300-200 | 0.000000 | 0/12/0 |
| preprocessing | 3 | 300-100 | -0.000000 | 1/10/1 |

Group labels follow the frozen p/r/feature-map indices in the manifest. These subgroup patterns are descriptive and do not estimate all factorial interactions.

## CPU execution and recovery

Resource experiments: {"available_ram_at_freeze": 6308532224, "benchmarks": [{"configuration_ids": ["pilot-p0-r0-m0", "pilot-p0-r3-m3", "pilot-p1-r0-m1", "pilot-p1-r3-m0", "pilot-p2-r2-m0", "pilot-p3-r1-m0"], "fit_wall_span_seconds": 7.0, "fits_per_second_fit_span": 0.8571428571428571, "fits_per_second_including_excel": 0.5297933389993965, "max_scheduler_delay_seconds": 0.025523400167003274, "max_worker_rss_bytes": 194306048, "mean_cpu_percent": 32.28636363636364, "min_available_ram": 6130585600, "new_fits": 6, "peak_cpu_percent": 78.2, "peak_memory_percent": 63.7, "samples": 44, "seed_pair_id": 1, "skipped": false, "threads_per_worker": 1, "valid_fits": 6, "wall_seconds_including_excel": 11.32517070020549, "workers": 1}, {"configuration_ids": ["pilot-p0-r0-m0", "pilot-p0-r3-m3", "pilot-p1-r0-m1", "pilot-p1-r3-m0", "pilot-p2-r2-m0", "pilot-p3-r1-m0"], "fit_wall_span_seconds": 5.0, "fits_per_second_fit_span": 1.2, "fits_per_second_including_excel": 0.6219078095821499, "max_scheduler_delay_seconds": 0.02388480002991855, "max_worker_rss_bytes": 191328256, "mean_cpu_percent": 41.555263157894736, "min_available_ram": 6061834240, "new_fits": 6, "peak_cpu_percent": 61.9, "peak_memory_percent": 64.1, "samples": 38, "seed_pair_id": 2, "skipped": false, "threads_per_worker": 1, "valid_fits": 6, "wall_seconds_including_excel": 9.647732200101018, "workers": 2}, {"configuration_ids": ["pilot-p0-r0-m0", "pilot-p0-r3-m3", "pilot-p1-r0-m1", "pilot-p1-r3-m0", "pilot-p2-r2-m0", "pilot-p3-r1-m0"], "fit_wall_span_seconds": 4.0, "fits_per_second_fit_span": 1.5, "fits_per_second_including_excel": 0.5347285232299959, "max_scheduler_delay_seconds": 0.018396399915218353, "max_worker_rss_bytes": 189378560, "mean_cpu_percent": 44.42727272727273, "min_available_ram": 5894713344, "new_fits": 6, "peak_cpu_percent": 72.9, "peak_memory_percent": 65.1, "samples": 44, "seed_pair_id": 3, "skipped": false, "threads_per_worker": 1, "valid_fits": 6, "wall_seconds_including_excel": 11.220647000009194, "workers": 3}], "cpu_only": true, "frozen": "2026-11-08T02:26:15+00:00", "logical_processors": 8, "memory_margin_factor_per_worker": 2, "physical_cores": 4, "reason": "Highest measured fit throughput among safely measured settings, subject to live RAM reserve, twice per-worker RSS, and scheduler delay; no CPU utilization target forced. Matched configuration panels use different declared seeds so timing is descriptive.", "selected_workers": 3, "system_ram_reserve_bytes": 1014017802, "threads_per_worker": 1}.

All execution was CPU-only with one BLAS/OpenMP thread per worker and independent initialization/sampling seeds. Resource selection used measured throughput, CPU load, available RAM and responsiveness, never validation results. Invariant encoder states and internal-split preprocessing were reused; every budget used a fresh optimizer/model/sampler from the same original vector. The verified numerical engine was reused unchanged.

COBYLA optimizer-state restoration remains unsupported. An interrupted fit restarts from its original initialization and seeds under a new attempt ID. Technical failures alone receive at most one retry; all poor and budget-exhausted outcomes are retained. Receipts are verified and atomically saved before the sole coordinator updates Excel.

## Recommendation

Tested budget recommendation: 100. The interval for 200 minus 100 supports an incremental gain below the proposed 0.5-point margin. The 100-evaluation budget is the lower-cost working choice for this pilot.

Use another predeclared internal split as the more useful next robustness test before committing to a full-grid budget revision. This one conditional split does not establish a universally optimal budget. Larger untested budgets may still improve some configurations, and four-qubit findings do not determine budgets for larger widths. No full-grid, seven-qubit or larger-budget runs were launched.

Full four-qubit grid: retain the existing 100-evaluation evidence with its budget-limited labels. Treat the pilot recommendation as provisional planning evidence; confirm its practical balance on another internal split before scheduling a complete revised grid.

## Output and resume

Output: `D:\PHYSIQUE MATHEMATIQUE\doctora\4 EME ANNE\articles\breast canser\wdbc_vqc_budget_pilot_20261108`.

```powershell
& 'D:\PHYSIQUE MATHEMATIQUE\doctora\4 EME ANNE\articles\breast canser\wdbc_vqc_budget_pilot_20261108\resume.ps1'
```

The launcher performs one startup checkpoint inventory and skips verified fits. Tables and receipts are authoritative; Excel can be rebuilt without repeating completed fits.

# Q9 four-qubit hardware execution — 2026-11-11

Status: COMPLETE_VERIFIED.

IBM Fez, physical qubits [114, 23, 58, 121]. 48 training / 24 held-out cases. ZFeatureMap reps=1, 256 shots, training-only MinMax and LR-RFE(k=4), C=1 precomputed SVC. Native prediction; malignant source label 0. No entangling gates, mitigation, PSD repair, threshold tuning or cohort changes.

|Method|Correct /24|Accuracy|Malignant probability AUC|
|---|---:|---:|---:|
|Ideal exact|23/24|95.8333%|1.00000000|
|Ideal 256 shots|24/24|100.0000%|1.00000000|
|IBM Fez 256 shots|23/24|95.8333%|1.00000000|

All 2,280 pairs / 583,680 returned shots verified across 10 jobs. Actual service-reported QPU usage: 180.000s; account remaining after execution: 420.000s. Initial transient zero usage for the pilot settled to 4 seconds; original report and reconciliation are both preserved. Subsequent jobs wait for positive usage metadata; unresolved usage blocks new submissions.

## Jobs

|Chunk|IBM ID|Pairs|Actual QPU seconds|Allowed seconds|
|---|---|---:|---:|---:|
|job-000|dbpr1t5t7fqc7389sasg|32|4.0|9.0|
|job-001|dbpr37s4148s739v2dgg|256|20.0|30.0|
|job-002|dbpr3nlt7fqc7389sdf0|256|20.0|30.0|
|job-003|dbpr3ugqtr0c738es20g|256|20.0|30.0|
|job-004|dbpr45oqtr0c738es290|256|20.0|30.0|
|job-005|dbpr4d0ou4as73cu42p0|256|20.0|30.0|
|job-006|dbpr540qtr0c738es3e0|256|20.0|30.0|
|job-007|dbpr5b8qtr0c738es3o0|256|20.0|30.0|
|job-008|dbpr5ik4148s739v2g9g|256|20.0|30.0|
|job-009|dbpr5poqtr0c738es47g|200|16.0|25.0|

This is a new small fixed-cohort hardware feasibility check of a historically selected configuration, not independent validation. Accuracy changes in 4.1667 percentage-point increments on 24 cases. Wilson intervals and complete clinical metrics are saved in metrics.json. Local finite-shot scores are a single simulation realization. Hardware differences reflect hardware noise and sampling as well as their interaction with the fitted kernel model.

Raw counts and job metadata: hardware/jobs/. Predictions, continuous malignant scores, kernels, saved SVC, clinical metrics, spectrum and verification are in hardware/. The ledger retains all job IDs, per-job allowances, service usage, forecast and timing evidence. Scientific settings and the original notebook were preserved. No commit or push was performed.

The train diagonal is fixed to1 and off-diagonal symmetry is reused, as frozen before execution. Finite-shot/hardware negative kernel eigenvalues, if present, are retained and reported in kernel_spectrum.json. Calibration and a backend-layout/circuit identity are frozen in the plan and preflight.

The saved raw hardware training kernel has 20 eigenvalues below -1e-10; minimum eigenvalue -0.266371862602. No PSD projection was applied, as required by the frozen protocol.

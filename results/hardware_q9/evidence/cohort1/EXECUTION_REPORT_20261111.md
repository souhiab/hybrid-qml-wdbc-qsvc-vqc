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

## Local SVC and spectrum audit (2026-11-11)

All saved SVCs have fit_status_=0. Exact / finite-shot / hardware n_iter_ values are17 /24 /25 respectively. These are binary decision-fit iteration counts, not the internal probability-calibration fit counts. Original structured warnings were not captured; none was found in the original hardware console. All36 explicit local diagnostic refits (identity plus11 permutations per kernel) captured zero convergence warnings and returned fit_status_=0.

|Kernel|Minimum eigenvalue|Negative count below -1e-10|Total negative magnitude|Full absolute spectral magnitude|Negative fraction|
|---|---:|---:|---:|---:|---:|
|Exact|2.82880581819e-09|0|0.000000000|48.000000000|0.000000%|
|Ideal256shots|-0.258079623234|19|2.519208249|53.038416499|4.749780%|
|Hardware256shots|-0.266371862602|20|2.338091791|52.676183582|4.438613%|

Negative fraction is sum(abs(negative eigenvalues below -1e-10)) divided by sum(abs(all eigenvalues)). No universal acceptable threshold is claimed. The hardware negative mass is slightly lower than this single ideal finite-shot realization despite its larger negative-eigenvalue count; this does not establish that hardware is less noisy.

Reverse order and10 prespecified random permutations (seed20261111) per kernel used K[P,P], y[P], K_test[:,P]. No training or held-out native class predictions changed. Maximum held-out decision-margin changes were local_exact: 0.000509832995, local_shots: 0.12663883, hardware: 0.000802127252. The finite-shot kernel therefore showed greater margin sensitivity in this panel even though its class predictions were unchanged. Hardware maximum malignant-probability change was 0.0318581072. Row ordering also changes internal probability-calibration folds, so probability differences do not isolate the decision solver. Identity replays reproduced saved outputs. These exploratory checks do not prove a unique/global optimum or general solver stability. No original production fit or score was replaced; no QPU time was used.

Full audit: ../solver_spectrum_audit_20261111/SOLVER_SPECTRUM_REPORT.md. Eigenvalues, per-fit warnings, exact permutations, scores and prediction arrays are saved alongside it.

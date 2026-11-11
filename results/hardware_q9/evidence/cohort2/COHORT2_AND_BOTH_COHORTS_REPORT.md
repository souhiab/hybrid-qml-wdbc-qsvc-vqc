# Q9 disjoint-cohort internal robustness check — 2026-11-11

Status: COMPLETE_VERIFIED. Further hardware experimentation stopped.

Cohort2 was frozen before fitting:72 WDBC rows sampled from the497 remaining after excluding all72 from cohort1. Stratified48training /24held-out split, selection and split seeds20261111. Raw-data hashes and ordered source-row IDs are saved. No overlap between cohorts. IDs are original sklearn row ordinals, not clinical identifiers.

Same four-qubit ZFeatureMap(reps1,alpha2), no entanglement argument or extra input rescaling,256shots, C1 precomputed SVC, same frozen SVC/calibration settings, native prediction, malignant source label0. MinMax[0,1],clip=False and LR(l1/saga,seed42)-RFE4 were refitted only on the new48training cases. No mitigation, PSD repair, normalization, threshold tuning, cohort reselection or performance-dependent adaptation. Training diagonal fixed1; training symmetry reused.

## Both cohorts reported separately

|Cohort|Method|Correct /24|Accuracy|Malignant AUC|Sensitivity|Specificity|
|---|---|---:|---:|---:|---:|---:|
|cohort1|local_exact|23/24|95.8333%|1.00000000|88.8889%|100.0000%|
|cohort1|local_shots|24/24|100.0000%|1.00000000|100.0000%|100.0000%|
|cohort1|hardware|23/24|95.8333%|1.00000000|88.8889%|100.0000%|
|cohort2|local_exact|20/24|83.3333%|0.94814815|66.6667%|93.3333%|
|cohort2|local_shots|19/24|79.1667%|0.94074074|66.6667%|86.6667%|
|cohort2|hardware|21/24|87.5000%|0.95555556|77.7778%|93.3333%|

Cohort2 selected features: mean concave points, worst concave points, worst symmetry, worst fractal dimension.

Cohort2:10valid jobs,2280pairs,583680returnedshots. Settled QPU usage: 180.000s. Cohort1 usage: 180.000s. Cumulative: 360.000s of600s; final account allowance 240.000s.120s reserve preserved. All prior frozen evidence (189files, including the first solver/spectrum audit) remained unchanged.

## Job records

|Chunk|IBM job ID|Pairs|Settled QPU seconds|Allowance|
|---|---|---:|---:|---:|
|job-000|dbpsk1oqtr0c738eu140|32|4.0|9.0|
|job-001|dbq3c4dt7fqc738a9kag|256|20.0|30.0|
|job-002|dbq3d7tt7fqc738a9mjg|256|20.0|30.0|
|job-003|dbq3deoou4as73cuh5s0|256|20.0|30.0|
|job-004|dbq3dloou4as73cuh68g|256|20.0|30.0|
|job-005|dbq3du5t7fqc738a9nq0|256|20.0|30.0|
|job-006|dbq3e58qtr0c738f9ch0|256|20.0|30.0|
|job-007|dbq3edoou4as73cuh8i0|256|20.0|30.0|
|job-008|dbq3emdt7fqc738a9q7g|256|20.0|30.0|
|job-009|dbq3euk4148s739vfplg|200|16.0|25.0|

## Interpretation

The second cohort is an additional internal robustness check, not an independent external validation. The configuration was historically selected using WDBC. Both results are retained regardless of performance. Different cohorts have different training samples and fitted preprocessing/models; results are not pooled into one fitted-model test score. Each24-case test accuracy changes by4.1667percentage points per patient. Repeated hardware measurements or good scores do not remove historical selection bias.

All raw counts and job IDs:hardware/jobs/. Kernels, predictions, malignant scores, metrics, SVC and fit diagnostics:local_exact/,local_shots/,hardware/. Protocol, hashes, exclusion IDs, split records, preprocessing evidence, environment and execution source are saved locally. No manuscript edit, commit or push.

No third cohort, larger dataset, additional shot stage or other QPU experiment is scheduled or authorized by this controller.

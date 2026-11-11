# Corrections and limitations

The original workflow fitted learned preprocessing before separating evaluation cases. The corrected studies fit scaling and supervised RFE using training rows and training labels only. Shared-fold tuning refits preprocessing inside inner folds. Historical notebook outputs are not corrected evidence.

Malignant is source label 0. Saved clinical metrics use malignant scores and malignant-positive confusion counts. Archived aggregates with an unverified orientation must not be compared as if their definitions agree.

Configuration selection used this same WDBC dataset. Corrected fixed-split scores and shared-fold evaluations are internal exploratory results, not independent validation or proof of corrected maxima across untested grids. Pauli/ZZ aliases are collapsed only where exact circuit equivalence was verified. Z receives no entanglement argument.

VQC budget exhaustion is a valid budget-limited outcome, not convergence. Seven-qubit pilot convergence at 300 was 41/48; eleven-qubit pilot convergence at 300 was 24/48. The selected main-grid caps are study decisions with exceptions to the pilot termination target. Full COBYLA and sampler-state restoration was unsupported: interrupted fits restart from the original initialization and seeds, with attempts retained.

Cache identity/integrity and Windows locking problems required technical recovery. Receipts, rather than transient status text or Excel backups, establish completed fits. Recovery must preserve verified fits. A single coordinator writes workbooks. The shared-fold canonical workbook is WDBC_QSVC_Classical_Shared_Folds_Master.xlsx; its recovery copy is not authoritative.

The optional native accelerator has separate numerical equivalence evidence. Its benchmark does not establish that every archived study used that engine. Preserve complex128 arithmetic, shots, circuit ordering and sampler RNG semantics when changing performance settings.

IBM job state and wall time are distinct from charged QPU usage. A transient zero usage after completion was subsequently settled; unresolved usage blocks further submission. The two hardware cohorts consumed 360 seconds in total. No further hardware experiment is part of this release.

Finite-shot and hardware kernels retain negative eigenvalues: no PSD projection, mitigation or threshold tuning was applied. Solver fit_status_=0 establishes successful reported termination, not convexity, uniqueness or generalization. First-cohort original structured warnings were not captured; later replay/permutation checks cannot establish their historical absence. Calibration snapshots do not provide per-job pulse traces.

The 17-qubit 600-cap stage was incomplete at this publication snapshot and is excluded from completed-study results. No final 17-qubit budget recommendation is asserted. Failed/interrupted attempts are distinct from retained budget-exhausted results.

No authoritative manuscript was supplied for this repository update. Manuscript PDFs, excerpts and manuscript integration drafts are excluded.

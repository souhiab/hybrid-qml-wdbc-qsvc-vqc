# Reproducing and verifying the evidence

Start with `python tools/verify_results.py`. This reads saved predictions, recomputes accuracy, malignant AUC and confusion counts, verifies coverage and archive hashes, and never trains or contacts IBM. Install NumPy in a separate environment if needed.

`results/STUDIES.json` is the completed-study index. Each simulator study has normalized metrics and patient-level predictions. Its evidence.zip retains original per-fit records, training histories, circuits and parameters where available. Large reproducible caches and classical binary models are omitted; original receipts may reference those locally retained files. Hardware evidence includes raw counts, job IDs, kernels and fitted SVCs. Only load serialized models from trusted sources, using the recorded environment.

`implementations/` contains the original validated study execution snapshots, manifests, split records and environment provenance. These snapshots preserve historical settings and may contain original machine paths and source locks. They are not a promise of one-command execution after relocation. Use a new output directory, recover dependencies from each study's recorded environment, and resolve paths before any new scientific execution. Do not overwrite published evidence or run controllers against existing local experiments.

The simulator environment used Qiskit 1.4.2 and Qiskit Machine Learning 0.8.4. Hardware preparation used Qiskit 2.2.1, Runtime 0.42.0, scikit-learn 1.6.1 and Python 3.11. Keep these environments separate; exact package records are included with studies. Do not upgrade a running experiment's environment.

`qsvm_on_wdbc/budgeted_qpu/` contains the current guarded hardware implementation. Importing it does not submit jobs. Credentials must stay outside this repository. Existing two-cohort results are final for this hardware exercise; this update authorizes no additional submissions. Legacy notebooks remain at their original paths for historical inspection and are not the default reproduction entrypoint.

Clinical positive class is malignant (source label 0). Native class decisions are preserved; probabilities are not substituted for native SVC decisions. VQC retains all seed outcomes, including budget exhaustion. Repeated folds and repeated predictions of the same patient must not be treated as independent patients.

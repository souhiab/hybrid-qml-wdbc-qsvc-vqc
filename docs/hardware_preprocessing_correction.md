# Hardware notebook preprocessing correction

`qsvm_on_wdbc/REAL_QPU.ipynb` now keeps raw features until each outer
training/held-out partition is formed. The correction covers its fixed-feature,
manual-feature and learned RFE CV variants, including the real-QPU paths.
The existing fixed-split cells already fitted scaling and RFE on training data
and are preserved.

## Corrected access boundaries

- StandardScaler fits only the relevant training rows.
- Learned RFE fits scaled training rows and training labels only. Feature names
  may consequently differ across outer and inner folds.
- Inner C selection refits scaling/RFE separately for every inner training fold
  and constructs corresponding training/validation kernels. It no longer slices
  an outer-preprocessed kernel for inner validation.
- The first C in the original candidate order wins an exact inner-score tie.
- Cross-kernel centering reads the uncentered training-kernel statistics. The
  previous outer code passed an already-centered matrix, losing those means.

The C grids, class weights, threshold rules, original label/score orientation,
stratified subset selection, fold seeds, feature-map settings, physical-layout
selection, shot/adaptive-shot rules, mitigation, sampler and job functions remain
unchanged. This notebook retains its historical benign-positive convention;
these edits do not relabel archived metrics as malignant-positive.

## Evidence and recovery

Each corrected variant writes to a separate `_train_only_cell<number>` output
directory. Historical matrices, checkpoints and results are not valid inputs
for the corrected preprocessing and remain untouched. Outer/inner preprocessing
audit files record training identities, transformed-array hashes and selected
features; a changed identity in the same namespace raises an error before its
kernel is reused. Inner kernels have distinct negative fold IDs in the existing
checkpoint/log system, and are reused only across C values in that inner fold.

Correct nested evaluation requires additional kernel computations and therefore
can increase the cost of a future hardware run. No new hardware results are
provided by this code correction. Archived scores retain their original
preprocessing provenance.

## Remaining interpretation limits

Fixed/manual feature lists retain any historical selection on this dataset;
fold-local scaling cannot undo that selection history. The optional farthest-
point cohort construction, exploratory shot-sweep stopping rule and historical
resume-summary aggregation behavior are preserved. Their scientific interpretation
requires separate review; this patch does not turn historical exploration into
independent validation or repair previously saved aggregate results.

## Validation

Run `python scripts/test_hardware_preprocessing.py`. Tests use synthetic data
and mock kernel builders. They verify evaluation-outlier isolation, exact
inner-training scaler/RFE access, kernel reuse scopes, preprocessing-identity
rejection, the C tie rule, all seven nested adapters and centering calculations.
No notebook experiment cell, WDBC training fit or quantum hardware job is run.

"""Partition-local preprocessing for the hardware and matching simulator notebooks.

Importing this module performs no data loading, training or hardware submission.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def training_only_output_dir(original, variant):
    """Keep corrected kernels/checkpoints separate from historical evidence."""
    suffix = "_train_only_" + variant
    value = str(original)
    return value if value.endswith(suffix) else value + suffix


def _digest(value):
    value = np.ascontiguousarray(value)
    description = json.dumps([value.shape, value.dtype.str]).encode()
    return hashlib.sha256(description + value.tobytes()).hexdigest()


def prepare_partition(X_train, X_evaluation, y_train, *,
                      n_features_to_select=None, estimator_params=None,
                      feature_names=None, audit_path=None):
    """Fit only training features/labels; transform evaluation features afterward.

    Fixed manual/historically selected columns are passed as raw input. Their
    selection history is not erased by fitting the scaler within each partition.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_evaluation = np.asarray(X_evaluation, dtype=float)
    y_train = np.asarray(y_train)
    if X_train.ndim != 2 or X_evaluation.ndim != 2:
        raise ValueError("Expected two raw feature matrices")
    if X_train.shape[1] != X_evaluation.shape[1] or len(y_train) != len(X_train):
        raise ValueError("Training/evaluation dimensions or training labels differ")
    scaler = StandardScaler()
    scaled_training = scaler.fit_transform(X_train)
    selector = None
    if n_features_to_select is not None:
        estimator = LogisticRegression(**dict(estimator_params or {}))
        selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
        training = selector.fit_transform(scaled_training, y_train)
        selected = np.flatnonzero(selector.support_)
    else:
        training = scaled_training
        selected = np.arange(X_train.shape[1])
    # No evaluation rows or labels are consulted by any fit above.
    evaluation = scaler.transform(X_evaluation)
    if selector is not None:
        evaluation = selector.transform(evaluation)
    names = list(feature_names) if feature_names is not None else list(range(X_train.shape[1]))
    audit = {
        "protocol": "partition-local-preprocessing-v1",
        "training_rows": len(X_train), "evaluation_rows": len(X_evaluation),
        "raw_training_sha256": _digest(X_train),
        "training_labels_sha256": _digest(y_train),
        "transformed_training_sha256": _digest(training),
        "transformed_evaluation_sha256": _digest(evaluation),
        "scaler_parameters": scaler.get_params(),
        "rfe_estimator_parameters": estimator_params,
        "n_features_to_select": n_features_to_select,
        "selected_indices": selected.tolist(),
        "selected_names": [names[i] for i in selected],
        "evaluation_rows_in_fit": False, "evaluation_labels_in_fit": False,
    }
    if audit_path is not None:
        destination = Path(audit_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(audit, sort_keys=True, indent=2)
        if destination.exists() and json.loads(destination.read_text()) != json.loads(encoded):
            raise ValueError("Preprocessing identity changed; use a new output directory")
        destination.write_text(encoded, encoding="utf-8")
    return training, evaluation, audit


def nested_select_C(X_outer_training, y_outer_training, Cs, *, seed,
                    preprocessing, kernel_builder, class_weight=None,
                    audit_dir=None, scope=1):
    """Refit preprocessing and build new kernels inside every inner split.

    kernel_builder receives transformed inner-training/evaluation features and
    a distinct negative scope ID. Never slice an outer-preprocessed kernel for
    inner validation. Each inner kernel pair is reused only across candidate C.
    """
    X = np.asarray(X_outer_training)
    y = np.asarray(y_outer_training)
    splits = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    panel = []
    for fold, (train, validation) in enumerate(splits.split(X, y), 1):
        identity = -(int(scope) * 1000 + fold)
        path = Path(audit_dir) / f"preprocessing_inner_{identity}.json" if audit_dir else None
        train_values, validation_values, _ = prepare_partition(
            X[train], X[validation], y[train], audit_path=path, **preprocessing)
        K_train, K_validation = kernel_builder(train_values, validation_values, identity)
        K_train = np.asarray(K_train)
        K_validation = np.asarray(K_validation)
        if K_train.shape != (len(train), len(train)):
            raise ValueError("Inner training kernel shape mismatch")
        if K_validation.shape != (len(validation), len(train)):
            raise ValueError("Inner validation kernel shape mismatch")
        panel.append((K_train, K_validation, y[train], y[validation]))
    best_score, best_C = -np.inf, Cs[0]
    for C in Cs:
        scores = []
        for K_train, K_validation, labels_train, labels_validation in panel:
            classifier = SVC(kernel="precomputed", C=C, class_weight=class_weight)
            classifier.fit(K_train, labels_train)
            scores.append(roc_auc_score(labels_validation,
                                        classifier.decision_function(K_validation)))
        mean = float(np.mean(scores))
        if mean > best_score:  # Preserve the first-C rule for exact ties.
            best_score, best_C = mean, C
    return best_C, best_score

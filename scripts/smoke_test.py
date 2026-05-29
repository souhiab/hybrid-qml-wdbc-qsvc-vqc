"""Lightweight smoke test for the hybrid QML WDBC pipeline.

Verifies that:
  1. the WDBC dataset loads from scikit-learn,
  2. preprocessing (scaling) and a fixed, stratified train/test split run,
  3. RFE feature selection produces a subset,
  4. a small classical baseline (Logistic Regression) trains and scores,
  5. (optional) a minimal QSVC sanity check runs if the quantum stack is
     installed.

Runs in well under a minute and needs no quantum hardware.

Usage:
    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys

RANDOM_STATE = 42


def main() -> int:
    # 1. Dataset ------------------------------------------------------------
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X, y = data.data, data.target
    assert X.shape == (569, 30), f"unexpected WDBC shape: {X.shape}"
    print(f"[OK] Loaded WDBC: X={X.shape}, classes={sorted(set(y))}")

    # 2. Split + scaling ----------------------------------------------------
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)
    print(f"[OK] Split + scaled: train={X_tr.shape[0]}, test={X_te.shape[0]}")

    # 3. RFE feature selection (k = 7) -------------------------------------
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression

    k = 7
    rfe = RFE(
        LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        n_features_to_select=k,
    ).fit(X_tr_s, y_tr)
    sel = [name for name, keep in zip(data.feature_names, rfe.support_) if keep]
    assert len(sel) == k
    print(f"[OK] RFE selected {k} features: {sel}")

    # 4. Classical baseline -------------------------------------------------
    from sklearn.metrics import accuracy_score, roc_auc_score

    clf = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    clf.fit(X_tr_s[:, rfe.support_], y_tr)
    pred = clf.predict(X_te_s[:, rfe.support_])
    proba = clf.predict_proba(X_te_s[:, rfe.support_])[:, 1]
    acc = accuracy_score(y_te, pred)
    auc = roc_auc_score(y_te, proba)
    print(f"[OK] Classical LR baseline (k={k}): acc={acc:.4f}, auc={auc:.4f}")
    assert acc > 0.9, f"baseline accuracy unexpectedly low: {acc:.4f}"

    # 5. Optional minimal QSVC ---------------------------------------------
    try:
        from qiskit.circuit.library import ZZFeatureMap
        from qiskit_machine_learning.kernels import FidelityStatevectorKernel
        from qiskit_machine_learning.algorithms.classifiers import QSVC
    except Exception as exc:  # noqa: BLE001
        print(f"[SKIP] Quantum stack not installed ({exc.__class__.__name__}); "
              "skipping QSVC check.")
    else:
        kq = 4
        idx = list(range(kq))
        # tiny subset to keep the statevector kernel fast
        n = 40
        fmap = ZZFeatureMap(feature_dimension=kq, reps=1, entanglement="linear")
        qk = FidelityStatevectorKernel(feature_map=fmap)
        qsvc = QSVC(quantum_kernel=qk)
        qsvc.fit(X_tr_s[:n, idx], y_tr[:n])
        q_acc = accuracy_score(y_te[:n], qsvc.predict(X_te_s[:n, idx]))
        print(f"[OK] Minimal QSVC sanity check: acc={q_acc:.4f} on {n} samples")

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

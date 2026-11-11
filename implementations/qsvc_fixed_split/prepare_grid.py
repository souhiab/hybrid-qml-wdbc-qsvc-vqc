from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from threadpoolctl import threadpool_info


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
REPO = PROJECT / "implementations"
PRIOR_FIG3 = PROJECT / "wdbc_qsvc_figure3_train_only_20261107"
PRIOR_MAIN = PROJECT / "wdbc_qsvc_leakage_check_20261106" / "manuscript_correction_20261106"
SOURCE = ROOT / "source"
RECORDS = ROOT / "records"
PREP = ROOT / "preprocessing"
IMPORTED = ROOT / "imported_existing"
MANIFEST = ROOT / "execution_manifest.csv"
CALIBRATION_SEED = 20261107
EXPECTED_WIDTHS = (4, 7)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    os.replace(temp, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def run_git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, text=True, capture_output=True, check=True
    ).stdout.strip()


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def package_versions() -> dict[str, str]:
    names = [
        "numpy", "pandas", "scikit-learn", "scipy", "qiskit",
        "qiskit-machine-learning", "psutil", "threadpoolctl",
    ]
    return {name: importlib.metadata.version(name) for name in names}


def clinical_metrics(y_source: np.ndarray, pred_source: np.ndarray, malignant_score: np.ndarray) -> dict:
    y_m = (np.asarray(y_source) == 0).astype(int)
    p_m = (np.asarray(pred_source) == 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_m, p_m, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_m, p_m)),
        "roc_auc": float(roc_auc_score(y_m, malignant_score)),
        "sensitivity": float(recall_score(y_m, p_m)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(precision_score(y_m, p_m, zero_division=0)),
        "f1": float(f1_score(y_m, p_m, zero_division=0)),
        "TP_malignant": int(tp),
        "FN_malignant": int(fn),
        "FP_malignant": int(fp),
        "TN_benign": int(tn),
        "correct_count": int(tp + tn),
        "n": int(len(y_m)),
    }


def stable_unique(values) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def effective_key(row: pd.Series) -> tuple:
    width = int(row["n_features"])
    scaling = str(row["Scaling"])
    estimator = str(row["Estimator"])
    reps = int(row["Reps"])
    fmap = str(row["Feature_Map"])
    if fmap == "ZFeatureMap":
        return width, scaling, estimator, "Z", "", reps
    return width, scaling, estimator, "Pauli_ZZ", str(row["Entanglement"]), reps


def config_id(width: int, scaling: str, estimator: str, family: str, topology: str, reps: int) -> str:
    scaler_token = "standard" if scaling == "Standardization" else "minmax"
    estimator_token = "lr" if estimator == "Logistic Regression" else "rf"
    encoder = "z" if family == "Z" else f"zz-{topology}"
    return f"fs-k{width:02d}-{scaler_token}-{estimator_token}-{encoder}-r{reps}"


def selector(estimator: str):
    if estimator == "Logistic Regression":
        return LogisticRegression(penalty="l1", solver="saga", max_iter=5000, random_state=42)
    if estimator == "Random Forest":
        return RandomForestClassifier(random_state=42)
    raise ValueError(estimator)


def preprocessing_id(width: int, scaling: str, estimator: str) -> str:
    return config_id(width, scaling, estimator, "Z", "", 1).rsplit("-z-r1", 1)[0]


def archived_feature_names(classical: pd.DataFrame, width: int, scaling: str, estimator: str) -> list[str]:
    selected = classical[
        (classical["n_features"].astype(int) == width)
        & (classical["Scaling"] == scaling)
        & (classical["Estimator"] == estimator)
    ]
    if len(selected) != 1:
        raise RuntimeError(f"Archived feature row not unique: {width}/{scaling}/{estimator}")
    return list(ast.literal_eval(selected.iloc[0]["Selected Features"]))


def source_run_path(row_id: int, width: int) -> Path:
    fig3_ids = {145, 152, 195, 226, 253, 260, 361, 372, 373, 433, 513, 517}
    table2 = {157: 4, 288: 7, 392: 11, 470: 17}
    if row_id in fig3_ids:
        return PRIOR_FIG3 / "runs" / f"row{row_id:04d}" / f"k{width:02d}"
    if row_id in table2:
        return PRIOR_MAIN / "runs" / "table2" / f"k{width:02d}"
    if row_id == 519:
        return PRIOR_MAIN / "runs" / "section4_1_auc" / "k17"
    raise KeyError(row_id)


def snapshot_sources() -> dict:
    copy_map = {
        PRIOR_FIG3 / "source" / "data" / "wdbc_arrays.npz": SOURCE / "data" / "wdbc_arrays.npz",
        PRIOR_FIG3 / "source" / "data" / "wdbc_sklearn.csv": SOURCE / "data" / "wdbc_sklearn.csv",
        PRIOR_FIG3 / "source" / "historical" / "split_idx.json": SOURCE / "historical" / "split_idx.json",
        PRIOR_FIG3 / "source" / "historical" / "quantum_each_results.csv": SOURCE / "historical" / "quantum_each_results.csv",
        PRIOR_FIG3 / "source" / "historical" / "classical_rfe_fixed_counts.csv": SOURCE / "historical" / "classical_rfe_fixed_counts.csv",
        PRIOR_FIG3 / "source" / "figure3_evidence" / "inputs" / "ARCHIVED_552_CONFIGURATION_ROWS.csv": SOURCE / "historical" / "ARCHIVED_552_CONFIGURATION_ROWS.csv",
        PRIOR_FIG3 / "source" / "figure3_evidence" / "inputs" / "historical_548.csv": SOURCE / "historical" / "historical_548.csv",
        PRIOR_FIG3 / "source" / "runner" / "run_leakage_check.py": SOURCE / "historical_code" / "run_leakage_check.py",
        PRIOR_FIG3 / "source" / "environment" / "quantum_adapter_environment_lock.txt": SOURCE / "environment" / "quantum_adapter_environment_lock.txt",
        PRIOR_FIG3 / "records" / "library_source_equivalence.json": SOURCE / "environment" / "library_source_equivalence.json",
        REPO / "quantum_no_scaling_qsvc_results.csv": SOURCE / "historical" / "quantum_no_scaling_qsvc_results.csv",
    }
    repeated = PROJECT / "PAPAER 1" / "03_CONFIRMATORY_EXPERIMENTS" / "03_QSVC" / "qv3_confirmatory_results" / "repeated_oof_summary.json"
    copy_map[repeated] = SOURCE / "repeated_cv" / "repeated_oof_summary.json"
    for source, destination in copy_map.items():
        if not source.exists():
            raise FileNotFoundError(source)
        copy_file(source, destination)
    rows = []
    for source, destination in copy_map.items():
        rows.append({
            "source_path": str(source),
            "isolated_path": str(destination.relative_to(ROOT)),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        })
    write_csv(RECORDS / "source_inventory.csv", pd.DataFrame(rows))
    return {"files": len(rows), "inventory": "records/source_inventory.csv"}


def repository_and_machine_record() -> None:
    status = run_git("status", "--porcelain=v1", "--untracked-files=all")
    remote = run_git("remote", "get-url", "origin")
    commit = run_git("rev-parse", "HEAD")
    repo_record = {
        "recorded_at": now_iso(),
        "repository": str(REPO),
        "remote_url": remote,
        "commit": commit,
        "status_porcelain": status.splitlines(),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }
    write_json(RECORDS / "repository_start_state.json", repo_record)
    vm = psutil.virtual_memory()
    machine = {
        "recorded_at": now_iso(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python": sys.version,
        "physical_cpu_cores": psutil.cpu_count(logical=False),
        "logical_cpu_cores": psutil.cpu_count(logical=True),
        "ram_total_bytes": vm.total,
        "ram_available_bytes": vm.available,
        "packages": package_versions(),
        "threadpools": threadpool_info(),
    }
    write_json(RECORDS / "machine_preflight.json", machine)


def verify_dataset_and_split() -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    with np.load(SOURCE / "data" / "wdbc_arrays.npz", allow_pickle=False) as data:
        X = np.asarray(data["X"], dtype=np.float64)
        y = np.asarray(data["y"], dtype=np.int64)
        feature_names = data["feature_names"].astype(str).tolist()
    split = json.loads((SOURCE / "historical" / "split_idx.json").read_text(encoding="utf-8"))
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    test_idx = np.asarray(split["test_idx"], dtype=np.int64)
    generated_train, generated_test = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )
    checks = {
        "verified_at": now_iso(),
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "train_count": int(len(train_idx)),
        "test_count": int(len(test_idx)),
        "ordered_train_matches_seed42_stratified_split": bool(np.array_equal(train_idx, generated_train)),
        "ordered_test_matches_seed42_stratified_split": bool(np.array_equal(test_idx, generated_test)),
        "test_unique_count": int(len(np.unique(test_idx))),
        "test_malignant_label0": int(np.sum(y[test_idx] == 0)),
        "test_benign_label1": int(np.sum(y[test_idx] == 1)),
        "train_index_sha256": array_hash(train_idx),
        "test_index_sha256": array_hash(test_idx),
        "train_labels_sha256": array_hash(y[train_idx]),
        "test_labels_sha256": array_hash(y[test_idx]),
        "positive_class": "malignant",
        "positive_source_label": 0,
    }
    required = [
        checks["train_count"] == 455,
        checks["test_count"] == 114,
        checks["test_unique_count"] == 114,
        checks["test_malignant_label0"] == 42,
        checks["test_benign_label1"] == 72,
        checks["ordered_train_matches_seed42_stratified_split"],
        checks["ordered_test_matches_seed42_stratified_split"],
    ]
    checks["passed"] = bool(all(required))
    write_json(RECORDS / "split_verification.json", checks)
    if not checks["passed"]:
        raise RuntimeError(f"Split verification failed: {checks}")
    return X, y, feature_names, train_idx, test_idx


def archive_groups() -> tuple[pd.DataFrame, dict[tuple, dict]]:
    archive = pd.read_csv(SOURCE / "historical" / "quantum_each_results.csv")
    mapping = pd.read_csv(SOURCE / "historical" / "ARCHIVED_552_CONFIGURATION_ROWS.csv")
    if len(archive) != 552 or len(mapping) != 552:
        raise RuntimeError("Expected the 552-row archive and mapping")
    archive.insert(0, "source_row_id", np.arange(1, len(archive) + 1))
    normalized = mapping.rename(columns={
        "archive_row_1_based": "source_row_id",
        "scaling": "Scaling",
        "estimator": "Estimator",
        "feature_map": "Feature_Map",
        "topology_label": "Entanglement",
        "repetitions": "Reps",
    })
    joined = archive.merge(
        normalized[["source_row_id", "identity_group", "canonical_gate_sha256"]],
        on="source_row_id", how="left", validate="one_to_one",
    )
    groups: dict[tuple, dict] = {}
    for key, group in joined.groupby(joined.apply(effective_key, axis=1), sort=False):
        group = group.sort_values("source_row_id", kind="stable")
        accuracies = group["QSVC_Test_Acc"].astype(float)
        f1s = group["QSVC_F1"].astype(float)
        aucs = group["QSVC_AUC"].astype(float)
        counts = group["QSVC_TP"].astype(int) + group["QSVC_TN"].astype(int)
        if accuracies.nunique() != 1 or f1s.nunique() != 1 or counts.nunique() != 1:
            raise RuntimeError(f"Alias conflict in discrete metrics for {key}")
        representative = group.iloc[0]
        groups[key] = {
            "source_row_ids": [int(value) for value in group["source_row_id"]],
            "nominal_feature_map_labels": stable_unique(group["Feature_Map"]),
            "nominal_topology_labels": stable_unique(group["Entanglement"]),
            "identity_groups": stable_unique(group["identity_group"]),
            "canonical_gate_sha256": stable_unique(group["canonical_gate_sha256"]),
            "representative_source_row_id": int(representative["source_row_id"]),
            "archived_accuracy_reported": float(representative["QSVC_Test_Acc"]),
            "archived_correct_count": int(counts.iloc[0]),
            "archived_auc_representative": float(representative["QSVC_AUC"]),
            "archived_auc_min": float(aucs.min()),
            "archived_auc_max": float(aucs.max()),
            "archived_auc_alias_conflict": bool(aucs.nunique() != 1),
            "archived_auc_values_by_source_id": json.dumps(
                {str(int(r.source_row_id)): float(r.QSVC_AUC) for r in group.itertuples()}, sort_keys=True
            ),
            "archived_f1_reported": float(representative["QSVC_F1"]),
        }
    audit = {
        "verified_at": now_iso(),
        "archive_rows": len(joined),
        "width_row_counts": joined.groupby("n_features").size().astype(int).to_dict(),
        "effective_group_counts": {
            str(k): sum(1 for key in groups if key[0] == k) for k in sorted(joined["n_features"].unique())
        },
        "alias_groups_with_auc_conflict": sum(value["archived_auc_alias_conflict"] for value in groups.values()),
        "alias_groups_with_accuracy_conflict": 0,
        "alias_definition": "ZFeatureMap topology labels collapse; PauliFeatureMap and ZZFeatureMap collapse only with the same topology, reps, width, scaler and RFE estimator.",
    }
    write_json(RECORDS / "archive_alias_audit.json", audit)
    return joined, groups


def build_manifest(groups: dict[tuple, dict]) -> pd.DataFrame:
    reuse_ids = {4: {157, 152, 145, 226}, 7: {195, 260, 253, 288}}
    rows = []
    for width in EXPECTED_WIDTHS:
        for scaling in ("Standardization", "Normalization"):
            for estimator in ("Logistic Regression", "Random Forest"):
                encoders = [("Z", "", reps) for reps in range(1, 5)]
                encoders += [
                    ("Pauli_ZZ", topology, reps)
                    for topology in ("linear", "full", "circular")
                    for reps in range(1, 5)
                ]
                for family, topology, reps in encoders:
                    key = (width, scaling, estimator, family, topology, reps)
                    if key not in groups:
                        raise RuntimeError(f"Missing archived effective group {key}")
                    evidence = groups[key]
                    matched_reuse = sorted(reuse_ids[width].intersection(evidence["source_row_ids"]))
                    if len(matched_reuse) > 1:
                        raise RuntimeError(f"Multiple reusable rows for {key}: {matched_reuse}")
                    reuse_id = matched_reuse[0] if matched_reuse else None
                    rows.append({
                        "manifest_order": len(rows) + 1,
                        "configuration_id": config_id(width, scaling, estimator, family, topology, reps),
                        "width": width,
                        "scaler": "StandardScaler" if scaling == "Standardization" else "MinMaxScaler",
                        "archived_scaling_label": scaling,
                        "rfe_estimator": estimator,
                        "preprocessing_id": preprocessing_id(width, scaling, estimator),
                        "effective_encoder_family": "ZFeatureMap" if family == "Z" else "Pauli/ZZ equivalent",
                        "execution_constructor": "ZFeatureMap" if family == "Z" else "ZZFeatureMap",
                        "topology": topology,
                        "repetitions": reps,
                        "nominal_feature_map_labels": "|".join(evidence["nominal_feature_map_labels"]),
                        "nominal_topology_labels": "|".join(evidence["nominal_topology_labels"]),
                        "source_row_ids": "|".join(map(str, evidence["source_row_ids"])),
                        "representative_source_row_id": evidence["representative_source_row_id"],
                        "archived_accuracy_reported": evidence["archived_accuracy_reported"],
                        "archived_correct_count": evidence["archived_correct_count"],
                        "archived_auc_representative": evidence["archived_auc_representative"],
                        "archived_auc_min": evidence["archived_auc_min"],
                        "archived_auc_max": evidence["archived_auc_max"],
                        "archived_auc_alias_conflict": evidence["archived_auc_alias_conflict"],
                        "archived_auc_values_by_source_id": evidence["archived_auc_values_by_source_id"],
                        "archived_f1_reported": evidence["archived_f1_reported"],
                        "reuse_source_row_id": reuse_id if reuse_id is not None else "",
                        "execution_action": "reuse_verified" if reuse_id is not None else "run_new",
                        "probability_calibration_seed": "uncontrolled_None" if reuse_id is not None else CALIBRATION_SEED,
                        "svc_settings": json.dumps({"C": 1.0, "probability": True, "class_weight": None, "shrinking": True, "tol": 0.001}),
                        "kernel_settings": json.dumps({"analytic": True, "shots": None, "fidelity": "squared statevector overlap", "enforce_psd": True, "psd_projection_expected": False}),
                        "split_source": "source/historical/split_idx.json",
                        "status": "locked_reuse" if reuse_id is not None else "locked_pending",
                    })
    manifest = pd.DataFrame(rows)
    if len(manifest) != 128 or manifest["configuration_id"].nunique() != 128:
        raise RuntimeError("Manifest must contain 128 unique effective configurations")
    counts = manifest.groupby(["width", "execution_action"]).size().unstack(fill_value=0)
    for width in EXPECTED_WIDTHS:
        if int(counts.loc[width, "reuse_verified"]) != 4 or int(counts.loc[width, "run_new"]) != 60:
            raise RuntimeError(f"Unexpected reuse/pending count at k={width}: {counts.loc[width].to_dict()}")
    write_csv(MANIFEST, manifest)
    write_json(RECORDS / "manifest_lock.json", {
        "locked_at": now_iso(),
        "manifest": str(MANIFEST),
        "manifest_sha256": sha256_file(MANIFEST),
        "rows": len(manifest),
        "configuration_ids_unique": int(manifest["configuration_id"].nunique()),
        "counts": counts.astype(int).to_dict(),
        "new_probability_calibration_seed": CALIBRATION_SEED,
        "old_probability_calibration_seed": None,
        "execution_order": "all k=4 pending evaluations, verify 64/64, then all k=7 pending evaluations",
    })
    return manifest


def create_preprocessing_caches(
    manifest: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> pd.DataFrame:
    classical = pd.read_csv(SOURCE / "historical" / "classical_rfe_fixed_counts.csv")
    rows = []
    for values in manifest[["width", "scaler", "archived_scaling_label", "rfe_estimator", "preprocessing_id"]].drop_duplicates().itertuples(index=False):
        started = time.perf_counter()
        width = int(values.width)
        scaler = StandardScaler() if values.scaler == "StandardScaler" else MinMaxScaler()
        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]
        y_train = y[train_idx]
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            scaler.fit(X_train_raw)
            X_train_scaled = scaler.transform(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw)
            rfe = RFE(estimator=selector(values.rfe_estimator), n_features_to_select=width, step=1)
            rfe.fit(X_train_scaled, y_train)
        X_train = np.ascontiguousarray(rfe.transform(X_train_scaled), dtype=np.float64)
        X_test = np.ascontiguousarray(rfe.transform(X_test_scaled), dtype=np.float64)
        selected_indices = np.flatnonzero(rfe.support_).astype(int)
        selected_names = [feature_names[index] for index in selected_indices]
        archived_names = archived_feature_names(classical, width, values.archived_scaling_label, values.rfe_estimator)
        cache_path = PREP / f"{values.preprocessing_id}.npz"
        np.savez_compressed(
            cache_path,
            X_train=X_train,
            X_test=X_test,
            y_train=y[train_idx],
            y_test=y[test_idx],
            train_idx=train_idx,
            test_idx=test_idx,
            selected_indices=selected_indices,
            selected_names=np.asarray(selected_names, dtype=str),
        )
        scaler_parameters = {}
        for name in ("mean_", "scale_", "var_", "data_min_", "data_max_", "data_range_", "min_"):
            if hasattr(scaler, name):
                scaler_parameters[name] = np.asarray(getattr(scaler, name)).tolist()
        audit = {
            "created_at": now_iso(),
            "preprocessing_id": values.preprocessing_id,
            "width": width,
            "scaler": values.scaler,
            "scaler_fit_rows": 455,
            "scaler_fit_index_sha256": array_hash(train_idx),
            "rfe_estimator": values.rfe_estimator,
            "rfe_settings": {
                key: (repr(value) if key == "estimator" else value)
                for key, value in rfe.get_params(deep=False).items()
            },
            "rfe_fit_rows": 455,
            "rfe_fit_index_sha256": array_hash(train_idx),
            "rfe_fit_labels_sha256": array_hash(y_train),
            "test_indices_in_fit": False,
            "test_labels_in_fit": False,
            "selected_feature_indices": selected_indices.tolist(),
            "selected_feature_names": selected_names,
            "archived_leaked_feature_names": archived_names,
            "selection_changed_from_archive": selected_names != archived_names,
            "scaler_parameters": scaler_parameters,
            "warnings": [{"category": item.category.__name__, "message": str(item.message)} for item in captured],
            "cache_relative_path": str(cache_path.relative_to(ROOT)),
            "cache_sha256": sha256_file(cache_path),
            "X_train_sha256": array_hash(X_train),
            "X_test_sha256": array_hash(X_test),
            "runtime_seconds": time.perf_counter() - started,
        }
        write_json(PREP / f"{values.preprocessing_id}.audit.json", audit)
        rows.append({
            "preprocessing_id": values.preprocessing_id,
            "width": width,
            "scaler": values.scaler,
            "rfe_estimator": values.rfe_estimator,
            "selected_feature_names": "|".join(selected_names),
            "selection_changed_from_archive": audit["selection_changed_from_archive"],
            "cache_relative_path": audit["cache_relative_path"],
            "cache_sha256": audit["cache_sha256"],
            "runtime_seconds": audit["runtime_seconds"],
            "verified_train_only": True,
        })
    frame = pd.DataFrame(rows)
    if len(frame) != 8:
        raise RuntimeError(f"Expected eight preprocessing caches, found {len(frame)}")
    write_csv(RECORDS / "preprocessing_inventory.csv", frame)
    return frame


def verify_existing(manifest: pd.DataFrame, y: np.ndarray, test_idx: np.ndarray) -> pd.DataFrame:
    source_ids = [157, 152, 145, 226, 195, 260, 253, 288, 373, 392, 361, 372, 470, 513, 433, 517, 519]
    archive = pd.read_csv(SOURCE / "historical" / "quantum_each_results.csv")
    rows = []
    for row_id in source_ids:
        source_row = archive.iloc[row_id - 1]
        width = int(source_row["n_features"])
        source_path = source_run_path(row_id, width)
        destination = IMPORTED / f"row{row_id:04d}" / f"k{width:02d}"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source_path, destination)
        predictions = pd.read_csv(destination / "test_predictions.csv")
        if len(predictions) != 114 or predictions["sample_index"].nunique() != 114:
            raise RuntimeError(f"Existing row {row_id} lacks 114 unique predictions")
        if not np.array_equal(predictions["sample_index"].to_numpy(dtype=int), test_idx):
            raise RuntimeError(f"Existing row {row_id} test order differs")
        if not np.array_equal(predictions["true_source_label"].to_numpy(dtype=int), y[test_idx]):
            raise RuntimeError(f"Existing row {row_id} test labels differ")
        metrics = clinical_metrics(
            predictions["true_source_label"].to_numpy(dtype=int),
            predictions["predicted_source_label"].to_numpy(dtype=int),
            predictions["malignant_probability"].to_numpy(dtype=float),
        )
        saved = json.loads((destination / "metrics.json").read_text(encoding="utf-8"))
        corrected = saved["corrected"]
        numeric_checks = {
            key: abs(float(metrics[key]) - float(corrected[key]))
            for key in ("accuracy", "roc_auc", "sensitivity", "specificity", "precision", "f1")
        }
        fit = json.loads((destination / "fit_access_audit.json").read_text(encoding="utf-8"))
        split = json.loads((destination / "split_ids.json").read_text(encoding="utf-8"))
        config = json.loads((destination / "configuration.json").read_text(encoding="utf-8"))
        passed = (
            max(numeric_checks.values()) <= 1e-12
            and not bool(fit["test_indices_in_any_fit"])
            and not bool(fit["test_labels_in_any_fit"])
            and split["test_idx"] == [int(value) for value in test_idx]
        )
        if not passed:
            raise RuntimeError(f"Existing row {row_id} verification failed")
        key = effective_key(source_row)
        cid = config_id(*key)
        manifest_match = manifest[manifest["configuration_id"] == cid]
        in_new_grid = len(manifest_match) == 1
        rows.append({
            "source_row_id": row_id,
            "configuration_id": cid,
            "width": width,
            "in_k4_k7_grid": in_new_grid,
            "source_artifact_path": str(source_path),
            "imported_artifact_path": str(destination.relative_to(ROOT)),
            "test_predictions_sha256": sha256_file(destination / "test_predictions.csv"),
            "configuration_sha256": sha256_file(destination / "configuration.json"),
            "accuracy": metrics["accuracy"],
            "correct_count": metrics["correct_count"],
            "malignant_roc_auc": metrics["roc_auc"],
            "runtime_seconds": float(saved["runtime"]["total_seconds"]),
            "calibration_seed": "uncontrolled_None",
            "metrics_max_absolute_error": max(numeric_checks.values()),
            "fit_access_verified_train_only": True,
            "test_order_verified": True,
            "verification_status": "verified_reusable",
        })
    frame = pd.DataFrame(rows)
    if len(frame) != 17 or frame["configuration_id"].nunique() != 17:
        raise RuntimeError("Expected 17 unique verified existing fixed-split configurations")
    grid_counts = frame[frame["width"].isin(EXPECTED_WIDTHS)].groupby("width").size().to_dict()
    if grid_counts != {4: 4, 7: 4}:
        raise RuntimeError(f"Expected four reusable rows at k4/k7: {grid_counts}")
    write_csv(RECORDS / "existing_results_inventory.csv", frame)
    write_json(RECORDS / "existing_results_verification.json", {
        "verified_at": now_iso(),
        "rows": len(frame),
        "unique_configurations": int(frame["configuration_id"].nunique()),
        "by_width": frame.groupby("width").size().astype(int).to_dict(),
        "all_passed": True,
    })
    return frame


def local_statevector_kernel(feature_map, values: np.ndarray) -> np.ndarray:
    statevectors = []
    statevector_type = FidelityStatevectorKernel(feature_map=feature_map)._statevector_type
    for row in values:
        statevectors.append(np.asarray(statevector_type(feature_map.assign_parameters(row)).data, dtype=np.complex128))
    bank = np.asarray(statevectors)
    result = np.abs(np.conjugate(bank) @ bank.T) ** 2
    np.fill_diagonal(result, 1.0)
    return result


def verify_kernel_equivalence() -> None:
    rng = np.random.default_rng(20261107)
    values = rng.uniform(0.05, 0.95, size=(4, 4))
    checks = []
    constructors = [("ZFeatureMap", "", ZFeatureMap(4, reps=2))]
    for topology in ("linear", "full", "circular"):
        constructors.append(("ZZFeatureMap", topology, ZZFeatureMap(4, reps=2, entanglement=topology)))
        constructors.append(("PauliFeatureMap", topology, PauliFeatureMap(4, reps=2, entanglement=topology, paulis=["Z", "ZZ"], alpha=2.0)))
    matrices = {}
    for name, topology, fmap in constructors:
        reference = FidelityStatevectorKernel(
            feature_map=fmap, cache_size=None, auto_clear_cache=False, shots=None, enforce_psd=True
        ).evaluate(values)
        optimized = local_statevector_kernel(fmap, values)
        error = float(np.max(np.abs(reference - optimized)))
        key = f"{name}:{topology or 'none'}"
        matrices[key] = optimized
        checks.append({"constructor": name, "topology": topology, "max_abs_error_vs_FidelityStatevectorKernel": error})
    alias_checks = []
    for topology in ("linear", "full", "circular"):
        error = float(np.max(np.abs(matrices[f"ZZFeatureMap:{topology}"] - matrices[f"PauliFeatureMap:{topology}"])))
        alias_checks.append({"topology": topology, "max_abs_kernel_difference": error})
    tolerance = 1e-12
    passed = all(item["max_abs_error_vs_FidelityStatevectorKernel"] <= tolerance for item in checks) and all(
        item["max_abs_kernel_difference"] <= tolerance for item in alias_checks
    )
    record = {
        "verified_at": now_iso(),
        "tolerance": tolerance,
        "sample_shape": list(values.shape),
        "feature_order": "input column order unchanged",
        "Pauli_terms": ["Z", "ZZ"],
        "alpha": 2.0,
        "data_map_func": "Qiskit default self_product",
        "constructor_checks": checks,
        "Pauli_ZZ_alias_checks": alias_checks,
        "passed": passed,
    }
    write_json(RECORDS / "kernel_equivalence_audit.json", record)
    if not passed:
        raise RuntimeError("Kernel equivalence audit failed")


def main() -> None:
    repository_and_machine_record()
    snapshot_sources()
    X, y, feature_names, train_idx, test_idx = verify_dataset_and_split()
    _, groups = archive_groups()
    manifest = build_manifest(groups)
    preprocessing = create_preprocessing_caches(manifest, X, y, feature_names, train_idx, test_idx)
    existing = verify_existing(manifest, y, test_idx)
    verify_kernel_equivalence()
    write_json(RECORDS / "prelaunch_verification.json", {
        "verified_at": now_iso(),
        "status": "passed",
        "manifest_rows": len(manifest),
        "k4_total": int((manifest["width"] == 4).sum()),
        "k4_reused": int(((manifest["width"] == 4) & (manifest["execution_action"] == "reuse_verified")).sum()),
        "k4_pending_new": int(((manifest["width"] == 4) & (manifest["execution_action"] == "run_new")).sum()),
        "k7_total": int((manifest["width"] == 7).sum()),
        "k7_reused": int(((manifest["width"] == 7) & (manifest["execution_action"] == "reuse_verified")).sum()),
        "k7_pending_new": int(((manifest["width"] == 7) & (manifest["execution_action"] == "run_new")).sum()),
        "preprocessing_cache_count": len(preprocessing),
        "verified_existing_fixed_split_count": len(existing),
        "verified_existing_unique_configuration_count": int(existing["configuration_id"].nunique()),
        "new_probability_calibration_seed": CALIBRATION_SEED,
        "old_probability_calibration_seed": None,
        "test_access_policy": "test features are transformed only; test labels are used only after final prediction for metrics",
        "width_gate": "k=7 may start only after k=4 coverage checkpoint verifies 64 completed configurations",
    })
    print(json.dumps({"status": "prepared", "manifest_rows": len(manifest), "preprocessing_caches": len(preprocessing), "existing_verified": len(existing)}, indent=2))


if __name__ == "__main__":
    main()

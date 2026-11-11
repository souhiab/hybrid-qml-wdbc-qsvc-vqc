from __future__ import annotations

import hashlib
import json
import os
import platform
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
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from threadpoolctl import threadpool_info

import prepare_grid as shared


ROOT = Path(__file__).resolve().parent
RECORDS = ROOT / "records"
SOURCE = ROOT / "source"
PREP = ROOT / "preprocessing"
K11_MANIFEST = ROOT / "execution_manifest_k11.csv"
COMBINED_MANIFEST = ROOT / "execution_manifest.csv"
WIDTH = 11
REUSE_IDS = {373, 392, 361, 372}
CALIBRATION_SEED = 20261107


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=shared.json_default) + "\n", encoding="utf-8")
    os.replace(temp, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def sha256_file(path: Path) -> str:
    return shared.sha256_file(path)


def run_git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=shared.REPO, text=True, capture_output=True, check=True).stdout.strip()


def build_k11_manifest(groups: dict[tuple, dict]) -> pd.DataFrame:
    rows = []
    for scaling in ("Standardization", "Normalization"):
        for estimator in ("Logistic Regression", "Random Forest"):
            encoders = [("Z", "", reps) for reps in range(1, 5)]
            encoders += [
                ("Pauli_ZZ", topology, reps)
                for topology in ("linear", "full", "circular")
                for reps in range(1, 5)
            ]
            for family, topology, reps in encoders:
                key = (WIDTH, scaling, estimator, family, topology, reps)
                if key not in groups:
                    raise RuntimeError(f"Missing k=11 archived effective group {key}")
                evidence = groups[key]
                matched = sorted(REUSE_IDS.intersection(evidence["source_row_ids"]))
                if len(matched) > 1:
                    raise RuntimeError(f"Multiple reusable records for {key}: {matched}")
                reuse_id = matched[0] if matched else None
                rows.append({
                    "manifest_order": 128 + len(rows) + 1,
                    "configuration_id": shared.config_id(WIDTH, scaling, estimator, family, topology, reps),
                    "width": WIDTH,
                    "scaler": "StandardScaler" if scaling == "Standardization" else "MinMaxScaler",
                    "archived_scaling_label": scaling,
                    "rfe_estimator": estimator,
                    "preprocessing_id": shared.preprocessing_id(WIDTH, scaling, estimator),
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
    if len(manifest) != 64 or manifest["configuration_id"].nunique() != 64:
        raise RuntimeError("k=11 manifest must contain 64 unique effective configurations")
    counts = manifest["execution_action"].value_counts().to_dict()
    if counts != {"run_new": 60, "reuse_verified": 4}:
        raise RuntimeError(f"Unexpected k=11 reuse/pending counts: {counts}")
    return manifest


def create_k11_preprocessing_caches(
    manifest: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> pd.DataFrame:
    classical = pd.read_csv(SOURCE / "historical" / "classical_rfe_fixed_counts.csv")
    rows = []
    combos = manifest[["width", "scaler", "archived_scaling_label", "rfe_estimator", "preprocessing_id"]].drop_duplicates()
    for values in combos.itertuples(index=False):
        started = time.perf_counter()
        scaler = StandardScaler() if values.scaler == "StandardScaler" else MinMaxScaler()
        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]
        y_train = y[train_idx]
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            scaler.fit(X_train_raw)
            X_train_scaled = scaler.transform(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw)
            rfe = RFE(estimator=shared.selector(values.rfe_estimator), n_features_to_select=WIDTH, step=1)
            rfe.fit(X_train_scaled, y_train)
        X_train = np.ascontiguousarray(rfe.transform(X_train_scaled), dtype=np.float64)
        X_test = np.ascontiguousarray(rfe.transform(X_test_scaled), dtype=np.float64)
        selected_indices = np.flatnonzero(rfe.support_).astype(int)
        selected_names = [feature_names[index] for index in selected_indices]
        archived_names = shared.archived_feature_names(classical, WIDTH, values.archived_scaling_label, values.rfe_estimator)
        cache_path = PREP / f"{values.preprocessing_id}.npz"
        np.savez_compressed(
            cache_path, X_train=X_train, X_test=X_test,
            y_train=y[train_idx], y_test=y[test_idx],
            train_idx=train_idx, test_idx=test_idx,
            selected_indices=selected_indices,
            selected_names=np.asarray(selected_names, dtype=str),
        )
        scaler_parameters = {}
        for name in ("mean_", "scale_", "var_", "data_min_", "data_max_", "data_range_", "min_"):
            if hasattr(scaler, name):
                scaler_parameters[name] = np.asarray(getattr(scaler, name)).tolist()
        audit = {
            "created_at": now_iso(),
            "continuation": "k11",
            "preprocessing_id": values.preprocessing_id,
            "width": WIDTH,
            "scaler": values.scaler,
            "scaler_fit_rows": 455,
            "scaler_fit_index_sha256": shared.array_hash(train_idx),
            "rfe_estimator": values.rfe_estimator,
            "rfe_settings": {key: (repr(value) if key == "estimator" else value) for key, value in rfe.get_params(deep=False).items()},
            "rfe_fit_rows": 455,
            "rfe_fit_index_sha256": shared.array_hash(train_idx),
            "rfe_fit_labels_sha256": shared.array_hash(y_train),
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
            "X_train_sha256": shared.array_hash(X_train),
            "X_test_sha256": shared.array_hash(X_test),
            "runtime_seconds": time.perf_counter() - started,
        }
        write_json(PREP / f"{values.preprocessing_id}.audit.json", audit)
        rows.append({
            "preprocessing_id": values.preprocessing_id,
            "width": WIDTH,
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
    if len(frame) != 4:
        raise RuntimeError(f"Expected four k=11 preprocessing caches, found {len(frame)}")
    prior = pd.read_csv(RECORDS / "preprocessing_inventory.csv")
    combined = pd.concat([prior[prior["width"].astype(int) != WIDTH], frame], ignore_index=True)
    if len(combined) != 12 or combined["preprocessing_id"].nunique() != 12:
        raise RuntimeError("Combined preprocessing inventory must have 12 unique rows")
    write_csv(RECORDS / "preprocessing_inventory.csv", combined)
    write_csv(RECORDS / "preprocessing_inventory_k11.csv", frame)
    return frame


def verify_reused_k11(manifest: pd.DataFrame, y: np.ndarray, test_idx: np.ndarray) -> pd.DataFrame:
    inventory = pd.read_csv(RECORDS / "existing_results_inventory.csv", keep_default_na=False)
    expected = manifest[manifest["execution_action"] == "reuse_verified"].copy()
    rows = []
    for item in expected.itertuples(index=False):
        source_id = int(item.reuse_source_row_id)
        matched = inventory[inventory["source_row_id"].astype(int) == source_id]
        if len(matched) != 1 or matched.iloc[0]["configuration_id"] != item.configuration_id:
            raise RuntimeError(f"Reusable inventory mismatch for source row {source_id}")
        artifact = ROOT / matched.iloc[0]["imported_artifact_path"]
        predictions = pd.read_csv(artifact / "test_predictions.csv")
        fit = json.loads((artifact / "fit_access_audit.json").read_text(encoding="utf-8"))
        metrics = json.loads((artifact / "metrics.json").read_text(encoding="utf-8"))
        recomputed = shared.clinical_metrics(
            predictions["true_source_label"].to_numpy(int),
            predictions["predicted_source_label"].to_numpy(int),
            predictions["malignant_probability"].to_numpy(float),
        )
        errors = [abs(float(recomputed[key]) - float(metrics["corrected"][key])) for key in ("accuracy", "roc_auc", "sensitivity", "specificity", "precision", "f1")]
        passed = (
            len(predictions) == 114
            and predictions["sample_index"].nunique() == 114
            and predictions["sample_index"].tolist() == test_idx.tolist()
            and predictions["true_source_label"].tolist() == y[test_idx].tolist()
            and max(errors) <= 1e-12
            and not fit["test_indices_in_any_fit"]
            and not fit["test_labels_in_any_fit"]
        )
        if not passed:
            raise RuntimeError(f"Reusable row {source_id} failed continuation verification")
        rows.append({
            "source_row_id": source_id,
            "configuration_id": item.configuration_id,
            "artifact_path": matched.iloc[0]["imported_artifact_path"],
            "accuracy": recomputed["accuracy"],
            "correct_count": recomputed["TP_malignant"] + recomputed["TN_benign"],
            "malignant_roc_auc": recomputed["roc_auc"],
            "max_metric_error": max(errors),
            "calibration_seed": "uncontrolled_None",
            "fit_access_verified": True,
            "verification_status": "verified_reusable",
        })
    frame = pd.DataFrame(rows)
    if len(frame) != 4 or frame["configuration_id"].nunique() != 4:
        raise RuntimeError("Expected four verified reusable k=11 configurations")
    write_csv(RECORDS / "existing_results_inventory_k11.csv", frame)
    return frame


def local_kernel(feature_map, values: np.ndarray) -> np.ndarray:
    kernel = FidelityStatevectorKernel(feature_map=feature_map, cache_size=None, auto_clear_cache=False, shots=None, enforce_psd=True)
    statevectors = np.asarray([
        np.asarray(kernel._statevector_type(feature_map.assign_parameters(row)).data, dtype=np.complex128)
        for row in values
    ])
    result = np.abs(np.conjugate(statevectors) @ statevectors.T) ** 2
    np.fill_diagonal(result, 1.0)
    return result


def verify_k11_kernel_equivalence() -> dict:
    rng = np.random.default_rng(20261107)
    values = rng.uniform(0.05, 0.95, size=(3, WIDTH))
    constructors = [("ZFeatureMap", "", ZFeatureMap(WIDTH, reps=2))]
    for topology in ("linear", "full", "circular"):
        constructors.append(("ZZFeatureMap", topology, ZZFeatureMap(WIDTH, reps=2, entanglement=topology)))
        constructors.append(("PauliFeatureMap", topology, PauliFeatureMap(WIDTH, reps=2, entanglement=topology, paulis=["Z", "ZZ"], alpha=2.0)))
    matrices = {}
    checks = []
    for name, topology, fmap in constructors:
        reference = FidelityStatevectorKernel(feature_map=fmap, cache_size=None, auto_clear_cache=False, shots=None, enforce_psd=True).evaluate(values)
        optimized = local_kernel(fmap, values)
        error = float(np.max(np.abs(reference - optimized)))
        matrices[f"{name}:{topology or 'none'}"] = optimized
        checks.append({"constructor": name, "topology": topology, "max_abs_error": error})
    aliases = []
    for topology in ("linear", "full", "circular"):
        aliases.append({"topology": topology, "max_abs_kernel_difference": float(np.max(np.abs(matrices[f"ZZFeatureMap:{topology}"] - matrices[f"PauliFeatureMap:{topology}"])))})
    tolerance = 1e-12
    passed = max(item["max_abs_error"] for item in checks) <= tolerance and max(item["max_abs_kernel_difference"] for item in aliases) <= tolerance
    record = {
        "verified_at": now_iso(), "width": WIDTH, "sample_shape": list(values.shape), "tolerance": tolerance,
        "constructor_checks": checks, "Pauli_ZZ_alias_checks": aliases,
        "Pauli_terms": ["Z", "ZZ"], "alpha": 2.0, "data_map_func": "Qiskit default self_product",
        "passed": passed,
    }
    write_json(RECORDS / "kernel_equivalence_audit_k11.json", record)
    if not passed:
        raise RuntimeError("k=11 kernel equivalence failed")
    return record


def main() -> None:
    workbook = ROOT / "WDBC_QSVC_Training_Only_Master.xlsx"
    if not workbook.exists():
        raise FileNotFoundError(workbook)
    status = run_git("status", "--porcelain=v1", "--untracked-files=all")
    write_json(RECORDS / "k11_repository_start_state.json", {
        "recorded_at": now_iso(), "commit": run_git("rev-parse", "HEAD"),
        "remote_url": run_git("remote", "get-url", "origin"),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "status_line_count": len(status.splitlines()),
    })
    vm = psutil.virtual_memory()
    write_json(RECORDS / "k11_machine_preflight.json", {
        "recorded_at": now_iso(), "platform": platform.platform(), "python": sys.version,
        "python_executable": sys.executable, "physical_cpu_cores": psutil.cpu_count(logical=False),
        "logical_cpu_cores": psutil.cpu_count(logical=True), "ram_total_bytes": vm.total,
        "ram_available_bytes": vm.available, "threadpools": threadpool_info(),
    })
    _, groups = shared.archive_groups()
    k11_manifest = build_k11_manifest(groups)
    prior = pd.read_csv(COMBINED_MANIFEST, keep_default_na=False)
    if len(prior) == 192:
        prior = prior[prior["width"].astype(int) != WIDTH].copy()
    if len(prior) != 128 or set(prior["width"].astype(int)) != {4, 7}:
        raise RuntimeError("The continuation expected the verified 128-row k=4/k=7 manifest")
    combined = pd.concat([prior, k11_manifest], ignore_index=True)
    if len(combined) != 192 or combined["configuration_id"].nunique() != 192:
        raise RuntimeError("Combined manifest must contain 192 unique configurations")
    write_csv(K11_MANIFEST, k11_manifest)
    write_csv(COMBINED_MANIFEST, combined)
    X, y, feature_names, train_idx, test_idx = shared.verify_dataset_and_split()
    preprocessing = create_k11_preprocessing_caches(k11_manifest, X, y, feature_names, train_idx, test_idx)
    reused = verify_reused_k11(k11_manifest, y, test_idx)
    kernel = verify_k11_kernel_equivalence()
    source_files = [
        ROOT / "run_scaled_grid.py", ROOT / "prepare_grid.py", ROOT / "consolidate_results.py",
        SOURCE / "environment" / "quantum_adapter_environment_lock.txt",
        SOURCE / "historical" / "quantum_each_results.csv", SOURCE / "historical" / "split_idx.json",
        workbook,
    ]
    write_csv(RECORDS / "k11_continuation_source_hashes.csv", pd.DataFrame([
        {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in source_files
    ]))
    write_json(RECORDS / "k11_manifest_lock.json", {
        "locked_at": now_iso(), "manifest_path": str(K11_MANIFEST.relative_to(ROOT)),
        "manifest_sha256": sha256_file(K11_MANIFEST), "combined_manifest_sha256": sha256_file(COMBINED_MANIFEST),
        "rows": len(k11_manifest), "reused": len(reused), "pending_new": 60,
        "preprocessing_caches": len(preprocessing), "kernel_equivalence_passed": kernel["passed"],
        "new_probability_calibration_seed": CALIBRATION_SEED,
        "execution_scope": "k=11 only; k=17 excluded",
    })
    print(json.dumps({"status": "k11_prepared", "manifest_rows": 64, "reused": 4, "pending": 60, "preprocessing_caches": 4}, indent=2))


if __name__ == "__main__":
    main()

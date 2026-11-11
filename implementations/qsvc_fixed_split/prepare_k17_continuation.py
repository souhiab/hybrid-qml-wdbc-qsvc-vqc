from __future__ import annotations

import gc
import hashlib
import importlib.metadata
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
import run_scaled_grid as runner


ROOT = Path(__file__).resolve().parent
RECORDS = ROOT / "records"
SOURCE = ROOT / "source"
PREP = ROOT / "preprocessing"
K17_MANIFEST = ROOT / "execution_manifest_k17.csv"
COMBINED_MANIFEST = ROOT / "execution_manifest.csv"
WIDTH = 17
REUSE_IDS = {433, 470, 513, 517, 519}
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


def run_git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=shared.REPO, text=True, capture_output=True, check=True).stdout.strip()


def archived_fields(evidence: dict | None, family: str, topology: str) -> dict:
    if evidence is None:
        return {
            "nominal_feature_map_labels": "ZFeatureMap" if family == "Z" else "PauliFeatureMap|ZZFeatureMap",
            "nominal_topology_labels": "" if family == "Z" else topology,
            "source_row_ids": "",
            "representative_source_row_id": "",
            "archived_accuracy_reported": "",
            "archived_correct_count": "",
            "archived_auc_representative": "",
            "archived_auc_min": "",
            "archived_auc_max": "",
            "archived_auc_alias_conflict": "",
            "archived_auc_values_by_source_id": "",
            "archived_f1_reported": "",
        }
    return {
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
    }


def build_manifest(groups: dict[tuple, dict]) -> pd.DataFrame:
    rows = []
    for scaling in ("Standardization", "Normalization"):
        for estimator in ("Logistic Regression", "Random Forest"):
            encoders = [("Z", "", reps) for reps in range(1, 5)]
            encoders += [("Pauli_ZZ", topology, reps) for topology in ("linear", "full", "circular") for reps in range(1, 5)]
            for family, topology, reps in encoders:
                key = (WIDTH, scaling, estimator, family, topology, reps)
                evidence = groups.get(key)
                archived = archived_fields(evidence, family, topology)
                matched = sorted(REUSE_IDS.intersection(evidence["source_row_ids"])) if evidence else []
                if len(matched) > 1:
                    raise RuntimeError(f"Multiple reusable records for {key}: {matched}")
                reuse_id = matched[0] if matched else None
                rows.append({
                    "manifest_order": 192 + len(rows) + 1,
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
                    **archived,
                    "reuse_source_row_id": reuse_id if reuse_id is not None else "",
                    "execution_action": "reuse_verified" if reuse_id is not None else "run_new",
                    "probability_calibration_seed": "uncontrolled_None" if reuse_id is not None else CALIBRATION_SEED,
                    "svc_settings": json.dumps({"C": 1.0, "probability": True, "class_weight": None, "shrinking": True, "tol": 0.001}),
                    "kernel_settings": json.dumps({"analytic": True, "shots": None, "fidelity": "squared statevector overlap", "enforce_psd": True, "psd_projection_expected": False}),
                    "split_source": "source/historical/split_idx.json",
                    "status": "locked_reuse" if reuse_id is not None else ("locked_pending" if evidence else "locked_pending_no_archived_mapping"),
                })
    manifest = pd.DataFrame(rows)
    if len(manifest) != 64 or manifest["configuration_id"].nunique() != 64:
        raise RuntimeError("k=17 manifest must contain 64 unique effective configurations")
    counts = manifest["execution_action"].value_counts().to_dict()
    if counts != {"run_new": 59, "reuse_verified": 5}:
        raise RuntimeError(f"Unexpected k=17 counts: {counts}")
    if int((manifest["source_row_ids"] == "").sum()) != 5:
        raise RuntimeError("Expected five k=17 configurations without archived source mappings")
    return manifest


def create_preprocessing(manifest: pd.DataFrame, X: np.ndarray, y: np.ndarray, feature_names: list[str], train_idx: np.ndarray, test_idx: np.ndarray) -> pd.DataFrame:
    classical = pd.read_csv(SOURCE / "historical" / "classical_rfe_fixed_counts.csv")
    rows = []
    combos = manifest[["width", "scaler", "archived_scaling_label", "rfe_estimator", "preprocessing_id"]].drop_duplicates()
    for values in combos.itertuples(index=False):
        started = time.perf_counter()
        scaler = StandardScaler() if values.scaler == "StandardScaler" else MinMaxScaler()
        X_train_raw, X_test_raw, y_train = X[train_idx], X[test_idx], y[train_idx]
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw)
            rfe = RFE(estimator=shared.selector(values.rfe_estimator), n_features_to_select=WIDTH, step=1)
            rfe.fit(X_train_scaled, y_train)
        X_train = np.ascontiguousarray(rfe.transform(X_train_scaled), dtype=np.float64)
        X_test = np.ascontiguousarray(rfe.transform(X_test_scaled), dtype=np.float64)
        selected_indices = np.flatnonzero(rfe.support_).astype(int)
        selected_names = [feature_names[index] for index in selected_indices]
        archived_names = shared.archived_feature_names(classical, WIDTH, values.archived_scaling_label, values.rfe_estimator)
        cache_path = PREP / f"{values.preprocessing_id}.npz"
        np.savez_compressed(cache_path, X_train=X_train, X_test=X_test, y_train=y[train_idx], y_test=y[test_idx], train_idx=train_idx, test_idx=test_idx, selected_indices=selected_indices, selected_names=np.asarray(selected_names, dtype=str))
        parameters = {}
        for name in ("mean_", "scale_", "var_", "data_min_", "data_max_", "data_range_", "min_"):
            if hasattr(scaler, name):
                parameters[name] = np.asarray(getattr(scaler, name)).tolist()
        audit = {
            "created_at": now_iso(), "continuation": "k17", "preprocessing_id": values.preprocessing_id, "width": WIDTH,
            "scaler": values.scaler, "scaler_fit_rows": 455, "scaler_fit_index_sha256": shared.array_hash(train_idx),
            "rfe_estimator": values.rfe_estimator, "rfe_settings": {key: (repr(value) if key == "estimator" else value) for key, value in rfe.get_params(deep=False).items()},
            "rfe_fit_rows": 455, "rfe_fit_index_sha256": shared.array_hash(train_idx), "rfe_fit_labels_sha256": shared.array_hash(y_train),
            "test_indices_in_fit": False, "test_labels_in_fit": False,
            "selected_feature_indices": selected_indices.tolist(), "selected_feature_names": selected_names,
            "archived_leaked_feature_names": archived_names, "selection_changed_from_archive": selected_names != archived_names,
            "scaler_parameters": parameters, "warnings": [{"category": item.category.__name__, "message": str(item.message)} for item in captured],
            "cache_relative_path": str(cache_path.relative_to(ROOT)), "cache_sha256": shared.sha256_file(cache_path),
            "X_train_sha256": shared.array_hash(X_train), "X_test_sha256": shared.array_hash(X_test), "runtime_seconds": time.perf_counter() - started,
        }
        write_json(PREP / f"{values.preprocessing_id}.audit.json", audit)
        rows.append({"preprocessing_id": values.preprocessing_id, "width": WIDTH, "scaler": values.scaler, "rfe_estimator": values.rfe_estimator, "selected_feature_names": "|".join(selected_names), "selection_changed_from_archive": audit["selection_changed_from_archive"], "cache_relative_path": audit["cache_relative_path"], "cache_sha256": audit["cache_sha256"], "runtime_seconds": audit["runtime_seconds"], "verified_train_only": True})
    frame = pd.DataFrame(rows)
    if len(frame) != 4:
        raise RuntimeError("Expected four k=17 preprocessing caches")
    prior = pd.read_csv(RECORDS / "preprocessing_inventory.csv")
    combined = pd.concat([prior[prior["width"].astype(int) != WIDTH], frame], ignore_index=True)
    if len(combined) != 16 or combined["preprocessing_id"].nunique() != 16:
        raise RuntimeError("Combined preprocessing inventory must have 16 rows")
    write_csv(RECORDS / "preprocessing_inventory.csv", combined)
    write_csv(RECORDS / "preprocessing_inventory_k17.csv", frame)
    return frame


def verify_reused(manifest: pd.DataFrame, y: np.ndarray, test_idx: np.ndarray) -> pd.DataFrame:
    inventory = pd.read_csv(RECORDS / "existing_results_inventory.csv", keep_default_na=False)
    rows = []
    for item in manifest[manifest["execution_action"] == "reuse_verified"].itertuples(index=False):
        source_id = int(item.reuse_source_row_id)
        matched = inventory[inventory["source_row_id"].astype(int) == source_id]
        if len(matched) != 1 or matched.iloc[0]["configuration_id"] != item.configuration_id:
            raise RuntimeError(f"Reusable inventory mismatch for source row {source_id}")
        artifact = ROOT / matched.iloc[0]["imported_artifact_path"]
        predictions = pd.read_csv(artifact / "test_predictions.csv")
        fit = json.loads((artifact / "fit_access_audit.json").read_text(encoding="utf-8"))
        metrics = json.loads((artifact / "metrics.json").read_text(encoding="utf-8"))
        recomputed = shared.clinical_metrics(predictions["true_source_label"].to_numpy(int), predictions["predicted_source_label"].to_numpy(int), predictions["malignant_probability"].to_numpy(float))
        errors = [abs(float(recomputed[key]) - float(metrics["corrected"][key])) for key in ("accuracy", "roc_auc", "sensitivity", "specificity", "precision", "f1")]
        passed = len(predictions) == 114 and predictions["sample_index"].tolist() == test_idx.tolist() and predictions["true_source_label"].tolist() == y[test_idx].tolist() and max(errors) <= 1e-12 and not fit["test_indices_in_any_fit"] and not fit["test_labels_in_any_fit"]
        if not passed:
            raise RuntimeError(f"Reusable row {source_id} failed k=17 verification")
        rows.append({"source_row_id": source_id, "configuration_id": item.configuration_id, "artifact_path": matched.iloc[0]["imported_artifact_path"], "accuracy": recomputed["accuracy"], "correct_count": recomputed["correct_count"], "malignant_roc_auc": recomputed["roc_auc"], "max_metric_error": max(errors), "calibration_seed": "uncontrolled_None", "fit_access_verified": True, "verification_status": "verified_reusable"})
    frame = pd.DataFrame(rows)
    if len(frame) != 5:
        raise RuntimeError("Expected five verified reusable k=17 configurations")
    write_csv(RECORDS / "existing_results_inventory_k17.csv", frame)
    return frame


def benchmark_and_verify(cache_path: Path) -> dict:
    with np.load(cache_path, allow_pickle=False) as cache:
        train_values = np.asarray(cache["X_train"][:3], dtype=np.float64)
        test_values = np.asarray(cache["X_test"][:2], dtype=np.float64)
    feature_map = ZZFeatureMap(WIDTH, reps=1, entanglement="full")
    original = FidelityStatevectorKernel(feature_map=feature_map, cache_size=None, auto_clear_cache=False, shots=None, enforce_psd=True)
    started = time.perf_counter()
    reference_train = original.evaluate(train_values)
    reference_test = original.evaluate(test_values, train_values)
    original_seconds = time.perf_counter() - started
    del original
    gc.collect()
    logger = runner.RunLogger(RECORDS / "k17_implementation_benchmark.log")
    started = time.perf_counter()
    train_bank, state_seconds = runner.statevector_bank(feature_map, train_values, logger, "benchmark train")
    optimized_train, train_audit = runner.blocked_fidelity(train_bank, train_bank, symmetric=True)
    optimized_test, test_audit, test_state_seconds = runner.streamed_cross_fidelity(feature_map, test_values, train_bank, logger, batch_size=2)
    optimized_seconds = time.perf_counter() - started
    max_error = max(float(np.max(np.abs(reference_train - optimized_train))), float(np.max(np.abs(reference_test - optimized_test))))
    state_bytes = 455 * (1 << WIDTH) * np.dtype(np.complex128).itemsize
    test_bytes = 114 * (1 << WIDTH) * np.dtype(np.complex128).itemsize
    try:
        aer_version = importlib.metadata.version("qiskit-aer")
    except importlib.metadata.PackageNotFoundError:
        aer_version = None
    record = {
        "verified_at": now_iso(), "width": WIDTH, "feature_map": "ZZFeatureMap", "topology": "full", "repetitions": 1,
        "sample_train_rows": 3, "sample_test_rows": 2,
        "original_implementation": "FidelityStatevectorKernel.evaluate",
        "optimized_implementation": "single-allocation train bank, symmetric blocked train kernel, streamed test cross-kernel",
        "original_seconds": original_seconds, "optimized_seconds": optimized_seconds, "speedup_original_over_optimized": original_seconds / optimized_seconds,
        "max_absolute_kernel_error": max_error, "tolerance": 1e-12, "passed": max_error <= 1e-12,
        "optimized_component_seconds": {"train_statevectors": state_seconds, "train_kernel": train_audit["seconds"], "test_statevectors": test_state_seconds, "test_kernel": test_audit["seconds"]},
        "estimated_old_peak_array_bytes": 2 * state_bytes + test_bytes,
        "estimated_optimized_peak_array_bytes": state_bytes + 4 * (1 << WIDTH) * np.dtype(np.complex128).itemsize,
        "estimated_array_memory_reduction_fraction": 1 - ((state_bytes + 4 * (1 << WIDTH) * np.dtype(np.complex128).itemsize) / (2 * state_bytes + test_bytes)),
        "selected_backend": "qiskit.quantum_info.Statevector",
        "selected_backend_gpu_supported": False,
        "qiskit_aer_installed_version": aer_version,
        "gpu_note": "The validated selected backend is CPU-only. Switching to an Aer GPU backend would be a separate implementation change and was not used.",
    }
    write_json(RECORDS / "k17_implementation_benchmark.json", record)
    if not record["passed"]:
        raise RuntimeError("k=17 optimized implementation failed numerical validation")
    return record


def main() -> None:
    if GetProcess := [process for process in psutil.process_iter(["pid", "name", "cmdline"]) if process.info["pid"] != os.getpid() and "python" in (process.info["name"] or "").lower() and any("run_k17_grid.py" in part for part in (process.info["cmdline"] or []))]:
        raise RuntimeError(f"A k=17 runner is already active: {[process.info['pid'] for process in GetProcess]}")
    status = run_git("status", "--porcelain=v1", "--untracked-files=all")
    write_json(RECORDS / "k17_repository_start_state.json", {"recorded_at": now_iso(), "commit": run_git("rev-parse", "HEAD"), "remote_url": run_git("remote", "get-url", "origin"), "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(), "status_line_count": len(status.splitlines())})
    vm = psutil.virtual_memory()
    write_json(RECORDS / "k17_machine_preflight.json", {"recorded_at": now_iso(), "platform": platform.platform(), "python": sys.version, "python_executable": sys.executable, "physical_cpu_cores": psutil.cpu_count(logical=False), "logical_cpu_cores": psutil.cpu_count(logical=True), "cpu_percent_before_launch": psutil.cpu_percent(interval=1.0, percpu=True), "ram_total_bytes": vm.total, "ram_available_bytes": vm.available, "ram_used_percent": vm.percent, "threadpools": threadpool_info()})
    _, groups = shared.archive_groups()
    manifest = build_manifest(groups)
    prior = pd.read_csv(COMBINED_MANIFEST, keep_default_na=False)
    prior = prior[prior["width"].astype(int) != WIDTH].copy()
    if len(prior) != 192 or set(prior["width"].astype(int)) != {4, 7, 11}:
        raise RuntimeError("Expected the verified 192-row k=4/k=7/k=11 manifest")
    combined = pd.concat([prior, manifest], ignore_index=True)
    if len(combined) != 256 or combined["configuration_id"].nunique() != 256:
        raise RuntimeError("Combined manifest must contain 256 unique configurations")
    write_csv(K17_MANIFEST, manifest)
    write_csv(COMBINED_MANIFEST, combined)
    X, y, feature_names, train_idx, test_idx = shared.verify_dataset_and_split()
    preprocessing = create_preprocessing(manifest, X, y, feature_names, train_idx, test_idx)
    reused = verify_reused(manifest, y, test_idx)
    benchmark = benchmark_and_verify(ROOT / preprocessing.iloc[0]["cache_relative_path"])
    source_files = [ROOT / "run_scaled_grid.py", ROOT / "prepare_k17_continuation.py", ROOT / "run_k11_grid.py", SOURCE / "environment" / "quantum_adapter_environment_lock.txt", SOURCE / "historical" / "quantum_each_results.csv", SOURCE / "historical" / "split_idx.json"]
    write_csv(RECORDS / "k17_continuation_source_hashes.csv", pd.DataFrame([{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": shared.sha256_file(path)} for path in source_files]))
    write_json(RECORDS / "k17_manifest_lock.json", {"locked_at": now_iso(), "manifest_path": str(K17_MANIFEST.relative_to(ROOT)), "manifest_sha256": shared.sha256_file(K17_MANIFEST), "combined_manifest_sha256": shared.sha256_file(COMBINED_MANIFEST), "rows": 64, "reused": len(reused), "pending_new": 59, "archived_mapping_missing": int((manifest["source_row_ids"] == "").sum()), "preprocessing_caches": len(preprocessing), "optimized_kernel_validation_passed": benchmark["passed"], "new_probability_calibration_seed": CALIBRATION_SEED, "execution_scope": "k=17 only"})
    print(json.dumps({"status": "k17_prepared", "manifest_rows": 64, "reused": 5, "pending": 59, "missing_archived_mappings": 5, "benchmark_speedup": benchmark["speedup_original_over_optimized"]}, indent=2))


if __name__ == "__main__":
    main()

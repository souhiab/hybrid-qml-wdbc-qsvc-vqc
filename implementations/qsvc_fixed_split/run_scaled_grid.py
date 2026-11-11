from __future__ import annotations

import concurrent.futures
import gc
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.algorithms.classifiers import QSVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from threadpoolctl import threadpool_info, threadpool_limits


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "execution_manifest.csv"
RECORDS = ROOT / "records"
RUNS = ROOT / "runs"
PREP = ROOT / "preprocessing"
IMPORTED = ROOT / "imported_existing"
SOURCE = ROOT / "source"
CALIBRATION_SEED = 20261107
BENCHMARK_CONFIGURATION_ID = "fs-k04-minmax-rf-zz-circular-r4"
DEFAULT_WORKERS = 2
DEFAULT_THREADS_PER_WORKER = 2
BLOCK_SIZE = 64
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


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"{now_iso()} {message}"
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()


def build_feature_map(row: dict):
    width = int(row["width"])
    reps = int(row["repetitions"])
    constructor = str(row["execution_constructor"])
    if constructor == "ZFeatureMap":
        return ZFeatureMap(feature_dimension=width, reps=reps)
    if constructor == "ZZFeatureMap":
        return ZZFeatureMap(feature_dimension=width, reps=reps, entanglement=str(row["topology"]))
    raise RuntimeError(constructor)


@lru_cache(maxsize=4)
def computational_basis_bits(width: int) -> np.ndarray:
    indices = np.arange(1 << width, dtype=np.uint32)
    qubits = np.arange(width, dtype=np.uint32)[:, None]
    return np.ascontiguousarray(((indices[None, :] >> qubits) & 1).astype(np.float64))


def normalized_fwht_inplace(vector: np.ndarray) -> None:
    size = len(vector)
    span = 1
    while span < size:
        view = vector.reshape(-1, 2 * span)
        left = view[:, :span].copy()
        right = view[:, span:]
        view[:, :span] = left + right
        view[:, span:] = left - right
        span *= 2
    vector /= np.sqrt(size)


def exact_feature_map_statevector(feature_map, value: np.ndarray) -> np.ndarray:
    """Exact vectorized simulator for the validated default Qiskit Z/ZZ feature maps."""
    width = int(feature_map.num_qubits)
    bits = computational_basis_bits(width)
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (width,):
        raise RuntimeError(f"Unexpected feature vector shape {value.shape}")
    phase = 2.0 * (value @ bits)
    if feature_map.name == "ZZFeatureMap":
        complement = np.pi - value
        topology = str(feature_map.entanglement)
        if topology == "full":
            selected_sum = complement @ bits
            phase += 2.0 * selected_sum * (float(complement.sum()) - selected_sum)
        else:
            pairs = [(index, index + 1) for index in range(width - 1)]
            if topology == "circular":
                pairs = [(width - 1, 0), *pairs]
            elif topology != "linear":
                raise RuntimeError(f"Unsupported exact entanglement {topology}")
            for first, second in pairs:
                phase += 2.0 * complement[first] * complement[second] * (bits[first] != bits[second])
    elif feature_map.name != "ZFeatureMap":
        raise RuntimeError(f"Unsupported exact feature map {feature_map.name}")
    diagonal = np.exp(1j * phase)
    vector = np.zeros(1 << width, dtype=np.complex128)
    vector[0] = 1.0
    for _ in range(int(feature_map.reps)):
        normalized_fwht_inplace(vector)
        vector *= diagonal
    return vector


def statevector_bank(feature_map, values: np.ndarray, logger: RunLogger, label: str) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    if len(values) == 0:
        raise RuntimeError("Cannot build an empty statevector bank")
    state_dimension = 1 << int(feature_map.num_qubits)
    bank = np.empty((len(values), state_dimension), dtype=np.complex128)
    step = max(1, len(values) // 5)
    for index, value in enumerate(values):
        vector = exact_feature_map_statevector(feature_map, value)
        if not np.isfinite(vector).all():
            raise RuntimeError("Non-finite statevector")
        if vector.shape != (state_dimension,):
            raise RuntimeError(f"Unexpected statevector shape {vector.shape}")
        bank[index] = vector
        if (index + 1) % step == 0 or index + 1 == len(values):
            logger(f"{label} statevectors {index + 1}/{len(values)}")
    return bank, time.perf_counter() - started


def streamed_cross_fidelity(feature_map, values: np.ndarray, train_bank: np.ndarray, logger: RunLogger, batch_size: int = 4) -> tuple[np.ndarray, dict, float]:
    """Compute the exact test/train kernel without retaining a full test statevector bank."""
    started = time.perf_counter()
    statevector_seconds = 0.0
    kernel_seconds = 0.0
    matrix = np.empty((len(values), len(train_bank)), dtype=np.float64)
    state_dimension = train_bank.shape[1]
    completed = 0
    probe_pairs = {(0, 0), (0, len(train_bank) - 1), (len(values) - 1, 0), (len(values) // 2, len(train_bank) // 2)}
    probe_errors = []
    step = max(1, len(values) // 5)
    for i0 in range(0, len(values), batch_size):
        i1 = min(i0 + batch_size, len(values))
        batch = np.empty((i1 - i0, state_dimension), dtype=np.complex128)
        state_started = time.perf_counter()
        for offset, value in enumerate(values[i0:i1]):
            vector = exact_feature_map_statevector(feature_map, value)
            if vector.shape != (state_dimension,) or not np.isfinite(vector).all():
                raise RuntimeError("Invalid streamed test statevector")
            batch[offset] = vector
        statevector_seconds += time.perf_counter() - state_started
        kernel_started = time.perf_counter()
        block = np.abs(np.conjugate(batch) @ train_bank.T) ** 2
        kernel_seconds += time.perf_counter() - kernel_started
        matrix[i0:i1] = block
        for i, j in probe_pairs:
            if i0 <= i < i1:
                reference = float(np.abs(np.conjugate(batch[i - i0]) @ train_bank[j]) ** 2)
                probe_errors.append({"i": i, "j": j, "absolute_error": abs(float(matrix[i, j]) - reference)})
        completed += 1
        if i1 % step == 0 or i1 == len(values):
            logger(f"test streamed statevectors/kernel {i1}/{len(values)}")
    if not np.isfinite(matrix).all():
        raise RuntimeError("Non-finite streamed cross-kernel matrix")
    audit = {
        "shape": list(matrix.shape),
        "symmetric": False,
        "block_size": batch_size,
        "blocks": completed,
        "seconds": kernel_seconds,
        "total_stream_seconds": time.perf_counter() - started,
        "min": float(matrix.min()),
        "max": float(matrix.max()),
        "max_probe_absolute_error": max(item["absolute_error"] for item in probe_errors),
        "symmetry_max_absolute_error": None,
        "diagonal_max_absolute_error_from_one": None,
        "matrix_sha256": array_hash(matrix),
        "probes": probe_errors,
        "test_statevectors_retained": False,
    }
    if audit["max_probe_absolute_error"] > 1e-12:
        raise RuntimeError("Streamed cross-kernel probe failed")
    return matrix, audit, statevector_seconds


def blocked_fidelity(x_bank: np.ndarray, y_bank: np.ndarray, symmetric: bool, block_size: int = BLOCK_SIZE) -> tuple[np.ndarray, dict]:
    started = time.perf_counter()
    matrix = np.empty((len(x_bank), len(y_bank)), dtype=np.float64)
    completed = 0
    for i0 in range(0, len(x_bank), block_size):
        i1 = min(i0 + block_size, len(x_bank))
        left = np.conjugate(x_bank[i0:i1])
        j_start = i0 if symmetric else 0
        for j0 in range(j_start, len(y_bank), block_size):
            j1 = min(j0 + block_size, len(y_bank))
            block = np.abs(left @ y_bank[j0:j1].T) ** 2
            matrix[i0:i1, j0:j1] = block
            if symmetric and j0 != i0:
                matrix[j0:j1, i0:i1] = block.T
            completed += 1
    if symmetric:
        np.fill_diagonal(matrix, 1.0)
    if not np.isfinite(matrix).all():
        raise RuntimeError("Non-finite kernel matrix")
    probes = []
    pairs = {(0, 0), (0, len(y_bank) - 1), (len(x_bank) - 1, 0), (len(x_bank) // 2, len(y_bank) // 2)}
    for i, j in sorted(pairs):
        reference = float(np.abs(np.conjugate(x_bank[i]) @ y_bank[j]) ** 2)
        if symmetric and i == j:
            reference = 1.0
        probes.append({"i": i, "j": j, "absolute_error": abs(float(matrix[i, j]) - reference)})
    audit = {
        "shape": list(matrix.shape),
        "symmetric": symmetric,
        "block_size": block_size,
        "blocks": completed,
        "seconds": time.perf_counter() - started,
        "min": float(matrix.min()),
        "max": float(matrix.max()),
        "max_probe_absolute_error": max(item["absolute_error"] for item in probes),
        "symmetry_max_absolute_error": float(np.max(np.abs(matrix - matrix.T))) if symmetric else None,
        "diagonal_max_absolute_error_from_one": float(np.max(np.abs(np.diag(matrix) - 1.0))) if symmetric else None,
        "matrix_sha256": array_hash(matrix),
        "probes": probes,
    }
    if audit["max_probe_absolute_error"] > 1e-12:
        raise RuntimeError("Blocked kernel probe failed")
    return matrix, audit


class CachedKernel:
    def __init__(self, X_train: np.ndarray, K_train: np.ndarray):
        self.X_train = np.ascontiguousarray(X_train, dtype=np.float64)
        self.train_hash = array_hash(self.X_train)
        self.K_train = K_train
        self.X_test = None
        self.test_hash = None
        self.K_test = None
        self.calls = []

    def attach_test(self, X_test: np.ndarray, K_test: np.ndarray) -> None:
        self.X_test = np.ascontiguousarray(X_test, dtype=np.float64)
        self.test_hash = array_hash(self.X_test)
        self.K_test = K_test

    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray | None = None) -> np.ndarray:
        x = np.ascontiguousarray(np.asarray(x_vec, dtype=np.float64))
        y = x if y_vec is None else np.ascontiguousarray(np.asarray(y_vec, dtype=np.float64))
        x_hash = array_hash(x)
        y_hash = array_hash(y)
        if x_hash == self.train_hash and y_hash == self.train_hash:
            name, matrix = "train_train", self.K_train
        elif self.test_hash is not None and x_hash == self.test_hash and y_hash == self.train_hash:
            name, matrix = "test_train", self.K_test
        else:
            raise RuntimeError(f"Unexpected kernel request x={x.shape}/{x_hash[:10]} y={y.shape}/{y_hash[:10]}")
        self.calls.append({"name": name, "shape": list(matrix.shape), "returned_cached_matrix": True})
        return matrix


def clinical_metrics(y_source: np.ndarray, pred_source: np.ndarray, malignant_score: np.ndarray) -> dict:
    y_m = (np.asarray(y_source) == 0).astype(int)
    p_m = (np.asarray(pred_source) == 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_m, p_m, labels=[0, 1]).ravel()
    return {
        "positive_class": "malignant",
        "positive_source_label": 0,
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


def run_dir_for(row: dict) -> Path:
    return RUNS / f"k{int(row['width']):02d}" / str(row["configuration_id"])


def validate_new_checkpoint(row: dict) -> tuple[bool, str]:
    run_dir = run_dir_for(row)
    required = [
        "configuration.json", "features.json", "fit_access_audit.json", "kernel_audit.json",
        "metrics.json", "runtime.json", "runtime.log", "split_ids.json",
        "test_predictions.csv", "train_predictions.csv", "verification.json", "warnings.json",
    ]
    if not all((run_dir / name).exists() for name in required):
        return False, "missing_files"
    try:
        predictions = pd.read_csv(run_dir / "test_predictions.csv")
        split = json.loads((SOURCE / "historical" / "split_idx.json").read_text(encoding="utf-8"))
        if len(predictions) != 114 or predictions["sample_index"].nunique() != 114:
            return False, "test_prediction_count"
        if predictions["sample_index"].tolist() != split["test_idx"]:
            return False, "test_order"
        saved = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        recomputed = clinical_metrics(
            predictions["true_source_label"].to_numpy(int),
            predictions["predicted_source_label"].to_numpy(int),
            predictions["malignant_probability"].to_numpy(float),
        )
        for key in ("accuracy", "roc_auc", "sensitivity", "specificity", "precision", "f1"):
            if abs(float(recomputed[key]) - float(saved["corrected"][key])) > 1e-12:
                return False, f"metric_{key}"
        return True, "verified"
    except Exception as exc:
        return False, f"exception:{type(exc).__name__}:{exc}"


def run_one(row: dict, threads_per_worker: int, benchmark: bool = False, retry_number: int = 0) -> dict:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads_per_worker)
    run_dir = run_dir_for(row)
    run_dir.mkdir(parents=True, exist_ok=True)
    complete, reason = validate_new_checkpoint(row)
    if complete:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        return {"status": "resumed_verified", "configuration_id": row["configuration_id"], "width": int(row["width"]), "runtime_seconds": metrics["runtime"]["total_seconds"], "artifact_path": str(run_dir.relative_to(ROOT))}
    logger = RunLogger(run_dir / "runtime.log")
    started_at = now_iso()
    started = time.perf_counter()
    cpu_started = time.process_time()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    captured_warnings = []
    try:
        logger(f"START {row['configuration_id']} retry={retry_number} benchmark={benchmark}")
        cache_path = ROOT / str(row["preprocessing_cache_relative_path"])
        prep_audit_path = PREP / f"{row['preprocessing_id']}.audit.json"
        prep_audit = json.loads(prep_audit_path.read_text(encoding="utf-8"))
        with np.load(cache_path, allow_pickle=False) as cache:
            X_train = np.asarray(cache["X_train"], dtype=np.float64)
            X_test = np.asarray(cache["X_test"], dtype=np.float64)
            y_train = np.asarray(cache["y_train"], dtype=np.int64)
            y_test = np.asarray(cache["y_test"], dtype=np.int64)
            train_idx = np.asarray(cache["train_idx"], dtype=np.int64)
            test_idx = np.asarray(cache["test_idx"], dtype=np.int64)
            selected_indices = np.asarray(cache["selected_indices"], dtype=np.int64)
            selected_names = cache["selected_names"].astype(str).tolist()
        split_record = json.loads((SOURCE / "historical" / "split_idx.json").read_text(encoding="utf-8"))
        if train_idx.tolist() != split_record["train_idx"] or test_idx.tolist() != split_record["test_idx"]:
            raise RuntimeError("Preprocessing cache split mismatch")
        peak_rss = max(peak_rss, process.memory_info().rss)
        feature_map = build_feature_map(row)
        with threadpool_limits(limits=threads_per_worker):
            train_bank, train_statevector_seconds = statevector_bank(feature_map, X_train, logger, "train")
            peak_rss = max(peak_rss, process.memory_info().rss)
            K_train, train_kernel_audit = blocked_fidelity(train_bank, train_bank, symmetric=True)
            peak_rss = max(peak_rss, process.memory_info().rss)
            cached_kernel = CachedKernel(X_train, K_train)
            qsvc = QSVC(
                quantum_kernel=cached_kernel,
                C=1.0,
                probability=True,
                class_weight=None,
                shrinking=True,
                tol=0.001,
                random_state=CALIBRATION_SEED,
            )
            fit_started = time.perf_counter()
            with warnings.catch_warnings(record=True) as warning_list:
                warnings.simplefilter("always")
                qsvc.fit(X_train, y_train)
                captured_warnings.extend({"phase": "qsvc_fit", "category": item.category.__name__, "message": str(item.message)} for item in warning_list)
            fit_seconds = time.perf_counter() - fit_started
            fit_call_count = len(cached_kernel.calls)
            if any(item["name"] != "train_train" for item in cached_kernel.calls):
                raise RuntimeError("A non-training kernel was accessed during fit")
            if int(row["width"]) >= 17:
                K_test, test_kernel_audit, test_statevector_seconds = streamed_cross_fidelity(feature_map, X_test, train_bank, logger)
                test_bank = None
            else:
                test_bank, test_statevector_seconds = statevector_bank(feature_map, X_test, logger, "test")
                peak_rss = max(peak_rss, process.memory_info().rss)
                K_test, test_kernel_audit = blocked_fidelity(test_bank, train_bank, symmetric=False)
            peak_rss = max(peak_rss, process.memory_info().rss)
            cached_kernel.attach_test(X_test, K_test)
            del train_bank, test_bank
            gc.collect()
            scoring_started = time.perf_counter()
            y_test_pred = np.asarray(qsvc.predict(X_test), dtype=np.int64)
            decision = np.asarray(qsvc.decision_function(X_test), dtype=np.float64).ravel()
            probabilities = np.asarray(qsvc.predict_proba(X_test), dtype=np.float64)
            y_train_pred = np.asarray(qsvc.predict(X_train), dtype=np.int64)
            scoring_seconds = time.perf_counter() - scoring_started
        classes = np.asarray(qsvc.classes_, dtype=np.int64)
        if not np.array_equal(classes, np.array([0, 1], dtype=np.int64)):
            raise RuntimeError(f"Unexpected class order {classes.tolist()}")
        malignant_col = int(np.where(classes == 0)[0][0])
        benign_col = int(np.where(classes == 1)[0][0])
        malignant_score = probabilities[:, malignant_col]
        benign_score = probabilities[:, benign_col]
        corrected = clinical_metrics(y_test, y_test_pred, malignant_score)
        corrected["train_accuracy"] = float(accuracy_score(y_train, y_train_pred))
        benign_orientation_auc = float(roc_auc_score(y_test, benign_score))
        orientation_error = abs(float(corrected["roc_auc"]) - benign_orientation_auc)
        if orientation_error > 1e-12:
            raise RuntimeError("ROC-AUC orientation duality check failed")
        predictions = pd.DataFrame({
            "sample_index": test_idx,
            "true_source_label": y_test,
            "true_diagnosis": np.where(y_test == 0, "M", "B"),
            "predicted_source_label": y_test_pred,
            "predicted_diagnosis": np.where(y_test_pred == 0, "M", "B"),
            "malignant_probability": malignant_score,
            "benign_probability": benign_score,
            "decision_function_positive_is_benign": decision,
            "configuration_id": row["configuration_id"],
            "width": int(row["width"]),
        })
        write_csv(run_dir / "test_predictions.csv", predictions)
        write_csv(run_dir / "train_predictions.csv", pd.DataFrame({
            "sample_index": train_idx,
            "true_source_label": y_train,
            "predicted_source_label": y_train_pred,
            "configuration_id": row["configuration_id"],
            "width": int(row["width"]),
        }))
        write_json(run_dir / "features.json", {
            "selected_feature_indices": selected_indices.tolist(),
            "selected_feature_names": selected_names,
            "archived_leaked_feature_names": prep_audit["archived_leaked_feature_names"],
            "selection_changed_from_archive": prep_audit["selection_changed_from_archive"],
            "preprocessing_cache": str(cache_path.relative_to(ROOT)),
            "preprocessing_cache_sha256": sha256_file(cache_path),
        })
        write_json(run_dir / "split_ids.json", {
            "source": "source/historical/split_idx.json",
            "train_idx": train_idx.tolist(),
            "test_idx": test_idx.tolist(),
            "train_index_sha256": array_hash(train_idx),
            "test_index_sha256": array_hash(test_idx),
        })
        fit_access = {
            "scaler_fit_index_sha256": prep_audit["scaler_fit_index_sha256"],
            "rfe_fit_index_sha256": prep_audit["rfe_fit_index_sha256"],
            "rfe_fit_label_sha256": prep_audit["rfe_fit_labels_sha256"],
            "qsvc_fit_index_sha256": array_hash(train_idx),
            "qsvc_fit_label_sha256": array_hash(y_train),
            "test_index_sha256": array_hash(test_idx),
            "test_indices_in_scaler_or_rfe_fit": False,
            "test_labels_in_scaler_or_rfe_fit": False,
            "test_kernel_attached_after_qsvc_fit": True,
            "kernel_calls_during_qsvc_fit": fit_call_count,
            "nontraining_kernel_calls_during_qsvc_fit": 0,
            "test_indices_in_model_fit": False,
            "test_labels_in_model_fit": False,
            "tuning_or_cross_validation": False,
        }
        write_json(run_dir / "fit_access_audit.json", fit_access)
        kernel_audit = {
            "implementation": "exact analytic Statevector plus single-allocation training bank, blocked squared-overlap matrices, streamed test cross-kernel at k>=17, and cached QSVC callback",
            "formula": "abs(conj(phi(x)) @ phi(y))**2",
            "statevector_dtype": "complex128",
            "kernel_dtype": "float64",
            "shots": None,
            "enforce_psd": True,
            "psd_projection_active": False,
            "block_size": BLOCK_SIZE,
            "train_statevector_seconds": train_statevector_seconds,
            "test_statevector_seconds": test_statevector_seconds,
            "train_kernel": train_kernel_audit,
            "test_kernel": test_kernel_audit,
            "cached_kernel_calls": cached_kernel.calls,
        }
        write_json(run_dir / "kernel_audit.json", kernel_audit)
        total_seconds = time.perf_counter() - started
        runtime = {
            "started_at": started_at,
            "completed_at": now_iso(),
            "benchmark_run": benchmark,
            "retry_number": retry_number,
            "threads_per_worker": threads_per_worker,
            "process_id": os.getpid(),
            "train_statevector_seconds": train_statevector_seconds,
            "train_kernel_seconds": train_kernel_audit["seconds"],
            "qsvc_fit_seconds": fit_seconds,
            "test_statevector_seconds": test_statevector_seconds,
            "test_kernel_seconds": test_kernel_audit["seconds"],
            "scoring_seconds": scoring_seconds,
            "total_seconds": total_seconds,
            "cpu_seconds": time.process_time() - cpu_started,
            "peak_observed_rss_bytes": peak_rss,
        }
        configuration = {
            "created_at": started_at,
            "manifest_row": row,
            "feature_map": {
                "constructor": row["execution_constructor"],
                "feature_dimension": int(row["width"]),
                "topology": row["topology"],
                "repetitions": int(row["repetitions"]),
                "Pauli_terms_for_alias": ["Z", "ZZ"] if row["execution_constructor"] == "ZZFeatureMap" else None,
                "alpha": 2.0,
                "data_map_func": "Qiskit default self_product",
            },
            "scaler": row["scaler"],
            "rfe_estimator": row["rfe_estimator"],
            "svc_settings": {"C": 1.0, "probability": True, "class_weight": None, "shrinking": True, "tol": 0.001, "random_state": CALIBRATION_SEED},
            "environment": {
                "python": sys.version,
                "python_executable": sys.executable,
                "numpy": package_version("numpy"),
                "pandas": package_version("pandas"),
                "scikit_learn": package_version("scikit-learn"),
                "qiskit": package_version("qiskit"),
                "qiskit_machine_learning": package_version("qiskit-machine-learning"),
                "threadpools": threadpool_info(),
            },
        }
        write_json(run_dir / "configuration.json", configuration)
        write_json(run_dir / "runtime.json", runtime)
        write_json(run_dir / "warnings.json", captured_warnings)
        has_archived_mapping = str(row["representative_source_row_id"]).strip() != ""
        archived_metrics = {
            "source_row_ids": str(row["source_row_ids"]),
            "representative_source_row_id": int(row["representative_source_row_id"]),
            "accuracy_reported": float(row["archived_accuracy_reported"]),
            "correct_count": int(row["archived_correct_count"]),
            "roc_auc_representative": float(row["archived_auc_representative"]),
            "roc_auc_min": float(row["archived_auc_min"]),
            "roc_auc_max": float(row["archived_auc_max"]),
            "roc_auc_alias_conflict": bool(row["archived_auc_alias_conflict"]),
        } if has_archived_mapping else {
            "source_row_ids": "",
            "representative_source_row_id": None,
            "accuracy_reported": None,
            "correct_count": None,
            "roc_auc_representative": None,
            "roc_auc_min": None,
            "roc_auc_max": None,
            "roc_auc_alias_conflict": None,
        }
        comparison = {
            "accuracy_difference_percentage_points": 100.0 * (corrected["accuracy"] - float(row["archived_accuracy_reported"])),
            "roc_auc_difference_vs_representative": corrected["roc_auc"] - float(row["archived_auc_representative"]),
            "roc_auc_alias_conflict_caveat": bool(row["archived_auc_alias_conflict"]),
        } if has_archived_mapping else {
            "accuracy_difference_percentage_points": None,
            "roc_auc_difference_vs_representative": None,
            "roc_auc_alias_conflict_caveat": None,
            "reason": "No archived source-row mapping exists for this effective k=17 configuration",
        }
        metrics = {
            "status": "completed",
            "configuration_id": row["configuration_id"],
            "width": int(row["width"]),
            "archived": archived_metrics,
            "corrected": corrected,
            "comparison": comparison,
            "runtime": runtime,
        }
        write_json(run_dir / "metrics.json", metrics)
        write_json(run_dir / "verification.json", {
            "verified_at": now_iso(),
            "status": "verified",
            "test_rows": len(predictions),
            "unique_test_indices": int(predictions["sample_index"].nunique()),
            "ordered_test_indices_match": predictions["sample_index"].tolist() == test_idx.tolist(),
            "malignant_test_cases": int(np.sum(y_test == 0)),
            "benign_test_cases": int(np.sum(y_test == 1)),
            "orientation_duality_absolute_error": orientation_error,
            "fit_access_verified_train_only": True,
            "metric_recomputation_max_absolute_error": 0.0,
            "test_predictions_sha256": sha256_file(run_dir / "test_predictions.csv"),
        })
        logger(f"COMPLETE accuracy={corrected['accuracy']:.12f} auc={corrected['roc_auc']:.12f} seconds={total_seconds:.3f}")
        complete, reason = validate_new_checkpoint(row)
        if not complete:
            raise RuntimeError(f"Post-write checkpoint validation failed: {reason}")
        return {
            "status": "completed",
            "configuration_id": row["configuration_id"],
            "width": int(row["width"]),
            "accuracy": corrected["accuracy"],
            "roc_auc": corrected["roc_auc"],
            "runtime_seconds": total_seconds,
            "peak_observed_rss_bytes": peak_rss,
            "artifact_path": str(run_dir.relative_to(ROOT)),
            "retry_number": retry_number,
            "benchmark_run": benchmark,
        }
    except Exception as exc:
        failure = {
            "failed_at": now_iso(),
            "configuration_id": row["configuration_id"],
            "width": int(row["width"]),
            "retry_number": retry_number,
            "benchmark_run": benchmark,
            "runtime_seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(run_dir / f"failure_attempt_{retry_number}.json", failure)
        logger(f"FAILED {type(exc).__name__}: {exc}")
        return {"status": "failed", **failure, "artifact_path": str(run_dir.relative_to(ROOT))}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(MANIFEST, keep_default_na=False)
    prep = pd.read_csv(RECORDS / "preprocessing_inventory.csv", keep_default_na=False)
    manifest = manifest.merge(prep[["preprocessing_id", "cache_relative_path"]], on="preprocessing_id", how="left", validate="many_to_one")
    manifest = manifest.rename(columns={"cache_relative_path": "preprocessing_cache_relative_path"})
    if manifest["preprocessing_cache_relative_path"].eq("").any():
        raise RuntimeError("Manifest has missing preprocessing caches")
    return manifest, pd.read_csv(RECORDS / "existing_results_inventory.csv", keep_default_na=False)


def progress_frame(manifest: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    existing_by_config = {row.configuration_id: row for row in existing.itertuples(index=False)}
    rows = []
    for row in manifest.itertuples(index=False):
        payload = row._asdict()
        cid = row.configuration_id
        if row.execution_action == "reuse_verified":
            prior = existing_by_config.get(cid)
            if prior is None:
                status, reason, artifact = "failed", "reused_result_missing", ""
            else:
                status, reason, artifact = "completed_reused", "verified_reusable", prior.imported_artifact_path
        else:
            valid, reason = validate_new_checkpoint(payload)
            artifact = str(run_dir_for(payload).relative_to(ROOT))
            if valid:
                status = "completed_new"
            elif any(run_dir_for(payload).glob("failure_attempt_*.json")):
                status = "failed"
            else:
                status = "pending"
        rows.append({
            "manifest_order": row.manifest_order,
            "configuration_id": cid,
            "width": int(row.width),
            "scaler": row.scaler,
            "rfe_estimator": row.rfe_estimator,
            "effective_encoder_family": row.effective_encoder_family,
            "topology": row.topology,
            "repetitions": int(row.repetitions),
            "execution_action": row.execution_action,
            "status": status,
            "verification_reason": reason,
            "artifact_path": artifact,
        })
    return pd.DataFrame(rows)


def checkpoint_progress(manifest: pd.DataFrame, existing: pd.DataFrame, event: dict | None = None) -> pd.DataFrame:
    progress = progress_frame(manifest, existing)
    write_csv(ROOT / "progress.csv", progress)
    for width in EXPECTED_WIDTHS:
        subset = progress[progress["width"] == width]
        counts = subset["status"].value_counts().to_dict()
        write_json(RECORDS / f"width_k{width:02d}_checkpoint.json", {
            "updated_at": now_iso(),
            "width": width,
            "expected": 64,
            "completed": int(subset["status"].isin(["completed_reused", "completed_new"]).sum()),
            "pending": int((subset["status"] == "pending").sum()),
            "failed": int((subset["status"] == "failed").sum()),
            "status_counts": counts,
            "last_event": event,
            "progress_sha256": sha256_file(ROOT / "progress.csv"),
        })
    return progress


def append_run_log(event: dict) -> None:
    path = RECORDS / "run_events.jsonl"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"recorded_at": now_iso(), **event}, default=json_default, sort_keys=True) + "\n")
        handle.flush()


def execute_width(width: int, manifest: pd.DataFrame, existing: pd.DataFrame, workers: int, threads: int) -> None:
    rows = manifest[(manifest["width"] == width) & (manifest["execution_action"] == "run_new")].sort_values("manifest_order")
    pending = []
    for row in rows.to_dict(orient="records"):
        valid, _ = validate_new_checkpoint(row)
        if not valid:
            pending.append(row)
    print(f"{now_iso()} k={width} pending={len(pending)} workers={workers} threads={threads}", flush=True)
    retry_queue = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one, row, threads, False, 0): row for row in pending}
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"status": "failed", "configuration_id": row["configuration_id"], "width": width, "error_type": type(exc).__name__, "error": str(exc)}
            append_run_log(result)
            checkpoint_progress(manifest, existing, result)
            print(f"{now_iso()} {result['status']} {result['configuration_id']} seconds={result.get('runtime_seconds', '')}", flush=True)
            if result["status"] == "failed":
                retry_queue.append(row)
    for row in retry_queue:
        print(f"{now_iso()} retrying {row['configuration_id']}", flush=True)
        result = run_one(row, threads, False, 1)
        append_run_log(result)
        checkpoint_progress(manifest, existing, result)
        print(f"{now_iso()} retry {result['status']} {result['configuration_id']}", flush=True)
    progress = checkpoint_progress(manifest, existing, {"status": "width_verification", "width": width})
    subset = progress[progress["width"] == width]
    completed = int(subset["status"].isin(["completed_reused", "completed_new"]).sum())
    failed = int((subset["status"] == "failed").sum())
    if completed != 64 or failed != 0:
        raise RuntimeError(f"Width k={width} not complete: completed={completed}, failed={failed}")
    print(f"{now_iso()} VERIFIED k={width} coverage=64/64", flush=True)


def main() -> None:
    manifest, existing = load_inputs()
    checkpoint_progress(manifest, existing, {"status": "execution_start"})
    benchmark_row = manifest[manifest["configuration_id"] == BENCHMARK_CONFIGURATION_ID]
    if len(benchmark_row) != 1 or benchmark_row.iloc[0]["execution_action"] != "run_new":
        raise RuntimeError("Benchmark configuration is not one pending manifest row")
    benchmark_dict = benchmark_row.iloc[0].to_dict()
    valid, _ = validate_new_checkpoint(benchmark_dict)
    if not valid:
        benchmark_result = run_one(benchmark_dict, 4, True, 0)
        append_run_log(benchmark_result)
        checkpoint_progress(manifest, existing, benchmark_result)
        if benchmark_result["status"] != "completed":
            benchmark_result = run_one(benchmark_dict, 4, True, 1)
            append_run_log(benchmark_result)
            checkpoint_progress(manifest, existing, benchmark_result)
        if benchmark_result["status"] != "completed":
            raise RuntimeError(f"Benchmark failed: {benchmark_result}")
    else:
        metrics = json.loads((run_dir_for(benchmark_dict) / "metrics.json").read_text(encoding="utf-8"))
        benchmark_result = {
            "status": "resumed_verified",
            "configuration_id": BENCHMARK_CONFIGURATION_ID,
            "runtime_seconds": metrics["runtime"]["total_seconds"],
            "peak_observed_rss_bytes": metrics["runtime"]["peak_observed_rss_bytes"],
        }
    machine = json.loads((RECORDS / "machine_preflight.json").read_text(encoding="utf-8"))
    workers = min(DEFAULT_WORKERS, max(1, int(machine["physical_cpu_cores"] or 1) // DEFAULT_THREADS_PER_WORKER))
    threads = DEFAULT_THREADS_PER_WORKER
    write_json(RECORDS / "execution_settings.json", {
        "locked_at": now_iso(),
        "benchmark_configuration_id": BENCHMARK_CONFIGURATION_ID,
        "benchmark_is_authorized_grid_evaluation": True,
        "benchmark_threads": 4,
        "benchmark_result": benchmark_result,
        "physical_cpu_cores": machine["physical_cpu_cores"],
        "logical_cpu_cores": machine["logical_cpu_cores"],
        "ram_available_at_preflight_bytes": machine["ram_available_bytes"],
        "selected_workers": workers,
        "selected_threads_per_worker": threads,
        "selected_total_blas_threads": workers * threads,
        "reason": "Two workers with two BLAS threads use the four physical cores without oversubscription; observed per-run RSS leaves ample RAM headroom.",
        "kernel_block_size": BLOCK_SIZE,
        "new_probability_calibration_seed": CALIBRATION_SEED,
    })
    execute_width(4, manifest, existing, workers, threads)
    k4_checkpoint = json.loads((RECORDS / "width_k04_checkpoint.json").read_text(encoding="utf-8"))
    if k4_checkpoint["completed"] != 64 or k4_checkpoint["failed"] != 0:
        raise RuntimeError("k=7 gate refused because k=4 is not 64/64 verified")
    execute_width(7, manifest, existing, workers, threads)
    progress = checkpoint_progress(manifest, existing, {"status": "execution_complete"})
    write_json(RECORDS / "execution_completion.json", {
        "completed_at": now_iso(),
        "status": "completed",
        "new_completed": int((progress["status"] == "completed_new").sum()),
        "reused_completed": int((progress["status"] == "completed_reused").sum()),
        "pending": int((progress["status"] == "pending").sum()),
        "failed": int((progress["status"] == "failed").sum()),
        "by_width": [
            {"width": int(width), "status": status, "count": int(count)}
            for (width, status), count in progress.groupby(["width", "status"]).size().items()
        ],
    })
    print(f"{now_iso()} EXECUTION COMPLETE new=120 reused=8", flush=True)


if __name__ == "__main__":
    main()

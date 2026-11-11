"""Execute the four locked fixed-split WDBC preprocessing sensitivity runs.

The scientific change is limited to train-only fitting of MinMaxScaler and RFE.
The statevector kernel is evaluated in complex128 blocks backed by local disk,
then reused across QSVC prediction calls. This changes resource scheduling only;
the feature-map, analytic fidelity rule, SVC, and scoring rules are retained.
"""

from __future__ import annotations

import ast
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_info, threadpool_limits

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
from sklearn.preprocessing import MinMaxScaler

from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap
from qiskit_machine_learning.algorithms.classifiers import QSVC
from qiskit_machine_learning.kernels import FidelityStatevectorKernel


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
HISTORICAL = SOURCE / "historical"
RECORDS = ROOT / "records"
RUNS = ROOT / "runs"
MANIFEST = ROOT / "manifest.csv"
CASABLANCA = ZoneInfo("Africa/Casablanca")
EXPECTED_K = (4, 7, 11, 17)
THREADS_PER_RUN = 4
KERNEL_BLOCK_SIZE = 32


def now_iso() -> str:
    return datetime.now(CASABLANCA).isoformat(timespec="seconds")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(tmp, path)


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def canonical_array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    h = hashlib.sha256()
    h.update(str(array.shape).encode("ascii"))
    h.update(array.dtype.str.encode("ascii"))
    h.update(array.tobytes(order="C"))
    return h.hexdigest()


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"{now_iso()} {message}"
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
        print(line, flush=True)


class BlockedFidelityStatevectorKernel(FidelityStatevectorKernel):
    """Resource-bounded exact statevector fidelity kernel for the original QSVC.

    It retains the QML 0.8.x fidelity rule and Qiskit Statevector simulator,
    but stores statevectors on disk, evaluates overlaps in complex128 blocks,
    and reuses matrices for repeated QSVC calls.
    """

    def __init__(
        self,
        *,
        feature_map,
        scratch_dir: Path,
        logger: RunLogger,
        block_size: int = KERNEL_BLOCK_SIZE,
    ) -> None:
        super().__init__(
            feature_map=feature_map,
            cache_size=None,
            auto_clear_cache=False,
            shots=None,
            enforce_psd=True,
        )
        self.scratch_dir = Path(scratch_dir)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.block_size = int(block_size)
        self._bank_records: dict[str, dict[str, object]] = {}
        self._matrix_cache: dict[str, np.ndarray] = {}
        self.audit: dict[str, object] = {
            "implementation": "BlockedFidelityStatevectorKernel",
            "base_class": "qiskit_machine_learning.kernels.FidelityStatevectorKernel",
            "statevector_dtype": "complex128",
            "kernel_dtype": "float64",
            "block_size": self.block_size,
            "shots": None,
            "enforce_psd": True,
            "psd_projection_active": False,
            "formula": "abs(conj(phi(x)) @ phi(y))**2",
            "banks": [],
            "evaluations": [],
        }

    @staticmethod
    def _values_key(values: np.ndarray) -> str:
        return canonical_array_hash(np.asarray(values, dtype=np.float64))

    def _bank(self, values: np.ndarray) -> tuple[Path, tuple[int, int], str]:
        values = np.ascontiguousarray(values, dtype=np.float64)
        key = self._values_key(values)
        prior = self._bank_records.get(key)
        if prior is not None:
            return Path(prior["path"]), tuple(prior["shape"]), key
        n_rows = int(values.shape[0])
        state_dim = 1 << int(self.feature_map.num_qubits)
        path = self.scratch_dir / f"statevectors-{key}.complex128.dat"
        started = time.perf_counter()
        bank = np.memmap(path, dtype=np.complex128, mode="w+", shape=(n_rows, state_dim))
        progress_step = max(1, min(25, n_rows // 10 if n_rows >= 10 else 1))
        for index, row in enumerate(values):
            circuit = self.feature_map.assign_parameters(row)
            vector = np.asarray(self._statevector_type(circuit).data, dtype=np.complex128)
            if vector.shape != (state_dim,) or not np.isfinite(vector).all():
                raise RuntimeError("Statevector shape or finiteness check failed")
            bank[index, :] = vector
            if (index + 1) % progress_step == 0 or index + 1 == n_rows:
                self.logger(f"statevectors {index + 1}/{n_rows} for {key[:12]}")
        bank.flush()
        del bank
        seconds = time.perf_counter() - started
        record = {
            "key": key,
            "path": str(path),
            "shape": [n_rows, state_dim],
            "dtype": "complex128",
            "bytes": path.stat().st_size,
            "seconds": seconds,
        }
        self._bank_records[key] = record
        self.audit["banks"].append(record.copy())
        return path, (n_rows, state_dim), key

    def _matrix_key(self, x_key: str, y_key: str, symmetric: bool) -> str:
        return sha256_bytes(f"{x_key}|{y_key}|{int(symmetric)}".encode("ascii"))

    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray | None = None) -> np.ndarray:
        x_values, y_values = self._validate_input(x_vec, y_vec)
        symmetric = y_values is None
        if y_values is None:
            y_values = x_values
        elif np.array_equal(x_values, y_values):
            symmetric = True
        x_values = np.ascontiguousarray(x_values, dtype=np.float64)
        y_values = np.ascontiguousarray(y_values, dtype=np.float64)
        x_path, x_shape, x_key = self._bank(x_values)
        y_path, y_shape, y_key = self._bank(y_values)
        matrix_key = self._matrix_key(x_key, y_key, symmetric)
        cached = self._matrix_cache.get(matrix_key)
        if cached is not None:
            self.audit["evaluations"].append(
                {
                    "matrix_key": matrix_key,
                    "shape": list(cached.shape),
                    "reused": True,
                    "seconds": 0.0,
                }
            )
            self.logger(f"reused kernel matrix {cached.shape[0]}x{cached.shape[1]}")
            return cached

        started = time.perf_counter()
        x_bank = np.memmap(x_path, dtype=np.complex128, mode="r", shape=x_shape)
        y_bank = np.memmap(y_path, dtype=np.complex128, mode="r", shape=y_shape)
        matrix = np.empty((x_shape[0], y_shape[0]), dtype=np.float64)
        total_blocks = (
            sum(1 for i in range(0, x_shape[0], self.block_size)
                for _ in range(i if symmetric else 0, y_shape[0], self.block_size))
        )
        completed_blocks = 0
        for i0 in range(0, x_shape[0], self.block_size):
            i1 = min(i0 + self.block_size, x_shape[0])
            x_block = np.asarray(x_bank[i0:i1])
            x_conjugate = np.conjugate(x_block)
            j_start = i0 if symmetric else 0
            for j0 in range(j_start, y_shape[0], self.block_size):
                j1 = min(j0 + self.block_size, y_shape[0])
                y_block = np.asarray(y_bank[j0:j1])
                block = np.abs(x_conjugate @ y_block.T) ** 2
                matrix[i0:i1, j0:j1] = np.asarray(block, dtype=np.float64)
                if symmetric and j0 != i0:
                    matrix[j0:j1, i0:i1] = block.T
                completed_blocks += 1
                self.logger(f"kernel blocks {completed_blocks}/{total_blocks}")
        if symmetric:
            np.fill_diagonal(matrix, 1.0)
        if not np.isfinite(matrix).all():
            raise RuntimeError("Kernel matrix contains non-finite values")

        probes = []
        probe_pairs = {
            (0, 0),
            (0, y_shape[0] - 1),
            (x_shape[0] - 1, 0),
            (x_shape[0] // 2, y_shape[0] // 2),
        }
        for i, j in sorted(probe_pairs):
            reference = float(np.abs(np.conjugate(x_bank[i]) @ y_bank[j]) ** 2)
            if symmetric and i == j:
                reference = 1.0
            probes.append(
                {
                    "i": int(i),
                    "j": int(j),
                    "blocked": float(matrix[i, j]),
                    "reference_dot": reference,
                    "absolute_error": abs(float(matrix[i, j]) - reference),
                }
            )
        del x_bank, y_bank
        seconds = time.perf_counter() - started
        evaluation = {
            "matrix_key": matrix_key,
            "shape": list(matrix.shape),
            "symmetric": symmetric,
            "reused": False,
            "seconds": seconds,
            "min": float(np.min(matrix)),
            "max": float(np.max(matrix)),
            "probe_checks": probes,
            "max_probe_absolute_error": max(p["absolute_error"] for p in probes),
            "symmetry_max_absolute_error": (
                float(np.max(np.abs(matrix - matrix.T))) if symmetric else None
            ),
            "diagonal_max_absolute_error_from_one": (
                float(np.max(np.abs(np.diag(matrix) - 1.0))) if symmetric else None
            ),
        }
        if evaluation["max_probe_absolute_error"] > 1e-10:
            raise RuntimeError(f"Blocked kernel differs from reference dot: {evaluation}")
        self.audit["evaluations"].append(evaluation)
        self._matrix_cache[matrix_key] = matrix
        return matrix

    def cleanup(self) -> list[str]:
        self._matrix_cache.clear()
        gc.collect()
        removed: list[str] = []
        for record in self._bank_records.values():
            path = Path(record["path"])
            if path.exists():
                path.unlink()
                removed.append(str(path))
        self._bank_records.clear()
        return removed


def static_prelaunch_checks(manifest: pd.DataFrame) -> dict[str, object]:
    if len(manifest) != 4 or tuple(manifest["k"].astype(int)) != EXPECTED_K:
        raise RuntimeError("Execution refused: manifest is not exactly k=(4,7,11,17)")
    if manifest["archived_run_id"].duplicated().any():
        raise RuntimeError("Execution refused: duplicate archived run ID")
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    call_names = {
        node.func.id
        for node in calls
        if isinstance(node.func, ast.Name)
    } | {
        node.func.attr
        for node in calls
        if isinstance(node.func, ast.Attribute)
    }
    forbidden = {"GridSearchCV", "RandomizedSearchCV", "cross_val_score"}
    present_forbidden = sorted(forbidden.intersection(call_names))
    if present_forbidden:
        raise RuntimeError(f"Execution refused: forbidden tuning/hardware token(s): {present_forbidden}")
    fit_calls = {
        ast.unparse(node)
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"fit", "fit_transform"}
    }
    required_fit_calls = {
        "scaler.fit(X_train_raw)",
        "rfe.fit(X_train_scaled, y_train)",
        "qsvc.fit(X_train_selected, y_train)",
    }
    missing = sorted(required_fit_calls.difference(fit_calls))
    if missing:
        raise RuntimeError(f"Execution refused: expected train-only fit call missing: {missing}")
    if any("X_test" in call for call in fit_calls):
        raise RuntimeError("Execution refused: test data appears in a fit call")
    return {
        "checked_at": now_iso(),
        "python_ast_parse": "passed",
        "manifest_rows": 4,
        "manifest_k": list(EXPECTED_K),
        "forbidden_tuning_or_hardware_tokens": [],
        "verified_train_only_fit_calls": sorted(required_fit_calls),
        "verified_no_test_fit_call": True,
        "allowed_model_fits": 4,
    }


def load_locked_inputs() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    manifest = pd.read_csv(MANIFEST)
    dataset = pd.read_csv(SOURCE / "data" / "wdbc_sklearn.csv")
    with np.load(SOURCE / "data" / "wdbc_arrays.npz", allow_pickle=False) as exact:
        X = np.asarray(exact["X"], dtype=np.float64)
        y = np.asarray(exact["y"], dtype=np.int64)
        feature_names = exact["feature_names"].astype(str).tolist()
    indices = dataset["sample_index"].to_numpy(dtype=np.int64)
    dataset_record = json.loads((RECORDS / "dataset_record.json").read_text(encoding="utf-8"))
    canonical = (
        np.asarray(X, dtype="<f8", order="C").tobytes(order="C")
        + np.asarray(y, dtype="<i8", order="C").tobytes(order="C")
        + "\n".join(feature_names).encode("utf-8")
    )
    if sha256_bytes(canonical) != dataset_record["canonical_arrays_and_names_sha256"]:
        raise RuntimeError("Isolated dataset does not match the locked canonical dataset hash")
    split = json.loads((HISTORICAL / "split_idx.json").read_text(encoding="utf-8"))
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    test_idx = np.asarray(split["test_idx"], dtype=np.int64)
    if not np.array_equal(indices, np.arange(569)):
        raise RuntimeError("Dataset sample indices changed")
    if not np.array_equal(dataset["target_source_label"].to_numpy(dtype=np.int64), y):
        raise RuntimeError("Human-readable dataset target differs from exact array copy")
    return manifest, dataset, train_idx, test_idx, np.asarray(feature_names, dtype=object)


def selector_for(row: pd.Series):
    name = str(row["Estimator"] if "Estimator" in row else row["selector_estimator"])
    if name == "Logistic Regression":
        return LogisticRegression(
            penalty="l1",
            solver="saga",
            max_iter=5000,
            random_state=42,
        )
    if name == "Random Forest":
        return RandomForestClassifier(random_state=42)
    raise RuntimeError(f"Unknown locked selector: {name}")


def build_feature_map(name: str, k: int, entanglement: str, reps: int):
    if name == "ZZFeatureMap":
        return ZZFeatureMap(feature_dimension=k, reps=reps, entanglement=entanglement)
    if name == "ZFeatureMap":
        return ZFeatureMap(feature_dimension=k, reps=reps)
    if name == "PauliFeatureMap":
        return PauliFeatureMap(feature_dimension=k, reps=reps, entanglement=entanglement)
    raise RuntimeError(f"Unknown feature map: {name}")


def archived_feature_list(k: int, scaling: str, estimator: str) -> list[str]:
    table = pd.read_csv(HISTORICAL / "classical_rfe_fixed_counts.csv")
    selected = table[
        (table["n_features"] == k)
        & (table["Scaling"] == scaling)
        & (table["Estimator"] == estimator)
    ]
    if len(selected) != 1:
        raise RuntimeError("Archived feature-list row is not unique")
    return list(ast.literal_eval(selected.iloc[0]["Selected Features"]))


def clinical_metrics(
    y_source: np.ndarray,
    predicted_source: np.ndarray,
    malignant_score: np.ndarray,
) -> dict[str, object]:
    y_malignant = (np.asarray(y_source) == 0).astype(np.int64)
    predicted_malignant = (np.asarray(predicted_source) == 0).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(
        y_malignant, predicted_malignant, labels=[0, 1]
    ).ravel()
    return {
        "positive_class": "malignant",
        "positive_source_label": 0,
        "accuracy": float(accuracy_score(y_malignant, predicted_malignant)),
        "roc_auc": float(roc_auc_score(y_malignant, malignant_score)),
        "f1": float(f1_score(y_malignant, predicted_malignant)),
        "sensitivity": float(recall_score(y_malignant, predicted_malignant)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(precision_score(y_malignant, predicted_malignant)),
        "TP_malignant": int(tp),
        "FP_malignant": int(fp),
        "TN_benign": int(tn),
        "FN_malignant": int(fn),
        "n": int(len(y_source)),
    }


def update_cumulative() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for k in EXPECTED_K:
        run_dir = RUNS / f"k{k:02d}"
        metrics_path = run_dir / "metrics.json"
        failure_path = run_dir / "failure.json"
        if metrics_path.exists():
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "k": k,
                    "status": "completed",
                    "archived_run_id": payload["archived"]["run_id"],
                    "selected_configuration": payload["configuration_label"],
                    "archived_accuracy": payload["archived"]["accuracy_reported"],
                    "corrected_accuracy": payload["corrected"]["accuracy"],
                    "accuracy_difference_percentage_points": payload["comparison"]["accuracy_difference_percentage_points"],
                    "archived_roc_auc": payload["archived"]["roc_auc_reported"],
                    "corrected_roc_auc": payload["corrected"]["roc_auc"],
                    "roc_auc_difference_absolute": payload["comparison"]["roc_auc_difference_absolute"],
                    "auc_comparable": payload["comparison"]["auc_comparable"],
                    "runtime_seconds": payload["runtime"]["total_seconds"],
                    "output_path": str(run_dir),
                    "failure": "",
                }
            )
        elif failure_path.exists():
            payload = json.loads(failure_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "k": k,
                    "status": "failed",
                    "archived_run_id": payload.get("archived_run_id", ""),
                    "selected_configuration": payload.get("configuration_label", ""),
                    "archived_accuracy": payload.get("archived_accuracy"),
                    "corrected_accuracy": None,
                    "accuracy_difference_percentage_points": None,
                    "archived_roc_auc": payload.get("archived_roc_auc"),
                    "corrected_roc_auc": None,
                    "roc_auc_difference_absolute": None,
                    "auc_comparable": False,
                    "runtime_seconds": payload.get("runtime_seconds"),
                    "output_path": str(run_dir),
                    "failure": payload.get("error", ""),
                }
            )
        else:
            rows.append(
                {
                    "k": k,
                    "status": "pending",
                    "archived_run_id": "",
                    "selected_configuration": "",
                    "archived_accuracy": None,
                    "corrected_accuracy": None,
                    "accuracy_difference_percentage_points": None,
                    "archived_roc_auc": None,
                    "corrected_roc_auc": None,
                    "roc_auc_difference_absolute": None,
                    "auc_comparable": False,
                    "runtime_seconds": None,
                    "output_path": str(run_dir),
                    "failure": "",
                }
            )
    frame = pd.DataFrame(rows)
    write_csv_atomic(ROOT / "cumulative_results.csv", frame)
    return frame


def run_one(
    row: pd.Series,
    dataset: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_names: np.ndarray,
) -> None:
    k = int(row["k"])
    run_dir = RUNS / f"k{k:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    failure_path = run_dir / "failure.json"
    if metrics_path.exists() or failure_path.exists():
        print(f"{now_iso()} k={k} already has a terminal record; not rerunning", flush=True)
        return
    logger = RunLogger(run_dir / "runtime.log")
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    started_at = now_iso()
    archived_run_id = str(row["archived_run_id"])
    fmap_name = str(row["feature_map"])
    entanglement = str(row["entanglement"])
    reps = int(row["repetitions"])
    feature_selection = json.loads(row["feature_selection_or_reduction"])
    estimator_name = str(feature_selection["estimator"])
    selector_name = (
        "Logistic Regression" if estimator_name == "LogisticRegression" else "Random Forest"
    )
    configuration_label = (
        f"MinMaxScaler | {selector_name} RFE | {fmap_name}/{entanglement}/reps={reps}"
    )
    kernel: BlockedFidelityStatevectorKernel | None = None
    scratch = run_dir / "scratch"
    try:
        logger(f"START k={k} archived_run_id={archived_run_id}")
        if scratch.exists():
            raise RuntimeError("Scratch directory already exists before first attempt")
        scratch.mkdir(parents=True)
        runtime_vm = psutil.virtual_memory()
        execution_config = {
            "started_at": started_at,
            "k": k,
            "archived_run_id": archived_run_id,
            "feature_map": fmap_name,
            "entanglement": entanglement,
            "repetitions": reps,
            "selector": feature_selection,
            "scaler": json.loads(row["scaler"]),
            "QSVC_settings": json.loads(row["svc_settings"]),
            "kernel_settings": json.loads(row["simulator_and_kernel_settings"]),
            "concurrent_runs": 1,
            "threads_per_run": THREADS_PER_RUN,
            "kernel_block_size": KERNEL_BLOCK_SIZE,
            "ram_available_bytes_at_start": runtime_vm.available,
            "environment": {
                "python": sys.version,
                "numpy": package_version("numpy"),
                "pandas": package_version("pandas"),
                "scikit_learn": package_version("scikit-learn"),
                "qiskit": package_version("qiskit"),
                "qiskit_machine_learning": package_version("qiskit-machine-learning"),
                "scipy": package_version("scipy"),
                "threadpools": threadpool_info(),
            },
            "manifest_row": json.loads(row.to_json()),
        }
        write_json(run_dir / "configuration.json", execution_config)
        write_json(
            run_dir / "split_ids.json",
            {
                "source": "source/historical/split_idx.json",
                "source_sha256": sha256_file(HISTORICAL / "split_idx.json"),
                "train_idx": [int(x) for x in train_idx],
                "test_idx": [int(x) for x in test_idx],
            },
        )

        with np.load(SOURCE / "data" / "wdbc_arrays.npz", allow_pickle=False) as exact:
            X_all = np.asarray(exact["X"], dtype=np.float64)
            y_all = np.asarray(exact["y"], dtype=np.int64)
        X_train_raw = X_all[train_idx]
        y_train = y_all[train_idx]
        X_test_raw = X_all[test_idx]
        preprocessing_started = time.perf_counter()
        captured_warnings: list[dict[str, str]] = []
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            scaler = MinMaxScaler()
            scaler.fit(X_train_raw)
            X_train_scaled = scaler.transform(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw)
            rfe = RFE(
                estimator=selector_for(pd.Series({"Estimator": selector_name})),
                n_features_to_select=k,
                step=1,
            )
            rfe.fit(X_train_scaled, y_train)
            X_train_selected = np.ascontiguousarray(
                rfe.transform(X_train_scaled), dtype=np.float64
            )
            X_test_selected = np.ascontiguousarray(
                rfe.transform(X_test_scaled), dtype=np.float64
            )
            captured_warnings.extend(
                {
                    "category": item.category.__name__,
                    "message": str(item.message),
                    "phase": "preprocessing",
                }
                for item in warning_list
            )
        preprocessing_seconds = time.perf_counter() - preprocessing_started
        selected_indices = np.flatnonzero(rfe.support_).astype(int)
        selected_names = feature_names[selected_indices].tolist()
        archived_names = archived_feature_list(k, "Normalization", selector_name)
        feature_payload = {
            "k": k,
            "corrected_selected_feature_indices": selected_indices.tolist(),
            "corrected_selected_feature_names": selected_names,
            "archived_leaked_selected_feature_names": archived_names,
            "selection_changed": selected_names != archived_names,
            "feature_order_rule": "original sklearn WDBC column order",
            "scaler_fit_rows": [int(x) for x in train_idx],
            "rfe_fit_rows": [int(x) for x in train_idx],
            "rfe_fit_labels_sha256": canonical_array_hash(y_train),
        }
        write_json(run_dir / "features.json", feature_payload)
        fit_access_audit = {
            "train_row_count": len(train_idx),
            "test_row_count": len(test_idx),
            "scaler_fit_index_sha256": canonical_array_hash(train_idx),
            "rfe_fit_index_sha256": canonical_array_hash(train_idx),
            "rfe_fit_label_sha256": canonical_array_hash(y_train),
            "qsvc_fit_index_sha256": canonical_array_hash(train_idx),
            "qsvc_fit_label_sha256": canonical_array_hash(y_train),
            "test_index_sha256": canonical_array_hash(test_idx),
            "test_indices_in_any_fit": False,
            "test_labels_in_any_fit": False,
            "tuning_or_cross_validation": False,
        }
        write_json(run_dir / "fit_access_audit.json", fit_access_audit)
        logger(
            f"preprocessing complete in {preprocessing_seconds:.3f}s; selected={selected_names}"
        )

        feature_map = build_feature_map(fmap_name, k, entanglement, reps)
        kernel = BlockedFidelityStatevectorKernel(
            feature_map=feature_map,
            scratch_dir=scratch,
            logger=logger,
            block_size=KERNEL_BLOCK_SIZE,
        )
        qsvc = QSVC(quantum_kernel=kernel, probability=True)
        fit_started = time.perf_counter()
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            qsvc.fit(X_train_selected, y_train)
            captured_warnings.extend(
                {
                    "category": item.category.__name__,
                    "message": str(item.message),
                    "phase": "qsvc_fit",
                }
                for item in warning_list
            )
        fit_seconds = time.perf_counter() - fit_started
        logger(f"QSVC fit complete in {fit_seconds:.3f}s")

        scoring_started = time.perf_counter()
        y_train_pred = np.asarray(qsvc.predict(X_train_selected), dtype=np.int64)
        y_test_pred = np.asarray(qsvc.predict(X_test_selected), dtype=np.int64)
        decision = np.asarray(qsvc.decision_function(X_test_selected), dtype=np.float64).ravel()
        probabilities = np.asarray(qsvc.predict_proba(X_test_selected), dtype=np.float64)
        classes = np.asarray(qsvc.classes_, dtype=np.int64)
        if not np.array_equal(classes, np.array([0, 1], dtype=np.int64)):
            raise RuntimeError(f"Unexpected QSVC class order: {classes.tolist()}")
        malignant_column = int(np.where(classes == 0)[0][0])
        benign_column = int(np.where(classes == 1)[0][0])
        malignant_score = probabilities[:, malignant_column]
        benign_score = probabilities[:, benign_column]
        y_test = y_all[test_idx]
        corrected = clinical_metrics(y_test, y_test_pred, malignant_score)
        benign_auc_check = float(roc_auc_score(y_test, benign_score))
        orientation_duality_error = abs(corrected["roc_auc"] - benign_auc_check)
        if orientation_duality_error > 1e-12:
            raise RuntimeError("Malignant/benign dual ROC-AUC orientation check failed")
        scoring_seconds = time.perf_counter() - scoring_started
        logger(f"test scoring complete in {scoring_seconds:.3f}s")

        predictions = pd.DataFrame(
            {
                "sample_index": test_idx,
                "true_source_label": y_test,
                "true_diagnosis": np.where(y_test == 0, "M", "B"),
                "predicted_source_label": y_test_pred,
                "predicted_diagnosis": np.where(y_test_pred == 0, "M", "B"),
                "malignant_probability": malignant_score,
                "benign_probability": benign_score,
                "decision_function_positive_is_benign": decision,
                "k": k,
                "archived_run_id": archived_run_id,
            }
        )
        write_csv_atomic(run_dir / "test_predictions.csv", predictions)
        train_predictions = pd.DataFrame(
            {
                "sample_index": train_idx,
                "true_source_label": y_train,
                "predicted_source_label": y_train_pred,
                "k": k,
                "archived_run_id": archived_run_id,
            }
        )
        write_csv_atomic(run_dir / "train_predictions.csv", train_predictions)
        write_json(run_dir / "warnings.json", captured_warnings)
        write_json(run_dir / "kernel_audit.json", kernel.audit)

        total_seconds = time.perf_counter() - started_wall
        cpu_seconds = time.process_time() - started_cpu
        archived_accuracy = float(row["archived_test_accuracy_reported"])
        archived_auc = float(row["archived_roc_auc_reported"])
        metrics = {
            "completed_at": now_iso(),
            "status": "completed",
            "k": k,
            "configuration_label": configuration_label,
            "archived": {
                "run_id": archived_run_id,
                "artifact_path": str(row["artifact_path"]),
                "accuracy_reported": archived_accuracy,
                "accuracy_from_counts": float(row["archived_test_accuracy_from_counts"]),
                "roc_auc_reported": archived_auc,
                "metric_orientation": str(row["archived_metric_definition"]),
            },
            "corrected": {
                **corrected,
                "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
                "benign_orientation_auc_numeric_check": benign_auc_check,
                "orientation_duality_absolute_error": orientation_duality_error,
            },
            "comparison": {
                "accuracy_difference_percentage_points": 100.0 * (corrected["accuracy"] - archived_accuracy),
                "roc_auc_difference_absolute": corrected["roc_auc"] - archived_auc,
                "accuracy_comparable": True,
                "auc_comparable": True,
                "auc_comparability_reason": (
                    "Archived code used benign label 1 with P(benign); corrected clinical AUC uses malignant label 1 with P(malignant). "
                    "Reversing both label and score preserves ROC-AUC, verified numerically for the corrected scores."
                ),
            },
            "features": feature_payload,
            "runtime": {
                "started_at": started_at,
                "completed_at": now_iso(),
                "preprocessing_seconds": preprocessing_seconds,
                "qsvc_fit_seconds": fit_seconds,
                "scoring_seconds": scoring_seconds,
                "total_seconds": total_seconds,
                "cpu_seconds": cpu_seconds,
                "concurrent_runs": 1,
                "threads_per_run": THREADS_PER_RUN,
            },
            "output_files": {
                "configuration": "configuration.json",
                "split_ids": "split_ids.json",
                "features": "features.json",
                "test_predictions_and_scores": "test_predictions.csv",
                "train_predictions": "train_predictions.csv",
                "fit_access_audit": "fit_access_audit.json",
                "kernel_audit": "kernel_audit.json",
                "warnings": "warnings.json",
                "runtime_log": "runtime.log",
            },
        }
        write_json(metrics_path, metrics)
        write_json(run_dir / "runtime.json", metrics["runtime"])
        removed = kernel.cleanup()
        kernel = None
        gc.collect()
        if scratch.exists():
            shutil.rmtree(scratch)
        logger(f"removed {len(removed)} temporary statevector bank(s)")
        logger(
            f"COMPLETE k={k} accuracy={corrected['accuracy']:.8f} auc={corrected['roc_auc']:.8f} total={total_seconds:.3f}s"
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started_wall
        failure = {
            "failed_at": now_iso(),
            "status": "failed",
            "k": k,
            "archived_run_id": archived_run_id,
            "configuration_label": configuration_label,
            "archived_accuracy": float(row["archived_test_accuracy_reported"]),
            "archived_roc_auc": float(row["archived_roc_auc_reported"]),
            "runtime_seconds": elapsed,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(failure_path, failure)
        logger(f"FAILED k={k} after {elapsed:.3f}s: {type(exc).__name__}: {exc}")
        if kernel is not None:
            try:
                kernel.cleanup()
            except Exception as cleanup_error:
                logger(f"scratch cleanup warning: {cleanup_error}")
        gc.collect()
        if scratch.exists():
            try:
                shutil.rmtree(scratch)
            except Exception as cleanup_error:
                logger(f"scratch directory retained after cleanup failure: {cleanup_error}")
    finally:
        update_cumulative()


def main() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(name) != str(THREADS_PER_RUN):
            raise RuntimeError(f"{name} must be set to {THREADS_PER_RUN} before launch")
    manifest, dataset, train_idx, test_idx, feature_names = load_locked_inputs()
    static_audit = static_prelaunch_checks(manifest)
    write_json(RECORDS / "static_prelaunch_audit.json", static_audit)
    update_cumulative()
    summary = {
        "started_at": now_iso(),
        "allowed_runs": 4,
        "allowed_k": list(EXPECTED_K),
        "attempted": [],
    }
    with threadpool_limits(limits=THREADS_PER_RUN):
        for _, row in manifest.iterrows():
            k = int(row["k"])
            summary["attempted"].append(k)
            write_json(RECORDS / "execution_summary.json", summary)
            run_one(row, dataset, train_idx, test_idx, feature_names)
    cumulative = update_cumulative()
    summary["completed_at"] = now_iso()
    summary["terminal_statuses"] = cumulative[["k", "status"]].to_dict(orient="records")
    summary["completed_count"] = int((cumulative["status"] == "completed").sum())
    summary["failed_count"] = int((cumulative["status"] == "failed").sum())
    write_json(RECORDS / "execution_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

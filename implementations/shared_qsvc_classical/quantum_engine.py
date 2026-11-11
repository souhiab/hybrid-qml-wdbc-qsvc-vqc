from __future__ import annotations
import hashlib,time
from functools import lru_cache
import numpy as np
from qiskit.circuit.library import ZFeatureMap,ZZFeatureMap
BLOCK_SIZE=64

def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()

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

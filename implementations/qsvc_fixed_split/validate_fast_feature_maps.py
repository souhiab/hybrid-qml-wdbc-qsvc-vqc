from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel

import run_scaled_grid as optimized


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "records" / "k17_fast_simulator_equivalence.json"


def main() -> None:
    rng = np.random.default_rng(20261107)
    checks = []
    reference_total = 0.0
    optimized_total = 0.0
    logger = optimized.RunLogger(ROOT / "records" / "k17_fast_simulator_validation.log")
    for width in (4, 7, 11, 17):
        repetitions = (1, 2, 3, 4) if width == 4 else (1, 4)
        for reps in repetitions:
            constructors = [("ZFeatureMap", "", ZFeatureMap(width, reps=reps))]
            constructors += [("ZZFeatureMap", topology, ZZFeatureMap(width, reps=reps, entanglement=topology)) for topology in ("linear", "full", "circular")]
            values = rng.uniform(0.05, 0.95, size=(2, width))
            for name, topology, feature_map in constructors:
                started = time.perf_counter()
                reference = FidelityStatevectorKernel(feature_map=feature_map, cache_size=None, auto_clear_cache=False, shots=None, enforce_psd=True).evaluate(values)
                reference_seconds = time.perf_counter() - started
                started = time.perf_counter()
                bank, state_seconds = optimized.statevector_bank(feature_map, values, logger, f"k{width} {name} {topology or 'none'} r{reps}")
                candidate, audit = optimized.blocked_fidelity(bank, bank, symmetric=True)
                optimized_seconds = time.perf_counter() - started
                error = float(np.max(np.abs(reference - candidate)))
                checks.append({"width": width, "constructor": name, "topology": topology, "repetitions": reps, "sample_rows": 2, "reference_seconds": reference_seconds, "optimized_seconds": optimized_seconds, "speedup": reference_seconds / optimized_seconds, "max_absolute_kernel_error": error, "statevector_seconds": state_seconds, "kernel_seconds": audit["seconds"]})
                reference_total += reference_seconds
                optimized_total += optimized_seconds
    tolerance = 1e-12
    record = {
        "validated_at": optimized.now_iso(),
        "method": "exact vectorized phase layers and normalized fast Walsh-Hadamard transforms for default Qiskit ZFeatureMap/ZZFeatureMap",
        "precision": "complex128 statevectors; float64 kernels",
        "shots": None,
        "cases": len(checks),
        "widths": [4, 7, 11, 17],
        "tolerance": tolerance,
        "max_absolute_kernel_error": max(item["max_absolute_kernel_error"] for item in checks),
        "reference_total_seconds": reference_total,
        "optimized_total_seconds": optimized_total,
        "aggregate_speedup": reference_total / optimized_total,
        "passed": max(item["max_absolute_kernel_error"] for item in checks) <= tolerance,
        "checks": checks,
    }
    temp = OUTPUT.with_suffix(".json.tmp")
    temp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, OUTPUT)
    print(json.dumps({key: record[key] for key in ("cases", "max_absolute_kernel_error", "reference_total_seconds", "optimized_total_seconds", "aggregate_speedup", "passed")}, indent=2))
    if not record["passed"]:
        raise RuntimeError("Fast feature-map simulator failed equivalence validation")


if __name__ == "__main__":
    main()

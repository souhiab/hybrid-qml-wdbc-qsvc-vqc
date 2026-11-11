from __future__ import annotations

import gc
import json
from pathlib import Path

import psutil

import run_scaled_grid as base


ROOT = Path(__file__).resolve().parent
RECORDS = ROOT / "records"
WIDTH = 11
BENCHMARK_CONFIGURATION_ID = "fs-k11-minmax-rf-zz-full-r4"
BENCHMARK_THREADS = 4
PRODUCTION_THREADS = 2


def main() -> None:
    base.EXPECTED_WIDTHS = (WIDTH,)
    manifest, existing = base.load_inputs()
    width_rows = manifest[manifest["width"].astype(int) == WIDTH]
    if len(width_rows) != 64:
        raise RuntimeError(f"Expected 64 k=11 manifest rows, found {len(width_rows)}")
    if int((width_rows["execution_action"] == "reuse_verified").sum()) != 4:
        raise RuntimeError("Expected four k=11 reusable rows")
    if int((width_rows["execution_action"] == "run_new").sum()) != 60:
        raise RuntimeError("Expected 60 k=11 new rows")
    base.checkpoint_progress(manifest, existing, {"status": "k11_execution_start"})
    benchmark_match = manifest[manifest["configuration_id"] == BENCHMARK_CONFIGURATION_ID]
    if len(benchmark_match) != 1 or benchmark_match.iloc[0]["execution_action"] != "run_new":
        raise RuntimeError("The locked k=11 benchmark is not one pending effective configuration")
    benchmark_row = benchmark_match.iloc[0].to_dict()
    valid, _ = base.validate_new_checkpoint(benchmark_row)
    if valid:
        metrics = json.loads((base.run_dir_for(benchmark_row) / "metrics.json").read_text(encoding="utf-8"))
        benchmark_result = {
            "status": "resumed_verified",
            "configuration_id": BENCHMARK_CONFIGURATION_ID,
            "runtime_seconds": metrics["runtime"]["total_seconds"],
            "peak_observed_rss_bytes": metrics["runtime"]["peak_observed_rss_bytes"],
        }
    else:
        benchmark_result = base.run_one(benchmark_row, BENCHMARK_THREADS, True, 0)
        base.append_run_log(benchmark_result)
        base.checkpoint_progress(manifest, existing, benchmark_result)
        if benchmark_result["status"] != "completed":
            benchmark_result = base.run_one(benchmark_row, BENCHMARK_THREADS, True, 1)
            base.append_run_log(benchmark_result)
            base.checkpoint_progress(manifest, existing, benchmark_result)
        if benchmark_result["status"] != "completed":
            raise RuntimeError(f"k=11 benchmark failed: {benchmark_result}")
    gc.collect()
    available = psutil.virtual_memory().available
    peak = int(benchmark_result["peak_observed_rss_bytes"])
    physical = int(psutil.cpu_count(logical=False) or 1)
    memory_workers = max(1, available // max(1, peak * 3))
    workers = max(1, min(2, physical // PRODUCTION_THREADS, memory_workers))
    reason = (
        f"Benchmark peak RSS was {peak} bytes and post-benchmark available RAM was {available} bytes. "
        f"Selected {workers} worker(s) with {PRODUCTION_THREADS} BLAS threads per worker, reserving a three-times-peak memory margin per worker and not exceeding {physical} physical cores."
    )
    base.write_json(RECORDS / "execution_settings_k11.json", {
        "locked_at": base.now_iso(),
        "width": WIDTH,
        "benchmark_configuration_id": BENCHMARK_CONFIGURATION_ID,
        "benchmark_is_authorized_grid_evaluation": True,
        "benchmark_threads": BENCHMARK_THREADS,
        "benchmark_result": benchmark_result,
        "physical_cpu_cores": physical,
        "logical_cpu_cores": psutil.cpu_count(logical=True),
        "ram_available_after_benchmark_bytes": available,
        "selected_workers": workers,
        "selected_threads_per_worker": PRODUCTION_THREADS,
        "selected_total_blas_threads": workers * PRODUCTION_THREADS,
        "kernel_block_size": base.BLOCK_SIZE,
        "new_probability_calibration_seed": base.CALIBRATION_SEED,
        "reason": reason,
    })
    print(f"{base.now_iso()} k=11 benchmark complete seconds={benchmark_result['runtime_seconds']:.3f} peak_rss={peak} workers={workers} threads={PRODUCTION_THREADS}", flush=True)
    base.execute_width(WIDTH, manifest, existing, workers, PRODUCTION_THREADS)
    progress = base.checkpoint_progress(manifest, existing, {"status": "k11_execution_complete"})
    subset = progress[progress["width"].astype(int) == WIDTH]
    completion = {
        "completed_at": base.now_iso(),
        "status": "completed",
        "width": WIDTH,
        "expected": 64,
        "new_completed": int((subset["status"] == "completed_new").sum()),
        "reused_completed": int((subset["status"] == "completed_reused").sum()),
        "pending": int((subset["status"] == "pending").sum()),
        "failed": int((subset["status"] == "failed").sum()),
        "scope_exclusion": "k=17 not executed",
    }
    base.write_json(RECORDS / "execution_completion_k11.json", completion)
    if completion["new_completed"] != 60 or completion["reused_completed"] != 4 or completion["pending"] or completion["failed"]:
        raise RuntimeError(f"k=11 completion gate failed: {completion}")
    print(f"{base.now_iso()} K11 EXECUTION COMPLETE new=60 reused=4 coverage=64/64", flush=True)


if __name__ == "__main__":
    main()

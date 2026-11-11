from __future__ import annotations

import gc
import json
from pathlib import Path

import psutil

import run_scaled_grid as base


ROOT = Path(__file__).resolve().parent
RECORDS = ROOT / "records"
WIDTH = 17
BENCHMARK_CONFIGURATION_ID = "fs-k17-minmax-rf-zz-full-r4"
BENCHMARK_THREADS = 4
SYSTEM_HEADROOM_BYTES = 1_500_000_000


def main() -> None:
    base.EXPECTED_WIDTHS = (WIDTH,)
    manifest, existing = base.load_inputs()
    width_rows = manifest[manifest["width"].astype(int) == WIDTH]
    if len(width_rows) != 64:
        raise RuntimeError(f"Expected 64 k=17 manifest rows, found {len(width_rows)}")
    if int((width_rows["execution_action"] == "reuse_verified").sum()) != 5 or int((width_rows["execution_action"] == "run_new").sum()) != 59:
        raise RuntimeError("Expected five k=17 reusable and 59 new rows")
    base.checkpoint_progress(manifest, existing, {"status": "k17_execution_start"})
    benchmark_match = manifest[manifest["configuration_id"] == BENCHMARK_CONFIGURATION_ID]
    if len(benchmark_match) != 1 or benchmark_match.iloc[0]["execution_action"] != "run_new":
        raise RuntimeError("The locked k=17 benchmark is not one pending effective configuration")
    benchmark_row = benchmark_match.iloc[0].to_dict()
    valid, _ = base.validate_new_checkpoint(benchmark_row)
    if valid:
        metrics = json.loads((base.run_dir_for(benchmark_row) / "metrics.json").read_text(encoding="utf-8"))
        benchmark_result = {"status": "resumed_verified", "configuration_id": BENCHMARK_CONFIGURATION_ID, "runtime_seconds": metrics["runtime"]["total_seconds"], "peak_observed_rss_bytes": metrics["runtime"]["peak_observed_rss_bytes"]}
    else:
        benchmark_result = base.run_one(benchmark_row, BENCHMARK_THREADS, True, 0)
        base.append_run_log(benchmark_result)
        base.checkpoint_progress(manifest, existing, benchmark_result)
        if benchmark_result["status"] != "completed":
            benchmark_result = base.run_one(benchmark_row, BENCHMARK_THREADS, True, 1)
            base.append_run_log(benchmark_result)
            base.checkpoint_progress(manifest, existing, benchmark_result)
        if benchmark_result["status"] != "completed":
            raise RuntimeError(f"k=17 benchmark failed: {benchmark_result}")
    gc.collect()
    vm = psutil.virtual_memory()
    available = int(vm.available)
    peak = int(benchmark_result["peak_observed_rss_bytes"])
    physical = int(psutil.cpu_count(logical=False) or 1)
    safe_memory = max(0, available - SYSTEM_HEADROOM_BYTES)
    memory_workers = max(1, safe_memory // max(1, int(peak * 1.75)))
    workers = max(1, min(2, physical // 2, memory_workers))
    threads = 2 if workers >= 2 else min(4, physical)
    reason = (
        f"The official k=17 benchmark observed peak RSS {peak} bytes. After it completed, {available} bytes were available; "
        f"{SYSTEM_HEADROOM_BYTES} bytes were reserved for system stability and a 1.75x per-worker memory margin was applied. "
        f"Selected {workers} worker(s) with {threads} BLAS threads each, using at most {workers * threads} of {physical} physical cores."
    )
    base.write_json(RECORDS / "execution_settings_k17.json", {
        "locked_at": base.now_iso(), "width": WIDTH,
        "benchmark_configuration_id": BENCHMARK_CONFIGURATION_ID, "benchmark_is_authorized_grid_evaluation": True,
        "benchmark_threads": BENCHMARK_THREADS, "benchmark_result": benchmark_result,
        "physical_cpu_cores": physical, "logical_cpu_cores": psutil.cpu_count(logical=True),
        "ram_total_bytes": vm.total, "ram_available_after_benchmark_bytes": available, "system_headroom_bytes": SYSTEM_HEADROOM_BYTES,
        "selected_workers": workers, "selected_threads_per_worker": threads, "selected_total_blas_threads": workers * threads,
        "kernel_block_size": base.BLOCK_SIZE, "test_stream_batch_size": 4,
        "new_probability_calibration_seed": base.CALIBRATION_SEED, "reason": reason,
    })
    print(f"{base.now_iso()} k=17 benchmark seconds={benchmark_result['runtime_seconds']:.3f} peak_rss={peak} workers={workers} threads={threads}", flush=True)
    base.execute_width(WIDTH, manifest, existing, workers, threads)
    progress = base.checkpoint_progress(manifest, existing, {"status": "k17_execution_complete"})
    subset = progress[progress["width"].astype(int) == WIDTH]
    completion = {
        "completed_at": base.now_iso(), "status": "completed", "width": WIDTH, "expected": 64,
        "new_completed": int((subset["status"] == "completed_new").sum()),
        "reused_completed": int((subset["status"] == "completed_reused").sum()),
        "pending": int((subset["status"] == "pending").sum()), "failed": int((subset["status"] == "failed").sum()),
    }
    base.write_json(RECORDS / "execution_completion_k17.json", completion)
    if completion["new_completed"] != 59 or completion["reused_completed"] != 5 or completion["pending"] or completion["failed"]:
        raise RuntimeError(f"k=17 completion gate failed: {completion}")
    print(f"{base.now_iso()} K17 EXECUTION COMPLETE new=59 reused=5 coverage=64/64", flush=True)


if __name__ == "__main__":
    main()

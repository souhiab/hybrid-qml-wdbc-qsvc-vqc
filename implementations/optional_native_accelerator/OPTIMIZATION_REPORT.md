# Seventeen-qubit execution optimization ? 2026-11-09

The compiled C++ backend applies the same RY and controlled-NOT gates directly to C-contiguous complex128 statevectors. It removes per-gate full-array copies. Compilation uses /O2 /fp:strict, double arithmetic, no fast-math and no OpenMP. The existing finite-shot sampler, circuit definitions, seeds, loss, precision, COBYLA budgets and preprocessing remain unchanged. Only CPU execution is used.

Numerical verification passed 108 finite-shot probability checks, including 96 actual seventeen-qubit circuit/seed/parameter-vector cases and random multirow states. Qiskit reference statevectors matched within tolerance; maximum measured amplitude error was approximately 3.13e-17.

A saved fit's full 364-training-case forward pass was replayed at its saved final parameters, with zero optimizer fits and no test-case evaluation. Previous engine: 16.270 s; native: 2.277 s; measured single-worker speedup: 7.14x. The native probabilities exactly matched the saved training probabilities, and the source prediction file remained unchanged.

Worker-throughput benchmarks tested one through five workers, one BLAS/OpenMP thread each. Five was fastest in the measured synthetic workload. The controller starts at the measured ceiling while checking available memory before every launch, reserves at least 2 GB or 15% of physical RAM, applies a 1.6 GB per-launch allowance and reduces slots on pressure. Changing whether VS Code is open does not override measured available RAM. Workers are independent and have independent RNGs. This is a bounded, measured concurrency choice, not a guarantee that all cores remain saturated.

These are replay/inner-loop speedups, not a measured full new-fit speedup. Encoding, checkpointing, process startup and optimizer termination also affect runtime. Floating-point gate results are tolerance-equivalent; exact bit identity is not claimed for every possible input. Tested sampled probability arrays are exactly equal.

All 26 existing completed outcomes are retained. Interrupted attempts retain their files and restart only from original initialization under a new attempt ID. Resume finishes cap100, then48 fresh cap300 fits, then48 fresh cap600 fits, and saves the matched comparison. No completed fit is rerun.

Windows restarted after the last pilot workers were launched, explaining the stale running status and silent process disappearance. Closing an editor is distinct from restarting Windows; unfinished COBYLA fits cannot be continued exactly after shutdown.

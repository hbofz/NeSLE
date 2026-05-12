# Legacy benchmarks

Historical benchmark scripts from the project's phase-6 throughput study. Kept for reproducibility of the published results in `docs/phase6-report.md`, but not part of the current performance verification flow.

For current benchmarks, see the parent directory:

- `../gpu_vs_cpu.py` — head-to-head CPU vs GPU env-step throughput
- `../verify_correctness.py` — falsifiability tests proving the CUDA kernel really runs N independent emulators
- `../phase5_benchmark.py` — broader env-count / observation-mode sweeps (still current)

| File | What it does |
|---|---|
| `phase6_console_ablation.py` | Phase 6 console-mode throughput ablation (frame-skip × env-count) |
| `plot_phase6.py` | Plotting script for the SVGs in `docs/assets/` |

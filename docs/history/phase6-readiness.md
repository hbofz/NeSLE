# Phase 6 Readiness Audit

Date: 2026-05-01

This file is a historical entry gate for Phase 6. The current final Phase 6
status lives in `docs/phase6-report.md`.

## Status

At Phase 6 entry, the project already had a working artifact: the public Python
API could build the CUDA extension, construct a ROM-backed `cuda-console` vector
backend, and step a real Super Mario Bros. ROM on an A100. The benchmark
harness also recorded native CPU, legacy `gym-super-mario-bros`, synthetic
CUDA, raw CUDA kernel, and ROM-backed CUDA-console rows.

## Verification Snapshot

- Local Python syntax checks passed for `src/nesle/env.py` and
  `benchmarks/phase5_benchmark.py`.
- `git diff --check` passed.
- `PYTHON=.venv/bin/python sh scripts/verify_phase4.sh` passed.
- `PYTHON=.venv/bin/python sh scripts/verify_phase5.sh` passed.
- `sh scripts/verify.sh` passed locally; CUDA extension build is skipped on the
  Mac when `nvcc` is unavailable.
- On Colab A100, `NESLE_CUDA_ARCH=sm_80 PYTHON=python3
  sh scripts/build_cuda_extension.sh` passed.
- On Colab A100, the real-ROM `cuda-console` reset/step/render smoke passed.

## Performance Readout

| Path | Best observed throughput | What it means |
| --- | ---: | --- |
| Native NeSLE CPU | ~213 env-steps/sec | Real CPU-side emulator baseline |
| `gym-super-mario-bros` | ~423 env-steps/sec | Legacy CPU emulator baseline |
| `cuda-console` with RGB obs copy | 3,650.5 env-steps/sec at 128 envs | Real CPU/PPU loop on GPU plus RGB render/copy |
| `cuda-console` no obs copy | 1,139,578.8 env-steps/sec at 128 envs | Real CPU/PPU loop on GPU, rewards/dones copied only |
| Synthetic CUDA reward no-copy | 102,991,288.4 env-steps/sec at 16,384 envs | Calibration kernel, not full NES emulation |
| Raw CUDA reward kernel | 2,633,530,000 env-steps/sec at 16,384 envs | Bare kernel ceiling |

## Historical Optimization Targets

These were the targets identified before Phase 6 implementation. They are kept
here to show why Phase 6 focused on observation transfer and reproducibility.

1. Expose render cadence and no-copy training modes through `NesleVecEnv`.
   Status: addressed by CUDA copy flags, `step_reward`, and RGB observation
   cadence support.

2. Add compact observation options.
   Status: addressed for normal vector stepping by `observation_mode="ram"`.
   GPU-resident visual feature tensors remain future work for image policies.

3. Broaden and package scaling runs.
   Status: addressed by the Phase 6 reproduction script, Dockerfile, tracked
   A100 JSON, and SVG plots.

4. Optimize per-env serial execution.
   Each environment currently uses one CUDA thread to run its CPU/PPU loop. That
   proves correctness but leaves room for warp-level scheduling, instruction
   dispatch tuning, memory layout tuning, and render kernel tiling.

5. Add reset-cache and frame-skip ablations.
   Status: frame-skip and copy-mode ablations are in the Phase 6 report; deeper
   reset-cache ablations remain future research.

6. Package reproducibility.
   Status: addressed by `docker/cuda.Dockerfile` and
   `scripts/reproduce_phase6.sh`.

## Bottom Line

The project entered Phase 6 with the CUDA loop working and full RGB host
observation copies identified as the dominant bottleneck. Phase 6 now addresses
that path with `observation_mode="ram"`, which keeps normal vector stepping
while copying compact CPU RAM observations instead of full RGB frames.

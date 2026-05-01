# Phase 6 Readiness Audit

Date: 2026-05-01

## Status

Phase 6 can start from a working artifact: the public Python API can build the
CUDA extension, construct a ROM-backed `cuda-console` vector backend, and step a
real Super Mario Bros. ROM on an A100. The benchmark harness also records native
CPU, legacy `gym-super-mario-bros`, synthetic CUDA, raw CUDA kernel, and
ROM-backed CUDA-console rows.

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

## Optimization Remaining

1. Expose render cadence and no-copy training modes through `NesleVecEnv`.
   The probe shows a roughly 312x gap at 128 envs between RGB-copy stepping and
   no-observation-copy stepping.

2. Add GPU-resident observation options.
   Returning full RGB NumPy frames every step is the main bottleneck. Phase 6
   should test device tensors, compact RAM/state features, frame stacks rendered
   every N steps, and lazy `render()` calls for evaluation.

3. Broaden scaling runs.
   Current real-console numbers cover up to 128 envs in the quick audit. Phase 6
   should run longer, warmed benchmarks for 256, 512, 1024, and 2048 envs with
   and without observation copies.

4. Optimize per-env serial execution.
   Each environment currently uses one CUDA thread to run its CPU/PPU loop. That
   proves correctness but leaves room for warp-level scheduling, instruction
   dispatch tuning, memory layout tuning, and render kernel tiling.

5. Add reset-cache and frame-skip ablations.
   Phase 3 has reset-cache primitives. Phase 6 needs benchmark rows showing the
   effect of reset restore, frame-skip choices, render cadence, and env count.

6. Package reproducibility.
   The artifact builds on Colab A100, but Phase 6 should add a CUDA container or
   one-command setup path so external reviewers do not depend on the ad hoc SSH
   tunnel.

## Bottom Line

The project entered Phase 6 with the CUDA loop working and full RGB host
observation copies identified as the dominant bottleneck. Phase 6 now addresses
that path with `observation_mode="ram"`, which keeps normal vector stepping
while copying compact CPU RAM observations instead of full RGB frames.

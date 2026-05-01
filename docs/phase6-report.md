# Phase 6 Research Package

Date: 2026-05-01

## Summary

Phase 6 packages NeSLE as a reproducible research artifact. The repository now
contains:

- a CUDA Dockerfile for NVIDIA builds,
- a one-command Phase 6 reproduction script,
- a CUDA-console ablation benchmark,
- tracked A100 benchmark data,
- generated SVG plots,
- explicit limitations and next optimization targets.

The central result is that the full ROM-backed CUDA CPU/PPU console loop works
through the Python vector API. The main remaining performance bottleneck is not
getting the NES loop onto CUDA; it is copying full RGB observations back to host
memory on every step.

## Reproduction

On an NVIDIA CUDA machine:

```sh
python -m pip install -e '.[dev,rl]'
NESLE_CUDA_ARCH=sm_80 PYTHON=python3 sh scripts/build_cuda_extension.sh
NESLE_ROM_PATH="/path/to/Super Mario Bros. (World).nes" sh scripts/reproduce_phase6.sh
```

With Docker:

```sh
docker build -f docker/cuda.Dockerfile -t nesle-cuda .
docker run --gpus all --rm -v "$PWD:/workspace/nesle" -w /workspace/nesle \
  -e NESLE_ROM_PATH="/workspace/nesle/Super Mario Bros. (World).nes" \
  nesle-cuda sh scripts/reproduce_phase6.sh
```

The ROM is not vendored. The benchmark expects a local mapper-0 Super Mario
Bros. iNES file.

## Methodology

The Phase 6 ablation uses `benchmarks/phase6_console_ablation.py`, which
constructs `nesle._cuda_core.CudaBatch(num_envs, frameskip, rom_bytes)` and
steps the real CUDA CPU/PPU console loop. It compares three modes:

- `rgb`: step, render, and copy RGB observations to host.
- `render_only`: step and render on device, but do not copy RGB observations.
- `no_copy`: step the real console loop and copy only rewards/done flags.

All rows below were collected on a Colab A100-SXM4-80GB with CUDA 12.8.

## Main Result

![Phase 6 copy gap](assets/phase6-copy-gap.svg)

| Mode | Envs | Env steps/sec | Training frames/sec | Peak GPU util | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| RGB obs copy | 1 | 136.3 | 545.0 | 74% | 5009 MiB |
| RGB obs copy | 8 | 825.3 | 3,301.3 | 87% | 5009 MiB |
| RGB obs copy | 32 | 1,671.7 | 6,686.7 | 88% | 5015 MiB |
| RGB obs copy | 128 | 3,598.8 | 14,395.0 | 83% | 5033 MiB |
| render only | 1 | 137.7 | 550.9 | 88% | 5009 MiB |
| render only | 8 | 865.4 | 3,461.7 | 86% | 5009 MiB |
| render only | 32 | 1,797.7 | 7,190.9 | 86% | 5015 MiB |
| render only | 128 | 4,284.2 | 17,136.6 | 88% | 5033 MiB |
| no obs copy | 1 | 8,278.5 | 33,114.2 | 87% | 5009 MiB |
| no obs copy | 8 | 65,449.4 | 261,797.6 | 87% | 5009 MiB |
| no obs copy | 32 | 248,793.3 | 995,173.2 | 81% | 5015 MiB |
| no obs copy | 128 | 1,008,196.2 | 4,032,784.8 | 96% | 5033 MiB |

At 128 environments and frame-skip 4, the no-copy `cuda-console` path is about
280x faster than the RGB-copy Python vector path. This is the immediate Phase 6
optimization target.

## Frame-Skip Ablation

![Phase 6 frameskip ablation](assets/phase6-frameskip.svg)

| Frame-skip | Envs | Mode | Env steps/sec | Training frames/sec |
| ---: | ---: | --- | ---: | ---: |
| 1 | 128 | no obs copy | 40,788.0 | 40,788.0 |
| 2 | 128 | no obs copy | 1,082,570.7 | 2,165,141.4 |
| 4 | 128 | no obs copy | 1,008,196.2 | 4,032,784.8 |
| 8 | 128 | no obs copy | 863,579.2 | 6,908,633.8 |

Frame-skip 2-8 keeps env-step throughput near the million-step/sec range in
no-copy mode while increasing training-frame throughput. Frame-skip 1 is much
slower because every environment step lands on the expensive startup path more
often in this short benchmark.

## API Surface

`NesleVecEnv.step(...)` keeps the SB3-compatible RGB observation contract.
Phase 6 adds `NesleVecEnv.step_reward(actions)` for CUDA-only high-throughput
loops that need rewards and done flags without copying RGB observations every
step. Evaluation and debugging can still call `render()`.

## Limitations

- Mapper support is limited to NROM/Super Mario Bros.
- The PPU is sufficient for current gates, but it is not a full cycle-perfect
  NES renderer.
- `step_reward` is not an SB3 `VecEnv` method; SB3 training still uses the RGB
  observation path unless a custom training loop or wrapper is added.
- The CUDA-console implementation currently maps one environment to one CUDA
  thread for the serial CPU/PPU loop. This proves the GPU path but leaves
  scheduling and memory-layout optimization on the table.
- The ROM is not redistributed. Reproducers must provide their own legal ROM
  file.
- Colab SSH was used for A100 measurements; the Dockerfile is provided so the
  artifact can move to a cleaner NVIDIA environment.

## Phase 6 Conclusion

Phase 6 is complete as a research package. NeSLE now has a reproducible CUDA
artifact, public benchmark scripts, tracked A100 data, generated plots, a
documented no-copy training path, and explicit limitations. Future work should
turn `step_reward` into an SB3-friendly training wrapper, add GPU-resident
observation tensors, and optimize the per-environment serial console loop.

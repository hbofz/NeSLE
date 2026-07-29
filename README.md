# NeSLE

[![CI](https://github.com/hbofz/NeSLE/actions/workflows/ci.yml/badge.svg)](https://github.com/hbofz/NeSLE/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**A GPU-native NES emulator and reinforcement-learning stack for Super Mario Bros.**
NeSLE runs thousands of independent NES instances in parallel on one CUDA GPU —
6502 CPU, PPU, and bus emulated inside a CUDA kernel, one thread per env — and
trains PPO agents against them with observations that never leave the device.

**Key results** (see [Benchmarks](#benchmarks) for methodology and commands):

- On a **GTX 1050 Ti (4 GB)**: 29,491 env-steps/s at 4,096 parallel envs —
  **88× a single-env CPU baseline** (measured 2026-07-28 on this repo's HEAD).
- On an **A100 (80 GB)**: ~560k env-steps/s with RAM observations at 128 envs
  ([full report](docs/phase6-report.md)); a 75M-timestep GPU-resident PPO run on
  World 7-1 finished in **40 minutes** at 65,536 envs, sustaining ~31k
  env-steps/s end-to-end while the value head learned
  (`explained_variance` 0 → 0.66).

## Install

Requires Python ≥ 3.10, a CUDA GPU, and a C++20 toolchain.

```sh
git clone https://github.com/hbofz/NeSLE.git && cd NeSLE
python -m pip install -e '.[dev,rl]'

# Build the CUDA extension for your GPU architecture:
NESLE_CUDA_ARCH=sm_61 sh scripts/build_cuda_extension.sh   # GTX 10-series
NESLE_CUDA_ARCH=sm_80 sh scripts/build_cuda_extension.sh   # A100
NESLE_CUDA_ARCH=sm_90 sh scripts/build_cuda_extension.sh   # H100
```

On Windows the build script doesn't apply — follow
[`docs/build-windows.md`](docs/build-windows.md).

**Bring your own legally obtained Super Mario Bros. ROM** (iNES, mapper 0,
usually named `Super Mario Bros. (World).nes`). ROMs are not included and are
never committed (`*.nes` is gitignored).

### Hardware requirements

| Requirement | Detail |
|---|---|
| GPU | Any CUDA GPU; tested on GTX 1050 Ti (sm_61, 4 GB) and A100 (sm_80, 80 GB) |
| CUDA Toolkit | 12.x. **Pascal (GTX 10-series) requires CTK 12.x** — CTK 13+ dropped sm_61 |
| PyTorch | CUDA build required for training paths (`pip install torch --index-url https://download.pytorch.org/whl/cu126`) |
| OS | Linux, Windows 11 (both tested); macOS builds the CPU core only |

## Quickstart

Train a small PPO agent on World 1-1 in a few minutes (Stable-Baselines3 path):

```sh
python examples/sb3_train.py "Super Mario Bros. (World).nes" \
    --backend cuda --sb3-device cpu \
    --observation-mode ram --action-space simple \
    --reset-state-path docs/data/smb_level1_1.state \
    --num-envs 8 --timesteps 16384 --n-steps 128 --batch-size 256 \
    --max-episode-steps 256 --model-path nesle_ppo_smoke
```

For scale, use the GPU-resident PPO path — observations, rollouts, GAE, and the
optimizer all stay on device via DLPack:

```sh
python examples/native_ppo_train.py "Super Mario Bros. (World).nes" \
    --reset-state-path docs/data/smb_level1_1.state \
    --action-space mario --reward-mode smart \
    --num-envs 2048 --total-timesteps 25_000_000 \
    --n-steps 128 --batch-size 8192 \
    --checkpoint-path checkpoints/native_ppo.pt
```

Full options and troubleshooting: [`docs/training.md`](docs/training.md).

## What's in the box

- **CUDA-batched NROM emulator** (`cpp/bindings/cuda_module.cu`,
  `cpp/src/cuda/kernels.cu`). One CUDA thread per env runs 6502 + PPU + bus +
  OAM-DMA. Frame-skip happens inside the kernel.
- **Snapshot reset.** Bundled FCEUX-format save states
  (`docs/data/smb_level{1..8}_1.state`) drop every env directly into gameplay;
  auto-reset on done restores the snapshot in a single kernel launch.
- **Two PPO entry points:**
  - `examples/sb3_train.py` — Stable-Baselines3 PPO, the natural starting point.
  - `examples/native_ppo_train.py` → `nesle.native_ppo` — GPU-resident PPO that
    bypasses SB3's CPU rollout buffer via DLPack / `__cuda_array_interface__`.
- **Shaped reward on GPU.** `--reward-mode smart` runs a dense
  progress/checkpoint/death shaped reward entirely on device
  (`TorchSmartMarioReward`); `--reward-mode minimal` keeps the original sparse
  x-progress reward.
- **Curated action spaces** including `--action-space mario` (11 actions,
  matches the vendored Mario RL project's controller space).
- **Curriculum support.** Pass `--reset-state-paths` with the 8 bundled saves
  and envs are round-robin-assigned across worlds.
- **Native CPU backend** (`nesle._core.NativeConsole`) for single-env debugging
  and parity testing.
- **Sibling baseline project** at
  [`project/mario-rl-ram/`](project/mario-rl-ram/) — the CPU Stable-Retro
  Mario training stack NeSLE is benchmarked against.

## Verification

```sh
python -m pytest tests/                   # 68 tests; GPU/ROM tests auto-skip when absent
python benchmarks/verify_correctness.py   # falsifiability — confirms the batched kernel
                                          # runs N independent emulators (not N copies of env 0)
```

## Benchmarks

Reproduce the GPU-vs-CPU sweep (ROM at repo root, ~3 min):

```sh
python benchmarks/gpu_vs_cpu.py
```

GTX 1050 Ti / 4 GB, Windows 11, CTK 12.9, frameskip 4 — measured 2026-07-28:

| Backend | Envs | Env-steps/s | vs CPU |
|---|---:|---:|---:|
| native CPU | 1 | 335 | 1.00× |
| `cuda-console` | 64 | 1,066 | 3.2× |
| `cuda-console` | 256 | 4,086 | 12.2× |
| `cuda-console` | 1,024 | 15,003 | 44.8× |
| `cuda-console` | 4,096 | **29,491** | **88.1×** |

The GPU crosses over the single-env CPU at ~64 envs and scales near-linearly to
~2k envs. An earlier run of the same script recorded 102× against a slower CPU
baseline ([details](docs/benchmark-gpu-vs-cpu.md)) — the GPU-side numbers agree
within ~1%.

A100 / 80 GB (recorded 2026-05, [full report](docs/phase6-report.md)):

| Mode | Envs | Steps/s |
|---|---:|---:|
| `cuda-console`, RGB obs to host | 128 | ~3.6k |
| `cuda-console`, RAM obs to host | 128 | ~560k |
| `cuda-console`, no host copy | 128 | ~1.0M |

End-to-end PPO with the GPU-resident loop on the A100 sustained ~31k
env-steps/s for a 75M-timestep run (40 min wall-clock).

## Known limitations

- **NROM (mapper 0) only.** SMB works; other mappers aren't wired up.
- **Title-screen state machine has a PPU-timing bug.** The snapshot-reset path
  bypasses it cleanly; a real fix is deferred. See
  [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).
- **One CUDA thread per env** — simple to reason about, but leaves performance
  on the table at very large batches (warp divergence).
- The smart reward is tuned for World 1-1; other levels currently reuse its
  defaults.

## Project layout

```text
cpp/            C++/CUDA emulator core (headers, kernels, pybind11 bindings)
src/nesle/      Python package: env, actions, rewards, ROM parsing, native PPO
tests/          65+ Python tests + ad-hoc C++ tests (tests/cpp/)
examples/       Training / evaluation entry points
benchmarks/     Throughput + correctness benchmarks (legacy sweeps in legacy/)
scripts/        Build + verification scripts (legacy one-offs in legacy/)
docs/           Guides, benchmark reports, bundled save states (docs/data/)
project/        Vendored mario-rl-ram baseline (CPU Stable-Retro stack)
docker/         CUDA build/test image
```

## Documents

- [Training guide](docs/training.md) — primary entry point for RL work
- [Architecture](docs/architecture.md) — system design
- [Windows CUDA build](docs/build-windows.md)
- [A100 benchmark report](docs/phase6-report.md)
- [GPU vs CPU benchmark (1050 Ti)](docs/benchmark-gpu-vs-cpu.md)
- [Research notes](docs/research-notes.md) — design rationale, NES hardware background
- [CPU validation](docs/cpu-validation.md) — Klaus 6502 functional test gate
- [Headless runner](docs/headless-runner.md) — low-level ROM runner for debugging
- [Project history](docs/history/) — archived phase-by-phase development docs

## Citation

```bibtex
@misc{nesle2026,
  author       = {Azzam, Hamzah},
  title        = {NeSLE: a GPU-native NES emulator and reinforcement-learning environment},
  year         = {2026},
  howpublished = {\url{https://github.com/hbofz/NeSLE}}
}
```

## License and disclaimer

MIT — see [LICENSE](LICENSE).

This project is provided for research and educational purposes. It is not
affiliated with or endorsed by Nintendo. No game ROMs are distributed with this
repository; users must supply their own legally obtained copies.

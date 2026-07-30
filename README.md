# NeSLE

[![CI](https://github.com/hbofz/NeSLE/actions/workflows/ci.yml/badge.svg)](https://github.com/hbofz/NeSLE/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**A GPU-native NES emulator and reinforcement-learning stack for Super Mario Bros.**
NeSLE runs thousands of independent NES instances in parallel on one CUDA GPU —
6502 CPU, PPU, and bus emulated inside a CUDA kernel, one thread per env — and
trains PPO agents against them with observations that never leave the device.

**Key results** (see [Benchmarks](#benchmarks) for methodology and commands):

- On an **A100 (80 GB)**: **3,142,205 env-steps/s at 65,536 parallel envs**
  (peak; 12.6M NES frames/s ≈ 209,000× real-time), 311,844 at 4,096 envs —
  **~1,422× that machine's CPU** — measured 2026-07-30 on a Colab A100 from a
  clean clone with all 69 tests green.
- On a **GTX 1050 Ti (4 GB)**: 180,437 env-steps/s at 4,096 envs —
  **576× a single-env CPU baseline**.
- These numbers are ~5× the previous release's, from the fully-executed
  optimization program in [docs/gpu-scaling.md](docs/gpu-scaling.md):
  table-driven 6502 decode, register-resident PPU state, lazy event
  settlement — each change measured, and the dead ends (shared-memory zero
  page, wavefront dispatch) retired with published measurements rather than
  silently dropped.

## Install

Requires Python ≥ 3.10, a CUDA GPU, and a C++20 toolchain.

```sh
git clone https://github.com/hbofz/NeSLE.git
cd NeSLE
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -e '.[dev,rl]'

# The rl extra pulls CPU-only torch from PyPI. For training you need the CUDA
# build — --force-reinstall is required (same version number, so pip would
# otherwise silently keep the CPU wheel):
python -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126

# Build the CUDA extension (all platforms; auto-detects nvcc, MSVC, and GPU arch):
python scripts/build_cuda_extension.py
```

The build script detects your GPU's architecture via torch; override with
`NESLE_CUDA_ARCH=sm_80` (A100) / `sm_90` (H100) if building for another
machine. `scripts/build_cuda_extension.sh` remains for POSIX scripting/Docker,
and [`docs/build-windows.md`](docs/build-windows.md) documents the manual
Windows recipe if the script can't find your toolchain. Shell snippets in this README
use `sh` syntax; on PowerShell, join the multi-line commands onto one line (or
replace the trailing `\` with a backtick) and drop the single quotes around
`.[dev,rl]`. Budget ~9 GB of transient disk for the CUDA torch wheel install.

**Bring your own legally obtained Super Mario Bros. ROM** (iNES, mapper 0,
usually named `Super Mario Bros. (World).nes`). ROMs are not included and are
never committed (`*.nes` is gitignored).

### Hardware requirements

| Requirement | Detail |
|---|---|
| GPU | Any CUDA GPU; tested on GTX 1050 Ti (sm_61, 4 GB) and A100 (sm_80, 80 GB) |
| CUDA Toolkit | 12.x. **Pascal (GTX 10-series) requires CTK 12.x** — CTK 13+ dropped sm_61 |
| PyTorch | CUDA build required for training paths (see Install — `--force-reinstall` from the cu126 index) |
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

### What the trained agent looks like

![Native PPO agent clearing World 1-1](docs/assets/agent-flag-run.gif)

A 25M-timestep native-PPO run (2048 envs, smart reward, `mario` action space,
~2.5 h on a GTX 1050 Ti). The GIF above is the final life of the agent's best
episode out of 12: a World 1-1 clear — flag captured at x≈3,157, rendered at
real-time speed. Honest status on two fronts: (1) the clear is the tail of the
distribution — most episodes stall around x≈1,100 and die; (2) the recording
contains occasional 1-frame visual glitches (objects flashing in/out) from a
known renderer limitation — the CUDA PPU samples scroll/sprite state mid-frame
rather than at coherent frame boundaries (see KNOWN_ISSUES.md; game state and
RAM-based training are unaffected). Training metrics: episode return −26 →
~195, `explained_variance` 0 → 0.88. Evaluate and record any checkpoint
yourself:

```sh
python examples/native_ppo_eval.py "Super Mario Bros. (World).nes" \
    --checkpoint checkpoints/native_ppo.pt --gif-out my_agent.gif
```

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

GTX 1050 Ti / 4 GB, Windows 11, CTK 12.9, frameskip 4 — measured 2026-07-29:

| Backend | Envs | Env-steps/s | vs CPU |
|---|---:|---:|---:|
| native CPU | 1 | 321 | 1.00× |
| `cuda-console` | 64 | 1,299 | 4.0× |
| `cuda-console` | 256 | 4,839 | 15.1× |
| `cuda-console` | 1,024 | 17,591 | 54.8× |
| `cuda-console` | 4,096 | **35,488** | **110.5×** |

The GPU crosses over the single-env CPU at ~32 envs and scales near-linearly to
~2k envs. These numbers include the optimization pass documented in
[docs/gpu-scaling.md](docs/gpu-scaling.md) (+20% over the previous day's
measurement; earlier recorded runs and their CPU-baseline sensitivity are in
[the benchmark doc](docs/benchmark-gpu-vs-cpu.md)). For external scale: the
standard CPU stack (nes-py / gym-super-mario-bros) measures 132 env-steps/s on
the same machine (`benchmarks/nespy_baseline.py`), so 4,096 GPU envs ≈
**269× nes-py**.

A100-SXM4-80GB (Colab), device stepping via `step_device` — measured
2026-07-30 on this repo's HEAD:

| Envs | Env-steps/s | vs that VM's CPU (219/s) |
|---:|---:|---:|
| 4,096 | 311,844 | 1,422× |
| 16,384 | 1,050,710 | 4,798× |
| 32,768 | 1,931,460 | 8,819× |
| **65,536** | **3,142,205** | **14,348×** |
| 131,072 | 2,816,144 | past the saturation knee |

Peak ≈ 12.6M NES frames/s using 13 GB of the 80 GB card. At training scale
the PPO learner, not the emulator, is the bottleneck — by design. Earlier
per-mode ablations (RGB/RAM/no-copy): [phase-6 report](docs/phase6-report.md),
recorded 2026-05.

## Known limitations

- **NROM (mapper 0) only.** SMB works; other mappers aren't wired up.
- **Title-screen state machine has a PPU-timing bug.** The snapshot-reset path
  bypasses it cleanly; a real fix is deferred. See
  [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).
- **One CUDA thread per env** — simple to reason about, but warp divergence
  leaves performance on the table. Measured costs, prior art (CuLE), and a
  ranked redesign roadmap: [docs/gpu-scaling.md](docs/gpu-scaling.md) — its
  quick-win tier is already implemented; the deeper redesigns are open.

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
- [Design rationale](docs/design.md) — why the emulator lives inside the kernel
- [GPU scaling notes](docs/gpu-scaling.md) — beyond one-thread-per-env: measured costs, prior art, ranked roadmap
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

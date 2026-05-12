# NeSLE

**A GPU-native NES emulator and reinforcement-learning stack for Super Mario Bros.** NeSLE runs thousands of independent NES instances in parallel on one CUDA GPU and trains PPO agents against them. On a GTX 1050 Ti, the emulator alone hits **~29k env-steps/sec at 4096 parallel envs** (94× a single-env CPU baseline). On an A100, a real **75M-timestep PPO run on World 7-1 finished in 40 minutes** at 65,536 envs while the value head learned (`explained_variance` 0 → 0.66, `ep_return` 70 → 150).

## Quick start

Train a small PPO agent on World 1-1 in a few minutes:

```sh
# Stable-Baselines3 path — CPU rollout buffer, easiest to start with
python examples/sb3_train.py "Super Mario Bros. (World).nes" \
    --backend cuda --sb3-device cpu \
    --observation-mode ram --action-space simple \
    --reset-state-path docs/data/smb_level1_1.state \
    --num-envs 8 --timesteps 16384 --n-steps 128 --batch-size 256 \
    --max-episode-steps 256 --model-path nesle_ppo_smoke
```

For real scale (A100/H100), use the GPU-resident PPO path:

```sh
# Native PPO path — observations / rollouts / loss all on GPU via DLPack
python examples/native_ppo_train.py "Super Mario Bros. (World).nes" \
    --reset-state-path docs/data/smb_level1_1.state \
    --action-space simple --num-envs 1024 --total-timesteps 1_000_000 \
    --n-steps 128 --batch-size 8192 --hidden-size 256 \
    --checkpoint-path nesle_native_ppo.pt
```

Full options + Colab setup notes: **[`docs/training.md`](docs/training.md)**.

## What's in the box

- **CUDA-batched NROM emulator** (`cpp/bindings/cuda_module.cu`, `cpp/src/cuda/kernels.cu`). One CUDA thread per env runs 6502 + PPU + bus + OAM-DMA. Frame-skip happens inside the kernel.
- **Snapshot reset** (`docs/data/smb_level1_1.state` … `smb_level8_1.state`). Bundled FCEUX FCS save states bypass SMB's title-screen state machine; every env reset (including auto-reset on done) restores the snapshot in a single kernel launch.
- **Two PPO entry points:**
  - `examples/sb3_train.py` — Stable-Baselines3 PPO with VecMonitor, the natural starting point.
  - `examples/native_ppo_train.py` (delegates to `nesle.native_ppo`) — GPU-resident PPO that keeps observations, rollouts, GAE, and the optimizer on device using DLPack / `__cuda_array_interface__`. Bypasses SB3's CPU rollout buffer.
- **Curriculum support.** Pass `--reset-state-paths` (plural) with the 8 bundled saves and envs are round-robin-assigned across worlds.
- **Native CPU backend** (`nesle._core.NativeConsole`) for single-env debugging and parity testing.
- **65+ tests** covering FCS parser, env reset, render freshness, multi-level, GAE, and lifetime of device views.

## Build & install

```sh
python -m pip install -e '.[dev,rl]'
```

Build the CUDA extension after any C++ change:

```sh
# Linux/macOS — pick the right arch
NESLE_CUDA_ARCH=sm_80 sh scripts/build_cuda_extension.sh    # A100
NESLE_CUDA_ARCH=sm_90 sh scripts/build_cuda_extension.sh    # H100
```

**Windows + Pascal (e.g. GTX 1050 Ti):** the POSIX build script doesn't apply. CUDA Toolkit 13+ dropped Pascal support, so install a CTK 12.x sidecar alongside CTK 13. The manual `nvcc` recipe lives in the project memory as `windows_cuda_build_recipe.md` (Claude Code agents can read it; humans can use that file as a template).

PyTorch needs CUDA to use `--sb3-device cuda` or `native_ppo`. See `docs/training.md` § "PyTorch CUDA Setup".

## Verification

```sh
python -m pytest tests/                   # 65+ green
python benchmarks/gpu_vs_cpu.py           # GPU vs single-env CPU throughput
python benchmarks/verify_correctness.py   # falsifiability — confirms the batched kernel
                                          # runs N independent emulators (not a copy of env 0)
```

## Benchmarks

A100 / 80 GB (from `docs/phase6-report.md`):

| Mode | Envs | Steps/sec |
|---|---:|---:|
| `cuda-console`, RAM obs | 128 | ~560k |
| `cuda-console`, no-copy reward | 16,384 | ~103M |

GTX 1050 Ti / 4 GB (from `docs/benchmark-gpu-vs-cpu.md`):

| Mode | Envs | Env-steps/sec | vs CPU |
|---|---:|---:|---:|
| native CPU baseline | 1 | 308 | 1.00× |
| `cuda-console` | 32 | 535 | 1.74× |
| `cuda-console` | 256 | 4,027 | 13× |
| `cuda-console` | 4,096 | 29,058 | **94×** |

End-to-end PPO with the GPU-resident loop on A100 reached **31k env-steps/sec** sustained for a 75M-timestep training run (40 min wall-clock).

## Known limitations

- **NROM only.** SMB and a handful of other mapper-0 games. Other mappers aren't wired up.
- **Title-screen state machine has a PPU-timing bug.** Real fix is deferred; the snapshot-reset path bypasses it cleanly for SMB-style training. Documented in [`CLAUDE.md`](CLAUDE.md).
- **Default reward function is minimal.** `nesle.smb.compute_reward` clips x-delta to ±5 (so walking and running look identical to the agent) and only penalizes death by `-25`. Real training runs will need a shaped reward (e.g. uncap x-delta, add level-completion bonus); the infrastructure is there but the policy isn't shipped yet.
- **One CUDA thread per env.** Easy to reason about, leaves perf on the table for very large batches. See `docs/phase6-report.md` § "Next optimization targets".

## Documents

- [Training guide](docs/training.md) — primary entry point for RL work
- [Architecture](docs/architecture.md) — system design
- [A100 benchmark report](docs/phase6-report.md)
- [GPU vs CPU benchmark (1050 Ti)](docs/benchmark-gpu-vs-cpu.md)
- [Research notes](docs/research-notes.md) — design rationale and NES hardware background
- [CPU validation](docs/cpu-validation.md) — Klaus 6502 functional test gate
- [Headless runner](docs/headless-runner.md) — low-level ROM runner for debugging
- [Project history](docs/history/) — archived phase-by-phase development docs

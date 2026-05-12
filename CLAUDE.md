# NeSLE — Claude Code onboarding

NeSLE is a **GPU-native NES emulator + reinforcement-learning stack** focused on Super Mario Bros. The cuda-console backend runs thousands of independent NES instances in parallel on one GPU; PPO trains against them via either Stable-Baselines3 or a custom GPU-resident loop (`nesle.native_ppo`).

## Where to start

- **Train an agent:** [`docs/training.md`](docs/training.md) — the practical "make Mario learn" path.
- **System design:** [`docs/architecture.md`](docs/architecture.md) — how the layers fit together.
- **Throughput numbers:** [`docs/phase6-report.md`](docs/phase6-report.md) (A100) and [`docs/benchmark-gpu-vs-cpu.md`](docs/benchmark-gpu-vs-cpu.md) (GTX 1050 Ti).
- **Historical phase docs:** [`docs/history/`](docs/history/).

## Rebuilding the CUDA extension

The Python package wraps a C++/CUDA extension at `cpp/bindings/cuda_module.cu`. After any C++ change, rebuild:

- **Linux/macOS:** `sh scripts/build_cuda_extension.sh` (set `NESLE_CUDA_ARCH=sm_80` for A100, `sm_90` for H100).
- **Windows:** Linux script is POSIX-only; use the recipe in the project memory file `windows_cuda_build_recipe.md`. CTK 12.x is required for Pascal GPUs (sm_61); CTK 13+ dropped support.

## Running tests

```sh
.venv\Scripts\python.exe -m pytest tests/        # Windows
python -m pytest tests/                          # Linux/macOS
```

Expect 65+ green. Tests that need a CUDA GPU or the SMB ROM auto-skip when those aren't present.

## Known design choices and caveats

- **Title-screen bypass.** SMB's title→game transition stalls on our PPU (a real timing bug). The standard workaround — load any of the 8 bundled save states in `docs/data/smb_level{1..8}_1.state` via `reset_state_path=...` — drops every env directly into gameplay. This is the default path for training and there's no current reason to debug the title screen.
- **Reward function is intentionally minimal.** `nesle.smb.compute_reward` clips x-delta to ±5 (so walking and running look the same to the agent) and only penalizes death by `-25`. Real training runs will need a shaped reward; this is on the roadmap but out of scope of the emulator.
- **NROM only.** SMB is mapper 0; other mappers aren't wired up.

## Recent direction

The most recent additions are the GPU-resident PPO loop (`src/nesle/native_ppo.py`) and the DLPack / `__cuda_array_interface__` bridge in `cpp/bindings/cuda_module.cu` (`step_device`, `ram_device`, `reset_device`). These let observations stay on the GPU end-to-end — useful on A100/H100 where the SB3 path's CPU rollout buffer becomes a bottleneck.

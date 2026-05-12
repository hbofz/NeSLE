# NeSLE A100 Benchmark Campaign

This project can use NeSLE as the CUDA emulator and PPO engine for large A100
limits runs.  The Stable-Retro environment remains available for baseline
training, but `mario-nesle-bench` is the preferred path for measuring NeSLE
throughput and GPU-resident PPO.

## Colab Setup

Use an A100 80GB runtime and keep artifacts in Drive:

```text
notebooks/nesle_a100_benchmark.ipynb
```

The notebook is the easiest path: set the GitHub URL, Drive ROM path, and run
the cells from top to bottom.

For a shell-only Colab setup:

```bash
git clone https://github.com/hbofz/Nesle-codex.git /content/Nesle-codex
cd /content/Nesle-codex/project/mario-rl-ram
python -m pip install -e .
```

The benchmark CLI auto-discovers the NeSLE root when this repo is nested under
`Nesle-codex`.  If your layout is different, pass `--nesle-root`.

## One-Command Campaign

```bash
mario-nesle-bench all \
  --setup \
  --run-correctness \
  --cuda-arch sm_80 \
  --rom "roms/Super Mario Bros. (World).nes" \
  --snapshot ../../docs/data/smb_level1_1.state \
  --output-dir /content/drive/MyDrive/mario_rl/nesle_a100
```

This writes:

- `nesle_a100_preflight.json` when running `preflight`
- `nesle_a100_limits.json`
- `nesle_a100_limits.csv`
- `nesle_a100_native_ppo.json`
- `nesle_a100_report.md`
- native PPO checkpoints under `checkpoints/`

## What It Measures

`env-sweep` uses `nesle._cuda_core.CudaBatch` directly with snapshot reset and
no RGB copies.  It sweeps:

```text
1, 8, 32, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 98304, 131072
```

After that, it doubles the env count while free VRAM remains above 8GB, up to
`--max-extra-envs`.

`ppo-sweep` runs short NeSLE native PPO jobs at:

```text
1024, 4096, 8192, 16384, 32768, 65536, 98304, 131072
```

Each case defaults to two PPO updates with `n_steps=128`, `batch_size=8192`,
`hidden_size=256`, and NeSLE's `mario` action space, which mirrors this
project's 11 curated Mario controls.

## Individual Commands

Preflight only:

```bash
mario-nesle-bench preflight --setup --run-correctness \
  --rom "roms/Super Mario Bros. (World).nes" \
  --snapshot ../../docs/data/smb_level1_1.state
```

Environment throughput only:

```bash
mario-nesle-bench env-sweep \
  --rom "roms/Super Mario Bros. (World).nes" \
  --snapshot ../../docs/data/smb_level1_1.state
```

Native PPO short sweep:

```bash
mario-nesle-bench ppo-sweep \
  --rom "roms/Super Mario Bros. (World).nes" \
  --snapshot ../../docs/data/smb_level1_1.state
```

Long stress run:

```bash
mario-nesle-bench stress \
  --stress-envs 65536 \
  --stress-timesteps 75000000 \
  --rom "roms/Super Mario Bros. (World).nes" \
  --snapshot ../../docs/data/smb_level1_1.state
```

## Current Scope

- Phase 1 uses NeSLE's current minimal CUDA reward to isolate throughput.
- Phase 2 should port the smart reward profile into a Torch/GPU RAM reward
  adapter so native PPO can train with richer shaping without copying RAM to
  CPU.
- The benchmark uses NeSLE's compact RAM observations and GPU-resident native
  PPO.  Stable-Retro/SB3 remains the baseline/reference project path.

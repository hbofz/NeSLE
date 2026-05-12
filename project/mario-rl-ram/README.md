# Mario RL: RAM and CNN Agents for Super Mario Bros.

This project trains reinforcement learning agents to play **Super Mario Bros.**
with [Stable-Retro](https://github.com/Farama-Foundation/stable-retro) and
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/).  It supports two
observation pipelines:

- **RAM agent:** reads Stable-Retro's RAM vector and trains an MLP policy.
- **CNN agent:** reads game frames, preprocesses them to 84x84 grayscale frame
  stacks, and trains a CNN policy.

The project started with World 1-1 and now includes a per-level reward-profile
system, including a specialized World 7-1 profile for cannon and Hammer Bros.
training.

## ROM Setup

The project expects a Super Mario Bros. `.nes` ROM under `roms/`.  Import it
into Stable-Retro before running training or evaluation:

```bash
python -m stable_retro.import roms/
```

The `models/`, `runs/`, and `videos/` directories are ignored because they
contain large generated artifacts.

## Project Highlights

| Area | What this project does |
|---|---|
| Environment | Builds a Gymnasium-compatible Stable-Retro Mario environment. |
| Algorithms | Supports PPO and RecurrentPPO through Stable-Baselines3/SB3-Contrib. |
| Observations | Supports RAM vectors and pixel-frame stacks. |
| Actions | Uses a curated 11-action Mario controller space. |
| Rewards | Uses dense smart rewards with map-specific profiles. |
| Training | Provides local and Colab commands with checkpoint auto-resume. |
| Evaluation | Records videos and prints episode-level reward component totals. |
| NeSLE A100 | Benchmarks the GPU-native NeSLE emulator and native PPO path on A100. |

## Quick Start

Requires Python 3.10, 3.11, or 3.12.  The local setup uses `uv`:

```bash
uv sync
source .venv/bin/activate
python -m stable_retro.import roms/
mario-doctor
```

Smoke-test both observation modes:

```bash
mario-smoke --state Level1-1 --obs-mode ram --steps 300 --reward-mode smart
mario-smoke --state Level1-1 --obs-mode pixel --steps 300 --reward-mode smart
```

## Train

Small local smoke runs:

```bash
mario-train --obs-mode ram --state Level1-1 --timesteps 20000 --n-envs 2 --run-name local-ram
mario-train --obs-mode pixel --state Level1-1 --timesteps 20000 --n-envs 2 --run-name local-cnn
```

Longer Colab-style runs:

```bash
# RAM + MLP baseline
mario-train \
  --obs-mode ram \
  --state Level1-1 \
  --reward-mode smart \
  --timesteps 10000000 \
  --n-envs 16 \
  --run-name ram-1-1 \
  --device cpu

# Pixel + CNN agent
mario-train \
  --obs-mode pixel \
  --state Level1-1 \
  --reward-mode smart \
  --timesteps 5000000 \
  --n-envs 8 \
  --n-steps 128 \
  --batch-size 512 \
  --run-name cnn-1-1 \
  --device auto

# Specialized Level 7-1 training
mario-train \
  --obs-mode pixel \
  --state Level7-1 \
  --reward-mode smart \
  --timesteps 5000000 \
  --n-envs 8 \
  --run-name cnn-7-1 \
  --device auto
```

Training auto-resumes from the latest checkpoint in the selected run directory
unless `--no-auto-resume` is passed.

By default, training is locked to the selected `--state`.  `mario-train` uses
`--flag-lock` by default, so when Mario clears the flag and the game transitions
to the next level, the episode ends and resets back to the original state.  For
example, `--state Level7-1` repeatedly trains only on Level 7-1 unless
`--no-flag-lock` is used.

## NeSLE A100 Benchmarking

For the GPU-native NeSLE campaign, use the `mario-nesle-bench` CLI.  It runs
preflight checks, no-copy CUDA env throughput sweeps, short native-PPO sweeps,
and stress runs while writing JSON/CSV/report artifacts.

The ready-to-run Colab notebook is
[`notebooks/nesle_a100_benchmark.ipynb`](notebooks/nesle_a100_benchmark.ipynb).

```bash
mario-nesle-bench all \
  --setup \
  --run-correctness \
  --cuda-arch sm_80 \
  --rom "roms/Super Mario Bros. (World).nes" \
  --snapshot ../../docs/data/smb_level1_1.state \
  --output-dir /content/drive/MyDrive/mario_rl/nesle_a100
```

See [NeSLE A100 Benchmark Campaign](docs/NESLE_A100.md).

## Evaluate

```bash
mario-eval \
  --model models/cnn-1-1/final_model.zip \
  --obs-mode pixel \
  --state Level1-1 \
  --episodes 5 \
  --reward-mode smart \
  --video-dir videos/cnn-1-1
```

`mario-eval` prints score, coins, max x-position, final info, and
`smart_reward_total`, which is the sum of reward components across the whole
episode.

## Repository Structure

```text
src/mario_rl/
|-- env.py              # Environment factory used by every CLI/script
|-- wrappers.py         # Action, observation, frame-stack, and episode wrappers
|-- rewards.py          # Reward wrapper implementation
|-- levels.py           # State name -> reward profile lookup
|-- reward_profiles/    # Per-map reward tuning files
`-- scripts/            # mario-doctor, mario-smoke, mario-train, mario-eval

notebooks/
|-- ram_training.ipynb  # Colab notebook for RAM + MLP PPO
`-- cnn_training.ipynb  # Colab notebook for pixels + CNN PPO

docs/
|-- ARCHITECTURE.md     # How the system is wired internally
|-- ACTIONS.md          # Curated Mario action space
|-- REWARD.md           # Reward shaping and per-level profiles
|-- NESLE_A100.md       # NeSLE limits-first A100 benchmark campaign
|-- COLAB.md            # Colab training guide
`-- PC_SETUP.md         # Local/WSL setup guide
```

## Documentation Map

- [Architecture](docs/ARCHITECTURE.md): environment pipeline, wrappers, CLIs, and
  training flow.
- [Reward Design](docs/REWARD.md): smart reward components and per-level profile
  system.
- [Action Space](docs/ACTIONS.md): the discrete controller actions used for
  training.
- [NeSLE A100 Benchmark Campaign](docs/NESLE_A100.md): limits-first benchmark
  and native-PPO workflow for A100.
- [Colab Guide](docs/COLAB.md): recommended notebook/runtime workflow.
- [PC Setup](docs/PC_SETUP.md): local Windows/WSL/Linux setup and commands.

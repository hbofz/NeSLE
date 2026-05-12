# PC Setup Guide

This guide is for running the project outside Colab on Windows, WSL2, Linux, or
macOS.  WSL2 Ubuntu is the recommended Windows path.

## WSL2 Ubuntu

Install WSL2 from PowerShell as Administrator:

```powershell
wsl --install -d Ubuntu
```

Inside Ubuntu:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip build-essential ffmpeg
git clone https://github.com/hbofz/mario-rl-ram.git
cd mario-rl-ram
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Put your legally obtained `.nes` ROM in `roms/`, then import it:

```bash
mkdir -p roms
python -m stable_retro.import roms/
```

## macOS Or Linux With `uv`

```bash
uv sync
source .venv/bin/activate
python -m stable_retro.import roms/
```

Apple Silicon uses the `stable-retro-apple-silicon` dependency from
`pyproject.toml`.

## Native Windows

Native Windows is less tested for Stable-Retro.  If install/import fails, use
WSL2.

```powershell
git clone https://github.com/hbofz/mario-rl-ram.git
cd mario-rl-ram
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
python -m stable_retro.import roms/
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Verify Setup

```bash
mario-doctor
mario-smoke --state Level1-1 --obs-mode ram --steps 300 --reward-mode smart
mario-smoke --state Level1-1 --obs-mode pixel --steps 300 --reward-mode smart
```

## Local Training Examples

```bash
# RAM + MLP
mario-train \
  --obs-mode ram \
  --state Level1-1 \
  --reward-mode smart \
  --timesteps 20000000 \
  --n-envs 16 \
  --n-steps 512 \
  --batch-size 2048 \
  --run-name ram-1-1 \
  --model-dir models \
  --log-dir runs \
  --device cpu

# Pixel + CNN
mario-train \
  --obs-mode pixel \
  --state Level1-1 \
  --reward-mode smart \
  --timesteps 5000000 \
  --n-envs 8 \
  --n-steps 128 \
  --batch-size 512 \
  --run-name cnn-1-1 \
  --model-dir models \
  --log-dir runs \
  --device auto
```

If the machine struggles, reduce parallel environments:

```bash
--n-envs 8 --batch-size 1024
```

## Resume Training

Checkpoints save under `models/<run-name>/`.  Auto-resume is enabled by default
when you rerun the same command.  To force a specific checkpoint:

```bash
mario-train \
  --obs-mode pixel \
  --state Level1-1 \
  --reward-mode smart \
  --resume-from models/cnn-1-1/ppo_2500000_steps.zip \
  --timesteps 2500000 \
  --n-envs 8 \
  --run-name cnn-1-1 \
  --model-dir models \
  --log-dir runs \
  --device auto
```

The observation mode, action mode, algorithm, and level should match the
checkpoint.

## TensorBoard

```bash
tensorboard --logdir runs
```

Open the printed URL, usually `http://localhost:6006`.

## Evaluate And Make Videos

```bash
mario-eval \
  --model models/cnn-1-1/final_model.zip \
  --obs-mode pixel \
  --state Level1-1 \
  --episodes 5 \
  --reward-mode smart \
  --video-dir videos/cnn-1-1
```

Use a new `--video-dir` for each run because Gymnasium warns when overwriting
existing videos.

## Troubleshooting

- `Game not found`: run `python -m stable_retro.import roms/`.
- `Module not found`: activate the venv, then run `pip install -e .`.
- Training is slow: lower `--n-envs` or use Colab for pixel training.
- GPU is idle during RAM training: expected; emulator rollout collection is CPU
  bound.
- Evaluation fails to load a model: check the checkpoint path and `--obs-mode`.

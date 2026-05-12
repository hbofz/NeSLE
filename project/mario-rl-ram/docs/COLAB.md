# Colab Training Guide

Use Colab for longer runs, especially pixel/CNN training.  Store checkpoints,
TensorBoard logs, and videos in Google Drive so they survive runtime resets.

## 1. Runtime

- RAM + MLP: CPU is usually fine; GPU can still help PPO tensor operations.
- Pixel + CNN: use a GPU runtime. T4 works; A100/H100 is faster.

## 2. Mount Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

Recommended Drive layout:

```text
/content/drive/MyDrive/mario_rl/
|-- models/
|-- runs/
|-- videos/
`-- roms/       # optional Drive copy of your .nes ROM
```

## 3. Clone And Install

```bash
git clone https://github.com/hbofz/mario-rl-ram.git /content/mario-rl-ram
cd /content/mario-rl-ram
pip install -e .
```

If you are rerunning an existing Colab session:

```bash
git -C /content/mario-rl-ram pull
cd /content/mario-rl-ram
pip install -e .
```

Restart the runtime after installing if Colab has dependency conflicts loaded.

## 4. Import The ROM

Import the ROM from the repo's `roms/` folder:

```bash
python -m stable_retro.import /content/mario-rl-ram/roms/
```

If you keep your ROM in Drive instead, replace the path with
`/content/drive/MyDrive/mario_rl/roms/`.

Verify setup:

```bash
mario-doctor
mario-smoke --state Level1-1 --obs-mode ram --steps 300 --reward-mode smart
mario-smoke --state Level1-1 --obs-mode pixel --steps 300 --reward-mode smart
```

## 5. Train RAM + MLP

```bash
mario-train \
  --obs-mode ram \
  --state Level1-1 \
  --reward-mode smart \
  --action-mode mario \
  --timesteps 10000000 \
  --n-envs 16 \
  --n-steps 512 \
  --batch-size 2048 \
  --run-name ram-1-1 \
  --model-dir /content/drive/MyDrive/mario_rl/models \
  --log-dir /content/drive/MyDrive/mario_rl/runs \
  --device cpu
```

RAM observations train quickly because the policy is a small MLP.

## 6. Train Pixel + CNN

```bash
mario-train \
  --obs-mode pixel \
  --state Level1-1 \
  --reward-mode smart \
  --action-mode mario \
  --timesteps 5000000 \
  --n-envs 8 \
  --n-steps 128 \
  --batch-size 512 \
  --run-name cnn-1-1 \
  --model-dir /content/drive/MyDrive/mario_rl/models \
  --log-dir /content/drive/MyDrive/mario_rl/runs \
  --device auto
```

Pixel observations use NatureCNN and benefit strongly from GPU.

## 7. Train A Different Level

The `--state` argument chooses the level.  With `--reward-mode smart`, the
matching reward profile is selected automatically.

Training stays on that selected level by default.  `mario-train` uses
`--flag-lock`, so clearing the flag ends the episode and resets back to the
same `--state` instead of continuing into the next level.  Only use
`--no-flag-lock` if you intentionally want the game to continue after a level
clear.

Example Level 7-1 run:

```bash
mario-train \
  --obs-mode pixel \
  --state Level7-1 \
  --reward-mode smart \
  --action-mode mario \
  --timesteps 5000000 \
  --n-envs 8 \
  --n-steps 128 \
  --batch-size 512 \
  --run-name cnn-7-1 \
  --model-dir /content/drive/MyDrive/mario_rl/models \
  --log-dir /content/drive/MyDrive/mario_rl/runs \
  --device auto
```

## 8. Resume Training

Auto-resume is on by default.  If Colab disconnects, rerun the same
`mario-train` command.  The script scans the selected run directory for the
latest checkpoint and continues from it.

To resume a specific checkpoint:

```bash
mario-train \
  --obs-mode pixel \
  --state Level1-1 \
  --reward-mode smart \
  --resume-from /content/drive/MyDrive/mario_rl/models/cnn-1-1/ppo_2500000_steps.zip \
  --timesteps 2500000 \
  --n-envs 8 \
  --run-name cnn-1-1 \
  --model-dir /content/drive/MyDrive/mario_rl/models \
  --log-dir /content/drive/MyDrive/mario_rl/runs \
  --device auto
```

The algorithm, observation mode, action mode, and level should match the
checkpoint.

## 9. Evaluate And Record Videos

```bash
mario-eval \
  --model /content/drive/MyDrive/mario_rl/models/cnn-1-1/final_model.zip \
  --obs-mode pixel \
  --state Level1-1 \
  --episodes 5 \
  --reward-mode smart \
  --video-dir /content/drive/MyDrive/mario_rl/videos/cnn-1-1
```

For stochastic rollouts:

```bash
mario-eval \
  --model /content/drive/MyDrive/mario_rl/models/cnn-1-1/final_model.zip \
  --obs-mode pixel \
  --state Level1-1 \
  --episodes 20 \
  --reward-mode smart \
  --video-dir /content/drive/MyDrive/mario_rl/videos/cnn-1-1-stochastic \
  --no-deterministic
```

For a submission, useful videos are: random baseline, early checkpoint,
mid-training checkpoint, final deterministic rollout, and final stochastic
rollouts.

## 10. Optional Recurrent PPO

```bash
mario-train \
  --algo recurrent-ppo \
  --obs-mode ram \
  --state Level1-1 \
  --reward-mode smart \
  --timesteps 5000000 \
  --n-envs 8 \
  --run-name recurrent-ram-1-1 \
  --device cpu
```

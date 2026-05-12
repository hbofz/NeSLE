# Training

This document is the practical "make Mario learn" path for the current code.
The emulator is now past the title-screen/reset blocker: use snapshot reset,
RAM observations, and SB3 `VecMonitor` logging.

## Mental Model

There are two GPU layers:

- **NeSLE CUDA emulator:** implemented by `nesle._cuda_core` and selected with
  `backend="cuda"`. This runs thousands of NES instances on CUDA.
- **PyTorch policy training:** selected by SB3's `device` argument or
  `--sb3-device`. This controls where PPO's neural net runs.

These are independent. You can have NeSLE stepping envs on CUDA while PyTorch is
CPU-only if the wrong PyTorch wheel is installed. The current local venv has now
been switched to a CUDA wheel that works on the GTX 1050 Ti:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Observed locally:

```text
2.11.0+cu126
12.6
True
NVIDIA GeForce GTX 1050 Ti
```

So `backend="cuda"` runs the emulator on CUDA, and `--sb3-device cuda` places
SB3/PyTorch policy work on CUDA too.

For RAM observations with SB3's default `MlpPolicy`, `--sb3-device cpu` can be
faster even on an A100. NeSLE still runs the emulator on CUDA; only PPO's small
policy network and rollout update step stay on CPU. SB3 stores VecEnv rollouts
as CPU NumPy arrays, and a small MLP often does not provide enough work to
offset CPU-to-GPU transfer and kernel-launch overhead. Use `--sb3-device cuda`
mainly for RGB/CNN policies or after measuring that it wins for the current
configuration.

The custom native path bypasses SB3's CPU rollout buffer:

- `nesle._cuda_core.CudaBatch.step_device(...)` consumes CUDA action-mask tensors.
- `_cuda_core` exposes RAM/reward/done buffers through DLPack and
  `__cuda_array_interface__`.
- `nesle.native_ppo` converts those buffers to PyTorch CUDA tensors and keeps the
  PPO rollout buffer, GAE, clipped policy loss, value loss, entropy term, and
  optimizer step on CUDA.

That path is the preferred experiment when the goal is a truly GPU-resident RAM
policy loop. It still uses Python as the PPO coordinator, but the per-step
observations and rollout tensors stay on the GPU.

## PyTorch CUDA Setup

Use the official PyTorch selector for the current command:

```text
https://pytorch.org/get-started/locally/
```

General cleanup flow:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio
```

Then install the command shown by the selector for:

- OS: Windows
- Package: Pip
- Language: Python
- Compute platform: CUDA

For the local GTX 1050 Ti, `cu126` works:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
```

`cu128` was tested and is not compatible with this Pascal card: it detects the
GPU, but CUDA tensor ops fail with `no kernel image is available for execution
on the device` because the wheel does not include `sm_61` kernels.

If CUDA wheels are not available for the Python version in `.venv`, create a
Python 3.12 venv and install `.[dev,rl]` there. This project currently works in
Python 3.14 for CPU-side tests, but PyTorch CUDA wheel support can lag newer
Python versions.

After install, this must print `True` and a GPU name:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

## CUDA Toolkit Note

The local GPU is a GTX 1050 Ti (`sm_61`, Pascal). CUDA Toolkit 13.x dropped
offline compilation support for Pascal. To rebuild NeSLE's CUDA extension for
this card, use CUDA Toolkit 12.x and set:

```powershell
$env:NESLE_CUDA_ARCH = "sm_61"
```

For larger training machines:

```bash
export NESLE_CUDA_ARCH=sm_80  # A100
export NESLE_CUDA_ARCH=sm_90  # H100
```

## Single-Level Smoke

Start with W1-1. The snapshot lands in active gameplay, avoiding the old
title-screen START workaround.

```powershell
.\.venv\Scripts\python.exe examples\sb3_train.py "Super Mario Bros. (World).nes" `
  --backend cuda `
  --observation-mode ram `
  --reset-state-path docs\data\smb_level1_1.state `
  --action-space simple `
  --num-envs 512 `
  --timesteps 100000 `
  --n-steps 128 `
  --batch-size 256 `
  --max-episode-steps 512 `
  --model-path nesle_ppo_w1_1
```

If PyTorch CUDA is installed, add:

```powershell
  --sb3-device cuda
```

The script prints a startup line like:

```text
nesle_backend=cuda-console observation_mode=ram sb3_device=cpu torch=... torch_cuda=...
```

For full GPU training, expect `nesle_backend=cuda-console`, `sb3_device=cuda`,
and a real `torch_cuda` GPU name.

## CUDA-Native PPO Smoke

After rebuilding `_cuda_core`, run the custom PPO path:

```powershell
.\.venv\Scripts\python.exe examples\native_ppo_train.py "Super Mario Bros. (World).nes" `
  --reset-state-path docs\data\smb_level1_1.state `
  --action-space simple `
  --num-envs 1024 `
  --total-timesteps 100000 `
  --n-steps 128 `
  --batch-size 8192 `
  --max-episode-steps 512 `
  --checkpoint-path nesle_native_ppo.pt
```

For a larger CUDA box, increase `--num-envs` to 4096 or higher after checking
VRAM headroom. The script prints update FPS, PPO losses, approximate clip
fraction, explained variance, and recent episode returns/lengths.

## Evaluate A Model

```powershell
.\.venv\Scripts\python.exe examples\eval_smoke.py --model nesle_ppo_w1_1 --steps 500
```

Good signs:

- `max x_pos` advances meaningfully beyond the snapshot start.
- total reward trends positive.
- the action histogram uses right-moving actions.

Bad signs:

- action histogram collapses to `NOOP` or `left`.
- `max x_pos` barely moves.
- evaluation reward stays near zero or negative.

Those signs mean the infrastructure works but the policy has not learned yet.

## Multi-Level Curriculum

Use all bundled World N-1 snapshots:

```powershell
.\.venv\Scripts\python.exe examples\sb3_train.py "Super Mario Bros. (World).nes" `
  --backend cuda `
  --observation-mode ram `
  --reset-state-paths `
    docs\data\smb_level1_1.state docs\data\smb_level2_1.state `
    docs\data\smb_level3_1.state docs\data\smb_level4_1.state `
    docs\data\smb_level5_1.state docs\data\smb_level6_1.state `
    docs\data\smb_level7_1.state docs\data\smb_level8_1.state `
  --action-space simple `
  --num-envs 4096 `
  --timesteps 10000000 `
  --n-steps 128 `
  --batch-size 256 `
  --max-episode-steps 1024 `
  --model-path nesle_ppo_curriculum
```

Without `env_to_level`, envs are assigned round-robin across snapshot paths.
Use explicit `env_to_level` from Python when you want a fixed curriculum ratio.

## Debug Checklist

Run these before trusting a long training job:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe benchmarks\verify_correctness.py
.\.venv\Scripts\python.exe benchmarks\gpu_vs_cpu.py
```

If training logs do not show rollout metrics, make sure `examples/sb3_train.py`
is wrapping the env with `VecMonitor`. The current script does this already.

If `--sb3-device cuda` fails, check PyTorch first, not NeSLE:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Colab A100 Notebook

Use the checked-in notebook for the A100 run:

```text
notebooks/nesle_colab_a100_training.ipynb
```

It does the standard Colab flow:

- mounts Google Drive,
- clones or updates the repo,
- installs `.[dev,rl]`,
- builds `_cuda_core` with `NESLE_CUDA_ARCH=sm_80`,
- verifies snapshot reset,
- trains W1-1 and multi-level curriculum PPO,
- writes checkpoints, final models, and TensorBoard logs to Drive,
- resumes from the latest checkpoint,
- runs `examples/eval_smoke.py`.

If the GitHub repo is private, create a Colab secret named `GITHUB_TOKEN` with
read access to the repo before running the clone cell.

The notebook expects your ROM in Drive, by default:

```text
/content/drive/MyDrive/nesle/roms/Super Mario Bros. (World).nes
```

The ROM is intentionally not committed to Git.

The notebook defaults `SB3_DEVICE = 'cpu'` for RAM-observation PPO while keeping
`backend cuda` for the emulator. It also includes an SB3 device probe cell that
tests CPU versus CUDA policy placement on the current Colab runtime.

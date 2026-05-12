# NeSLE - Current State And Next Steps

> Context: the pre-training blockers are no longer the main problem. The CUDA
> emulator works, snapshot reset starts episodes in playable SMB levels, and
> local GPU throughput has been measured. The next milestone is turning this
> into a learning run, then moving it to a larger CUDA box with PyTorch on GPU.

---

## 1. What Works Now

- `backend="cuda"` routes the public Python vector API through the ROM-backed
  `cuda-console` backend.
- `observation_mode="ram"` returns compact 2 KB CPU RAM observations for SB3
  without copying full RGB frames every step.
- `reset_state_path` restores a Stable Retro/FCEUX `.state` directly onto the
  CUDA batch state, bypassing the fragile title-screen/start-sequence path.
- `reset_state_paths` supports curriculum training with multiple SMB levels.
  The current bundled states are `docs/data/smb_level1_1.state` through
  `docs/data/smb_level8_1.state`.
- `render()` is fresh after no-copy throughput steps; it launches the CUDA
  render kernel before copying frames back.
- `examples/sb3_train.py` wraps `VecMonitor`, so SB3 logs episode reward and
  length once episodes finish or truncate.
- `benchmarks/verify_correctness.py` falsifies the scary benchmark failure
  modes: different actions diverge, per-env CPU work is plausible, and random
  envs produce distinct RAM hashes.

---

## 2. Local Verification

On this Windows checkout:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe benchmarks\verify_correctness.py
.\.venv\Scripts\python.exe benchmarks\gpu_vs_cpu.py
```

Latest observed local results:

- Python tests: `60 passed`
- Correctness benchmark: all three checks passed
- GTX 1050 Ti throughput: about `29.7k env-steps/s` at 4096 envs, roughly
  `102x` native single-env CPU for this benchmark

The POSIX scripts under `scripts/` are still useful on Linux/WSL/Git Bash, but
plain PowerShell does not provide `sh`, `c++`, or `/tmp`.

---

## 3. Why PyTorch Is Not On GPU Yet

There are two separate GPU pieces:

- **NeSLE emulator GPU:** working. `_cuda_core` loads and `backend="cuda"`
  runs CUDA kernels.
- **PyTorch/SB3 policy GPU:** working in the current venv after switching from
  the CPU wheel to `torch==2.11.0+cu126`.

That means both environment stepping and PPO's neural network placement can run
on GPU when training is launched with `--sb3-device cuda`.

Check with:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Use the official PyTorch install selector for the current command:

```text
https://pytorch.org/get-started/locally/
```

On this machine, also keep in mind:

- GPU is GTX 1050 Ti, compute capability `sm_61`.
- `torch==2.11.0+cu128` detects the GPU but cannot execute kernels on this
  card (`no kernel image is available`) because that wheel targets newer SMs.
- `torch==2.11.0+cu126` works locally: `torch.cuda.is_available() == True` and
  CUDA tensor ops run on the GTX 1050 Ti.
- CUDA Toolkit 13.x dropped offline compilation support for Pascal GPUs like
  the 1050 Ti; rebuilding NeSLE CUDA from source for this card still needs
  CUDA Toolkit 12.x.

---

## 4. Recommended Local Training Smoke

Start from a snapshot, keep RAM observations, and force short episodes so
`VecMonitor` produces visible rollout metrics quickly:

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

Then evaluate:

```powershell
.\.venv\Scripts\python.exe examples\eval_smoke.py --model nesle_ppo_w1_1 --steps 500
```

Things to watch:

- `ep_rew_mean` should trend positive.
- `max x_pos` in `eval_smoke.py` should advance beyond the snapshot start.
- If the action histogram collapses to `left` or `NOOP`, policy learning has
  not happened yet even if the infrastructure is working.

---

## 5. Curriculum Run

Once W1-1 smoke behaves, use all bundled World N-1 snapshots:

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

For explicit per-env assignment from Python, pass `env_to_level`; otherwise the
wrapper assigns envs round-robin across the snapshot list.

---

## 6. Larger GPU Plan

For A100/H100 or Colab:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,rl]"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

export NESLE_CUDA_ARCH=sm_80  # A100; use sm_90 for H100
sh scripts/build_cuda_extension.sh
python -m pytest tests -q
```

Then rerun the local smoke with `--sb3-device cuda`. Expect:

- `nesle_backend=cuda-console`
- `sb3_device=cuda`
- `torch_cuda=<GPU name>`

For Colab A100 specifically, open:

```text
notebooks/nesle_colab_a100_training.ipynb
```

The notebook mounts Google Drive, uses your ROM from Drive, builds with
`NESLE_CUDA_ARCH=sm_80`, trains with periodic checkpoints to Drive, supports
resume, and runs evaluation/TensorBoard cells.

---

## 7. File Reference

| File | Purpose |
|------|---------|
| `src/nesle/env.py` | Core Python API, snapshot args, VecEnv + single env |
| `src/nesle/actions.py` | Action space definitions |
| `src/nesle/smb.py` | Mario RAM addresses + reward function |
| `cpp/include/nesle/fcs.hpp` | FCEUX FCS save-state parser |
| `cpp/bindings/cuda_module.cu` | CUDA Python binding (`CudaBatch`) |
| `cpp/src/cuda/kernels.cu` | CUDA kernel implementations |
| `docs/data/smb_level*_1.state` | Bundled Stable Retro level snapshots |
| `examples/sb3_train.py` | SB3 training script |
| `examples/eval_smoke.py` | Trained-model rollout smoke |
| `notebooks/nesle_colab_a100_training.ipynb` | Drive-backed Colab/A100 training notebook |
| `benchmarks/gpu_vs_cpu.py` | Local CPU-vs-GPU throughput benchmark |
| `benchmarks/verify_correctness.py` | Falsifiability checks for GPU throughput |
| `docs/training.md` | Training and PyTorch GPU setup notes |

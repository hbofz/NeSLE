# NeSLE - Next Steps (NVIDIA GPU Testing & Training)

> Context: The pre-training audit blockers have been fixed and verified.
> CUDA env stepping, reward-only stepping, per-env reset, terminal observations,
> and START-button actions are covered by tests. The next major milestone is
> large-GPU training on Colab/A100/H100 with both `backend="cuda"` and
> `--sb3-device cuda`.

---

## 1. Setup on NVIDIA Machine

```bash
git clone git@github.com:hbofz/Nesle-codex.git
cd Nesle-codex

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,rl]"
```

For Colab, verify PyTorch sees the GPU before training:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

## 2. Build the CUDA Extension

```bash
pip install -e .

# Set NESLE_CUDA_ARCH for the GPU:
# - GTX 1050 Ti: sm_61, requires CUDA Toolkit 12.x
# - A100: sm_80
# - H100: sm_90
export NESLE_CUDA_ARCH=sm_80
sh scripts/build_cuda_extension.sh
```

If the build fails, check:

- `nvcc --version`
- The GPU compute capability at https://developer.nvidia.com/cuda-gpus
- `NESLE_CUDA_ARCH`

Note: CUDA 13 drops offline compilation for Pascal GPUs such as GTX 1050 Ti.
Use CUDA Toolkit 12.x for `sm_61`.

---

## 3. Run Verification

```bash
export PYTHONPATH=src

python -m pytest tests/ -v
sh scripts/verify.sh
sh scripts/verify_cuda.sh

export NESLE_ROM_PATH="/path/to/Super Mario Bros. (World).nes"
sh scripts/verify_phase4.sh
sh scripts/verify_phase5.sh
sh scripts/verify_native_binding.sh
```

Specific GPU checks:

| Test | What It Validates |
|------|-------------------|
| `python -m pytest tests/test_critical_fixes.py -v` | START button, auto-reset, numpy actions, CUDA `step_reward()` auto-reset |
| `python -c "from nesle._cuda_core import CudaBatch; b = CudaBatch(4, 4); print(b.name)"` | CUDA extension loads |
| `python -c "import nesle; e = nesle.make_vec('ROM.nes', 4, backend='cuda'); e.reset(); r = e.step([0,0,0,0]); print('dones:', r[2])"` | CUDA ROM step works |
| Step until envs finish and inspect `infos` | `terminal_observation` and auto-reset work |
| `python -c "from nesle._cuda_core import CudaBatch; import numpy as np; b = CudaBatch(4, 4); b.reset(); b.reset_envs(np.array([1,0,0,0], dtype=np.uint8)); print('ok')"` | Per-env reset kernel works |

---

## 4. Audit Follow-Ups

Resolved:

- `step_reward()` now implements CUDA auto-reset and returns
  `terminal_observation` for done envs.
- `tests/conftest.py` adds `src` to `sys.path` for pytest.
- Unused `previous_mario_dying` CUDA tracking was removed from buffers,
  reset snapshots, tests, and tools.
- `docker/cuda.Dockerfile` now defaults to ROM-independent pytest instead of
  `verify.sh`, which can require `$NESLE_ROM_PATH`.

---

## 5. Start Agent Training

Start with a small smoke before long runs:

```bash
python examples/sb3_train.py /path/to/rom.nes \
    --backend cuda \
    --sb3-device cuda \
    --observation-mode ram \
    --action-space simple_with_start \
    --num-envs 8 \
    --timesteps 10000 \
    --n-steps 128 \
    --batch-size 256 \
    --model-path nesle_cuda_smoke
```

Then scale up on A100/H100:

```bash
python examples/sb3_train.py /path/to/rom.nes \
    --backend cuda \
    --sb3-device cuda \
    --observation-mode ram \
    --action-space simple_with_start \
    --num-envs 64 \
    --timesteps 5000000 \
    --n-steps 128 \
    --batch-size 256 \
    --model-path nesle_cuda_ppo
```

Training tips:

- `ram` observation mode is much faster than `rgb_array`; use it first.
- `simple_with_start` includes START, NOOP, right, right+A, right+B,
  right+A+B, A, and left.
- Monitor `ep_rew_mean`; Mario's x-position reward should trend positive.
- If the agent gets stuck, verify it presses START at the beginning of episodes.
- The training script prints both the NeSLE backend and SB3/PyTorch device. For
  Colab A100/H100 runs, expect `nesle_backend=cuda-console` and `sb3_device=cuda`.

---

## 6. File Reference

| File | Purpose |
|------|---------|
| `src/nesle/env.py` | Core env implementation (VecEnv + single env) |
| `src/nesle/actions.py` | Action space definitions |
| `src/nesle/smb.py` | Mario RAM addresses + reward function |
| `cpp/bindings/cuda_module.cu` | CUDA Python binding (`CudaBatch`) |
| `cpp/src/cuda/kernels.cu` | CUDA kernel implementations |
| `cpp/include/nesle/cuda/batch_step.cuh` | Reward logic + per-env reset functions |
| `examples/sb3_train.py` | SB3 training script |
| `tests/test_critical_fixes.py` | Tests for audit fixes |
| `scripts/verify.sh` | Full verification suite |
| `docs/architecture.md` | Architecture overview |
| `docs/phases.md` | Project phase history |

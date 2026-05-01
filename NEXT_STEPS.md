# NeSLE — Next Steps (NVIDIA GPU Testing & Training)

> **Context**: A full audit was performed and 3 critical issues were fixed in commit
> `a341713` ("Fix critical pre-training issues: CUDA auto-reset, START button, loop guard").
> All Python and C++ host-side tests pass. The CUDA device code compiles as C++ but
> needs `nvcc` compilation and GPU testing on your NVIDIA machine.

---

## 1. Setup on NVIDIA Machine

```bash
# Clone the repo
git clone git@github.com:hbofz/Nesle-codex.git
cd Nesle-codex

# Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,rl]"
```

---

## 2. Build the CUDA Extension

```bash
# Build the native C++ pybind11 module
pip install -e .

# Build the CUDA extension (requires nvcc + CUDA toolkit)
# Set NESLE_CUDA_ARCH to match your GPU (e.g., sm_75 for RTX 2080, sm_86 for RTX 3080)
export NESLE_CUDA_ARCH=sm_75  # adjust for your GPU
sh scripts/build_cuda_extension.sh
```

If the build fails, check:
- `nvcc --version` to confirm CUDA toolkit is installed
- Your GPU's compute capability at https://developer.nvidia.com/cuda-gpus
- Set `NESLE_CUDA_ARCH` accordingly (e.g., `sm_75`, `sm_86`, `sm_89`)

---

## 3. Run the Full Verification Suite

```bash
# Set PYTHONPATH for scripts that don't use pip install
export PYTHONPATH=src

# Run all Python tests (including the new critical fix tests)
python -m pytest tests/ -v

# Run the full verification script
sh scripts/verify.sh

# If you have a Super Mario Bros ROM, set the path and run ROM-dependent tests
export NESLE_ROM_PATH=/path/to/Super\ Mario\ Bros.\ \(World\).nes
sh scripts/verify_phase4.sh
sh scripts/verify_phase5.sh
sh scripts/verify_native_binding.sh
```

### What to Verify Specifically

These are the **new features from the latest commit** that need GPU validation:

| Test | What It Validates |
|------|-------------------|
| `python -m pytest tests/test_critical_fixes.py -v` | START button, auto-reset contract, numpy array actions |
| `python -c "from nesle._cuda_core import CudaBatch; b = CudaBatch(4, 4); print(b.name)"` | CUDA extension loads |
| `python -c "import nesle; e = nesle.make_vec('ROM.nes', 4, backend='cuda'); e.reset(); r = e.step([0,0,0,0]); print('dones:', r[2])"` | CUDA step works |
| Step until an env dies, verify `terminal_observation` is in info | Auto-reset works |
| `python -c "from nesle._cuda_core import CudaBatch; import numpy as np; b = CudaBatch(4, 4); b.reset(); b.reset_envs(np.array([1,0,0,0], dtype=np.uint8))"` | Per-env reset kernel works |

---

## 4. Important Issues Still Open (from the audit)

These were identified but not yet fixed (severity: Important/Minor):

### I3: `step_reward()` Needs Auto-Reset Too
The `step_reward()` method (CUDA reward-only fast path) still doesn't implement auto-reset.
If you plan to use `step_reward()` for high-throughput training, it needs the same
`terminal_observation` + `reset_envs()` treatment that `step()` now has.

### I4: Add `conftest.py` for pytest
Create `tests/conftest.py` with:
```python
import sys
sys.path.insert(0, "src")
```
This lets `pytest` find the `nesle` package without setting `PYTHONPATH` manually.

### M4: Remove Unused `previous_mario_dying` Tracking
The `previous_mario_dying` field in `BatchBuffers` is tracked but never read in the
reward formula. Harmless dead state that can be cleaned up.

### M6: Docker Default CMD
`docker/cuda.Dockerfile` CMD runs `verify.sh` which needs `$NESLE_ROM_PATH`.
Consider changing the default CMD to run only ROM-independent tests.

---

## 5. Start Agent Training

Once verification passes:

```bash
# Basic PPO training with RAM observations (fastest)
python examples/sb3_train.py \
    --rom /path/to/rom.nes \
    --backend native \
    --observation-mode ram \
    --action-space simple_with_start \
    --num-envs 8 \
    --total-timesteps 1000000

# CUDA backend (if extension built successfully)
python examples/sb3_train.py \
    --rom /path/to/rom.nes \
    --backend cuda \
    --observation-mode ram \
    --action-space simple_with_start \
    --num-envs 64 \
    --total-timesteps 5000000
```

### Training Tips
- Start with `--backend native` to validate the training loop works before switching to CUDA
- `ram` observation mode is much faster than `rgb_array` — use it for initial experiments
- `simple_with_start` includes 8 actions (START, NOOP, right, right+A, right+B, right+A+B, A, left)
- Monitor `ep_rew_mean` in TensorBoard/stdout — Mario's x-position reward should trend positive
- If the agent seems stuck, check that it's actually pressing START at the beginning of episodes

---

## 6. File Reference

Key files you'll be working with:

| File | Purpose |
|------|---------|
| `src/nesle/env.py` | Core env implementation (VecEnv + single env) |
| `src/nesle/actions.py` | Action space definitions |
| `src/nesle/smb.py` | Mario RAM addresses + reward function |
| `cpp/bindings/cuda_module.cu` | CUDA Python binding (CudaBatch) |
| `cpp/src/cuda/kernels.cu` | CUDA kernel implementations |
| `cpp/include/nesle/cuda/batch_step.cuh` | Reward logic + per-env reset functions |
| `examples/sb3_train.py` | SB3 training script |
| `tests/test_critical_fixes.py` | Tests for the latest fixes |
| `scripts/verify.sh` | Full verification suite |
| `docs/architecture.md` | Architecture overview |
| `docs/phases.md` | Project phase history |

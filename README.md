# NeSLE

NeSLE is a GPU-native NES learning environment aimed at running thousands of
Super Mario Bros. instances on NVIDIA GPUs behind a Gymnasium/SB3-compatible
Python API.

The repository is intentionally staged. Phase 0 established the target
architecture, project layout, NROM/iNES parsing, Mario RAM decoding, reward
extraction, action mappings, and tests. Phases 1 and 2 built the portable CPU,
PPU, input, rendering, and OpenEmu reference gates. Phase 3 moved the emulator
correctness contract into CUDA batch execution. Phase 4 added the Gymnasium/SB3
Python API. Phases 5 and 6 package the ROM-backed CUDA console backend,
benchmarks, A100 results, and high-throughput RAM-observation training path.
The current training branch adds Stable Retro/FCEUX snapshot resets, multi-level
curriculum startup, render freshness fixes, and SB3 logging/evaluation helpers.

## Current Status

The current path covers the completed CPU emulator, CUDA batch execution,
Gymnasium/SB3 Python API, ROM-backed CUDA console backend, Phase 6 benchmark
package, and the first practical training unblock:

- 2A03/6502 state and official-opcode execution core
- Flat 64 KB test bus, NROM memory-map smoke tests, and NES console CPU bus
- RAM, PPU register, APU/input, and PRG ROM mirroring behavior
- Basic NTSC PPU timing, vblank/NMI delivery, OAMDMA stalls, and frame stepping
- Coarse sprite-0-hit behavior for early Super Mario Bros. boot progress
- CPU RGB frame rendering for background and sprite tiles
- Deterministic action traces with Mario RAM, reward, RAM hash, and frame hash
- OpenEmu/Nestopia save-state rendering bridge for reference-frame debugging
- OpenEmu screenshot comparison gate for local Nestopia reference captures
- Headless `.nes` boot runner for NROM smoke tests
- C++ tests for CPU execution, stack calls, branches, arithmetic, and NROM reads
- CUDA smoke for 4096-env reward/done batches
- CUDA device smoke for the shared CPU core, batch console stepping, OAM DMA,
  PPU timing, PPU register-fed RGB rendering, and device-side reset snapshot
  restore
- `NesleEnv` and `NesleVecEnv` Python wrappers with Gymnasium-style single-env
  reset/step and SB3-style vector reset/step/auto-reset semantics
- Native C++ console binding hook plus deterministic Python compatibility
  backend for API development without a packaged native runtime
- ROM-backed `cuda-console` backend that advances the CPU/PPU console loop on
  CUDA
- `observation_mode="ram"` for normal vector stepping without full RGB host
  copies, plus `render()` for explicit RGB frame capture
- FCEUX/Stable Retro `.state` snapshot reset through `reset_state_path`, which
  starts episodes directly in playable SMB levels instead of relying on the
  fragile title-screen boot path
- Multi-level curriculum reset through `reset_state_paths` and optional
  `env_to_level`, with bundled World 1-1 through World 8-1 snapshots in
  `docs/data/`
- CUDA render freshness regression coverage: explicit `render()` now launches
  the render kernel after no-copy throughput steps
- SB3 training helper wraps `VecMonitor`, exposes snapshot/curriculum flags, and
  prints the NeSLE backend plus the PyTorch device
- GPU-vs-CPU throughput and falsifiability checks for the local GTX 1050 Ti

## Quick Verification

```sh
sh scripts/verify.sh
```

On Windows, the Python tests are the most reliable first gate:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

The shell scripts under `scripts/` are POSIX-oriented and expect `sh`, `c++`,
and `/tmp`; use Git Bash/WSL or translate the individual commands when running
on plain PowerShell.

The ROM is not vendored in the repository. Keep your local `.nes` file outside
Git or in Google Drive, then pass its path to the scripts/notebook.

Phase 4 API checks can also be run directly:

```sh
python -m pip install -e '.[dev,rl]'
sh scripts/verify_phase4.sh
sh scripts/verify_native_binding.sh
```

`verify_native_binding.sh` compiles and imports the pybind extension, exercises
`NativeConsole`, and runs the native Python backend when the selected Python has
a complete NumPy install.

With a local Super Mario Bros. `.nes` file, run the optional real-ROM gate:

```sh
NESLE_ROM_PATH="/path/to/Super Mario Bros. (World).nes" sh scripts/smoke_user_rom.sh
NESLE_ROM_PATH="/path/to/Super Mario Bros. (World).nes" sh scripts/smoke_phase2_user_rom.sh
NESLE_ROM_PATH="/path/to/Super Mario Bros. (World).nes" sh scripts/smoke_phase4_user_rom.sh
NESLE_ROM_PATH="/path/to/Super Mario Bros. (World).nes" sh scripts/render_openemu_state.sh
NESLE_ROM_PATH="/path/to/Super Mario Bros. (World).nes" sh scripts/compare_openemu_state.sh
```

On an NVIDIA CUDA machine, run the optional device smoke:

```sh
sh scripts/verify_cuda.sh
```

That smoke compiles the CUDA kernels, launches a 4096-env reward/done batch,
runs a tiny on-device NROM CPU trace through the batch CPU bus, steps an
integrated CPU/PPU console path through OAM DMA, and verifies device-side reset
snapshot restore and byte-for-byte CPU/GPU RGB frame parity for a synthetic
background+sprite scene.

Phase 5 throughput smoke:

```sh
python -m pip install -e '.[dev,rl]'
NESLE_ROM_PATH="/path/to/Super Mario Bros. (World).nes" sh scripts/verify_phase5.sh
```

Full Phase 5 benchmark runs write JSON or CSV rows under `benchmarks/results/`:

```sh
PYTHONPATH=src python benchmarks/phase5_benchmark.py \
  "Super Mario Bros. (World).nes" \
  --env-counts 1,8,32,128,512,1024,2048,4096 \
  --steps 200 \
  --modes step,render,inference \
  --output benchmarks/results/phase5.json
```

Install `.[legacy-mario]` and add `--include-legacy` only for the slower
`gym-super-mario-bros` comparison rows. The legacy extra pins NumPy and Gym to
the old API versions required by `nes-py`.

To separate raw CUDA kernel throughput from the current Python backend, run:

```sh
NESLE_CUDA_ARCH=sm_80 sh scripts/benchmark_cuda_kernels.sh \
  --env-counts 1024,4096,8192,16384
```

The Python API benchmark reports packaged backend throughput. The CUDA kernel
benchmark reports the lower-level GPU reward/render kernels separately from the
ROM-backed `cuda-console` path.

To build the optional CUDA Python backend on an NVIDIA machine:

```sh
python -m pip install -e '.[dev,rl]'
NESLE_CUDA_ARCH=sm_80 sh scripts/build_cuda_extension.sh
PYTHONPATH=src python benchmarks/phase5_benchmark.py \
  "Super Mario Bros. (World).nes" \
  --backend cuda \
  --env-counts 1,2,8,32 \
  --steps 10 \
  --warmup-steps 2
```

When a ROM is supplied through the Python vector API, `backend="cuda"` uses the
ROM-backed `cuda-console` path, which advances the CUDA batch CPU/PPU console
loop to frame boundaries. The lower-level two-argument `CudaBatch` constructor
is kept for synthetic reward/render kernel calibration.

To measure reward/done throughput without copying RGB observations back to the
host on every step:

```sh
PYTHONPATH=src python benchmarks/phase5_benchmark.py \
  "Super Mario Bros. (World).nes" \
  --backend cuda \
  --modes reward \
  --env-counts 128,512,2048,4096,8192,16384
```

Current A100 calibration notes are tracked in
[docs/phase5-results.md](docs/phase5-results.md).

Local GTX 1050 Ti benchmark:

```sh
python benchmarks/gpu_vs_cpu.py
python benchmarks/verify_correctness.py
```

The benchmark compares native CPU single-env stepping with batched
`cuda-console` stepping from the W1-1 snapshot. The correctness script verifies
different actions diverge, each env runs plausible CPU work, and randomized
envs evolve into distinct RAM states. Latest local numbers are tracked in
[docs/benchmark-gpu-vs-cpu.md](docs/benchmark-gpu-vs-cpu.md).

## Target API

The end state is:

```python
import nesle

env = nesle.make_vec(
    rom_path="Super Mario Bros. (World).nes",
    num_envs=4096,
    action_space="simple",
    backend="cuda",
    render_mode="rgb_array",
    observation_mode="ram",
    reset_state_path="docs/data/smb_level1_1.state",
)

obs = env.reset()
obs, rewards, dones, infos = env.step(actions)
frames = env.render()
```

`observation_mode="ram"` keeps the normal vector `reset()`/`step()` contract
while returning compact 2 KB CPU RAM observations instead of copying full RGB
frames every step. RGB frames remain available through `render()` for debugging,
evaluation, and video capture.

For curriculum training, pass multiple snapshot paths:

```python
env = nesle.make_vec(
    rom_path="Super Mario Bros. (World).nes",
    num_envs=4096,
    action_space="simple",
    backend="cuda",
    observation_mode="ram",
    reset_state_paths=[
        "docs/data/smb_level1_1.state",
        "docs/data/smb_level2_1.state",
        "docs/data/smb_level3_1.state",
        "docs/data/smb_level4_1.state",
        "docs/data/smb_level5_1.state",
        "docs/data/smb_level6_1.state",
        "docs/data/smb_level7_1.state",
        "docs/data/smb_level8_1.state",
    ],
)
```

By default envs are assigned round-robin across the provided snapshots. Pass
`env_to_level` for explicit per-env assignment.

For custom CUDA loops that only need rewards and done flags:

```python
rewards, dones, infos = env.step_reward(actions)
frames = env.render()
```

The vector wrapper follows SB3's `VecEnv` reset/step shape and auto-reset
contract, including `terminal_observation`. The single environment wrapper uses
Gymnasium's reset/step return convention when Gymnasium is installed.

An SB3 PPO starter is available at [examples/sb3_train.py](examples/sb3_train.py):

```sh
python -m pip install -e '.[rl]'
python examples/sb3_train.py "Super Mario Bros. (World).nes" \
  --backend cuda \
  --observation-mode ram \
  --reset-state-path docs/data/smb_level1_1.state \
  --num-envs 512 \
  --n-steps 128
```

The starter defaults to `observation_mode="ram"` and `MlpPolicy` so SB3 does
not build giant CPU rollout buffers from stacked RGB frames. Use
`--observation-mode rgb_array --policy CnnPolicy` only for explicit visual-policy
experiments; that path copies RGB frames back to host RAM every step.

For multi-level curriculum training:

```sh
python examples/sb3_train.py "Super Mario Bros. (World).nes" \
  --backend cuda \
  --observation-mode ram \
  --reset-state-paths \
    docs/data/smb_level1_1.state docs/data/smb_level2_1.state \
    docs/data/smb_level3_1.state docs/data/smb_level4_1.state \
    docs/data/smb_level5_1.state docs/data/smb_level6_1.state \
    docs/data/smb_level7_1.state docs/data/smb_level8_1.state \
  --num-envs 4096 \
  --timesteps 10000000 \
  --n-steps 128 \
  --batch-size 256 \
  --model-path nesle_ppo_curriculum
```

After training, run a quick rollout:

```sh
python examples/eval_smoke.py --model nesle_ppo_curriculum --steps 200
```

Important: `backend="cuda"` means NeSLE's emulator runs on CUDA. It does not
guarantee that SB3/PyTorch policy training runs on the GPU. Check
`torch.cuda.is_available()` and use `--sb3-device cuda` only after installing a
CUDA-enabled PyTorch wheel. For RAM observations with `MlpPolicy`,
`--sb3-device cpu` can still be faster because NeSLE returns CPU NumPy rollout
buffers and the policy network is small. See [Training](docs/training.md).

### CUDA-Native PPO

For a custom PPO loop that keeps NeSLE RAM observations, rollout buffers, PPO
loss computation, and the policy network on the GPU, build the CUDA extension
and run:

```sh
python examples/native_ppo_train.py "Super Mario Bros. (World).nes" \
  --reset-state-path docs/data/smb_level1_1.state \
  --num-envs 4096 \
  --total-timesteps 10000000 \
  --n-steps 128 \
  --batch-size 8192 \
  --checkpoint-path nesle_native_ppo.pt
```

This path uses `_cuda_core.CudaBatch.step_device(...)` and PyTorch's CUDA array
interface support to avoid the SB3 VecEnv/RolloutBuffer host-copy loop. It is
RAM-observation PPO only for now; RGB policy training should stay on the SB3
path until the renderer has a device-side frame stack and CNN input bridge.

For Colab/A100, use
[notebooks/nesle_colab_a100_training.ipynb](notebooks/nesle_colab_a100_training.ipynb).
It mounts Google Drive, uses your ROM from Drive, builds the CUDA extension for
`sm_80`, trains with checkpoints saved back to Drive, supports resume, and opens
TensorBoard. For a private GitHub repo, add a Colab secret named `GITHUB_TOKEN`
with read access before running the clone cell.

Legacy `nes-py` and `gym-super-mario-bros` comparison dependencies are kept in
the `legacy-mario` extra for benchmark work.

## Documents

- [Research notes](docs/research-notes.md)
- [Architecture](docs/architecture.md)
- [Phases](docs/phases.md)
- [Phase 6 readiness](docs/phase6-readiness.md)
- [Phase 6 report](docs/phase6-report.md)
- [CPU validation](docs/cpu-validation.md)
- [Headless runner](docs/headless-runner.md)
- [Phase 5 results](docs/phase5-results.md)
- [GPU vs CPU benchmark (GTX 1050 Ti)](docs/benchmark-gpu-vs-cpu.md)
- [Training](docs/training.md)
- [Colab A100 training notebook](notebooks/nesle_colab_a100_training.ipynb)

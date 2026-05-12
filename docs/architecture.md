# Architecture

## Goal

Run thousands of independent Super Mario Bros. environments on one NVIDIA A100,
with emulator state, observations, reward inputs, and reset state caches resident
on GPU.

## Modules

```text
nesle/
  python API: Gymnasium Env and SB3 VecEnv facade
  native bridge: pybind11 extension and CUDA console binding

cpp/
  core: ROM parsing, CPU/PPU/APU/input state, mapper interfaces
  cuda: batched state layout and kernels
  bindings: Python extension

benchmarks/
  nes-py comparison, FPS scaling, frame-skip, render/no-render modes,
  GPU-vs-CPU smoke, and correctness falsifiability checks
```

## Execution Model

The first GPU implementation should use one CUDA thread per environment for CPU
execution. This maps the sequential 6502 instruction stream naturally and keeps
debugging tractable. PPU rendering is separate because it writes many pixels and
has a different occupancy/register profile.

Per RL step:

1. Copy or write action masks into device input buffers.
2. Launch CPU/console kernel for `frameskip` raw frames.
3. During CPU execution, update CPU RAM, PPU registers, controller shift state,
   timers, NMI, OAMDMA, and minimal APU timing.
4. Render selected frames with a separate PPU kernel only when observation output
   requires pixels.
5. Launch reward/info kernel that reads Mario RAM addresses into compact arrays.
6. Auto-reset completed envs from cached initial states or FCEUX snapshot banks.
7. Return GPU tensors directly where possible; copy to NumPy only for Gym/SB3
   compatibility paths that require CPU arrays.

## State Layout

Use structure-of-arrays for hot batched state:

- CPU registers: `pc[n]`, `a[n]`, `x[n]`, `y[n]`, `sp[n]`, `p[n]`
- CPU timing: `cycles[n]`, `frame_cycles[n]`, `nmi_pending[n]`
- RAM: `cpu_ram[n][2048]`, env-major contiguous initially
- PPU registers: scalar arrays for control/mask/status/latches/scroll
- PPU memory: nametable RAM, palette RAM, OAM per env
- ROM: PRG and CHR in read-only device memory, shared by all envs
- Output: frame buffer `[n, 240, 256, 3]` or lower-resolution postprocessed
  buffers in device memory

The initial implementation favors correctness and debug visibility. Once the
CPU and PPU pass tests, profile alternatives: RAM tiling, grouping envs by PC,
multi-thread PPU rendering per env, and direct PyTorch tensor output.

## Mapper Strategy

Support mapper 0/NROM first:

- PRG ROM fixed at `$8000-$FFFF`
- NROM-128 mirrors 16 KB PRG into upper bank
- NROM-256 maps 32 KB PRG directly
- CHR ROM fixed at PPU `$0000-$1FFF`
- no mapper registers, no IRQs

After Mario throughput is proven, add mapper abstractions for other NES RL games.

## CPU Strategy

Build a single instruction implementation that can compile for CPU tests and
CUDA device code. Avoid divergent host-only behavior in the core instruction
functions. Test order:

1. Official opcode table metadata.
2. Unit tests for addressing modes and flags.
3. Klaus Dormann functional test on CPU.
4. Same test in a single CUDA thread.
5. Batched randomized differential tests against the CPU path.

## PPU Strategy

Correctness target is Mario, not every obscure PPU edge case on day one.

Required first:

- vblank/NMI timing
- `$2000-$2007` semantics used by Mario
- OAMDMA
- background rendering with scrolling
- 8x8 sprites, sprite priority, sprite 0 hit
- palette lookup into RGB output

Deferred until needed:

- rare open-bus decay behavior
- DMC/controller conflict
- tricky mid-scanline effects outside Mario
- non-NROM mapper scanline IRQs

## Reset Strategy

Follow CuLE's reset-cache idea, but the practical training path now uses
Stable Retro/FCEUX `.state` files as reset templates. Python loads raw or
gzip-wrapped FCS files, `cpp/include/nesle/fcs.hpp` parses CPU RAM, PRG RAM,
CPU registers, PPU registers, nametable RAM, palette RAM, and OAM, and the CUDA
binding uploads one or more snapshot templates to device memory.

For a single level, `reset_state_path` restores every env from the same
snapshot. For curriculum training, `reset_state_paths` uploads a snapshot bank
and `env_to_level[env]` selects the template used by each env. If no explicit
assignment is provided, Python assigns envs round-robin across the snapshots.

This avoids replaying fragile title-screen/start sequences and makes done-env
auto-reset cheap: the reset kernel copies the selected snapshot directly into
the env's device-resident emulator state.

## Python API

Expose two layers:

- `NesleEnv`: Gymnasium-compatible single environment for debugging and smoke
  tests.
- `NesleVecEnv`: SB3-compatible vector environment for training. It should return
  `obs, rewards, dones, infos`, auto-reset ended envs, and populate
  `terminal_observation`.

The vector API is the important performance path. The single env is a debugger.
For training, prefer `observation_mode="ram"` and `reset_state_path` or
`reset_state_paths`. RGB observations still work, but they copy full frames back
to host memory and should be reserved for debugging, videos, or visual-policy
experiments.

`backend="cuda"` controls NeSLE's emulator backend. SB3/PyTorch policy placement
is separate and is controlled by `--sb3-device` in `examples/sb3_train.py`.

## GPU-resident PPO path

For large-batch training where SB3's CPU rollout buffer is the bottleneck,
`nesle.native_ppo` provides a fully GPU-resident PPO loop. Observations, the
rollout buffer, the action sample, GAE, the policy/value loss, the optimizer
state, and the gradient update all stay on device. The only host roundtrip per
update is the small set of scalar log lines.

The bridge is provided by three methods on `nesle._cuda_core.CudaBatch`:

- `reset_device()` — runs the snapshot reset kernel and returns a
  `CudaDeviceArrayView` over the RAM observation buffer.
- `step_device(actions, auto_reset=True, synchronize=True)` — accepts a CUDA
  action tensor (uint8 mask or int64-encoded mask), launches the console step
  kernel, optionally fires the snapshot reset kernel for done envs, and returns
  a dict of `CudaDeviceArrayView` objects (`ram`, `rewards`, `dones`). When
  `auto_reset=True` the call always synchronizes (otherwise the next step's
  action copy would race the reset writes).
- `ram_device()`, `rewards_device()`, `last_done_device()` — direct device views
  for inspection between calls.

Each view implements both `__cuda_array_interface__` v3 and `__dlpack__`, so
PyTorch can build tensors directly via `torch.utils.dlpack.from_dlpack(view)`
without a host copy. The pybind layer uses `py::keep_alive<0, 1>()` to keep the
parent `CudaBatch` alive as long as any view (and any torch tensor built from
it) is reachable from Python — see the lifetime comment above
`CudaDeviceArrayView` in `cpp/bindings/cuda_module.cu`.

## Benchmark Plan

Benchmark modes:

- emulation only, random actions, no render
- render only where observations are requested
- full inference path with a small CNN policy on GPU
- SB3 PPO/A2C compatibility path

Compare against `nes-py`/`gym-super-mario-bros` at env counts:

```text
1, 8, 32, 128, 512, 1024, 2048, 4096, 8192
```

Report raw FPS, training-frame FPS, FPS/env, GPU utilization, memory footprint,
and reset rate.

The reproducible entrypoint is `benchmarks/phase5_benchmark.py`. Use the
`step`, `render`, and `inference` modes for NeSLE scaling runs, then rerun with
`--include-legacy` after installing `.[legacy-mario]` for CPU emulator
comparison rows using registered legacy env IDs such as `SuperMarioBros-v0`.
Use `scripts/benchmark_cuda_kernels.sh` for raw CUDA kernel scaling so benchmark
reports distinguish packaged Python backend throughput from lower-level GPU
reward/render capacity.
Use `scripts/build_cuda_extension.sh` to build the optional `nesle._cuda_core`
module; once present, `NesleVecEnv(..., backend="cuda")` runs the ROM-backed
CUDA batch CPU/PPU console loop through the public Python vector API. The
lower-level CUDA reward/render kernels remain available for calibration runs.

The local practical smoke is `benchmarks/gpu_vs_cpu.py`, which compares native
CPU single-env throughput against batched `cuda-console` stepping from the W1-1
snapshot. `benchmarks/verify_correctness.py` checks that the benchmark is doing
real per-env work by verifying action divergence, plausible instruction counts,
and independent RAM evolution.

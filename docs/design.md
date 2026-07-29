# NeSLE design: the emulator inside the kernel

This document explains *why* NeSLE is built the way it is — the reasoning
behind the architecture, not just its shape. For the component-level view see
[architecture.md](architecture.md); for measured numbers see
[phase6-report.md](phase6-report.md) and
[benchmark-gpu-vs-cpu.md](benchmark-gpu-vs-cpu.md).

## The problem: RL environments are a throughput problem

PPO-style training on a game needs billions of environment frames. A CPU NES
emulator produces a few hundred env-steps/s per core; even a 32-core machine
tops out around 10k. The usual fix (EnvPool, SubprocVecEnv) is *more CPU
processes* — which scales linearly at best and saturates quickly.

NeSLE's bet: the NES is small enough (2 KB RAM, ~1.79 MHz 6502, simple PPU)
that an *entire console* fits comfortably in the working set of a single GPU
thread. So instead of putting the neural network on the GPU and the
environments on the CPU — the standard split — put **everything** on the GPU
and give each environment one CUDA thread.

## Design decision 1: one thread per environment

Each CUDA thread runs a complete, independent NES: fetch-decode-execute for
the 6502, PPU state stepping, OAM DMA, controller latching. Frame-skip loops
happen inside the kernel, so one launch advances all N consoles by
`frameskip` frames.

- **Why not one warp/block per console?** The 6502 instruction stream is
  inherently serial — there is no intra-console parallelism worth extracting.
  Thread-per-env keeps the emulator code essentially identical to a normal
  C++ emulator (same headers compile for CPU tests and CUDA), which made it
  possible to validate the core against the Klaus 6502 functional tests before
  ever touching the GPU.
- **The cost:** warp divergence. 32 consoles share a warp; when their
  instruction streams diverge (they always do), threads serialize. This is the
  known headroom in the design — measured throughput still reaches ~30k
  env-steps/s on a GTX 1050 Ti and ~1M steps/s (no-copy) on an A100 at just
  128 envs, because the sheer batch width buries the inefficiency.

State is laid out as structure-of-arrays indexed by env id, so the frequent
fields (registers, cycle counters, RAM) coalesce across threads.

## Design decision 2: the copy gap decides the API

The phase-6 ablation measured the same kernel with four observation policies
on an A100 (128 envs, frameskip 4):

| What crosses PCIe per step | Env-steps/s |
|---|---:|
| Full RGB frames (184 KB/env) | ~3.6k |
| CPU RAM (2 KB/env) | ~560k |
| Nothing (rewards/dones only) | ~1.0M |

The emulator was never the bottleneck — **the observation copy was**, by two
orders of magnitude. That single measurement drove the whole API surface:

- `observation_mode="ram"` is the default training path (2 KB of RAM *is* the
  full game state; an MLP learns from it directly).
- `step_device()` / `ram_device()` / `rewards_device()` expose the state as
  device tensors via `__cuda_array_interface__` and DLPack, so PyTorch can
  consume observations **without any host copy at all**.
- Rendering is a separate, optional kernel (`render()`), not part of `step()`.

## Design decision 3: snapshot reset instead of boot

SMB's title-screen state machine stalls on NeSLE's PPU timing (a real,
documented bug — see KNOWN_ISSUES.md). Rather than chase cycle-accurate PPU
timing, reset restores a gzip FCEUX-format save state captured at the start
of a level, entirely on-device: the parsed snapshot lives in GPU memory and a
reset kernel copies it into any env that finished an episode, in the same
launch that steps the others. Auto-reset therefore costs nothing measurable.

This turned a bug workaround into a feature: `reset_state_paths` with eight
level snapshots gives round-robin multi-level curricula for free, and any
moment of gameplay can become a training start state.

## Design decision 4: keep PPO on the GPU too

With observations on-device, Stable-Baselines3's CPU rollout buffer becomes
the next bottleneck (it stores rollouts as host NumPy arrays). The native PPO
loop (`nesle.native_ppo`) keeps rollout buffers, GAE, the optimizer, and the
reward function on the GPU. The shaped Mario reward (`TorchSmartMarioReward`)
reads the batched RAM tensor directly — progress, checkpoints, coins, kills,
deaths computed as vectorized tensor ops over all envs at once.

Measured end-to-end on a 4 GB GTX 1050 Ti (2048 envs): ~25M timesteps in
~2.5 h wall-clock including learning, with the policy demonstrably improving
(episode return −26 → ~195, explained variance 0 → 0.88).

## What the design deliberately does not do

- **Cycle-perfect emulation.** The PPU is sufficient for SMB gameplay, not
  for timing-sensitive edge cases (hence the title-screen bug). Fidelity is
  bounded by "does the game play correctly," verified by falsifiability
  checks (`benchmarks/verify_correctness.py`) and by agents actually learning.
- **Mappers beyond NROM.** Each mapper multiplies supported games but adds
  divergent banking logic inside the hot kernel. NROM covers SMB, the target.
- **Audio.** No RL value for the cost.

## On adding mappers (MMC1 first)

Extending beyond NROM is the highest-leverage *capability* work: MMC1 alone
adds Zelda, Metroid, and Mega Man 2-class games. Honest scoping from reading
the current code:

- **Where it plugs in:** cartridge reads/writes flow through the batched bus
  (`cpp/include/nesle/cuda/batch_bus.cuh`); NROM is currently a fixed mapping.
  MMC1 needs per-env mapper state in the SoA layout (shift register, 5-bit
  load count, 4 bank-control registers) plus bank-translation on PRG/CHR
  access and runtime nametable-mirroring control.
- **The hard parts:** (1) every PRG/CHR access gains a bank indirection —
  measurable but small; (2) CHR-RAM support (MMC1 games commonly use CHR-RAM,
  which NROM's read-only CHR path doesn't model); (3) more warp divergence as
  envs' bank states drift apart.
- **The validation blocker:** this machine has a legally-obtained ROM for SMB
  only. Mapper logic can be unit-tested against synthetic iNES images (the
  existing C++ test style supports this), but claiming "game X works" without
  running game X would violate this project's standards. MMC1 implementation
  should land together with real-game validation by someone holding the ROMs.
- **Estimate:** the mapper state machine itself is well-documented and small
  (~200 lines device code + parsing); the CHR-RAM plumbing and validation are
  the real cost. A focused effort is likely a few days, not hours.

## The next frontier: killing the launch round-trip

Every `step()` is still one kernel launch plus a Python round-trip (~3 ms of
fixed overhead — the reason a *single* GPU env runs at only ~18 steps/s while
4096 of them hit 29k). The natural evolution:

1. **CUDA Graphs** over the step+inference sequence — cheap to try, cuts
   launch overhead.
2. **Persistent rollout kernel** — run the entire n-step rollout inside one
   launch. With a policy as small as the RAM MLP, even inference could move
   into the kernel, making the rollout loop fully GPU-resident. At that point
   the environment's cost per step approaches its arithmetic cost, which the
   A100 ablation shows is ~1M steps/s territory.

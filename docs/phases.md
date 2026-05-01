# Phases

## Phase 0: Skeleton And Research Grounding

Success criteria:

- Research notes capture CuLE, NES hardware scope, Mario RAM, and RL API choices.
- Repository has C++/CUDA/Python layout.
- iNES/NROM parsing and Mario RAM reward extraction have tests.

Status: implemented in this initial slice.

## Phase 1: CPU-Correct NES Core

Success criteria:

- 2A03 CPU core passes Klaus Dormann 6502 functional tests on CPU.
- CPU memory map implements RAM mirrors, PPU/APU/input register dispatch, and
  NROM PRG mapping.
- iNES loader rejects unsupported mapper/ROM combinations clearly.
- A minimal headless Mario boot reaches stable gameplay state on CPU.

Gate: no CUDA work beyond single-thread mirrors until the CPU core is testable.

Status: implemented. The first slice adds a portable official-opcode CPU core,
a flat 64 KB test bus, an NROM CPU memory map, smoke tests for CPU execution,
stack calls, branch timing, arithmetic, and memory mirrors, plus a flat-binary
runner for Klaus-style functional tests. The stock upstream Klaus binary now
passes in the MOS 6502 validation profile; the NES 2A03 profile intentionally
keeps decimal arithmetic disabled. The console CPU bus now covers CPU RAM
mirrors, PPU register dispatch, controller serial reads, OAMDMA, PRG RAM, and
NROM PRG ROM mapping. A basic NTSC timing loop now advances PPU dots from CPU
cycles, drives vblank/NMI timing, accounts for OAMDMA stalls, and can run a
synthetic NROM program across one frame. A headless `.nes` runner now loads
ROM files from disk and reports frame/instruction/cycle progress for NROM boot
smoke tests. PPU memory now maps CHR ROM, CHR RAM fallback, nametable mirrors,
palette mirrors, and a coarse sprite-0-hit signal through the same register
paths that games use. The optional user-ROM smoke test now accepts iNES and
mapper-0 NES 2.0 Super Mario Bros. ROMs, runs the local boot window, and checks
decoded Mario RAM for meaningful progress.

## Phase 2: Mario-Correct PPU And Input

Success criteria:

- CPU PPU implementation renders Super Mario Bros. frames that visually match a
  known-good emulator over sampled frames.
- Controller protocol produces correct input effects in Mario.
- Reward/info values match `gym-super-mario-bros` on the same action trace.

Gate: record deterministic action traces and compare RAM/frame hashes.

## Phase 3: CUDA Batched Emulator

Success criteria:

- Same CPU core compiles for device execution.
- Single CUDA-thread environment matches CPU trace for CPU RAM, PPU state, reward,
  done, and selected frame hashes.
- Batched execution works for at least 4096 envs with deterministic seeds.
- Reset cache works without startup-frame replay per episode.

Gate: correctness before throughput claims.

## Phase 4: Gymnasium And SB3 API

Success criteria:

- `NesleEnv` passes Gymnasium environment checks.
- `NesleVecEnv` follows SB3 VecEnv API, including auto-reset and
  `terminal_observation`.
- Existing SB3 training script can switch from `gym-super-mario-bros` to NeSLE
  with only environment construction changed.

Gate: keep a NumPy compatibility path even if the fast path returns GPU tensors.

## Phase 5: Throughput Benchmark

Success criteria:

- Benchmarks compare NeSLE and nes-py across increasing environment counts.
- Report emulation-only, render, and inference-path FPS.
- Include A100 GPU utilization and memory usage.
- The headline chart shows NeSLE scaling after CPU emulators saturate.

Gate: no paper claims until benchmark scripts are reproducible from a clean env.

## Phase 6: Research Package

Success criteria:

- Paper-quality methodology and plots.
- Ablations for reset cache, render cadence, env count, and frame-skip.
- Public artifact can build in an NVIDIA CUDA container.
- Limitations are explicit: initial mapper support is NROM/Super Mario Bros.

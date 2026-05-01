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

Status: local CPU gate implemented, with an OpenEmu reference-state visual gate.
The PPU can render RGB frames from
background and sprite pattern data, the headless runner accepts deterministic
per-frame controller masks, and the runner reports Mario RAM, accumulated reward
components, CPU RAM hashes, and RGB frame hashes. The optional Phase 2 user-ROM
smoke script compares a neutral trace against a Start, Right+B trace and asserts
that controller input changes Mario position, reward, RAM hash, and rendered
frame hash. The OpenEmu/Nestopia bridge parses local Nestopia save-state PPU
chunks and renders them through NeSLE, giving the project a reference-state path
for visual parity work. The optional OpenEmu comparison script normalizes the
OpenEmu save-state screenshot to native NES resolution, compares it with the
NeSLE render, and fails on excessive RGB drift or weak channel correlation.
External known-good emulator parity is not vendored; the emitted hashes,
OpenEmu state renderer, and screenshot comparison script are the comparison
surface for reference captures.

## Phase 3: CUDA Batched Emulator

Success criteria:

- Same CPU core compiles for device execution.
- Single CUDA-thread environment matches CPU trace for CPU RAM, PPU state, reward,
  done, and selected frame hashes.
- Batched execution works for at least 4096 envs with deterministic seeds.
- Reset cache works without startup-frame replay per episode.

Gate: correctness before throughput claims.

Status: CUDA correctness gate implemented. Batch buffers now carry the Mario
reward baseline needed by GPU stepping, and a host/device batch helper reads
per-env CPU RAM, computes SMB-style reward/done values, and advances the reward
baseline with the same semantics as the CPU `smb` module. The CUDA step kernel
uses that helper as its current per-env work item, and a host-side C++ parity
test compares the GPU-ready helper against the CPU reward implementation across
multiple environments. The optional CUDA smoke script compiles this kernel with
`nvcc` and launches it on NVIDIA hardware, checking device-computed reward/done
outputs over 4096 envs against expected SMB RAM cases. The GPU-ready batch CPU
bus covers CPU RAM mirrors, PRG RAM, NROM PRG ROM mapping, controller
strobe/shift reads, and a minimal PPU register surface, with host-side tests
comparing the memory-map behavior against the CPU console/controller path. The
same CPU core now compiles for device execution and the CUDA smoke runs an
on-device synthetic NROM CPU trace through reset, RAM writes, controller reads,
PRG RAM writes, and a loop PC check. The smoke also runs an integrated
device-side batch console trace through CPU execution, `$4014` OAM DMA,
513-cycle DMA stall accounting, and PPU scanline/dot advancement. A host-side
batch CPU step adapter now runs the existing CPU core over the batch bus and
checks instruction, register, cycle, RAM, PRG RAM, and controller parity against
`Console` on a synthetic NROM trace. A host batch runner now steps multiple envs
independently through that adapter and verifies per-env controller divergence
and RAM/PRG RAM isolation against independent `Console` instances. Batch PPU
timing now tracks per-env scanline/dot/frame state, vblank/NMI start,
pre-render status clearing, and the coarse sprite-0 signal, with parity tests
against the CPU `Ppu`. The batch console step now combines CPU stepping,
pre-instruction NMI service, PPU timing, and OAM DMA stalls/copies, with parity
tests against `Console` for NMI frame service and DMA behavior. A host reset-cache
contract now captures/restores CPU, CPU RAM, PRG RAM, PPU timing/register
memory, OAM, reward baselines, and done/reward slots, with deterministic rerun
coverage after restore. A device reset snapshot view now restores CPU state,
CPU RAM, PRG RAM, PPU timing/register state, scroll/open-bus/read-buffer state,
and OAM inside a CUDA kernel, and the CUDA smoke validates the restored
console/DMA trace on NVIDIA hardware. Batch PPU register writes now cover
`PPUCTRL`, `PPUMASK`, `OAMADDR`, `OAMDATA`, `PPUSCROLL`, `PPUADDR`, and
`PPUDATA` for nametable/palette updates. The CUDA render kernel now produces
RGB frames from CHR ROM, nametable RAM, palette RAM, and OAM; host tests compare
the batch renderer byte-for-byte against `Ppu::render_rgb_frame`, and the H100
smoke validates a CPU/GPU frame-hash match for a synthetic background+sprite
scene. Throughput-oriented tiled rendering and end-to-end Gym/SB3 integration
move to the next phases.

## Phase 4: Gymnasium And SB3 API

Success criteria:

- `NesleEnv` passes Gymnasium environment checks.
- `NesleVecEnv` follows SB3 VecEnv API, including auto-reset and
  `terminal_observation`.
- Existing SB3 training script can switch from `gym-super-mario-bros` to NeSLE
  with only environment construction changed.

Gate: keep a NumPy compatibility path even if the fast path returns GPU tensors.

Status: started. `NesleEnv` now exposes a Gymnasium-style single-env API and
`NesleVecEnv` exposes the SB3-style vector contract with `reset`, `step`,
`step_async`, `step_wait`, `render`, `get_images`, auto-reset, `reset_infos`,
seed/options plumbing, and `terminal_observation`. When SB3 is installed the
vector wrapper inherits from SB3's `VecEnv`; without SB3 it keeps the same
runtime method surface. The wrappers validate ROM paths, support the existing
right-only/simple/complex/raw/custom action spaces, and return NumPy RGB
observations, float rewards, done flags, and Mario RAM info dictionaries. A
native C++ console binding hook is available for packaged CPU execution, while
a deterministic Python compatibility backend keeps API tests and downstream
integration code runnable before the CUDA runtime is packaged into Python. The
`examples/sb3_train.py` script shows the intended PPO training entrypoint.

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

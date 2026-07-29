# Scaling beyond one-thread-per-env

Research notes toward a higher-throughput emulation core: what the current
design costs, what the literature says, and a ranked roadmap. Companion to
[design.md](design.md) ("the emulator inside the kernel").

## Where the design stands

One CUDA thread runs one complete NES (`cpp/src/cuda/kernels.cu:19-69`,
128 threads/block). CPU registers are cached in hardware registers for the
whole kernel (`load_cpu_state` once at entry). Measured throughput: ~29.5k
env-steps/s at 4,096 envs on a GTX 1050 Ti; ~1.0M steps/s (no-copy) at 128
envs on an A100.

The same design choice has published precedent: NVIDIA's **CuLE** (CUDA Atari,
NeurIPS 2020) maps one thread to one Atari 2600 — whose 6507 CPU is nearly
the same silicon as the NES's 6502. Their assessment applies verbatim: a
1-to-1 thread↔emulator mapping "is not the most computationally efficient
way … but it makes the implementation relatively straightforward."

## The two costs, and which one is bigger

**Divergence.** Every lane runs an env at a different program counter, so the
~151-case opcode switch (`cpp/include/nesle/cpu.hpp:316-485`) serializes over
the distinct opcodes present in each warp iteration, bus accesses take
different range branches (`batch_bus.cuh:148-292`), and per-frame events
(vblank snapshot copy, OAM DMA) hit lanes at different times as fully
serialized ~2.3 KB byte-loops. **However**: CuLE *measured* this on the same
ISA and found divergence costs ~30% in the worst case — painful, but nowhere
near the naive 32× serialization floor, because real game-state divergence
across a warp is partial. Anti-divergence work has a hard ceiling on its
payoff.

**Memory traffic.** This may be the larger cost and gets less attention:

- PPU bookkeeping scalars (`scanline`, `dot`, `status`, `ctrl`, `nmi_pending`,
  `frame`) are read-modify-written to **global memory ~6–10 times per emulated
  instruction** (`batch_ppu.cuh:126,193-198`, `batch_console.hpp:41`) — while
  CPU registers already show the fix: cache in registers, write back at kernel
  exit.
- Emulated CPU RAM is a contiguous 2 KB block per env (`state.cuh:146-153`),
  so warp lanes touching even the *same* NES address hit locations 2 KB apart:
  fully scattered, up to 32 transactions per warp access — and zero-page/stack
  (the 6502's hottest region) always takes this path.

## Ranked roadmap

### Tier 0 — quick wins, no redesign — **EXECUTED 2026-07-29, measured**

Three of the five items below were implemented and measured the same day
(GTX 1050 Ti, per-change A/B with the full test suite green at each step):

| Change | step @2048 envs | render @256 envs |
|---|---:|---:|
| (baseline, start of day) | 88.3 ms / 23.2k steps/s | 118.5 ms |
| + hybrid render dispatch (#2) | — | **41.4 ms (2.9×)** |
| + vectorized bulk copies (#3) | 80.6 ms / 25.4k (+9.5%) | — |
| + PPU hot-state register caching (#1) | **68.6 ms / 29.8k (+17%)** | — |
| **cumulative** | **+29% stepping** | **2.9× small-batch render** |

Effect on the official `benchmarks/gpu_vs_cpu.py` headline: 29,491 →
**35,488 env-steps/s at 4,096 envs (88× → 110× the CPU baseline)**. Honest
notes: the per-pixel render kernel loses to the serial one once the env count
alone saturates the GPU (321 ms vs 207 ms at 2,048 envs), so dispatch is
hybrid with the crossover at 512 envs; the refactored serial path also costs
~14% at 2,048 envs (207 → 237 ms) — accepted, as large-batch rendering serves
only RGB-observation training. Items #4 (zero page in shared memory) and #5
(hygiene pass) remain open.

Original plan:

1. **Register-cache the hot PPU scalars** exactly like `CpuState` already is
   (verified single-writer per env within a kernel). Eliminates most
   per-instruction global RMWs. Likely the single largest safe win.
2. **Parallelize the render kernel** — today one thread paints all 61,440
   pixels of its env's frame (`batch_render.cuh`); block-per-env /
   thread-per-pixel is the natural shape. (This is also where warp-level
   parallelism genuinely belongs — not in the CPU loop.)
3. **Vectorize the byte-loop copies** (vblank presentation snapshot, resets,
   OAM DMA) with 16-byte chunks: 2048 iterations → 128.
4. **Zero page + stack in shared memory**: 256 B × 128 threads = 32 KB smem,
   converting the hottest scattered traffic to on-chip.
5. **Hygiene**: strip always-true nullptr guards from the hot bus path, add
   `__restrict__` to `BatchBuffers` pointers, merge `scanline`/`dot` into one
   stored word.

*Measure first*: the binding already ships an opcode/PC histogram profiler
(`step_profile`, `cuda_module.cu`) — quantify opcode fan-out and hot PCs
before and after each change.

### Tier 1 — CPU/PPU kernel split (weeks; CuLE's shipped answer)

CuLE runs 6507 emulation in one kernel (buffering graphics-chip register
writes to global memory) and consumes the buffer in a second kernel — because
register pressure and divergence profiles differ between CPU and video work,
and observation-free frames can skip the second kernel entirely. NeSLE
interleaves PPU catch-up after every instruction; the PPU is already analytic
(event-horizon arithmetic, `batch_ppu.cuh:62-102`), so deferring it to
event/register-access boundaries is feasible. Correctness pivot: NMI delivery
at instruction boundaries.

### Tier 2 — table-driven decode (weeks; best divergence fix per unit risk)

Replace the monolithic 151-case switch with a 256-entry `__constant__` decode
table ({addressing mode, operation, cycles, flags}) driving a restructured
loop: uniform fetch → uniform table lookup → ~8-way effective-address switch →
shared operand load → ~30 short ALU cases (already factored as lambdas,
`cpu.hpp:205-312`). Divergence per iteration drops from 1-of-151 long cases to
two small switches, and memory ops decouple from ALU. Stays a single kernel;
bindings and tests unchanged. Risk: encoding page-cross/branch cycle rules in
the table — well-covered by the existing CPU test suite (Klaus functional
tests).

### Tier 3 — wavefront / opcode-binned interpreter (months; research-grade)

The structurally "correct" fix from GPU ray tracing ("Megakernels Considered
Harmful", Laine & Karras HPG 2013) and demonstrated for CHIP-8 by Octax
(JAX `lax.switch` vectorized dispatch; 1.4M frames/s — but that's a 35-opcode
ISA without cycle-accurate video timing). Requires compaction/queue
infrastructure and an interpreter rewrite; the payoff for a 256-opcode 6502
with a timed PPU is a hypothesis to validate with a small prototype (ALU
dispatch only) before committing. CuLE's ~30%-worst-case divergence number is
the sober counterweight to this tier's ambitions.

### Rejected: warp-per-env for the CPU loop

A serial 6502 instruction stream offers no work for 31 of 32 lanes —
throughput would drop roughly an order of magnitude. Lane-level parallelism
belongs to rendering and bulk copies (Tier 0, #2–3).

### Cheap orthogonal lever: state-correlation scheduling

CuLE measured throughput highest right after reset (envs share PC regions)
and decaying as states drift. NeSLE's snapshot resets already correlate the
population far more than ROM-boot Atari; grouping envs by reset state at the
rollout level (so warps tend to share game phase) is nearly free to prototype
and worth measuring, though its payoff is inferred, not published.

## Sources

- CuLE: [arXiv:1907.08467](https://arxiv.org/abs/1907.08467), code
  [NVlabs/cule](https://github.com/NVlabs/cule)
- Laine & Karras, "Megakernels Considered Harmful", HPG 2013
  ([PDF](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf))
- Octax (vectorized CHIP-8 RL env): [arXiv:2510.01764](https://arxiv.org/abs/2510.01764)
- Madrona batch-simulation engine: [madrona-engine.github.io](https://madrona-engine.github.io/)
- EnvPool (why general video-game envs resist GPU vectorization):
  [arXiv:2206.10558](https://arxiv.org/abs/2206.10558)
- GVM, GPU Java bytecode interpreter, ACM TOPLAS 2019
  ([PDF](https://users.ece.utexas.edu/~gligoric/papers/CelikETAL19GVM.pdf))

# Scaling beyond one-thread-per-env

Research notes toward a higher-throughput emulation core: what the current
design costs, what the literature says, and a ranked roadmap. Companion to
[design.md](design.md) ("the emulator inside the kernel").

## Where the design stands

One CUDA thread runs one complete NES (`cpp/src/cuda/kernels.cu:19-69`,
128 threads/block). CPU registers are cached in hardware registers for the
whole kernel (`load_cpu_state` once at entry). Measured throughput at the time
this note was written: ~29.5k env-steps/s at 4,096 envs on a GTX 1050 Ti;
~1.0M steps/s (no-copy) at 128 envs on an A100. **These are the pre-sprint
baseline.** The Tier 0 section below walks them to 180,437 on the same card;
current figures are in the [README](../README.md).

The same design choice has published precedent: NVIDIA's **CuLE** (CUDA Atari,
NeurIPS 2020) maps one thread to one Atari 2600 — whose 6507 CPU is nearly
the same silicon as the NES's 6502. Their assessment applies verbatim: a
1-to-1 thread↔emulator mapping "is not the most computationally efficient
way … but it makes the implementation relatively straightforward."

## The two costs, and which one is bigger

**Divergence.** Every lane runs an env at a different program counter, so the
151-entry decode table (`cpp/include/nesle/cpu.hpp:200-350`) and its
50-case operation switch (`cpu.hpp:590`) serializes over
the distinct opcodes present in each warp iteration, bus accesses take
different range branches (`batch_bus.cuh:148-292`), and per-frame events
(vblank snapshot copy, OAM DMA) hit lanes at different times as fully
serialized ~2.3 KB byte-loops. **However**: CuLE *measured* this on the same
ISA and found divergence costs ~30% in the worst case — painful, but nowhere
near the naive 32× serialization floor, because real game-state divergence
across a warp is partial. Anti-divergence work has a hard ceiling on its
payoff.

**Measured on NeSLE (2026-10-25), divergence costs far more than 30%.** On the
1050 Ti at 4,096 envs from the W1-1 snapshot, every env holding RIGHT runs at
177k env-steps/s, but each env drawing its own random action from
`simple_with_start` runs at **28.1k**, 6.3x slower (`gpu_vs_cpu.py --policy
random`; full sweep in `benchmark-gpu-vs-cpu.md`). Giving every env the *same*
random action restores 214k, so the gap is per-env action divergence, not the
actions themselves. The headline numbers (README, `benchmark-gpu-vs-cpu.md`) are
the all-envs-identical best case; an untrained stochastic policy sees roughly a
sixth of that on this card. See the state-correlation section below.

Where the 6.3x goes, from a throwaway build that counted, at the top of the
instruction loop, each converged lane group, its active lanes, and the distinct
opcodes it was about to run (4,096 envs, 100 steps):

| Policy | instr/env/step | loop passes/warp/step | active lanes/pass | distinct opcodes/pass |
|---|---:|---:|---:|---:|
| RIGHT held | 14,456 | 12,726 | 32.0 | 1.00 |
| one shared random action | 11,901 | 11,084 | 32.0 | 1.00 |
| per-env random | 12,422 | 20,827 | 19.0 | 8.86 |

Per-env random envs execute *fewer* instructions, so none of the gap is extra
game work. About 1.6x is lanes finishing their frame at different times (the
slowest lane in a warp averages 1.67x the mean). The other ~3.8x is the ~9
distinct opcodes per pass serializing through the decode switch, partly shared
by SIMT, so well under 9x.

Tried and rejected (2026-10-25): flattening the per-frame loop into one loop
over all frameskip windows, so a lane that finishes a frame starts the next
instead of waiting for its warp. It is bit-exact against main (same RAM,
rewards, dones and instruction counts over 800 steps x 1,024 envs, 8 levels,
frameskip 1 and 4), but **slower**: 28.0k to 25.0k env-steps/s with per-env
random actions and 176k to 166k with RIGHT held at 4,096 envs. The extra
branch in the hot loop costs more than the frame barrier it removes.

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
only RGB-observation training.

**Evening update — every remaining item resolved (all measured on the GTX
1050 Ti; behavior verified bit-exact via a fixed-seed training run whose
losses match pre-sprint to 4 decimals):**

| Item | Verdict | Stepping @2,048 envs after |
|---|---|---:|
| #5 hygiene (restrict, guard strip, merged frame_dot) | ✅ shipped, +10% | 92.8k steps/s |
| #4 zero page in shared memory | ❌ **rejected with data**: 1.9× regression on sm_61 — 32 KB/block collapses occupancy from 32 to 6 warps/SM; break-even needs ≤96 B/thread. Implementation preserved on its branch. | — |
| Tier 2: table-driven decode (+ forced render inlining) | ✅ shipped — the sprint's largest win (see below) | 84.4k steps/s |
| Tier 1: **lazy PPU settlement** (supersedes the kernel-split proposal below) | ✅ shipped, +9% | **100.8k steps/s** |

Official `benchmarks/gpu_vs_cpu.py` after the sprint: **180,437 env-steps/s
at 4,096 envs (576× CPU)** — 5.1× the morning's 35,488. A cautionary tale
worth keeping: the decode merge initially *slowed both render kernels 2.3×*
because the enlarged module blew nvcc's inlining heuristics — forcing
inlining on the render helper chain (`NESLE_CUDA_RENDER_INLINE`) fixed render
AND unlocked the decode change's full stepping value (57.6 → 24.3 ms in one
line). Watch for this whenever the module grows.

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

### Tier 1 — CPU/PPU kernel split — **RESOLVED as lazy settlement (shipped)**

The original CuLE-style two-kernel split was superseded by the Tier-0 work:
with PPU scalars register-resident, events analytic, and rendering already a
separate kernel, a literal split would only add launch overhead and buffer
traffic. The surviving substance shipped as **lazy PPU settlement**: the step
kernel accumulates cycles and advances the PPU only when the span reaches the
precomputed next-event distance (PPU scalar state changes exclusively at the
three timing events, so between settles the register-resident state the bus
reads is exact). Measured: +9%, taking local 2,048-env stepping past 100k
env-steps/s. Timing stays instruction-granular, identical to before.

Original proposal (kept for the record): CuLE runs 6507 emulation in one
kernel (buffering graphics-chip register writes) and consumes the buffer in a
second — because register pressure and divergence profiles differ between CPU
and video work.

### Tier 2 — table-driven decode — **SHIPPED (the sprint's largest win)**

Shipped exactly as designed: a 256-entry constexpr decode table drives an
11-case effective-address switch and a 49-case operation switch (all eight
branch opcodes share one case). Verified by an 8M-random-instruction
differential harness against the old interpreter (zero divergence, identical
bus traffic and exception messages) and cycle-exact Klaus functional counts.
Combined with the forced-inlining fix it took local 2,048-env stepping from
29.8k to 84.4k env-steps/s — and it is also what makes Tier 3's explicit
binning redundant (see below).

### Tier 3 — wavefront / opcode-binned interpreter — **NO-GO (measured)**

Prototyped and measured (`benchmarks/wavefront_prototype/`, full write-up in
its RESULTS.md): the real table-driven interpreter over a synthetic NROM bus,
instruction streams sampled from the *measured* SMB opcode distribution
(535.7M profiled executions), wavefront variants validated byte-for-byte
against the baseline. Verdict: opcode-binned dispatch is **3.2× slower** than
thread-per-env on realistic decorrelated streams (0.31× at every scale from
4,096 to 131,072 instances) and merely break-even (+4–6%) on fully correlated
streams where there is no divergence to remove. The loss is group
serialization, not binning overhead: a decorrelated warp holds ~20.6 distinct
opcodes, and hardware SIMT reconvergence in the baseline already shares
fetch/table-lookup/addressing across differing opcodes — explicit binning
serializes what the hardware was sharing. Ceiling analysis: even *free*
divergence removal caps at ~1.9× on the CPU-only prototype (~1.4× diluted by
real PPU/bus work) — from a 3.2× hole. The original proposal (Laine & Karras
wavefront, Octax's CHIP-8 result) is kept in the sources; the pattern does
not transfer to a 6502 with a table-driven core.

Useful byproduct: the measured 1.9× correlated-vs-decorrelated gap motivated
the state-correlation scheduling A/B below, which came back negative for
training.

### Rejected: warp-per-env for the CPU loop

A serial 6502 instruction stream offers no work for 31 of 32 lanes —
throughput would drop roughly an order of magnitude. Lane-level parallelism
belongs to rendering and bulk copies (Tier 0, #2–3).

### State-correlation scheduling: **NO-GO for training (measured 2026-10-25)**

CuLE measured throughput highest right after reset (envs share PC regions)
and decaying as states drift. The concrete instance in NeSLE
([#8](https://github.com/hbofz/NeSLE/issues/8)): with several reset states,
`env.py` assigns levels round-robin, so every 32-lane warp holds all 8 worlds.
Block assignment (a contiguous run of envs per level) makes each warp
single-level.

A/B on the GTX 1050 Ti, 8 world snapshots, 200 timed steps after 30 warmup,
done envs reset every step as training does, median of 3 interleaved runs:

| Policy | 256 | 1,024 | 2,048 | 4,096 | 8,192 |
|---|---:|---:|---:|---:|---:|
| RIGHT held (identical actions) | 2.18x | 2.17x | 2.13x | 2.23x | 3.14x |
| random `simple` per env | 0.94x | 0.94x | 0.95x | 0.96x | 1.00x |
| random `simple_with_start` per env (training default) | 0.93x | 0.94x | 0.94x | 0.95x | - |

(block throughput / round-robin throughput; the last cell was not run.)

Level homogeneity only pays when the actions are correlated too. Per-env
stochastic actions decorrelate a warp within a few frames whatever level it
started on, and block assignment then loses 4-7%. The likely cause, not
verified, is that it puts all of a costly level's warps together and leaves a
slow tail. Round-robin stays the default. A late-training policy that has
become near-deterministic could sit between the two rows; anyone wanting to
try that can already pass `env_to_level` with contiguous blocks.

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

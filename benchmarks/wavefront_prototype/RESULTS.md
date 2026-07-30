# Wavefront (opcode-binned) dispatch prototype — measured verdict

Tier-3 prototype from [docs/gpu-scaling.md](../../docs/gpu-scaling.md): does
opcode-binned "wavefront" execution beat the status-quo thread-per-env
interpreter for the 6502 workload? Standalone CUDA micro-benchmark, real
production interpreter (`cpp/include/nesle/cpu.hpp`), synthetic bus, real
measured SMB opcode distribution.

**Verdict: NO-GO.** On decorrelated instruction streams — the realistic
steady-state of a training population — warp-level opcode binning runs at
**0.31×** the baseline's throughput (3.2× *slower*), and coarser
operation-class binning at **0.43×** (2.3× slower). The only regime where
binning does not lose is fully correlated streams (all lanes at the same PC),
where there is nothing to bin and the baseline has no divergence to remove in
the first place. The measured ceiling for *any* anti-divergence scheme on this
workload is ~2× (see below), and this scheme moves in the wrong direction.

## Environment

- GPU: NVIDIA GeForce GTX 1050 Ti (sm_61, 6 SMs), Windows 11, WDDM
- CUDA 12.9 (`nvcc -std=c++20 -arch=sm_61 -O3`), MSVC 2022 via `vcvarsall x64`
- 128 threads/block (same launch shape as the production kernel)
- Primary run: K=1024 instructions per instance per launch, 1 warmup +
  5 timed launches. Stability check at K=2048 × 3 launches reproduced every
  ratio within ~2%.

## Real SMB opcode distribution (measured, not assumed)

Gathered with `profile_opcodes.py` via the production binding's
`CudaBatch.step_profile`: 256 envs from the W1-1 snapshot, frameskip 4,
plausible controller mix, 120 profiled steps = **535,681,514 opcode
executions**, 102 distinct opcodes. Top 20:

| # | opcode | mnemonic | share | # | opcode | mnemonic | share |
|--:|--------|----------|------:|--:|--------|----------|------:|
| 1 | 0xF0 | BEQ rel | 6.77% | 11 | 0xBD | LDA abs,X | 2.76% |
| 2 | 0xAD | LDA abs | 6.49% | 12 | 0x20 | JSR abs | 2.64% |
| 3 | 0xC8 | INY | 6.37% | 13 | 0x60 | RTS | 2.50% |
| 4 | 0xD0 | BNE rel | 5.93% | 14 | 0x18 | CLC | 2.17% |
| 5 | 0x29 | AND #imm | 5.71% | 15 | 0x90 | BCC rel | 2.16% |
| 6 | 0x85 | STA zp | 4.55% | 16 | 0x10 | BPL rel | 2.08% |
| 7 | 0xC9 | CMP #imm | 3.32% | 17 | 0xB5 | LDA zp,X | 1.98% |
| 8 | 0x4A | LSR A | 3.28% | 18 | 0x88 | DEY | 1.76% |
| 9 | 0x99 | STA abs,Y | 3.24% | 19 | 0xCA | DEX | 1.72% |
| 10 | 0xA5 | LDA zp | 3.03% | 20 | 0xA9 | LDA #imm | 1.61% |

Raw counts: [opcode_hist.json](opcode_hist.json) (all 256 opcodes).

## Prototype design

- **Interpreter**: the actual `nesle::cpu::step()` table-driven core, included
  unmodified from `cpp/include/nesle/cpu.hpp`.
- **Bus**: per-instance 2 KB RAM in global memory mirrored across
  0x0000–0x7FFF (NES layout: zero page / stack behave normally); one shared
  32 KB read-only code image at 0x8000–0xFFFF (mirrors NROM — all real envs
  share one SMB PRG; writes ≥ 0x8000 are dropped like writes into cartridge
  ROM). No PPU, no APU: pure CPU dispatch.
- **Code image**: 16,648 instructions sampled from the measured distribution;
  operands randomized within addressing-mode constraints (stores/RMW aim at
  RAM so they do real work; loads may hit ROM). Branch offsets are patched to
  forward instruction boundaries within range so the PC never leaves the
  region or lands mid-instruction; the image ends with `JMP $8000`, so streams
  run forever. **Exclusions** (5.30% of the measured stream, renormalized):
  JSR/RTS (unpaired sampling would pop garbage return addresses), BRK/RTI,
  `JMP ($nn)` (uncontrolled pointer target), and unofficial opcodes (the core
  traps on them). JSR/RTS are stack+memory ops much like PHA/PLA (included),
  so their absence should not bias the A-vs-B comparison materially.
- **Validation**: 3 × 1,000,000 instructions of the image run on the *host*
  through the same interpreter (throws instead of trapping) — no illegal
  opcodes, PC confined to the code region. The synthetic **dynamic** mix
  tracks the measured target well (largest deviation: PHA 3.2% vs 1.3%).
- **Equivalence**: both wavefront kernels reproduce the baseline's final CPU
  states and RAM **byte-for-byte** at 4096 instances (correlated and
  decorrelated) — same computation, different schedule.

Variants:

- **A (status quo)**: one thread per instance, K × `cpu::step()`; divergence
  is whatever the hardware makes of 32 unrelated PCs per warp.
- **B1 (wavefront, opcode bins)**: per instruction, lanes peek their next
  opcode; the warp iterates the distinct opcodes present via a
  ballot/shuffle loop (sm_61 has no `__match_any_sync`), each group executing
  `step()` with a warp-uniform opcode → fully uniform decode dispatch.
- **B2 (wavefront, op-class bins)**: same, but binned by the decode table's
  operation class (LDA/STA/ADC/…, 46 classes) — fewer, larger groups; the
  addressing-mode switch may still diverge within a group.

Correlated mode = all instances at the same PC, identical RAM/registers
(lockstep forever; CuLE's "right after reset" best case). Decorrelated mode =
random start boundary, random RAM and registers per instance (fully drifted
worst case). Real training populations live near the decorrelated end.

## Results (primary run: K=1024, 5 timed launches)

| n | streams | A instr/s | B1 instr/s | B2 instr/s | B1/A | B2/A | bins/warp B1 | bins/warp B2 |
|--:|---|--:|--:|--:|--:|--:|--:|--:|
| 4,096 | correlated | 2.676e9 | 2.828e9 | 2.458e9 | 1.057 | 0.919 | 1.00 | 1.00 |
| 4,096 | decorrelated | 7.563e8 | 2.422e8 | 3.529e8 | **0.320** | **0.467** | 20.61 | 13.89 |
| 32,768 | correlated | 2.702e9 | 2.821e9 | 2.896e9 | 1.044 | 1.072 | 1.00 | 1.00 |
| 32,768 | decorrelated | 1.363e9 | 4.275e8 | 5.784e8 | **0.314** | **0.424** | 20.60 | 13.87 |
| 131,072 | correlated | 2.647e9 | 2.753e9 | 2.815e9 | 1.040 | 1.063 | 1.00 | 1.00 |
| 131,072 | decorrelated | 1.442e9 | 4.503e8 | 6.167e8 | **0.312** | **0.428** | 20.60 | 13.87 |

Stability check (K=2048, 3 launches): B1/A decorrelated = 0.335 / 0.313 /
0.313 at the three sizes; all other ratios within ~2% of the table above.

### Binning overhead

In correlated mode every warp has exactly 1 bin, so B1-vs-A isolates the pure
cost of the peek + ballot/shuffle loop: it is noise-level (**B1/A =
1.04–1.06** — the binning machinery itself is nearly free, and slightly
positive here, presumably from the warmed fetch line). The entire decorrelated
loss therefore comes from **group serialization**, not from binning
bookkeeping.

### Why the wavefront loses

A decorrelated warp carries on average **20.6 distinct opcodes** (13.9
distinct op classes) among its 32 lanes. Opcode binning converts each
warp-instruction into ~20.6 *fully serialized* passes through `step()`. The
hardware's own SIMT reconvergence in variant A is strictly smarter than that:
the table-driven core funnels all 32 lanes through a shared fetch, a shared
table lookup, a ~10-way addressing-mode switch, and only then a ~46-way
operation switch — so lanes with different opcodes still share most of the
per-instruction work. Explicit binning forcibly serializes exactly the work
the hardware was already overlapping. Coarser bins (B2) recover part of the
loss (0.43×) but cannot cross 1.0: any binning scheme pays `bins × step()`
while the hardware pays `max-divergent-path ≤ bins × step()`.

### The ceiling for any anti-divergence scheme

Baseline A itself measures the total cost of decorrelation:
A_decorrelated / A_correlated = **0.54** at 131,072 (0.50 at 32,768; the
4,096-point is occupancy-limited on this 6-SM part and not comparable). So
even a hypothetical zero-cost scheme that removed *all* divergence — perfect
global compaction à la Laine & Karras — could buy at most **~1.9×** on this
CPU-only prototype. The real emulator interleaves PPU catch-up, bus range
dispatch, and per-frame events with every instruction, so dispatch divergence
is a *smaller* fraction of its runtime than it is here; CuLE's measured
~30% worst-case divergence cost on the same ISA caps the realistic payoff
nearer **1.4×**. A global-compaction wavefront would additionally pay CPU
state save/restore traffic per bin, and would start from the 3.2× hole
measured here for the warp-local version.

## Honest caveats

- The prototype has **no PPU/APU work**: dispatch divergence is a larger
  share of runtime here than in the real emulator, which *favors* the
  wavefront. It still lost — the real-emulator outcome would be worse.
- Warp-local binning only; no global compaction/queues (Laine-Karras). That
  design costs state movement the prototype doesn't model, and its best case
  is bounded by the ~1.9× ceiling above.
- JSR/RTS excluded (5.1% of the real stream); branch targets forward-only;
  RAM mirrored across 0x0000–0x7FFF (real bus has PPU/APU registers there).
- Single GPU (Pascal, no independent thread scheduling, no
  `__match_any_sync`). On Volta+ the binning loop gets cheaper primitives,
  but the group-serialization arithmetic — 20.6 serialized bins vs hardware
  reconvergence — is architecture-independent.

## Conclusions for the roadmap

1. **Tier 3 (wavefront interpreter): rejected on measurement.** Remove it
   from consideration for the CPU loop; the megakernel-decomposition idea
   remains valid only where work items are heavyweight (rendering), not for
   ~20-cycle interpreter steps.
2. The measured A_corr/A_decorr gap (~1.9×) confirms the cheap orthogonal
   lever already in the doc: **state-correlation scheduling** (grouping envs
   by reset cohort so warps share game phase) attacks the same gap with
   near-zero code risk.
3. The table-driven core (Tier 2, shipped) is doing its job: hardware
   reconvergence over the two-switch structure is precisely what makes
   explicit binning redundant.

## Reproduce

```powershell
# 1. (optional) re-measure the opcode distribution — needs the built pyd + ROM
.venv\Scripts\python.exe benchmarks\wavefront_prototype\profile_opcodes.py
.venv\Scripts\python.exe benchmarks\wavefront_prototype\make_hist_header.py

# 2. build + run the prototype (vcvarsall x64 + nvcc, standalone exe)
benchmarks\wavefront_prototype\build.ps1
benchmarks\wavefront_prototype\proto.exe            # default: K=1024, 5 launches
benchmarks\wavefront_prototype\proto.exe 2048 3     # K, timed launches
```

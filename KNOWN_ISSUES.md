# Known issues

Honest list of what's broken, deferred, or unverified. Kept current as of
2026-09-30.

## Deferred bugs

- **Renderer samples live PPU state mid-frame, causing transient visual
  artifacts in recordings** (found 2026-07-29 via a user report of GIF
  flicker; game state and RAM-based training are unaffected (render-only).
  One root cause, three observed symptom classes, all confirmed frame-by-frame:
  1. *HUD-less wrong-scroll frames* (~⅓ of frames in a scrolling recording):
     the whole frame renders at the playfield scroll, scrolling the status bar
     away. Real SMB holds the HUD still via a mid-frame scroll change
     (sprite-0 split) that the renderer doesn't emulate.
  2. *Objects flashing in/out*: sprite (OAM) state sampled mid-update.
  3. *Future level content materializing* (e.g. a flagpole appearing
     mid-level for one frame): SMB pre-writes upcoming columns into the
     second nametable; sampled mid-frame, the scroll/nametable-select state
     can expose them.
  **Fix implemented 2026-07-29:** stepping now freezes a per-env presentation
  snapshot at each vblank (OAM, nametable, palette, frame-start and frame-end
  scroll/ctrl), and the renderer draws from the snapshot with a two-region
  scroll split at sprite-0's bottom edge. Measured on a 500-frame scripted
  scrolling run: object-pop events fell 130 → 21 (−84%), and part of the
  residue is legitimate game animation (score popups, spawns). Remaining
  honest caveat: ~14% of frames during heavy action lose the status bar for
  one frame. These are SMB *lag frames* where the game skips its scroll
  reset (real hardware glitches these frames too, our timing makes them more
  frequent); the recorder's HUD filter drops them from GIFs.

- **Title-screen → gameplay transition stalls** (PPU timing bug). SMB's menu
  handler runs and controller polling works, but the edge-detect branch that
  advances `OperMode` never fires under our PPU timing. Worked around
  completely by snapshot reset (`docs/data/smb_level*.state`); fixing the PPU
  timing is real emulator-archaeology work with no training payoff, so it is
  deliberately deferred.

## Limitations by design

- **Mapper support is UxROM-family only.** iNES mappers `0, 2, 11, 30, 34, 94,
  180`. No MMC1/MMC3 etc. Per-mapper verification against real cartridges is
  recorded in *Cartridge banking vs real ROMs* below.
- **`scripts/build_cuda_extension.sh` is POSIX-only**, but
  `scripts/build_cuda_extension.py` is the cross-platform replacement and is
  what should be used everywhere: it locates nvcc, detects the GPU arch via
  torch, drives `vcvarsall` on Windows, and smoke-tests the artifact.
  Confirmed working on Windows with CUDA 12.8 and MSVC 19.44. Folding both into
  `setup.py` so a single build path serves all platforms is still the cleanest
  known improvement.
- **One CUDA thread per env.** Warp divergence leaves throughput on the table
  at very large batches; a warp-per-env or SoA-wavefront redesign is the known
  next optimization (see `docs/phase6-report.md`).
- **No multi-GPU support.** There is no `cudaSetDevice` and no env-to-device
  sharding anywhere in `cpp/`; a batch is one device times N envs launched as a
  single kernel, so a second GPU is simply idle. Arch flags depend on the GPU
  *model*, not on how many are present, so this is a missing feature rather
  than a build issue.

## Unverified claims (recorded, not reproduced on current hardware)

- ~~The A100 numbers were recorded in May 2026 and not re-verified.~~
  **Resolved 2026-09-01.** Independently re-measured on a Colab
  A100-SXM4-80GB from a clean clone: 311,770 env-steps/s at 4,096 envs and a
  peak of **3,274,290 env-steps/s at 65,536 envs**, with 69/69 tests passing
  and all three falsifiability checks green. Every v0.3.0 claim reproduced at
  or above its published value (4,096 to four significant figures). Also
  measured for the first time: crossover versus a single-env CPU at 8 envs, and
  ~195 KB of device memory per environment (12.2 GiB at the 65,536-env peak),
  which confirms the rough "13 GB" figure the README asserted before this run
  without any measurement behind it. Raw output:
  `docs/data/verification-2026-09-01-a100.json`.
  (The interim 2026-07-29 figures of 161,430 and 699,456 predate table-driven
  decode and lazy PPU settlement.) The May phase-6 mode-ablation numbers remain as recorded
  (different measurement modes), but the headline throughput is now current
  and verified.
- The C++ tests in `tests/cpp/` run in CI on Ubuntu (the `cpp-tests` job runs
  `scripts/run_cpp_tests.sh`). They are not run on Windows and are not visible
  to pytest, so a local `pytest` run does not cover them.

## Cartridge banking vs real ROMs (measured 2026-09-30)

Each ROM was run on the host `Console` and on the real CUDA kernel from a single
batch, then all 2 KiB of CPU RAM was compared byte-for-byte. The host was also
run in lockstep against the shared batch headers one CPU instruction at a time
(PC, opcode, cycle count, all RAM and PRG RAM, per instruction), which is what
makes a mapping bug distinguishable from a kernel bug.

| Mapper | ROM | Result |
| --- | --- | --- |
| 2 (UxROM) | `Contra (USA)`, NES 2.0 submapper 2, bus conflicts **on** | 60 frames, exact match |
| 2 (UxROM) | `Contra (U) [T-Rus uBAH009]`, plain iNES, bus conflicts off | 60 frames, exact match |
| 94 (UN1ROM) | `Senjou no Ookami (Japan)` | 60 frames, exact match |
| 34 (BNROM) | `Deadly Towers (USA)`, `Mashou (Japan)` | 60 frames, exact match |
| 180 | `Crazy Climber (Japan)` | mapping exact, kernel diverges (below) |
| 34 (NINA-001) | none available | **unverified** (below) |

Use `scripts/report_rom_mappers.py` to vet a new test cart before trusting its
filename: the mapper number lives only in the header, and translations of
mapper 94 and 180 boards are often re-tagged as plain mapper 2, because both
are the same UNROM PCB with different logic gates. It reports CRC32 (to pin the
exact revision), mapper, submapper, bank sizes and the support verdict.

- **Mapper 180: the mapping is correct; the production kernel diverges from
  the host by one RAM byte on `Crazy Climber`.** Host and shared batch headers
  stay in lockstep for all 60 frames at every instruction, so the
  window-at-top geometry (bank 0 fixed at `$8000`, switchable window at
  `$C000`, entry at `$8000`, exactly as the 74HC08 variant on the UNROM PCB)
  is right. The real kernel matches through frame 8; from frame 9 exactly one
  byte differs, `$0732`: host `0xD4` vs kernel `0xBB`, a constant offset of
  `0x19` (25) after which both sides decrement by 3 per frame. All other 2047
  RAM bytes, the PRG bank (5) and CHR RAM agree. A constant one-time offset is
  not mapping drift; it looks like the game sampling a PPU-timing-derived value
  once and getting a slightly different number, i.e. a discrepancy in the
  kernel's hot PPU path (`step_batch_console_instruction_hot`, register-resident
  state) rather than in the mapper. `Contra`, `Senjou no Ookami`, `Deadly
  Towers` and `Mashou` match for all 60 frames, so this is not a general kernel
  defect. Not investigated further.

- **Mapper 34 NINA-001 is unverified against a real cartridge.** The only
  available `Impossible Mission II` dump is CRC32 `F73D26D6`, and it does not
  run: the CPU leaves the code path and executes data in the fixed region. The
  evidence points at the *fixed* 24 KiB not being the last 24 KiB of the chip.
  Under the current assumption (`fixed = prg_size - 24K`, chip offset `0xA000`)
  the fixed region decodes to obvious data (`$B000` is `21 22 23 24 25 26`, a
  counting ramp); at chip offset `0x2000` the same addresses decode to code
  (`$AC3A` is `A9 00 85 76 20 08`, i.e. `LDA #$00` / `STA $76` / `JSR`), and the
  reset vectors then resolve to `prg[0x7FFA]`, which does hold a vector table
  (`03 80 06 80 09 80`). This has **not** been changed, for two reasons: the
  correct wiring could not be confirmed (nesdev.org returns HTTP 403 and no
  reference implementation was reachable), and the dump itself is suspect, since
  it carries vector tables in two places and its two 32 KiB halves agree on
  only 3% of bytes. The database copy of the same board is CRC32 `92A3D007`.
  Retest with that dump before touching the geometry; if it passes as-is, the
  layout is correct and the dump was the problem. Settling it properly means
  adding an explicit fixed-region base to `MapperLayout` instead of deriving
  `size - fixed_bytes` independently in `console.hpp` and `cuda_module.cu`.

- Four `Battletoads` dumps in the test folder are mapper 7 (AxROM), which is
  not supported. AxROM is a good next candidate precisely because it reuses the
  generalized window code (a 32 KiB window over the whole `$8000-$FFFF` space,
  no fixed bank, and the 8 KiB CHR RAM is already allocated), but adding it was
  left out of scope.

## Windows/WDDM training-throughput ceiling (measured 2026-07-29)

On Windows (WDDM driver model, GTX 1050 Ti), interleaving torch CUDA *kernels*
(gather/`copy_`/sampling; memcpy-class ops are exempt) with the emulator's
step-kernel launches costs ~100–200 ms per interleaved kernel class per step,
capping the native-PPO rollout at ~3k env-steps/s at 2048 envs even though raw
stepping does ~25k and policy inference alone takes ~2.5 ms. Reproduce with
`benchmarks/profile_native_ppo.py`. Measured to be independent of: sync flavor
(device sync vs event sync vs no sync), action-tensor allocation pattern
(fresh vs persistent), and cudart linkage (static vs shared). CUDA graphs on
the policy forward did not help. The pathology does not appear on Linux: the
A100 (Colab, Linux) training run sustained ~31k env-steps/s end-to-end. If you
train on Windows, this is the known ceiling; the suspected culprit is WDDM
command-buffer scheduling, and the practical fix is training on Linux/WSL or a
TCC-mode GPU.

## Environment quirks

- The CPU-baseline number in `benchmarks/gpu_vs_cpu.py` is sensitive to host
  background load (observed 290–335 env-steps/s across runs on the same
  machine); the GPU numbers are stable within ~1%.
- A stale `_cuda_core.pyd` fails silently (old behavior, no error). Rebuild
  after any `cpp/` change.

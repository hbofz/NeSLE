# Known issues

Honest list of what's broken, deferred, or unverified. Kept current as of
2026-07-28.

## Deferred bugs

- **Renderer samples live PPU state mid-frame, causing transient visual
  artifacts in recordings** (found 2026-07-29 via a user report of GIF
  flicker; game state and RAM-based training are unaffected — render-only).
  One root cause, three observed symptom classes, all confirmed frame-by-frame:
  1. *HUD-less wrong-scroll frames* (~⅓ of frames in a scrolling recording):
     the whole frame renders at the playfield scroll, scrolling the status bar
     away — real SMB holds the HUD still via a mid-frame scroll change
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
  one frame — these are SMB *lag frames* where the game skips its scroll
  reset (real hardware glitches these frames too, our timing makes them more
  frequent); the recorder's HUD filter drops them from GIFs.

- **Title-screen → gameplay transition stalls** (PPU timing bug). SMB's menu
  handler runs and controller polling works, but the edge-detect branch that
  advances `OperMode` never fires under our PPU timing. Worked around
  completely by snapshot reset (`docs/data/smb_level*.state`); fixing the PPU
  timing is real emulator-archaeology work with no training payoff, so it is
  deliberately deferred.

## Limitations by design

- **NROM (mapper 0) only.** No MMC1/MMC3 etc.
- **One CUDA thread per env.** Warp divergence leaves throughput on the table
  at very large batches; a warp-per-env or SoA-wavefront redesign is the known
  next optimization (see `docs/phase6-report.md`).
- **Smart reward is tuned for World 1-1.** Other levels run with 1-1's
  constants; per-level profiles exist only in the vendored CPU project
  (`project/mario-rl-ram/src/mario_rl/reward_profiles/`).
- **`scripts/build_cuda_extension.sh` is POSIX-only.** Windows uses the manual
  recipe in `docs/build-windows.md`. Porting both into `setup.py` so one build
  path serves all platforms is the cleanest known improvement.

## Unverified claims (recorded, not reproduced on current hardware)

- The A100 numbers (`docs/phase6-report.md`, README A100 table, 75M-step /
  40-min PPO run) were recorded in May 2026 on Colab A100 hardware. They cannot
  be re-verified on the local GTX 1050 Ti; re-run
  `scripts/legacy/reproduce_phase6.sh` on an A100 to refresh them.
- The C++ tests in `tests/cpp/` are compiled ad hoc by `scripts/verify.sh`
  (POSIX) and are not wired into pytest or CI on Windows.

## Windows/WDDM training-throughput ceiling (measured 2026-07-29)

On Windows (WDDM driver model, GTX 1050 Ti), interleaving torch CUDA *kernels*
(gather/`copy_`/sampling — memcpy-class ops are exempt) with the emulator's
step-kernel launches costs ~100–200 ms per interleaved kernel class per step,
capping the native-PPO rollout at ~3k env-steps/s at 2048 envs even though raw
stepping does ~25k and policy inference alone takes ~2.5 ms. Reproduce with
`benchmarks/profile_native_ppo.py`. Measured to be independent of: sync flavor
(device sync vs event sync vs no sync), action-tensor allocation pattern
(fresh vs persistent), and cudart linkage (static vs shared). CUDA graphs on
the policy forward did not help. The pathology does not appear on Linux — the
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

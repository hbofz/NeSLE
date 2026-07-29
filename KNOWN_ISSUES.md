# Known issues

Honest list of what's broken, deferred, or unverified. Kept current as of
2026-07-28.

## Deferred bugs

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

## Environment quirks

- The CPU-baseline number in `benchmarks/gpu_vs_cpu.py` is sensitive to host
  background load (observed 290–335 env-steps/s across runs on the same
  machine); the GPU numbers are stable within ~1%.
- A stale `_cuda_core.pyd` fails silently (old behavior, no error). Rebuild
  after any `cpp/` change.

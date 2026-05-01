# Headless Runner

The headless runner loads an iNES `.nes` file, constructs the CPU-side NROM
console, resets from `$FFFC`, and runs for a requested number of frames without
rendering.

Build:

```sh
c++ -std=c++20 -Icpp/include cpp/src/rom.cpp cpp/tools/run_nes_headless.cpp -o /tmp/nesle_run_nes_headless
```

Run:

```sh
/tmp/nesle_run_nes_headless "Super Mario Bros. (World).nes" \
  --allow-trap \
  --require-mario-target \
  --require-mario-boot \
  --frames 120 \
  --max-instructions 5000000 \
  --trace 32
```

The output reports status, completed frames, instruction count, CPU cycles, PC,
last opcode, PPU position, mapper, PRG/CHR sizes, and decoded Super Mario Bros.
RAM fields such as position, world/stage, timer, lives, status, death flags, and
flag-get, plus a conservative Mario boot plausibility check. It also reports
accumulated reward components, a CPU RAM hash, and a rendered RGB frame hash.
With `--trace N`, it prints the last `N` executed instructions with CPU and PPU
timing state. This is the bridge from synthetic NROM programs to real Super
Mario Bros. boot and action-trace debugging.

Real games can park the CPU in short idle loops while PPU status drives the
next step, so the Mario smoke path uses `--allow-trap` and runs a longer boot
window before checking decoded RAM.

Use `--actions` to apply controller masks per frame. Masks are decimal or
`0x`-prefixed bytes; `mask*count` repeats a value for several frames:

```sh
/tmp/nesle_run_nes_headless "Super Mario Bros. (World).nes" \
  --allow-trap \
  --require-mario-target \
  --require-mario-boot \
  --actions '0*120,8*4,0*16,130*180' \
  --max-instructions 15000000 \
  --trace 0
```

That sequence waits for boot, presses Start, releases briefly, then holds
Right+B. The hashes make the trace reproducible and suitable for comparison
against external reference-emulator captures.

The repo also includes an optional smoke gate that stays skipped unless a local
ROM path is provided:

```sh
NESLE_ROM_PATH="/path/to/Super Mario Bros. (World).nes" sh scripts/smoke_user_rom.sh
NESLE_ROM_PATH="/path/to/Super Mario Bros. (World).nes" sh scripts/smoke_phase2_user_rom.sh
```

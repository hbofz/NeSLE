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
  --frames 1 \
  --max-instructions 5000000 \
  --trace 32
```

The output reports status, completed frames, instruction count, CPU cycles, PC,
last opcode, PPU position, mapper, and PRG/CHR sizes. With `--trace N`, it also
prints the last `N` executed instructions with CPU and PPU timing state. This is
the Phase 1 bridge from synthetic NROM programs to real Super Mario Bros. boot
debugging.

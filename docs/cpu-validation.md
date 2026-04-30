# CPU Validation

The current Phase 1 CPU gate is Klaus Dormann-style functional testing. The
runner is intentionally generic: it loads a flat 6502 binary into a 64 KB test
bus, starts at a configured PC, and stops when the program counter traps by
looping on itself.

## Runner

Build the runner:

```sh
c++ -std=c++20 -Icpp/include cpp/tools/run_6502_binary.cpp -o /tmp/run_6502_binary
```

Run a Klaus binary:

```sh
/tmp/run_6502_binary path/to/6502_functional_test.bin \
  --load 0x0000 \
  --start 0x0400 \
  --success 0x3469 \
  --variant mos6502
```

The stock Klaus `6502_functional_test.bin` is a plain NMOS 6502 test. For the
NES CPU profile, use `--variant 2a03` and a decimal-disabled build of the test,
because the Ricoh 2A03 keeps the decimal flag but does not implement BCD
arithmetic. The full MOS 6502 profile exists only as a validation aid.

## Current Result

With the upstream stock binary downloaded to `/tmp` and kept out of the repo:

```text
/tmp/run_6502_binary /tmp/6502_functional_test.bin \
  --load 0x0000 --start 0x0400 --success 0x3469 --variant mos6502
success pc=0x3469 opcode=0x4c instructions=30646177 cycles=96241367
```

The same stock binary under `--variant 2a03` traps before success, as expected,
because the stock configuration enables decimal ADC/SBC checks that the NES CPU
does not implement in hardware.

Sources:

- Klaus Dormann tests: https://github.com/Klaus2m5/6502_65C02_functional_tests
- 6502 test program overview: https://www.nesdev.org/wiki/Visual6502wiki/6502TestPrograms
- 6502 opcode reference: https://www.nesdev.org/obelisk-6502-guide/reference.html

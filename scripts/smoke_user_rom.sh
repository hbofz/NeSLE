#!/usr/bin/env sh
set -eu

if [ -z "${NESLE_ROM_PATH:-}" ]; then
  echo "NESLE_ROM_PATH is not set; skipping user ROM smoke test."
  exit 0
fi

if [ ! -f "$NESLE_ROM_PATH" ]; then
  echo "NESLE_ROM_PATH does not point to a file: $NESLE_ROM_PATH" >&2
  exit 1
fi

c++ -std=c++20 -Icpp/include cpp/src/rom.cpp cpp/src/smb.cpp cpp/tools/run_nes_headless.cpp -o /tmp/nesle_run_nes_headless
/tmp/nesle_run_nes_headless "$NESLE_ROM_PATH" \
  --allow-trap \
  --require-mario-target \
  --require-mario-boot \
  --frames "${NESLE_SMOKE_FRAMES:-120}" \
  --max-instructions "${NESLE_SMOKE_MAX_INSTRUCTIONS:-5000000}" \
  --trace "${NESLE_SMOKE_TRACE:-64}"

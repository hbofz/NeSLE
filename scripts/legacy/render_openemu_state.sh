#!/usr/bin/env sh
set -eu

if [ -z "${NESLE_ROM_PATH:-}" ]; then
  echo "NESLE_ROM_PATH is not set; skipping OpenEmu/Nestopia render."
  exit 0
fi

if [ ! -f "$NESLE_ROM_PATH" ]; then
  echo "NESLE_ROM_PATH does not point to a file: $NESLE_ROM_PATH" >&2
  exit 1
fi

state_path="${NESLE_OPENEMU_STATE_PATH:-$HOME/Library/Application Support/OpenEmu/Save States/NES/Super Mario Bros. (World)/Auto Save State.oesavestate/State}"
output_path="${NESLE_OPENEMU_RENDER_PATH:-/tmp/nesle_openemu_state.ppm}"

if [ ! -f "$state_path" ]; then
  echo "OpenEmu/Nestopia state file was not found: $state_path" >&2
  exit 1
fi

c++ -std=c++20 -Icpp/include cpp/src/rom.cpp cpp/tools/render_nestopia_state.cpp -o /tmp/nesle_render_nestopia_state
/tmp/nesle_render_nestopia_state "$NESLE_ROM_PATH" "$state_path" "$output_path"

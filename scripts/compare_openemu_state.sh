#!/usr/bin/env sh
set -eu

if [ -z "${NESLE_ROM_PATH:-}" ]; then
  echo "NESLE_ROM_PATH is not set; skipping OpenEmu/Nestopia screenshot comparison."
  exit 0
fi

if [ ! -f "$NESLE_ROM_PATH" ]; then
  echo "NESLE_ROM_PATH does not point to a file: $NESLE_ROM_PATH" >&2
  exit 1
fi

state_path="${NESLE_OPENEMU_STATE_PATH:-$HOME/Library/Application Support/OpenEmu/Save States/NES/Super Mario Bros. (World)/Auto Save State.oesavestate/State}"
screenshot_path="${NESLE_OPENEMU_SCREENSHOT_PATH:-$HOME/Library/Application Support/OpenEmu/Save States/NES/Super Mario Bros. (World)/Auto Save State.oesavestate/ScreenShot}"
render_path="${NESLE_OPENEMU_RENDER_PATH:-/tmp/nesle_openemu_state.ppm}"
reference_path="${NESLE_OPENEMU_REFERENCE_PATH:-/tmp/nesle_openemu_reference_256.bmp}"
max_rgb_mae="${NESLE_OPENEMU_MAX_RGB_MAE:-45}"
min_rgb_corr="${NESLE_OPENEMU_MIN_RGB_CORR:-0.65}"

if [ ! -f "$state_path" ]; then
  echo "OpenEmu/Nestopia state file was not found: $state_path" >&2
  exit 1
fi

if [ ! -f "$screenshot_path" ]; then
  echo "OpenEmu screenshot was not found: $screenshot_path" >&2
  exit 1
fi

if ! command -v sips >/dev/null 2>&1; then
  echo "sips is required to normalize OpenEmu screenshots on macOS" >&2
  exit 1
fi

c++ -std=c++20 -Icpp/include cpp/src/rom.cpp cpp/tools/render_nestopia_state.cpp -o /tmp/nesle_render_nestopia_state
c++ -std=c++20 -Icpp/include cpp/tools/compare_rgb_frame.cpp -o /tmp/nesle_compare_rgb_frame

/tmp/nesle_render_nestopia_state "$NESLE_ROM_PATH" "$state_path" "$render_path"
sips -z 240 256 -s format bmp "$screenshot_path" --out "$reference_path" >/dev/null
/tmp/nesle_compare_rgb_frame "$render_path" "$reference_path" \
  --max-rgb-mae "$max_rgb_mae" \
  --min-rgb-corr "$min_rgb_corr"

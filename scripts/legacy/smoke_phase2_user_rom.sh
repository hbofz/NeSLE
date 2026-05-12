#!/usr/bin/env sh
set -eu

if [ -z "${NESLE_ROM_PATH:-}" ]; then
  echo "NESLE_ROM_PATH is not set; skipping Phase 2 user ROM smoke test."
  exit 0
fi

if [ ! -f "$NESLE_ROM_PATH" ]; then
  echo "NESLE_ROM_PATH does not point to a file: $NESLE_ROM_PATH" >&2
  exit 1
fi

c++ -std=c++20 -Icpp/include cpp/src/rom.cpp cpp/src/smb.cpp cpp/tools/run_nes_headless.cpp -o /tmp/nesle_run_nes_headless

neutral=$(/tmp/nesle_run_nes_headless "$NESLE_ROM_PATH" \
  --allow-trap \
  --require-mario-target \
  --require-mario-boot \
  --actions '0*120,8*4,0*16,0*180' \
  --max-instructions "${NESLE_PHASE2_MAX_INSTRUCTIONS:-15000000}" \
  --trace 0)

right=$(/tmp/nesle_run_nes_headless "$NESLE_ROM_PATH" \
  --allow-trap \
  --require-mario-target \
  --require-mario-boot \
  --actions '0*120,8*4,0*16,130*180' \
  --max-instructions "${NESLE_PHASE2_MAX_INSTRUCTIONS:-15000000}" \
  --trace 0)

field() {
  key="$1"
  line="$2"
  printf '%s\n' "$line" | tr ' ' '\n' | awk -F= -v key="$key" '$1 == key { print $2 }'
}

neutral_x=$(field mario_x "$neutral")
right_x=$(field mario_x "$right")
neutral_reward=$(field reward_total "$neutral")
right_reward=$(field reward_total "$right")
neutral_frame_hash=$(field frame_hash "$neutral")
right_frame_hash=$(field frame_hash "$right")

echo "$neutral"
echo "$right"

if [ "$right_x" -le "$neutral_x" ]; then
  echo "expected Right+B trace to move Mario farther than neutral: neutral=$neutral_x right=$right_x" >&2
  exit 1
fi

if [ "$right_reward" -le "$neutral_reward" ]; then
  echo "expected Right+B trace reward to exceed neutral: neutral=$neutral_reward right=$right_reward" >&2
  exit 1
fi

if [ "$right_frame_hash" = "$neutral_frame_hash" ]; then
  echo "expected input traces to produce different rendered frame hashes" >&2
  exit 1
fi

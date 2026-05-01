#!/usr/bin/env sh
set -eu

rom_path="${NESLE_ROM_PATH:-}"
if [ -z "$rom_path" ] && [ -f "Super Mario Bros. (World).nes" ]; then
  rom_path="Super Mario Bros. (World).nes"
fi

if [ -z "$rom_path" ]; then
  echo "NESLE_ROM_PATH is not set; skipping Phase 4 user ROM smoke test."
  exit 0
fi

if [ ! -f "$rom_path" ]; then
  echo "NESLE_ROM_PATH does not point to a file: $rom_path" >&2
  exit 1
fi

python_bin="${PYTHON:-python3}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/nesle-matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/private/tmp/nesle-cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME/fontconfig"
if ! "$python_bin" - <<'PY'
try:
    import numpy as np
    raise SystemExit(0 if hasattr(np, "uint8") else 1)
except ImportError:
    raise SystemExit(1)
PY
then
  echo "complete numpy is not available; skipping Phase 4 user ROM smoke test."
  exit 0
fi

PYTHONPATH=src "$python_bin" scripts/smoke_phase4_user_rom.py "$rom_path" \
  --backend "${NESLE_PHASE4_BACKEND:-auto}" \
  --num-envs "${NESLE_PHASE4_NUM_ENVS:-2}" \
  --steps "${NESLE_PHASE4_STEPS:-8}" \
  --frameskip "${NESLE_PHASE4_FRAMESKIP:-4}" \
  --action-space "${NESLE_PHASE4_ACTION_SPACE:-simple}"

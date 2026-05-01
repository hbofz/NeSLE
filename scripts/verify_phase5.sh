#!/usr/bin/env sh
set -eu

python_bin="${PYTHON:-python3}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/nesle-matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/private/tmp/nesle-cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME/fontconfig"

rom_path="${NESLE_ROM_PATH:-}"
if [ -z "$rom_path" ] && [ -f "Super Mario Bros. (World).nes" ]; then
  rom_path="Super Mario Bros. (World).nes"
fi

if [ -z "$rom_path" ]; then
  echo "NESLE_ROM_PATH is not set; skipping Phase 5 benchmark smoke."
  exit 0
fi

if ! "$python_bin" - <<'PY'
try:
    import numpy as np
    raise SystemExit(0 if hasattr(np, "uint8") else 1)
except ImportError:
    raise SystemExit(1)
PY
then
  echo "complete numpy is not available; skipping Phase 5 benchmark smoke."
  exit 0
fi

PYTHONPATH=src "$python_bin" benchmarks/phase5_benchmark.py "$rom_path" \
  --env-counts 1,2 \
  --steps 3 \
  --warmup-steps 1 \
  --modes step,render \
  --backend "${NESLE_PHASE5_BACKEND:-auto}" \
  --output /tmp/nesle_phase5_smoke.json

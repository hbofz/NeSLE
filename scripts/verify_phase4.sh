#!/usr/bin/env sh
set -eu

python_bin="${PYTHON:-python3}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/nesle-matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/private/tmp/nesle-cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME/fontconfig"
PYTHONPATH=src "$python_bin" scripts/verify_phase4.py

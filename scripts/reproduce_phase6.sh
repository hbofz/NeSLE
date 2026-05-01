#!/usr/bin/env sh
set -eu

python_bin="${PYTHON:-python3}"
rom_path="${NESLE_ROM_PATH:-${1:-}}"

if [ -z "$rom_path" ]; then
  echo "Usage: NESLE_ROM_PATH=/path/to/rom.nes sh scripts/reproduce_phase6.sh"
  echo "   or: sh scripts/reproduce_phase6.sh /path/to/rom.nes"
  exit 2
fi

if [ ! -f "$rom_path" ]; then
  echo "ROM path does not point to a file: $rom_path"
  exit 2
fi

pip_flags="${NESLE_PIP_INSTALL_FLAGS:-}"
"$python_bin" -m pip install $pip_flags -e '.[dev,rl]'
NESLE_CUDA_ARCH="${NESLE_CUDA_ARCH:-sm_80}" PYTHON="$python_bin" sh scripts/build_cuda_extension.sh

PYTHONPATH=src "$python_bin" benchmarks/phase6_console_ablation.py "$rom_path" \
  --env-counts "${NESLE_PHASE6_ENV_COUNTS:-1,8,32,128}" \
  --frameskips "${NESLE_PHASE6_FRAMESKIPS:-1,2,4,8}" \
  --steps "${NESLE_PHASE6_STEPS:-20}" \
  --warmup-steps "${NESLE_PHASE6_WARMUP_STEPS:-3}" \
  --modes "${NESLE_PHASE6_MODES:-rgb,render_only,no_copy}" \
  --output benchmarks/results/phase6_console_ablation.json

PYTHONPATH=src "$python_bin" benchmarks/plot_phase6.py \
  --input benchmarks/results/phase6_console_ablation.json \
  --output-dir benchmarks/results

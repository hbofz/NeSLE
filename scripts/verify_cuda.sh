#!/usr/bin/env sh
set -eu

nvcc_bin="${NVCC:-nvcc}"
if ! command -v "$nvcc_bin" >/dev/null 2>&1; then
  echo "nvcc is not available; skipping CUDA verification."
  exit 0
fi

cuda_arch="${NESLE_CUDA_ARCH:-sm_90}"
output_path="${NESLE_CUDA_SMOKE_BIN:-/tmp/nesle_cuda_smoke}"

"$nvcc_bin" -std=c++20 "-arch=$cuda_arch" -Icpp/include \
  cpp/src/cuda/kernels.cu cpp/tools/run_cuda_smoke.cu \
  -o "$output_path"

"$output_path"

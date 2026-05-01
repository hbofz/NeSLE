#!/usr/bin/env sh
set -eu

nvcc_bin="${NVCC:-}"
if [ -z "$nvcc_bin" ]; then
  for candidate in nvcc /usr/local/cuda/bin/nvcc /usr/local/cuda-12.8/bin/nvcc /usr/local/cuda-12.6/bin/nvcc; do
    if command -v "$candidate" >/dev/null 2>&1; then
      nvcc_bin="$candidate"
      break
    fi
  done
fi

if ! command -v "$nvcc_bin" >/dev/null 2>&1; then
  echo "nvcc is not available; skipping CUDA kernel benchmark."
  exit 0
fi

cuda_arch="${NESLE_CUDA_ARCH:-sm_80}"
output_path="${NESLE_CUDA_KERNEL_BENCH_BIN:-/tmp/nesle_cuda_kernel_benchmark}"

"$nvcc_bin" -std=c++20 "-arch=$cuda_arch" -Icpp/include \
  cpp/src/cuda/kernels.cu cpp/tools/benchmark_cuda_kernels.cu \
  -o "$output_path"

"$output_path" "$@"

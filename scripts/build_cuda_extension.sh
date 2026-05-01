#!/usr/bin/env sh
set -eu

python_bin="${PYTHON:-python3}"
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
  echo "nvcc is not available; skipping CUDA extension build."
  exit 0
fi

if ! "$python_bin" -m pybind11 --includes >/tmp/nesle_cuda_pybind11_includes.txt 2>/dev/null; then
  echo "pybind11 is not available for $python_bin; skipping CUDA extension build."
  exit 0
fi

includes=$(cat /tmp/nesle_cuda_pybind11_includes.txt)
ext_suffix=$("$python_bin" - <<'PY'
import sysconfig

print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")
PY
)
cuda_arch="${NESLE_CUDA_ARCH:-sm_80}"
output_path="${NESLE_CUDA_EXTENSION_PATH:-src/nesle/_cuda_core${ext_suffix}}"

"$nvcc_bin" -std=c++20 "-arch=$cuda_arch" -Icpp/include $includes \
  --compiler-options '-fPIC' --shared \
  cpp/src/rom.cpp cpp/src/cuda/kernels.cu cpp/bindings/cuda_module.cu \
  -o "$output_path"

PYTHONPATH=src "$python_bin" - <<'PY'
import nesle._cuda_core as cuda_core

batch = cuda_core.CudaBatch(2, 4)
obs = batch.reset()
result = batch.step([0x80, 0x00])
assert obs.shape == (2, 240, 256, 3)
assert result["obs"].shape == (2, 240, 256, 3)
assert result["rewards"].shape == (2,)
assert result["dones"].shape == (2,)
assert batch.ram().shape == (2, 2048)
rom = bytearray(b"NES\x1a" + bytes([2, 1, 0, 0]) + bytes(8))
rom.extend(bytes([0xEA]) * (32 * 1024))
rom[16 + 0x7FFC] = 0x00
rom[16 + 0x7FFD] = 0x80
rom.extend(bytes(8 * 1024))
console_batch = cuda_core.CudaBatch(1, 1, bytes(rom))
console_obs = console_batch.reset()
console_result = console_batch.step([0x00])
assert console_batch.name == "cuda-console"
assert console_obs.shape == (1, 240, 256, 3)
assert console_result["obs"].shape == (1, 240, 256, 3)
assert console_result["rewards"].shape == (1,)
assert console_result["dones"].shape == (1,)
assert console_batch.ram().shape == (1, 2048)
print("cuda_extension_check ok")
PY

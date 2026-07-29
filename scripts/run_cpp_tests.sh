#!/usr/bin/env sh
# Compile and run the C++ unit tests in tests/cpp/ with a host compiler.
# The test_cuda_* files exercise the CUDA batch code paths compiled as host
# C++ (the headers are host-compilable); no GPU or nvcc is required.
set -eu

CXX="${CXX:-c++}"
out_dir="${TMPDIR:-/tmp}"

run() {
  name=$1
  shift
  echo "== $name"
  "$CXX" -std=c++20 -Icpp/include "$@" "tests/cpp/$name.cpp" -o "$out_dir/nesle_$name"
  "$out_dir/nesle_$name"
}

run test_core cpp/src/rom.cpp cpp/src/smb.cpp
run test_cpu cpp/src/rom.cpp
run test_cpu_runner
run test_console cpp/src/rom.cpp
run test_headless cpp/src/rom.cpp cpp/src/smb.cpp
run test_cuda_batch cpp/src/smb.cpp
run test_cuda_bus cpp/src/rom.cpp
run test_cuda_cpu_step cpp/src/rom.cpp
run test_cuda_batch_runner cpp/src/rom.cpp
run test_cuda_ppu
run test_cuda_render
run test_cuda_batch_console cpp/src/rom.cpp
run test_cuda_reset_cache

echo "all C++ tests passed"

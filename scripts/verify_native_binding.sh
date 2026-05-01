#!/usr/bin/env sh
set -eu

python_bin="${PYTHON:-python3}"
cxx_bin="${CXX:-c++}"
output_path="${NESLE_NATIVE_BINDING_CHECK_BIN:-/tmp/nesle_core_build_check.so}"

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "python is not available; skipping native binding compile check."
  exit 0
fi

if ! "$python_bin" -m pybind11 --includes >/tmp/nesle_pybind11_includes.txt 2>/dev/null; then
  echo "pybind11 is not available for $python_bin; skipping native binding compile check."
  exit 0
fi

includes=$(cat /tmp/nesle_pybind11_includes.txt)
case "$(uname -s)" in
  Darwin*) ldflags="-shared -undefined dynamic_lookup" ;;
  *) ldflags="-shared -fPIC" ;;
esac

"$cxx_bin" -O0 -g -std=c++20 $includes -Icpp/include \
  cpp/bindings/module.cpp cpp/src/rom.cpp cpp/src/smb.cpp \
  $ldflags -o "$output_path"

echo "native_binding_check path=$output_path"

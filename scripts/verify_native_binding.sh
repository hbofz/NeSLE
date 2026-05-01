#!/usr/bin/env sh
set -eu

python_bin="${PYTHON:-python3}"
cxx_bin="${CXX:-c++}"
build_dir="${NESLE_NATIVE_BINDING_CHECK_DIR:-/tmp/nesle_native_binding_check}"

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "python is not available; skipping native binding compile check."
  exit 0
fi

if ! "$python_bin" -m pybind11 --includes >/tmp/nesle_pybind11_includes.txt 2>/dev/null; then
  echo "pybind11 is not available for $python_bin; skipping native binding compile check."
  exit 0
fi

includes=$(cat /tmp/nesle_pybind11_includes.txt)
ext_suffix=$("$python_bin" - <<'PY'
import sysconfig

print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")
PY
)
mkdir -p "$build_dir"
output_path="$build_dir/_core$ext_suffix"
case "$(uname -s)" in
  Darwin*) ldflags="-shared -undefined dynamic_lookup" ;;
  *) ldflags="-shared -fPIC" ;;
esac

"$cxx_bin" -O0 -g -std=c++20 $includes -Icpp/include \
  cpp/bindings/module.cpp cpp/src/rom.cpp cpp/src/smb.cpp \
  $ldflags -o "$output_path"

"$python_bin" - "$output_path" <<'PY'
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path


def make_rom() -> bytes:
    header = bytearray(b"NES\x1a")
    header.extend([2, 1, 0, 0])
    header.extend(b"\x00" * 8)
    prg = bytearray([0xEA] * (2 * 16 * 1024))
    prg[0x7FFC] = 0x00
    prg[0x7FFD] = 0x80
    return bytes(header + prg + bytearray(8 * 1024))


module_path = sys.argv[1]
spec = importlib.util.spec_from_file_location("_core", module_path)
if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load native module spec: {module_path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

metadata = module.parse_ines_metadata(make_rom())
assert metadata["is_nrom"]
console = module.NativeConsole(make_rom())
assert len(console.ram()) == 2048
assert len(console.frame()) == 240 * 256 * 3
step = console.step(0, 1, 50000)
assert step["instructions"] > 0
assert step["frames_completed"] >= 1
assert len(console.ram()) == 2048
assert len(console.frame()) == 240 * 256 * 3

try:
    import numpy as np

    if not hasattr(np, "uint8"):
        raise ImportError("numpy is present but incomplete")
except ImportError:
    print(
        f"native_binding_check path={module_path} instructions={step['instructions']} "
        "env_backend=skipped_numpy"
    )
else:
    sys.path.insert(0, "src")
    import nesle

    sys.modules["nesle._core"] = module
    setattr(nesle, "_core", module)
    with tempfile.TemporaryDirectory() as tmp:
        rom_path = Path(tmp) / "native.nes"
        rom_path.write_bytes(make_rom())
        env = nesle.make(str(rom_path), backend="native")
        obs, info = env.reset()
        assert obs.shape == (240, 256, 3)
        assert info["backend"] == "native"
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (240, 256, 3)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        env.close()
    print(
        f"native_binding_check path={module_path} instructions={step['instructions']} "
        "env_backend=native"
    )
PY

"""Build nesle._cuda_core for the current interpreter and GPU, on any OS.

Cross-platform replacement for scripts/build_cuda_extension.sh (which is
POSIX-only) and the manual Windows recipe in docs/build-windows.md.

    python scripts/build_cuda_extension.py            # auto-detect everything
    NESLE_CUDA_ARCH=sm_80 python scripts/build_cuda_extension.py

Environment overrides:
    NVCC                       path to nvcc (else: $PATH, then common install dirs)
    NESLE_CUDA_ARCH            e.g. sm_61 / sm_80 / sm_90 (else: detected via torch,
                               falling back to sm_80)
    NESLE_CUDA_EXTENSION_PATH  output path (else: src/nesle/_cuda_core<EXT_SUFFIX>)
    NESLE_REQUIRE_CUDA=1       exit non-zero when nvcc/pybind11 are missing instead
                               of skipping quietly (CI-friendly default matches the
                               shell script: skip with exit 0)
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCES = [
    "cpp/src/rom.cpp",
    "cpp/src/cuda/kernels.cu",
    "cpp/bindings/cuda_module.cu",
]


def _skip(message: str) -> "int":
    print(message)
    return 1 if os.environ.get("NESLE_REQUIRE_CUDA") == "1" else 0


def find_nvcc() -> str | None:
    explicit = os.environ.get("NVCC")
    if explicit:
        return explicit if Path(explicit).exists() else None
    found = shutil.which("nvcc")
    if found:
        return found
    patterns = [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12*/bin/nvcc",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12*\bin\nvcc.exe",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13*\bin\nvcc.exe",
        r"D:\Dev\CUDA\*\bin\nvcc.exe",
    ]
    candidates: list[str] = []
    for pattern in patterns:
        candidates.extend(sorted(glob.glob(pattern)))
    return candidates[-1] if candidates else None


def detect_arch() -> str:
    override = os.environ.get("NESLE_CUDA_ARCH")
    if override:
        return override
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return f"sm_{major}{minor}"
    except Exception:
        pass
    return "sm_80"


def find_vcvarsall() -> str | None:
    """Locate MSVC's vcvarsall.bat so nvcc can find cl.exe on Windows."""
    vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    if vswhere.exists():
        try:
            root = subprocess.run(
                [str(vswhere), "-latest", "-products", "*",
                 "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                 "-property", "installationPath"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            if root:
                candidate = Path(root) / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
                if candidate.exists():
                    return str(candidate)
        except (subprocess.CalledProcessError, OSError):
            pass
    for pattern in (
        r"C:\Program Files*\Microsoft Visual Studio\2022\*\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files*\Microsoft Visual Studio\2019\*\VC\Auxiliary\Build\vcvarsall.bat",
    ):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[-1]
    return None


def main() -> int:
    nvcc = find_nvcc()
    if nvcc is None:
        return _skip("nvcc is not available; cannot build nesle._cuda_core.")
    try:
        import pybind11
    except ImportError:
        return _skip("pybind11 is not available; cannot build nesle._cuda_core.")

    arch = detect_arch()
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or (".pyd" if os.name == "nt" else ".so")
    output = os.environ.get("NESLE_CUDA_EXTENSION_PATH") or str(REPO / "src" / "nesle" / f"_cuda_core{ext_suffix}")

    cmd = [
        nvcc, "-std=c++20", f"-arch={arch}",
        # Static cudart (nvcc default) on purpose: --cudart=shared makes the
        # extension require cudart64_*.dll on the DLL search path, so plain
        # `import nesle` breaks unless torch happens to be imported first.
        # (Shared cudart was also tested as a fix for the WDDM interleave
        # slowdown documented in KNOWN_ISSUES.md — it made no difference.)
        f"-I{REPO / 'cpp' / 'include'}",
        f"-I{pybind11.get_include()}",
        f"-I{sysconfig.get_paths()['include']}",  # Python.h on every platform
    ]
    if os.name == "nt":
        libs_dir = Path(sys.base_prefix) / "libs"
        py_lib = libs_dir / f"python{sys.version_info.major}{sys.version_info.minor}.lib"
        if not py_lib.exists():
            print(f"could not find {py_lib}; is this a standard CPython install?")
            return 1
        cmd += ["-shared"]
        cmd += [str(REPO / s) for s in SOURCES]
        cmd += ["-o", output, str(py_lib)]
    else:
        cmd += ["--compiler-options", "-fPIC", "--shared"]
        cmd += [str(REPO / s) for s in SOURCES]
        cmd += ["-o", output]

    print(f"nvcc:   {nvcc}")
    print(f"arch:   {arch}")
    print(f"output: {output}")

    if os.name == "nt" and shutil.which("cl") is None:
        vcvars = find_vcvarsall()
        if vcvars is None:
            print("MSVC cl.exe not on PATH and vcvarsall.bat not found; install VS 2022 Build Tools (C++ workload).")
            return 1
        quoted = subprocess.list2cmdline(cmd)
        # cmd /s /c with the whole command as one verbatim string survives the
        # nested quotes around both vcvarsall.bat and nvcc paths.
        full = f'cmd /s /c "call "{vcvars}" x64 && {quoted}"'
        result = subprocess.run(full, cwd=REPO)
    else:
        result = subprocess.run(cmd, cwd=REPO)
    if result.returncode != 0:
        print(f"nvcc failed with exit code {result.returncode}")
        return result.returncode

    # Smoke-test in a subprocess so this process never locks the built library
    # (a loaded .pyd on Windows blocks the next rebuild's link step). Load the
    # exact artifact we just built by path — `import nesle._cuda_core` would
    # test whatever pyd sits in src/nesle instead when output is overridden.
    smoke = (
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('_cuda_core', r'{output}')\n"
        "c = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(c)\n"
        "b = c.CudaBatch(2, 4)\n"
        "obs = b.reset()\n"
        "r = b.step([0x80, 0x00])\n"
        "assert obs.shape == (2, 240, 256, 3)\n"
        "assert r['rewards'].shape == (2,)\n"
        "assert b.ram().shape == (2, 2048)\n"
        "rom = bytearray(b'NES\\x1a' + bytes([2, 1, 0, 0]) + bytes(8))\n"
        "rom.extend(bytes([0xEA]) * (32 * 1024))\n"
        "rom[16 + 0x7FFC] = 0x00\n"
        "rom[16 + 0x7FFD] = 0x80\n"
        "rom.extend(bytes(8 * 1024))\n"
        "cb = c.CudaBatch(1, 1, bytes(rom))\n"
        "cb.reset()\n"
        "cb.step([0x00])\n"
        "assert cb.name == 'cuda-console'\n"
        "print('cuda_extension_check ok')\n"
    )
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    check = subprocess.run([sys.executable, "-c", smoke], env=env, cwd=REPO)
    return check.returncode


if __name__ == "__main__":
    raise SystemExit(main())

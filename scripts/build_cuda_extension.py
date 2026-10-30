"""Build nesle._cuda_core for the current interpreter and GPU, on any OS, and smoke-test it.

`pip install -e .` already builds the extension through setup.py, which shares
the toolchain logic below. Run this directly to rebuild quickly and exercise the
result on the GPU.

    python scripts/build_cuda_extension.py            # auto-detect everything
    NESLE_CUDA_ARCH=sm_80 python scripts/build_cuda_extension.py

Environment overrides:
    NVCC                       nvcc path or command name (else: $PATH, then common
                               install dirs). One that cannot be found, such as
                               NVCC=none, skips the build
    NESLE_CUDA_ARCH            e.g. sm_61 / sm_80 / sm_90 (else: detected via
                               nvidia-smi, then torch, falling back to sm_80)
    NESLE_CUDA_EXTENSION_PATH  output path (else: src/nesle/_cuda_core<EXT_SUFFIX>)
    NESLE_REQUIRE_CUDA=1       exit non-zero when nvcc/pybind11 are missing instead
                               of skipping quietly (CI-friendly default: skip with
                               exit 0)
"""

from __future__ import annotations

import glob
import os
import re
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


def _nvcc_release(nvcc: str) -> tuple[int, int]:
    """(major, minor) of a toolkit's nvcc, or (0, 0) when it cannot be probed."""
    try:
        out = subprocess.run(
            [nvcc, "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        return (0, 0)
    match = re.search(r"release (\d+)\.(\d+)", out)
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


# -std=c++20 needs a 12.x toolkit or newer.
_MIN_NVCC_RELEASE = (12, 0)

_SEARCH_PATTERNS = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\nvcc.exe",
    r"D:\Dev\CUDA\*\bin\nvcc.exe",
    "/usr/local/cuda/bin/nvcc",
    "/usr/local/cuda-12*/bin/nvcc",
    "/usr/local/cuda-13*/bin/nvcc",
]


def _discover_nvcc() -> list[tuple[tuple[int, int], str]]:
    """Installed toolkits as (release, path), newest first by full version.

    Probing costs a subprocess per toolkit, so this is only reached when the
    nvcc on PATH is missing or too old - see find_nvcc. Sorting on (major,
    minor) keeps the choice stable when several 12.x toolkits are installed.
    """
    found: set[str] = set()
    for pattern in _SEARCH_PATTERNS:
        found.update(glob.glob(pattern))
    probed = [(_nvcc_release(path), path) for path in found]
    probed.sort(reverse=True)
    return probed


def find_nvcc() -> str | None:
    """First usable nvcc, preferring a toolkit new enough for -std=c++20.

    PATH wins when it has a new enough nvcc, and in that case nothing else is
    probed at all. A PATH nvcc that is missing or older than 12.x falls through
    to the newest installed toolkit rather than failing, because an 11.x nvcc
    cannot compile this script's -std=c++20.
    """
    explicit = os.environ.get("NVCC")
    if explicit:
        return shutil.which(explicit)

    on_path = shutil.which("nvcc")
    if on_path and _nvcc_release(on_path) >= _MIN_NVCC_RELEASE:
        return on_path

    discovered = _discover_nvcc()
    for release, candidate in discovered:
        if release >= _MIN_NVCC_RELEASE:
            if on_path:
                print(
                    "nvcc on PATH is too old for -std=c++20; "
                    f"using {candidate} instead."
                )
            return candidate

    # Nothing is new enough. Prefer PATH so the compiler reports the version
    # problem itself, otherwise the newest toolkit found.
    return on_path or (discovered[0][1] if discovered else None)


def _nvidia_smi_arch() -> str | None:
    # WSL ships nvidia-smi outside the default PATH.
    search = os.pathsep.join([os.environ.get("PATH", ""), "/usr/lib/wsl/lib"])
    smi = shutil.which("nvidia-smi", path=search)
    if smi is None:
        return None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        return None
    match = re.match(r"\s*(\d+)\.(\d+)", out)
    return f"sm_{match.group(1)}{match.group(2)}" if match else None


def nvcc_too_old(nvcc: str) -> bool:
    """True when nvcc reports a release below 12.x; an unprobeable nvcc gets to try and fail."""
    return (0, 0) < _nvcc_release(nvcc) < _MIN_NVCC_RELEASE


def detect_arch() -> str:
    override = os.environ.get("NESLE_CUDA_ARCH")
    if override:
        return override
    # nvidia-smi first: pip builds in an isolated env where torch is not importable.
    from_smi = _nvidia_smi_arch()
    if from_smi:
        return from_smi
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


def compile_extension(nvcc: str, arch: str, output: str) -> int:
    """Run nvcc for nesle._cuda_core and return its exit code."""
    import pybind11

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
        return subprocess.run(full, cwd=REPO).returncode
    return subprocess.run(cmd, cwd=REPO).returncode


def main() -> int:
    nvcc = find_nvcc()
    if nvcc is None:
        return _skip("nvcc is not available; cannot build nesle._cuda_core.")
    try:
        import pybind11  # noqa: F401
    except ImportError:
        return _skip("pybind11 is not available; cannot build nesle._cuda_core.")

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or (".pyd" if os.name == "nt" else ".so")
    output = os.environ.get("NESLE_CUDA_EXTENSION_PATH") or str(REPO / "src" / "nesle" / f"_cuda_core{ext_suffix}")

    returncode = compile_extension(nvcc, detect_arch(), output)
    if returncode != 0:
        print(f"nvcc failed with exit code {returncode}")
        return returncode

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

# Installation

The short path is in the [README](../README.md). This document covers
platform-specific details: the PyTorch wheel, Windows, CUDA architectures, and
disk requirements.

## Requirements

| Requirement | Detail |
|---|---|
| Python | 3.10 or newer |
| GPU | Any CUDA GPU. Tested on GTX 1050 Ti (sm_61, 4 GB) and A100 (sm_80, 80 GB) |
| CUDA Toolkit | 12.x. Pascal (GTX 10-series) requires CTK 12.x, as CTK 13 dropped sm_61 |
| Toolchain | C++20 (GCC or Clang on Linux, MSVC on Windows) |
| PyTorch | CUDA build required for all training paths |
| OS | Linux and Windows 11 are tested. macOS builds the CPU core only |

Allow roughly 9 GB of transient disk for the CUDA PyTorch wheel install.

## PyTorch wheel

The `rl` extra installs CPU-only PyTorch from PyPI. The CUDA build carries the
same version number, so pip treats the requirement as satisfied and keeps the
CPU wheel. `--force-reinstall` is required:

```sh
python -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
```

Verify:

```sh
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

A correct install reports a `+cu126` style version and `True`. A CPU wheel here
still allows the emulator to run on CUDA, but the policy network will train on
CPU.

## Building the CUDA extension

```sh
python scripts/build_cuda_extension.py
```

The script detects nvcc, MSVC, and the local GPU architecture through PyTorch.
Override the architecture when building for a different target:

```sh
NESLE_CUDA_ARCH=sm_80 python scripts/build_cuda_extension.py   # A100
NESLE_CUDA_ARCH=sm_90 python scripts/build_cuda_extension.py   # H100
```

`scripts/build_cuda_extension.sh` is retained for POSIX scripting and Docker.

A stale `_cuda_core.pyd` or `.so` fails silently and retains the previous
behavior. Rebuild after any change under `cpp/`.

## Windows

PowerShell requires two adjustments to the README commands: join multi-line
commands onto a single line (or replace the trailing `\` with a backtick), and
omit the single quotes around `.[dev,rl]`.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .[dev,rl]
python -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
python scripts\build_cuda_extension.py
```

If the build script cannot locate the toolchain, use the manual recipe in
[build-windows.md](build-windows.md).

Windows has a known training-throughput ceiling under the WDDM driver model
that does not affect Linux. Review [KNOWN_ISSUES.md](../KNOWN_ISSUES.md) before
committing to long training runs on Windows.

## Docker

`docker/cuda.Dockerfile` provides a CUDA build and test image as an alternative
to a local toolchain.

## ROMs

NeSLE does not distribute ROMs. Supply a legally obtained Super Mario Bros. ROM
in iNES format, mapper 0, typically named `Super Mario Bros. (World).nes`. The
`*.nes` pattern is gitignored.

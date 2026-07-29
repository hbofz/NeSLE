# Building the CUDA extension on Windows

`scripts/build_cuda_extension.sh` is POSIX-only (it uses `-fPIC --shared`,
POSIX paths, and `/tmp`). On Windows, invoke `nvcc` directly through the MSVC
environment. This is the recipe used to build the extension this repo was
tested with (Windows 11, MSVC 2022 Build Tools, Python 3.14, GTX 1050 Ti).

## Prerequisites

- **Visual Studio 2022 Build Tools** with the C++ workload (`vcvarsall.bat`).
- **CUDA Toolkit 12.x.** For Pascal GPUs (sm_61, e.g. GTX 10-series) CTK 12.x
  is *required*: CTK 13+ dropped Pascal support. It is fine to install a 12.x
  "sidecar" alongside a newer toolkit — during installation, **skip the driver
  component** so your host driver is not downgraded.
- `pip install -e '.[dev]'` in your venv (provides pybind11).

## Build

PowerShell (adjust the five paths to your installation):

```powershell
$nvcc     = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe'
$vcvars   = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat'
$pybindInc = "$PWD\.venv\Lib\site-packages\pybind11\include"
$pyInc    = "$env:LOCALAPPDATA\Python\pythoncore-3.14-64\Include"   # your CPython include dir
$pyLib    = "$env:LOCALAPPDATA\Python\pythoncore-3.14-64\libs\python314.lib"
$out      = "$PWD\src\nesle\_cuda_core.cp314-win_amd64.pyd"          # match your EXT_SUFFIX

$cmd = "call `"$vcvars`" x64 && `"$nvcc`" -std=c++20 -arch=sm_61 -Icpp\include -I`"$pybindInc`" -I`"$pyInc`" -shared cpp\src\rom.cpp cpp\src\cuda\kernels.cu cpp\bindings\cuda_module.cu -o `"$out`" `"$pyLib`""
& cmd /c $cmd
```

Pick `-arch` for your GPU: `sm_61` (GTX 10-series), `sm_80` (A100),
`sm_86` (RTX 30-series), `sm_89` (RTX 40-series), `sm_90` (H100).
The output filename must match your interpreter's `EXT_SUFFIX`
(`python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"`).

## Verify

```powershell
.venv\Scripts\python.exe -c "from nesle._cuda_core import CudaBatch; b = CudaBatch(1, 4); print(hasattr(b, 'poke_ram'))"
```

should print `True`.

## Gotchas

- The built `.pyd` is held open by any Python process that imported it. Close
  all Python processes before rebuilding, or the link step fails with
  "Access is denied".
- `vcvarsall.bat` prints a `vswhere.exe` line to stderr that can look like an
  error in PowerShell — harmless unless the exit code is nonzero.
- Rebuild after **any** change under `cpp/`; a stale `.pyd` fails silently
  (old behavior, no error).
- On sm_61, nvcc 12.9 prints a deprecation warning ("support for architectures
  prior to sm_75 will be removed in a future release") — expected, harmless.
- The link step drops stray `_cuda_core*.lib`/`.exp` files next to the `.pyd`;
  they are ignorable (and gitignored) linker artifacts.
- The CUDA torch wheel is ~2.6 GB; `--force-reinstall` needs roughly 7–9 GB of
  transient free disk (wheel + cache + unpacked copy). Use `--no-cache-dir` if
  disk is tight.

# Build the wavefront-dispatch prototype as a standalone exe (not a .pyd).
# Same toolchain recipe as docs/build-windows.md: vcvarsall x64, then nvcc.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..')).Path
$nvcc = 'D:\Dev\CUDA\Cuda 12.9\bin\nvcc.exe'
$vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat'

$cmd = "call `"$vcvars`" x64 && `"$nvcc`" -std=c++20 -arch=sm_61 -O3 -I`"$repo\cpp\include`" `"$here\proto.cu`" -o `"$here\proto.exe`""
& cmd /c $cmd
exit $LASTEXITCODE

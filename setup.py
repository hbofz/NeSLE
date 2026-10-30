from __future__ import annotations

import os
import sys
from glob import glob
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import Extension, setup
from setuptools.errors import CompileError
from setuptools.modified import newer_group

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from build_cuda_extension import (  # noqa: E402
    SOURCES,
    compile_extension,
    detect_arch,
    find_nvcc,
    nvcc_too_old,
)

CUDA_EXTENSION = "nesle._cuda_core"
HEADERS = [path for path in glob("cpp/include/**", recursive=True) if os.path.isfile(path)]


class BuildExt(build_ext):
    def run(self):
        self.nvcc = find_nvcc()
        if self.nvcc is None or nvcc_too_old(self.nvcc):
            self._skip_cuda_core()
        super().run()

    def _skip_cuda_core(self):
        """Build the CPU core alone, so macOS and CI still get a working install."""
        message = f"no nvcc 12.x or newer available; cannot build {CUDA_EXTENSION}"
        if os.environ.get("NESLE_REQUIRE_CUDA") == "1":
            raise CompileError(message)
        self.warn(f"{message}; building the CPU core only")
        # A binary from an earlier build here would otherwise ship in the wheel.
        Path(self.build_lib, self.get_ext_filename(CUDA_EXTENSION)).unlink(missing_ok=True)
        # Later steps, like strict editable installs, expect every listed extension to exist.
        self.extensions = [ext for ext in self.extensions if ext.name != CUDA_EXTENSION]

    def build_extension(self, ext):
        if ext.name == CUDA_EXTENSION:
            self._build_cuda_core(ext)
        else:
            super().build_extension(ext)

    def _build_cuda_core(self, ext):
        # Absolute because vcvarsall can change directory before nvcc runs.
        output = os.path.abspath(self.get_ext_fullpath(ext.name))
        arch = detect_arch()
        # build_temp outlives the binary's inputs, so remember what it was built with.
        toolchain = f"{arch} {self.nvcc}"
        stamp = Path(self.build_temp) / "_cuda_core.toolchain"
        if (
            not self.force
            and stamp.is_file()
            and stamp.read_text() == toolchain
            and not newer_group(ext.sources + ext.depends, output)
        ):
            return

        self.mkpath(os.path.dirname(output))
        self.mkpath(self.build_temp)
        # A failed compile can leave a partial binary newer than its sources.
        stamp.unlink(missing_ok=True)
        # With a toolchain present, a failure stops the install: carrying on would
        # leave any previous _cuda_core in src/nesle silently in use.
        returncode = compile_extension(self.nvcc, arch, output)
        if returncode != 0:
            raise CompileError(
                f"nvcc failed with exit code {returncode}. Fix the toolchain, or set "
                "NVCC=none to install the CPU core only."
            )
        stamp.write_text(toolchain)
        # MSVC linker droppings; they do not belong in a wheel.
        for suffix in (".lib", ".exp"):
            Path(output).with_suffix(suffix).unlink(missing_ok=True)


ext_modules = [
    Pybind11Extension(
        "nesle._core",
        [
            "cpp/bindings/module.cpp",
            "cpp/src/rom.cpp",
            "cpp/src/smb.cpp",
        ],
        include_dirs=["cpp/include"],
        depends=HEADERS,
        cxx_std=20,
    ),
    Extension(CUDA_EXTENSION, SOURCES, depends=[*HEADERS, "scripts/build_cuda_extension.py"]),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": BuildExt})

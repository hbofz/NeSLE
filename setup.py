from __future__ import annotations

from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


ROOT = Path(__file__).parent

ext_modules = [
    Pybind11Extension(
        "nesle._core",
        [
            str(ROOT / "cpp" / "bindings" / "module.cpp"),
            str(ROOT / "cpp" / "src" / "rom.cpp"),
            str(ROOT / "cpp" / "src" / "smb.cpp"),
        ],
        include_dirs=[str(ROOT / "cpp" / "include")],
        cxx_std=20,
    )
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})

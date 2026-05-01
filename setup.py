from __future__ import annotations

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


ext_modules = [
    Pybind11Extension(
        "nesle._core",
        [
            "cpp/bindings/module.cpp",
            "cpp/src/rom.cpp",
            "cpp/src/smb.cpp",
        ],
        include_dirs=["cpp/include"],
        cxx_std=20,
    )
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})

#!/usr/bin/env sh
set -eu

PYTHONPATH=src python3 -m unittest discover -s tests
c++ -std=c++20 -Icpp/include cpp/src/rom.cpp cpp/src/smb.cpp tests/cpp/test_core.cpp -o /tmp/nesle_cpp_tests
/tmp/nesle_cpp_tests
c++ -std=c++20 -Icpp/include cpp/src/rom.cpp tests/cpp/test_cpu.cpp -o /tmp/nesle_cpu_tests
/tmp/nesle_cpu_tests

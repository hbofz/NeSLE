#!/usr/bin/env sh
set -eu

PYTHONPATH=src python3 -m unittest discover -s tests
c++ -std=c++20 -Icpp/include cpp/src/rom.cpp cpp/src/smb.cpp tests/cpp/test_core.cpp -o /tmp/nesle_cpp_tests
/tmp/nesle_cpp_tests
c++ -std=c++20 -Icpp/include cpp/src/rom.cpp tests/cpp/test_cpu.cpp -o /tmp/nesle_cpu_tests
/tmp/nesle_cpu_tests
c++ -std=c++20 -Icpp/include tests/cpp/test_cpu_runner.cpp -o /tmp/nesle_cpu_runner_tests
/tmp/nesle_cpu_runner_tests
c++ -std=c++20 -Icpp/include cpp/tools/run_6502_binary.cpp -o /tmp/nesle_run_6502_binary

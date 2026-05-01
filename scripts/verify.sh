#!/usr/bin/env sh
set -eu

PYTHONPATH=src python3 -m unittest discover -s tests
c++ -std=c++20 -Icpp/include cpp/src/rom.cpp cpp/src/smb.cpp tests/cpp/test_core.cpp -o /tmp/nesle_cpp_tests
/tmp/nesle_cpp_tests
c++ -std=c++20 -Icpp/include cpp/src/rom.cpp tests/cpp/test_cpu.cpp -o /tmp/nesle_cpu_tests
/tmp/nesle_cpu_tests
c++ -std=c++20 -Icpp/include tests/cpp/test_cpu_runner.cpp -o /tmp/nesle_cpu_runner_tests
/tmp/nesle_cpu_runner_tests
c++ -std=c++20 -Icpp/include cpp/src/rom.cpp tests/cpp/test_console.cpp -o /tmp/nesle_console_tests
/tmp/nesle_console_tests
c++ -std=c++20 -Icpp/include cpp/src/rom.cpp cpp/src/smb.cpp tests/cpp/test_headless.cpp -o /tmp/nesle_headless_tests
/tmp/nesle_headless_tests
c++ -std=c++20 -Icpp/include cpp/src/smb.cpp tests/cpp/test_cuda_batch.cpp -o /tmp/nesle_cuda_batch_tests
/tmp/nesle_cuda_batch_tests
c++ -std=c++20 -Icpp/include cpp/src/rom.cpp tests/cpp/test_cuda_bus.cpp -o /tmp/nesle_cuda_bus_tests
/tmp/nesle_cuda_bus_tests
c++ -std=c++20 -Icpp/include cpp/src/rom.cpp tests/cpp/test_cuda_cpu_step.cpp -o /tmp/nesle_cuda_cpu_step_tests
/tmp/nesle_cuda_cpu_step_tests
c++ -std=c++20 -Icpp/include cpp/src/rom.cpp tests/cpp/test_cuda_batch_runner.cpp -o /tmp/nesle_cuda_batch_runner_tests
/tmp/nesle_cuda_batch_runner_tests
c++ -std=c++20 -Icpp/include tests/cpp/test_cuda_ppu.cpp -o /tmp/nesle_cuda_ppu_tests
/tmp/nesle_cuda_ppu_tests
c++ -std=c++20 -Icpp/include cpp/src/rom.cpp tests/cpp/test_cuda_batch_console.cpp -o /tmp/nesle_cuda_batch_console_tests
/tmp/nesle_cuda_batch_console_tests
c++ -std=c++20 -Icpp/include cpp/tools/run_6502_binary.cpp -o /tmp/nesle_run_6502_binary
c++ -std=c++20 -Icpp/include cpp/src/rom.cpp cpp/src/smb.cpp cpp/tools/run_nes_headless.cpp -o /tmp/nesle_run_nes_headless
/tmp/nesle_run_nes_headless /tmp/nesle_headless_test.nes --frames 1 --max-instructions 50000 --trace 2
c++ -std=c++20 -Icpp/include cpp/src/rom.cpp cpp/tools/render_nestopia_state.cpp -o /tmp/nesle_render_nestopia_state
c++ -std=c++20 -Icpp/include cpp/tools/compare_rgb_frame.cpp -o /tmp/nesle_compare_rgb_frame

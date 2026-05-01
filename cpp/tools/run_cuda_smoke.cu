#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include "nesle/cuda/batch_cpu.hpp"
#include "nesle/cuda/batch_step.cuh"
#include "nesle/cuda/kernels.cuh"

namespace {

void check(cudaError_t error, const char* label) {
    if (error != cudaSuccess) {
        std::cerr << label << ": " << cudaGetErrorString(error) << '\n';
        std::exit(1);
    }
}

void set_digits(std::vector<std::uint8_t>& ram,
                std::size_t base,
                int hundreds,
                int tens,
                int ones) {
    ram[base + nesle::cuda::kMarioTimeDigits] = static_cast<std::uint8_t>(hundreds);
    ram[base + nesle::cuda::kMarioTimeDigits + 1] = static_cast<std::uint8_t>(tens);
    ram[base + nesle::cuda::kMarioTimeDigits + 2] = static_cast<std::uint8_t>(ones);
}

void seed_mario_ram(std::vector<std::uint8_t>& ram,
                    std::size_t env,
                    int x_page,
                    int x_screen,
                    int timer,
                    std::uint8_t player_state,
                    std::uint8_t y_viewport) {
    const auto base = env * nesle::cuda::kCpuRamBytes;
    ram[base + nesle::cuda::kMarioXPage] = static_cast<std::uint8_t>(x_page);
    ram[base + nesle::cuda::kMarioXScreen] = static_cast<std::uint8_t>(x_screen);
    ram[base + nesle::cuda::kMarioYViewport] = y_viewport;
    ram[base + nesle::cuda::kMarioPlayerState] = player_state;
    ram[base + nesle::cuda::kMarioLives] = 2;
    set_digits(ram, base, timer / 100, (timer / 10) % 10, timer % 10);
}

__global__ void cpu_trace_kernel(nesle::cuda::BatchBuffers buffers, std::uint8_t* passed) {
    nesle::cuda::reset_batch_cpu_env(buffers, 0);
    for (int i = 0; i < 11; ++i) {
        (void)nesle::cuda::step_batch_cpu_env(buffers, 0);
    }

    const auto* ram = nesle::cuda::env_cpu_ram(buffers, 0);
    const auto* prg_ram = nesle::cuda::env_prg_ram(buffers, 0);
    passed[0] = buffers.cpu.pc[0] == 0x8017 &&
                        buffers.cpu.a[0] == 0x01 &&
                        buffers.cpu.cycles[0] == 42 &&
                        ram[0x0002] == 0x11 &&
                        prg_ram[0] == 0x01
                    ? 1
                    : 0;
}

}  // namespace

int main() {
    constexpr std::uint32_t kNumEnvs = 4096;

    std::vector<std::uint8_t> host_ram(kNumEnvs * nesle::cuda::kCpuRamBytes, 0);
    std::vector<int> host_previous_x(kNumEnvs, 0);
    std::vector<int> host_previous_time(kNumEnvs, 400);
    std::vector<std::uint8_t> host_previous_dying(kNumEnvs, 0);
    std::vector<float> host_rewards(kNumEnvs, 0.0F);
    std::vector<std::uint8_t> host_done(kNumEnvs, 0);

    for (std::uint32_t env = 0; env < kNumEnvs; ++env) {
        const auto mode = env % 3;
        const auto previous_x = 40 + static_cast<int>(env % 64);
        const auto dx = mode == 0 ? 3 : mode == 1 ? 10 : 2;
        const auto player_state = static_cast<std::uint8_t>(mode == 2 ? 0x0B : 8);
        host_previous_x[env] = previous_x;
        seed_mario_ram(host_ram, env, 0, previous_x + dx, 399, player_state, 1);
    }

    std::uint8_t* device_ram = nullptr;
    int* device_previous_x = nullptr;
    int* device_previous_time = nullptr;
    std::uint8_t* device_previous_dying = nullptr;
    float* device_rewards = nullptr;
    std::uint8_t* device_done = nullptr;

    check(cudaMalloc(&device_ram, host_ram.size()), "cudaMalloc ram");
    check(cudaMalloc(&device_previous_x, host_previous_x.size() * sizeof(int)),
          "cudaMalloc previous_x");
    check(cudaMalloc(&device_previous_time, host_previous_time.size() * sizeof(int)),
          "cudaMalloc previous_time");
    check(cudaMalloc(&device_previous_dying, host_previous_dying.size()),
          "cudaMalloc previous_dying");
    check(cudaMalloc(&device_rewards, host_rewards.size() * sizeof(float)), "cudaMalloc rewards");
    check(cudaMalloc(&device_done, host_done.size()), "cudaMalloc done");

    check(cudaMemcpy(device_ram, host_ram.data(), host_ram.size(), cudaMemcpyHostToDevice),
          "cudaMemcpy ram");
    check(cudaMemcpy(device_previous_x,
                     host_previous_x.data(),
                     host_previous_x.size() * sizeof(int),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy previous_x");
    check(cudaMemcpy(device_previous_time,
                     host_previous_time.data(),
                     host_previous_time.size() * sizeof(int),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy previous_time");
    check(cudaMemcpy(device_previous_dying,
                     host_previous_dying.data(),
                     host_previous_dying.size(),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy previous_dying");
    check(cudaMemcpy(device_rewards,
                     host_rewards.data(),
                     host_rewards.size() * sizeof(float),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy rewards");
    check(cudaMemcpy(device_done, host_done.data(), host_done.size(), cudaMemcpyHostToDevice),
          "cudaMemcpy done");

    nesle::cuda::BatchBuffers buffers{};
    buffers.cpu.ram = device_ram;
    buffers.previous_mario_x = device_previous_x;
    buffers.previous_mario_time = device_previous_time;
    buffers.previous_mario_dying = device_previous_dying;
    buffers.rewards = device_rewards;
    buffers.done = device_done;

    nesle::cuda::launch_step_kernel(
        buffers,
        nesle::cuda::StepConfig{kNumEnvs, 1, false},
        nullptr);
    check(cudaGetLastError(), "launch_step_kernel");
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    check(cudaMemcpy(host_rewards.data(),
                     device_rewards,
                     host_rewards.size() * sizeof(float),
                     cudaMemcpyDeviceToHost),
          "cudaMemcpy rewards back");
    check(cudaMemcpy(host_done.data(), device_done, host_done.size(), cudaMemcpyDeviceToHost),
          "cudaMemcpy done back");
    check(cudaMemcpy(host_previous_x.data(),
                     device_previous_x,
                     host_previous_x.size() * sizeof(int),
                     cudaMemcpyDeviceToHost),
          "cudaMemcpy previous_x back");

    check(cudaFree(device_ram), "cudaFree ram");
    check(cudaFree(device_previous_x), "cudaFree previous_x");
    check(cudaFree(device_previous_time), "cudaFree previous_time");
    check(cudaFree(device_previous_dying), "cudaFree previous_dying");
    check(cudaFree(device_rewards), "cudaFree rewards");
    check(cudaFree(device_done), "cudaFree done");

    for (std::uint32_t env = 0; env < kNumEnvs; ++env) {
        const auto mode = env % 3;
        const auto previous_x = 40 + static_cast<int>(env % 64);
        const auto dx = mode == 0 ? 3 : mode == 1 ? 10 : 2;
        const auto expected_reward = mode == 0 ? 2.0F : mode == 1 ? -1.0F : -24.0F;
        const auto expected_done = static_cast<std::uint8_t>(mode == 2 ? 1 : 0);
        if (host_rewards[env] != expected_reward || host_done[env] != expected_done ||
            host_previous_x[env] != previous_x + dx) {
            std::cerr << "unexpected batched reward result env=" << env
                      << " reward=" << host_rewards[env]
                      << " done=" << static_cast<int>(host_done[env])
                      << " previous_x=" << host_previous_x[env] << '\n';
            return 1;
        }
    }

    std::cout << "cuda_smoke envs=" << kNumEnvs
              << " rewards=" << host_rewards[0] << "," << host_rewards[1] << "," << host_rewards[2]
              << " done=" << static_cast<int>(host_done[0]) << ","
              << static_cast<int>(host_done[1]) << "," << static_cast<int>(host_done[2])
              << '\n';

    constexpr std::size_t kPrgRomSize = 32 * 1024;
    std::vector<std::uint8_t> host_prg_rom(kPrgRomSize, 0xEA);
    std::size_t pc = 0;
    auto emit = [&](std::uint8_t value) {
        host_prg_rom[pc++] = value;
    };
    auto lda_imm = [&](std::uint8_t value) {
        emit(0xA9);
        emit(value);
    };
    auto sta_abs = [&](std::uint16_t address) {
        emit(0x8D);
        emit(static_cast<std::uint8_t>(address & 0x00FF));
        emit(static_cast<std::uint8_t>(address >> 8));
    };

    lda_imm(0x10);
    emit(0x85);  // STA $02
    emit(0x02);
    emit(0xE6);  // INC $02
    emit(0x02);
    lda_imm(1);
    sta_abs(0x4016);
    lda_imm(0);
    sta_abs(0x4016);
    emit(0xAD);  // LDA $4016
    emit(0x16);
    emit(0x40);
    emit(0x29);  // AND #$01
    emit(0x01);
    sta_abs(0x6000);
    emit(0x4C);  // JMP $8017
    emit(0x17);
    emit(0x80);
    host_prg_rom[0x7FFC] = 0x00;
    host_prg_rom[0x7FFD] = 0x80;

    std::vector<std::uint8_t> host_cpu_ram(nesle::cuda::kCpuRamBytes, 0);
    std::vector<std::uint8_t> host_prg_ram(nesle::cuda::kPrgRamBytes, 0);
    std::uint8_t host_action = 0x01;
    std::uint16_t host_pc = 0;
    std::uint8_t host_a = 0;
    std::uint8_t host_x = 0;
    std::uint8_t host_y = 0;
    std::uint8_t host_sp = 0;
    std::uint8_t host_p = 0;
    std::uint64_t host_cycles = 0;
    std::uint8_t host_controller_shift = 0;
    std::uint8_t host_controller_shift_count = 8;
    std::uint8_t host_controller_strobe = 0;
    std::uint8_t host_cpu_passed = 0;

    std::uint8_t* device_prg_rom = nullptr;
    std::uint8_t* device_cpu_ram = nullptr;
    std::uint8_t* device_prg_ram = nullptr;
    std::uint8_t* device_action = nullptr;
    std::uint16_t* device_pc = nullptr;
    std::uint8_t* device_a = nullptr;
    std::uint8_t* device_x = nullptr;
    std::uint8_t* device_y = nullptr;
    std::uint8_t* device_sp = nullptr;
    std::uint8_t* device_p = nullptr;
    std::uint64_t* device_cycles = nullptr;
    std::uint8_t* device_controller_shift = nullptr;
    std::uint8_t* device_controller_shift_count = nullptr;
    std::uint8_t* device_controller_strobe = nullptr;
    std::uint8_t* device_cpu_passed = nullptr;

    check(cudaMalloc(&device_prg_rom, host_prg_rom.size()), "cudaMalloc cpu prg_rom");
    check(cudaMalloc(&device_cpu_ram, host_cpu_ram.size()), "cudaMalloc cpu ram");
    check(cudaMalloc(&device_prg_ram, host_prg_ram.size()), "cudaMalloc cpu prg_ram");
    check(cudaMalloc(&device_action, sizeof(host_action)), "cudaMalloc action");
    check(cudaMalloc(&device_pc, sizeof(host_pc)), "cudaMalloc pc");
    check(cudaMalloc(&device_a, sizeof(host_a)), "cudaMalloc a");
    check(cudaMalloc(&device_x, sizeof(host_x)), "cudaMalloc x");
    check(cudaMalloc(&device_y, sizeof(host_y)), "cudaMalloc y");
    check(cudaMalloc(&device_sp, sizeof(host_sp)), "cudaMalloc sp");
    check(cudaMalloc(&device_p, sizeof(host_p)), "cudaMalloc p");
    check(cudaMalloc(&device_cycles, sizeof(host_cycles)), "cudaMalloc cycles");
    check(cudaMalloc(&device_controller_shift, sizeof(host_controller_shift)),
          "cudaMalloc controller_shift");
    check(cudaMalloc(&device_controller_shift_count, sizeof(host_controller_shift_count)),
          "cudaMalloc controller_shift_count");
    check(cudaMalloc(&device_controller_strobe, sizeof(host_controller_strobe)),
          "cudaMalloc controller_strobe");
    check(cudaMalloc(&device_cpu_passed, sizeof(host_cpu_passed)), "cudaMalloc cpu_passed");

    check(cudaMemcpy(device_prg_rom,
                     host_prg_rom.data(),
                     host_prg_rom.size(),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy cpu prg_rom");
    check(cudaMemcpy(device_cpu_ram,
                     host_cpu_ram.data(),
                     host_cpu_ram.size(),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy cpu ram");
    check(cudaMemcpy(device_prg_ram,
                     host_prg_ram.data(),
                     host_prg_ram.size(),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy cpu prg_ram");
    check(cudaMemcpy(device_action, &host_action, sizeof(host_action), cudaMemcpyHostToDevice),
          "cudaMemcpy action");
    check(cudaMemcpy(device_pc, &host_pc, sizeof(host_pc), cudaMemcpyHostToDevice),
          "cudaMemcpy pc");
    check(cudaMemcpy(device_a, &host_a, sizeof(host_a), cudaMemcpyHostToDevice),
          "cudaMemcpy a");
    check(cudaMemcpy(device_x, &host_x, sizeof(host_x), cudaMemcpyHostToDevice),
          "cudaMemcpy x");
    check(cudaMemcpy(device_y, &host_y, sizeof(host_y), cudaMemcpyHostToDevice),
          "cudaMemcpy y");
    check(cudaMemcpy(device_sp, &host_sp, sizeof(host_sp), cudaMemcpyHostToDevice),
          "cudaMemcpy sp");
    check(cudaMemcpy(device_p, &host_p, sizeof(host_p), cudaMemcpyHostToDevice),
          "cudaMemcpy p");
    check(cudaMemcpy(device_cycles, &host_cycles, sizeof(host_cycles), cudaMemcpyHostToDevice),
          "cudaMemcpy cycles");
    check(cudaMemcpy(device_controller_shift,
                     &host_controller_shift,
                     sizeof(host_controller_shift),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy controller_shift");
    check(cudaMemcpy(device_controller_shift_count,
                     &host_controller_shift_count,
                     sizeof(host_controller_shift_count),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy controller_shift_count");
    check(cudaMemcpy(device_controller_strobe,
                     &host_controller_strobe,
                     sizeof(host_controller_strobe),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy controller_strobe");
    check(cudaMemcpy(device_cpu_passed,
                     &host_cpu_passed,
                     sizeof(host_cpu_passed),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy cpu_passed");

    nesle::cuda::BatchBuffers cpu_buffers{};
    cpu_buffers.cpu.pc = device_pc;
    cpu_buffers.cpu.a = device_a;
    cpu_buffers.cpu.x = device_x;
    cpu_buffers.cpu.y = device_y;
    cpu_buffers.cpu.sp = device_sp;
    cpu_buffers.cpu.p = device_p;
    cpu_buffers.cpu.cycles = device_cycles;
    cpu_buffers.cpu.ram = device_cpu_ram;
    cpu_buffers.cpu.prg_ram = device_prg_ram;
    cpu_buffers.cpu.controller1_shift = device_controller_shift;
    cpu_buffers.cpu.controller1_shift_count = device_controller_shift_count;
    cpu_buffers.cpu.controller1_strobe = device_controller_strobe;
    cpu_buffers.cart.prg_rom = device_prg_rom;
    cpu_buffers.cart.prg_rom_size = static_cast<std::uint32_t>(host_prg_rom.size());
    cpu_buffers.action_masks = device_action;

    cpu_trace_kernel<<<1, 1>>>(cpu_buffers, device_cpu_passed);
    check(cudaGetLastError(), "cpu_trace_kernel");
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize cpu_trace");
    check(cudaMemcpy(&host_cpu_passed,
                     device_cpu_passed,
                     sizeof(host_cpu_passed),
                     cudaMemcpyDeviceToHost),
          "cudaMemcpy cpu_passed back");

    check(cudaFree(device_prg_rom), "cudaFree cpu prg_rom");
    check(cudaFree(device_cpu_ram), "cudaFree cpu ram");
    check(cudaFree(device_prg_ram), "cudaFree cpu prg_ram");
    check(cudaFree(device_action), "cudaFree action");
    check(cudaFree(device_pc), "cudaFree pc");
    check(cudaFree(device_a), "cudaFree a");
    check(cudaFree(device_x), "cudaFree x");
    check(cudaFree(device_y), "cudaFree y");
    check(cudaFree(device_sp), "cudaFree sp");
    check(cudaFree(device_p), "cudaFree p");
    check(cudaFree(device_cycles), "cudaFree cycles");
    check(cudaFree(device_controller_shift), "cudaFree controller_shift");
    check(cudaFree(device_controller_shift_count), "cudaFree controller_shift_count");
    check(cudaFree(device_controller_strobe), "cudaFree controller_strobe");
    check(cudaFree(device_cpu_passed), "cudaFree cpu_passed");

    if (host_cpu_passed != 1) {
        std::cerr << "CUDA CPU trace smoke failed\n";
        return 1;
    }
    std::cout << "cuda_cpu_trace instructions=11 pc=0x8017 ram_0002=0x11 prg_ram_6000=0x01\n";
    return 0;
}

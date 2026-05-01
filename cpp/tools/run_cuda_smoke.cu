#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include "nesle/cuda/batch_console.hpp"
#include "nesle/cuda/batch_cpu.hpp"
#include "nesle/cuda/batch_step.cuh"
#include "nesle/cuda/device_reset.cuh"
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

struct ConsoleTraceResult {
    std::uint8_t passed = 0;
    std::uint16_t pc = 0;
    std::uint32_t first_cpu_cycles = 0;
    std::uint32_t second_cpu_cycles = 0;
    std::uint32_t first_ppu_cycles = 0;
    std::uint32_t second_ppu_cycles = 0;
    std::uint32_t pending_dma_cycles = 0;
    std::uint64_t total_cpu_cycles = 0;
    std::int16_t scanline = 0;
    std::uint16_t dot = 0;
    std::uint8_t oam0 = 0;
    std::uint8_t oam1 = 0;
    std::uint8_t oam255 = 0;
};

__global__ void console_dma_trace_kernel(nesle::cuda::BatchBuffers buffers,
                                         ConsoleTraceResult* result) {
    nesle::cuda::reset_batch_cpu_env(buffers, 0);
    const auto first = nesle::cuda::step_batch_console_instruction(buffers, 0);
    const auto second = nesle::cuda::step_batch_console_instruction(buffers, 0);
    const auto* oam = nesle::cuda::env_oam(buffers, 0);

    result->pc = buffers.cpu.pc[0];
    result->first_cpu_cycles = first.cpu_cycles;
    result->second_cpu_cycles = second.cpu_cycles;
    result->first_ppu_cycles = first.ppu_cycles;
    result->second_ppu_cycles = second.ppu_cycles;
    result->pending_dma_cycles = buffers.cpu.pending_dma_cycles[0];
    result->total_cpu_cycles = buffers.cpu.cycles[0];
    result->scanline = buffers.ppu.scanline[0];
    result->dot = buffers.ppu.dot[0];
    result->oam0 = oam[0];
    result->oam1 = oam[1];
    result->oam255 = oam[255];
    result->passed = result->pc == 0x8005 &&
                             result->first_cpu_cycles == 2 &&
                             result->second_cpu_cycles == 517 &&
                             result->first_ppu_cycles == 6 &&
                             result->second_ppu_cycles == 1551 &&
                             result->pending_dma_cycles == 0 &&
                             result->total_cpu_cycles == 526 &&
                             result->scanline == 4 &&
                             result->dot == 193 &&
                             result->oam0 == 0xFF &&
                             result->oam1 == 0xFE &&
                             result->oam255 == 0x00
                         ? 1
                         : 0;
}

__global__ void device_reset_trace_kernel(nesle::cuda::BatchBuffers buffers,
                                          nesle::cuda::DeviceResetSnapshots snapshots,
                                          ConsoleTraceResult* result) {
    buffers.cpu.pending_dma_cycles[0] = 0;
    buffers.ppu.ctrl[0] = 0;
    buffers.ppu.mask[0] = 0;
    buffers.ppu.status[0] = 0;
    buffers.ppu.oam_addr[0] = 0;
    buffers.ppu.nmi_pending[0] = 0;
    buffers.ppu.scanline[0] = 0;
    buffers.ppu.dot[0] = 0;
    buffers.ppu.frame[0] = 0;

    nesle::cuda::reset_batch_cpu_env(buffers, 0);
    (void)nesle::cuda::step_batch_console_instruction(buffers, 0);
    (void)nesle::cuda::step_batch_console_instruction(buffers, 0);
    nesle::cuda::capture_device_reset_snapshot(buffers, snapshots, 0, 0);

    auto* ram = nesle::cuda::env_cpu_ram(buffers, 0);
    auto* prg_ram = nesle::cuda::env_prg_ram(buffers, 0);
    auto* oam = nesle::cuda::env_oam(buffers, 0);
    buffers.cpu.pc[0] = 0x1234;
    buffers.cpu.a[0] = 0x56;
    buffers.cpu.cycles[0] = 99;
    buffers.cpu.pending_dma_cycles[0] = 99;
    buffers.ppu.scanline[0] = 99;
    buffers.ppu.dot[0] = 99;
    ram[0x0200] = 0;
    prg_ram[0] = 0x5A;
    oam[0] = 0;
    oam[1] = 0;

    nesle::cuda::restore_device_reset_snapshot(buffers, snapshots, 0, 0);

    result->pc = buffers.cpu.pc[0];
    result->pending_dma_cycles = buffers.cpu.pending_dma_cycles[0];
    result->total_cpu_cycles = buffers.cpu.cycles[0];
    result->scanline = buffers.ppu.scanline[0];
    result->dot = buffers.ppu.dot[0];
    result->oam0 = oam[0];
    result->oam1 = oam[1];
    result->oam255 = oam[255];
    result->passed = result->pc == 0x8005 &&
                             buffers.cpu.a[0] == 0x02 &&
                             result->pending_dma_cycles == 0 &&
                             result->total_cpu_cycles == 526 &&
                             result->scanline == 4 &&
                             result->dot == 193 &&
                             ram[0x0200] == 0xFF &&
                             prg_ram[0] == 0 &&
                             result->oam0 == 0xFF &&
                             result->oam1 == 0xFE &&
                             result->oam255 == 0
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

    std::vector<std::uint8_t> host_console_prg_rom(kPrgRomSize, 0xEA);
    host_console_prg_rom[0] = 0xA9;  // LDA #$02
    host_console_prg_rom[1] = 0x02;
    host_console_prg_rom[2] = 0x8D;  // STA $4014
    host_console_prg_rom[3] = 0x14;
    host_console_prg_rom[4] = 0x40;
    host_console_prg_rom[5] = 0x4C;  // JMP $8005
    host_console_prg_rom[6] = 0x05;
    host_console_prg_rom[7] = 0x80;
    host_console_prg_rom[0x7FFC] = 0x00;
    host_console_prg_rom[0x7FFD] = 0x80;

    std::vector<std::uint8_t> host_console_cpu_ram(nesle::cuda::kCpuRamBytes, 0);
    for (std::size_t i = 0; i < nesle::cuda::kOamBytes; ++i) {
        host_console_cpu_ram[0x0200 + i] = static_cast<std::uint8_t>(0xFFu - i);
    }
    std::vector<std::uint8_t> host_console_prg_ram(nesle::cuda::kPrgRamBytes, 0);
    std::vector<std::uint8_t> host_console_oam(nesle::cuda::kOamBytes, 0);
    std::uint16_t host_console_pc = 0;
    std::uint8_t host_console_a = 0;
    std::uint8_t host_console_x = 0;
    std::uint8_t host_console_y = 0;
    std::uint8_t host_console_sp = 0;
    std::uint8_t host_console_p = 0;
    std::uint64_t host_console_cycles = 0;
    std::uint32_t host_console_dma_cycles = 0;
    std::uint8_t host_console_ppu_ctrl = 0;
    std::uint8_t host_console_ppu_mask = 0;
    std::uint8_t host_console_ppu_status = 0;
    std::uint8_t host_console_oam_addr = 0;
    std::uint8_t host_console_ppu_nmi_pending = 0;
    std::int16_t host_console_scanline = 0;
    std::uint16_t host_console_dot = 0;
    std::uint64_t host_console_frame = 0;
    ConsoleTraceResult host_console_result{};

    std::uint8_t* device_console_prg_rom = nullptr;
    std::uint8_t* device_console_cpu_ram = nullptr;
    std::uint8_t* device_console_prg_ram = nullptr;
    std::uint8_t* device_console_oam = nullptr;
    std::uint16_t* device_console_pc = nullptr;
    std::uint8_t* device_console_a = nullptr;
    std::uint8_t* device_console_x = nullptr;
    std::uint8_t* device_console_y = nullptr;
    std::uint8_t* device_console_sp = nullptr;
    std::uint8_t* device_console_p = nullptr;
    std::uint64_t* device_console_cycles = nullptr;
    std::uint32_t* device_console_dma_cycles = nullptr;
    std::uint8_t* device_console_ppu_ctrl = nullptr;
    std::uint8_t* device_console_ppu_mask = nullptr;
    std::uint8_t* device_console_ppu_status = nullptr;
    std::uint8_t* device_console_oam_addr = nullptr;
    std::uint8_t* device_console_ppu_nmi_pending = nullptr;
    std::int16_t* device_console_scanline = nullptr;
    std::uint16_t* device_console_dot = nullptr;
    std::uint64_t* device_console_frame = nullptr;
    ConsoleTraceResult* device_console_result = nullptr;

    check(cudaMalloc(&device_console_prg_rom, host_console_prg_rom.size()),
          "cudaMalloc console prg_rom");
    check(cudaMalloc(&device_console_cpu_ram, host_console_cpu_ram.size()),
          "cudaMalloc console cpu_ram");
    check(cudaMalloc(&device_console_prg_ram, host_console_prg_ram.size()),
          "cudaMalloc console prg_ram");
    check(cudaMalloc(&device_console_oam, host_console_oam.size()), "cudaMalloc console oam");
    check(cudaMalloc(&device_console_pc, sizeof(host_console_pc)), "cudaMalloc console pc");
    check(cudaMalloc(&device_console_a, sizeof(host_console_a)), "cudaMalloc console a");
    check(cudaMalloc(&device_console_x, sizeof(host_console_x)), "cudaMalloc console x");
    check(cudaMalloc(&device_console_y, sizeof(host_console_y)), "cudaMalloc console y");
    check(cudaMalloc(&device_console_sp, sizeof(host_console_sp)), "cudaMalloc console sp");
    check(cudaMalloc(&device_console_p, sizeof(host_console_p)), "cudaMalloc console p");
    check(cudaMalloc(&device_console_cycles, sizeof(host_console_cycles)),
          "cudaMalloc console cycles");
    check(cudaMalloc(&device_console_dma_cycles, sizeof(host_console_dma_cycles)),
          "cudaMalloc console dma_cycles");
    check(cudaMalloc(&device_console_ppu_ctrl, sizeof(host_console_ppu_ctrl)),
          "cudaMalloc console ppu_ctrl");
    check(cudaMalloc(&device_console_ppu_mask, sizeof(host_console_ppu_mask)),
          "cudaMalloc console ppu_mask");
    check(cudaMalloc(&device_console_ppu_status, sizeof(host_console_ppu_status)),
          "cudaMalloc console ppu_status");
    check(cudaMalloc(&device_console_oam_addr, sizeof(host_console_oam_addr)),
          "cudaMalloc console oam_addr");
    check(cudaMalloc(&device_console_ppu_nmi_pending,
                     sizeof(host_console_ppu_nmi_pending)),
          "cudaMalloc console ppu_nmi_pending");
    check(cudaMalloc(&device_console_scanline, sizeof(host_console_scanline)),
          "cudaMalloc console scanline");
    check(cudaMalloc(&device_console_dot, sizeof(host_console_dot)), "cudaMalloc console dot");
    check(cudaMalloc(&device_console_frame, sizeof(host_console_frame)),
          "cudaMalloc console frame");
    check(cudaMalloc(&device_console_result, sizeof(host_console_result)),
          "cudaMalloc console result");

    check(cudaMemcpy(device_console_prg_rom,
                     host_console_prg_rom.data(),
                     host_console_prg_rom.size(),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy console prg_rom");
    check(cudaMemcpy(device_console_cpu_ram,
                     host_console_cpu_ram.data(),
                     host_console_cpu_ram.size(),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy console cpu_ram");
    check(cudaMemcpy(device_console_prg_ram,
                     host_console_prg_ram.data(),
                     host_console_prg_ram.size(),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy console prg_ram");
    check(cudaMemcpy(device_console_oam,
                     host_console_oam.data(),
                     host_console_oam.size(),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy console oam");
    check(cudaMemcpy(device_console_pc,
                     &host_console_pc,
                     sizeof(host_console_pc),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy console pc");
    check(cudaMemcpy(device_console_a,
                     &host_console_a,
                     sizeof(host_console_a),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy console a");
    check(cudaMemcpy(device_console_x,
                     &host_console_x,
                     sizeof(host_console_x),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy console x");
    check(cudaMemcpy(device_console_y,
                     &host_console_y,
                     sizeof(host_console_y),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy console y");
    check(cudaMemcpy(device_console_sp,
                     &host_console_sp,
                     sizeof(host_console_sp),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy console sp");
    check(cudaMemcpy(device_console_p,
                     &host_console_p,
                     sizeof(host_console_p),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy console p");
    check(cudaMemcpy(device_console_cycles,
                     &host_console_cycles,
                     sizeof(host_console_cycles),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy console cycles");
    check(cudaMemcpy(device_console_dma_cycles,
                     &host_console_dma_cycles,
                     sizeof(host_console_dma_cycles),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy console dma_cycles");
    check(cudaMemcpy(device_console_ppu_ctrl,
                     &host_console_ppu_ctrl,
                     sizeof(host_console_ppu_ctrl),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy console ppu_ctrl");
    check(cudaMemcpy(device_console_ppu_mask,
                     &host_console_ppu_mask,
                     sizeof(host_console_ppu_mask),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy console ppu_mask");
    check(cudaMemcpy(device_console_ppu_status,
                     &host_console_ppu_status,
                     sizeof(host_console_ppu_status),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy console ppu_status");
    check(cudaMemcpy(device_console_oam_addr,
                     &host_console_oam_addr,
                     sizeof(host_console_oam_addr),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy console oam_addr");
    check(cudaMemcpy(device_console_ppu_nmi_pending,
                     &host_console_ppu_nmi_pending,
                     sizeof(host_console_ppu_nmi_pending),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy console ppu_nmi_pending");
    check(cudaMemcpy(device_console_scanline,
                     &host_console_scanline,
                     sizeof(host_console_scanline),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy console scanline");
    check(cudaMemcpy(device_console_dot,
                     &host_console_dot,
                     sizeof(host_console_dot),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy console dot");
    check(cudaMemcpy(device_console_frame,
                     &host_console_frame,
                     sizeof(host_console_frame),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy console frame");
    check(cudaMemcpy(device_console_result,
                     &host_console_result,
                     sizeof(host_console_result),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy console result");

    nesle::cuda::BatchBuffers console_buffers{};
    console_buffers.cpu.pc = device_console_pc;
    console_buffers.cpu.a = device_console_a;
    console_buffers.cpu.x = device_console_x;
    console_buffers.cpu.y = device_console_y;
    console_buffers.cpu.sp = device_console_sp;
    console_buffers.cpu.p = device_console_p;
    console_buffers.cpu.cycles = device_console_cycles;
    console_buffers.cpu.ram = device_console_cpu_ram;
    console_buffers.cpu.prg_ram = device_console_prg_ram;
    console_buffers.cpu.pending_dma_cycles = device_console_dma_cycles;
    console_buffers.ppu.ctrl = device_console_ppu_ctrl;
    console_buffers.ppu.mask = device_console_ppu_mask;
    console_buffers.ppu.status = device_console_ppu_status;
    console_buffers.ppu.oam_addr = device_console_oam_addr;
    console_buffers.ppu.nmi_pending = device_console_ppu_nmi_pending;
    console_buffers.ppu.scanline = device_console_scanline;
    console_buffers.ppu.dot = device_console_dot;
    console_buffers.ppu.frame = device_console_frame;
    console_buffers.ppu.oam = device_console_oam;
    console_buffers.cart.prg_rom = device_console_prg_rom;
    console_buffers.cart.prg_rom_size = static_cast<std::uint32_t>(host_console_prg_rom.size());

    console_dma_trace_kernel<<<1, 1>>>(console_buffers, device_console_result);
    check(cudaGetLastError(), "console_dma_trace_kernel");
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize console_dma_trace");
    check(cudaMemcpy(&host_console_result,
                     device_console_result,
                     sizeof(host_console_result),
                     cudaMemcpyDeviceToHost),
          "cudaMemcpy console result back");

    if (host_console_result.passed != 1) {
        std::cerr << "CUDA console DMA smoke failed"
                  << " pc=0x" << std::hex << host_console_result.pc << std::dec
                  << " cpu_cycles=" << host_console_result.first_cpu_cycles << ","
                  << host_console_result.second_cpu_cycles
                  << " ppu_cycles=" << host_console_result.first_ppu_cycles << ","
                  << host_console_result.second_ppu_cycles
                  << " scanline=" << host_console_result.scanline
                  << " dot=" << host_console_result.dot
                  << " oam=" << static_cast<int>(host_console_result.oam0) << ","
                  << static_cast<int>(host_console_result.oam1) << ","
                  << static_cast<int>(host_console_result.oam255) << '\n';
        return 1;
    }
    std::cout << "cuda_console_dma cpu_cycles="
              << host_console_result.first_cpu_cycles << ","
              << host_console_result.second_cpu_cycles
              << " ppu_cycles=" << host_console_result.first_ppu_cycles << ","
              << host_console_result.second_ppu_cycles
              << " oam=" << static_cast<int>(host_console_result.oam0) << ","
              << static_cast<int>(host_console_result.oam1) << ","
              << static_cast<int>(host_console_result.oam255)
              << " ppu=" << host_console_result.scanline << ":"
              << host_console_result.dot << '\n';

    nesle::cuda::DeviceResetSnapshots device_snapshots{};
    std::uint16_t* device_snapshot_pc = nullptr;
    std::uint8_t* device_snapshot_a = nullptr;
    std::uint8_t* device_snapshot_x = nullptr;
    std::uint8_t* device_snapshot_y = nullptr;
    std::uint8_t* device_snapshot_sp = nullptr;
    std::uint8_t* device_snapshot_p = nullptr;
    std::uint64_t* device_snapshot_cycles = nullptr;
    std::uint32_t* device_snapshot_dma_cycles = nullptr;
    std::uint8_t* device_snapshot_ppu_ctrl = nullptr;
    std::uint8_t* device_snapshot_ppu_mask = nullptr;
    std::uint8_t* device_snapshot_ppu_status = nullptr;
    std::uint8_t* device_snapshot_oam_addr = nullptr;
    std::uint8_t* device_snapshot_ppu_nmi_pending = nullptr;
    std::int16_t* device_snapshot_scanline = nullptr;
    std::uint16_t* device_snapshot_dot = nullptr;
    std::uint64_t* device_snapshot_frame = nullptr;
    std::uint8_t* device_snapshot_ram = nullptr;
    std::uint8_t* device_snapshot_prg_ram = nullptr;
    std::uint8_t* device_snapshot_oam = nullptr;
    ConsoleTraceResult host_reset_result{};

    check(cudaMalloc(&device_snapshot_pc, sizeof(std::uint16_t)), "cudaMalloc snapshot pc");
    check(cudaMalloc(&device_snapshot_a, sizeof(std::uint8_t)), "cudaMalloc snapshot a");
    check(cudaMalloc(&device_snapshot_x, sizeof(std::uint8_t)), "cudaMalloc snapshot x");
    check(cudaMalloc(&device_snapshot_y, sizeof(std::uint8_t)), "cudaMalloc snapshot y");
    check(cudaMalloc(&device_snapshot_sp, sizeof(std::uint8_t)), "cudaMalloc snapshot sp");
    check(cudaMalloc(&device_snapshot_p, sizeof(std::uint8_t)), "cudaMalloc snapshot p");
    check(cudaMalloc(&device_snapshot_cycles, sizeof(std::uint64_t)),
          "cudaMalloc snapshot cycles");
    check(cudaMalloc(&device_snapshot_dma_cycles, sizeof(std::uint32_t)),
          "cudaMalloc snapshot dma_cycles");
    check(cudaMalloc(&device_snapshot_ppu_ctrl, sizeof(std::uint8_t)),
          "cudaMalloc snapshot ppu_ctrl");
    check(cudaMalloc(&device_snapshot_ppu_mask, sizeof(std::uint8_t)),
          "cudaMalloc snapshot ppu_mask");
    check(cudaMalloc(&device_snapshot_ppu_status, sizeof(std::uint8_t)),
          "cudaMalloc snapshot ppu_status");
    check(cudaMalloc(&device_snapshot_oam_addr, sizeof(std::uint8_t)),
          "cudaMalloc snapshot oam_addr");
    check(cudaMalloc(&device_snapshot_ppu_nmi_pending, sizeof(std::uint8_t)),
          "cudaMalloc snapshot ppu_nmi_pending");
    check(cudaMalloc(&device_snapshot_scanline, sizeof(std::int16_t)),
          "cudaMalloc snapshot scanline");
    check(cudaMalloc(&device_snapshot_dot, sizeof(std::uint16_t)), "cudaMalloc snapshot dot");
    check(cudaMalloc(&device_snapshot_frame, sizeof(std::uint64_t)),
          "cudaMalloc snapshot frame");
    check(cudaMalloc(&device_snapshot_ram, nesle::cuda::kCpuRamBytes),
          "cudaMalloc snapshot ram");
    check(cudaMalloc(&device_snapshot_prg_ram, nesle::cuda::kPrgRamBytes),
          "cudaMalloc snapshot prg_ram");
    check(cudaMalloc(&device_snapshot_oam, nesle::cuda::kOamBytes),
          "cudaMalloc snapshot oam");

    device_snapshots.pc = device_snapshot_pc;
    device_snapshots.a = device_snapshot_a;
    device_snapshots.x = device_snapshot_x;
    device_snapshots.y = device_snapshot_y;
    device_snapshots.sp = device_snapshot_sp;
    device_snapshots.p = device_snapshot_p;
    device_snapshots.cycles = device_snapshot_cycles;
    device_snapshots.pending_dma_cycles = device_snapshot_dma_cycles;
    device_snapshots.ppu_ctrl = device_snapshot_ppu_ctrl;
    device_snapshots.ppu_mask = device_snapshot_ppu_mask;
    device_snapshots.ppu_status = device_snapshot_ppu_status;
    device_snapshots.ppu_oam_addr = device_snapshot_oam_addr;
    device_snapshots.ppu_nmi_pending = device_snapshot_ppu_nmi_pending;
    device_snapshots.ppu_scanline = device_snapshot_scanline;
    device_snapshots.ppu_dot = device_snapshot_dot;
    device_snapshots.ppu_frame = device_snapshot_frame;
    device_snapshots.ram = device_snapshot_ram;
    device_snapshots.prg_ram = device_snapshot_prg_ram;
    device_snapshots.oam = device_snapshot_oam;

    check(cudaMemcpy(device_console_result,
                     &host_reset_result,
                     sizeof(host_reset_result),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy reset result");
    device_reset_trace_kernel<<<1, 1>>>(console_buffers, device_snapshots, device_console_result);
    check(cudaGetLastError(), "device_reset_trace_kernel");
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize device_reset_trace");
    check(cudaMemcpy(&host_reset_result,
                     device_console_result,
                     sizeof(host_reset_result),
                     cudaMemcpyDeviceToHost),
          "cudaMemcpy reset result back");

    check(cudaFree(device_snapshot_pc), "cudaFree snapshot pc");
    check(cudaFree(device_snapshot_a), "cudaFree snapshot a");
    check(cudaFree(device_snapshot_x), "cudaFree snapshot x");
    check(cudaFree(device_snapshot_y), "cudaFree snapshot y");
    check(cudaFree(device_snapshot_sp), "cudaFree snapshot sp");
    check(cudaFree(device_snapshot_p), "cudaFree snapshot p");
    check(cudaFree(device_snapshot_cycles), "cudaFree snapshot cycles");
    check(cudaFree(device_snapshot_dma_cycles), "cudaFree snapshot dma_cycles");
    check(cudaFree(device_snapshot_ppu_ctrl), "cudaFree snapshot ppu_ctrl");
    check(cudaFree(device_snapshot_ppu_mask), "cudaFree snapshot ppu_mask");
    check(cudaFree(device_snapshot_ppu_status), "cudaFree snapshot ppu_status");
    check(cudaFree(device_snapshot_oam_addr), "cudaFree snapshot oam_addr");
    check(cudaFree(device_snapshot_ppu_nmi_pending), "cudaFree snapshot ppu_nmi_pending");
    check(cudaFree(device_snapshot_scanline), "cudaFree snapshot scanline");
    check(cudaFree(device_snapshot_dot), "cudaFree snapshot dot");
    check(cudaFree(device_snapshot_frame), "cudaFree snapshot frame");
    check(cudaFree(device_snapshot_ram), "cudaFree snapshot ram");
    check(cudaFree(device_snapshot_prg_ram), "cudaFree snapshot prg_ram");
    check(cudaFree(device_snapshot_oam), "cudaFree snapshot oam");

    if (host_reset_result.passed != 1) {
        std::cerr << "CUDA device reset smoke failed"
                  << " pc=0x" << std::hex << host_reset_result.pc << std::dec
                  << " cycles=" << host_reset_result.total_cpu_cycles
                  << " dma=" << host_reset_result.pending_dma_cycles
                  << " ppu=" << host_reset_result.scanline << ":" << host_reset_result.dot
                  << " oam=" << static_cast<int>(host_reset_result.oam0) << ","
                  << static_cast<int>(host_reset_result.oam1) << ","
                  << static_cast<int>(host_reset_result.oam255) << '\n';
        return 1;
    }
    std::cout << "cuda_device_reset pc=0x8005 cycles="
              << host_reset_result.total_cpu_cycles
              << " ppu=" << host_reset_result.scanline << ":"
              << host_reset_result.dot
              << " oam=" << static_cast<int>(host_reset_result.oam0) << ","
              << static_cast<int>(host_reset_result.oam1) << ","
              << static_cast<int>(host_reset_result.oam255) << '\n';

    check(cudaFree(device_console_prg_rom), "cudaFree console prg_rom");
    check(cudaFree(device_console_cpu_ram), "cudaFree console cpu_ram");
    check(cudaFree(device_console_prg_ram), "cudaFree console prg_ram");
    check(cudaFree(device_console_oam), "cudaFree console oam");
    check(cudaFree(device_console_pc), "cudaFree console pc");
    check(cudaFree(device_console_a), "cudaFree console a");
    check(cudaFree(device_console_x), "cudaFree console x");
    check(cudaFree(device_console_y), "cudaFree console y");
    check(cudaFree(device_console_sp), "cudaFree console sp");
    check(cudaFree(device_console_p), "cudaFree console p");
    check(cudaFree(device_console_cycles), "cudaFree console cycles");
    check(cudaFree(device_console_dma_cycles), "cudaFree console dma_cycles");
    check(cudaFree(device_console_ppu_ctrl), "cudaFree console ppu_ctrl");
    check(cudaFree(device_console_ppu_mask), "cudaFree console ppu_mask");
    check(cudaFree(device_console_ppu_status), "cudaFree console ppu_status");
    check(cudaFree(device_console_oam_addr), "cudaFree console oam_addr");
    check(cudaFree(device_console_ppu_nmi_pending), "cudaFree console ppu_nmi_pending");
    check(cudaFree(device_console_scanline), "cudaFree console scanline");
    check(cudaFree(device_console_dot), "cudaFree console dot");
    check(cudaFree(device_console_frame), "cudaFree console frame");
    check(cudaFree(device_console_result), "cudaFree console result");

    return 0;
}

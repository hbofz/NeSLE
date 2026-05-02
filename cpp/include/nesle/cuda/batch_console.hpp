#pragma once

#include <cstdint>

#include "nesle/cpu.hpp"
#include "nesle/cuda/batch_cpu.hpp"
#include "nesle/cuda/batch_ppu.cuh"

#ifdef __CUDACC__
#define NESLE_CUDA_BATCH_CONSOLE_HD __host__ __device__
#else
#define NESLE_CUDA_BATCH_CONSOLE_HD
#endif

namespace nesle::cuda {

constexpr std::uint32_t kPpuCyclesPerCpuCycle = 3;

struct BatchConsoleStepResult {
    cpu::StepResult cpu;
    std::uint32_t cpu_cycles = 0;
    std::uint32_t ppu_cycles = 0;
    std::uint32_t frames_completed = 0;
    bool nmi_serviced = false;
    bool nmi_started = false;
};

NESLE_CUDA_BATCH_CONSOLE_HD inline void clear_batch_ppu_nmi_pending(BatchBuffers& buffers,
                                                                    std::uint32_t env) noexcept {
    buffers.ppu.nmi_pending[env] = 0;
}

[[nodiscard]] NESLE_CUDA_BATCH_CONSOLE_HD inline BatchConsoleStepResult
step_batch_console_instruction(BatchBuffers& buffers,
                               std::uint32_t env,
                               cpu::CpuState& state) {
    const auto cycles_before = state.cycles;
    bool nmi_serviced = false;
    BatchCpuBus bus(buffers, env);

    if (buffers.ppu.nmi_pending[env] != 0) {
        clear_batch_ppu_nmi_pending(buffers, env);
        cpu::nmi(state, bus);
        nmi_serviced = true;
    }

    const auto cpu_step = cpu::step(state, bus);
    if (buffers.cpu.pending_dma_cycles != nullptr && buffers.cpu.pending_dma_cycles[env] != 0) {
        state.cycles += buffers.cpu.pending_dma_cycles[env];
        buffers.cpu.pending_dma_cycles[env] = 0;
    }

    const auto cpu_cycles = static_cast<std::uint32_t>(state.cycles - cycles_before);
    const auto ppu_cycles = cpu_cycles * kPpuCyclesPerCpuCycle;
    const auto ppu_step = batch_ppu_step_env(buffers, env, ppu_cycles);

    return BatchConsoleStepResult{
        cpu_step,
        cpu_cycles,
        ppu_cycles,
        ppu_step.frames_completed,
        nmi_serviced,
        ppu_step.nmi_started,
    };
}

[[nodiscard]] NESLE_CUDA_BATCH_CONSOLE_HD inline BatchConsoleStepResult
step_batch_console_instruction(BatchBuffers& buffers, std::uint32_t env) {
    auto state = load_cpu_state(buffers, env);
    const auto result = step_batch_console_instruction(buffers, env, state);
    store_cpu_state(buffers, env, state);
    return result;
}

}  // namespace nesle::cuda

#undef NESLE_CUDA_BATCH_CONSOLE_HD

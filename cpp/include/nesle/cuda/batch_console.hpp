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

// Hot variant: PPU scalars live in the caller's register-resident PpuHotState
// for the whole kernel; the bus and PPU catch-up read/write it instead of
// global memory.
[[nodiscard]] NESLE_CUDA_BATCH_CONSOLE_HD inline BatchConsoleStepResult
step_batch_console_instruction_hot(BatchBuffers& buffers,
                                   std::uint32_t env,
                                   cpu::CpuState& state,
                                   PpuHotState& hot) {
    const auto cycles_before = state.cycles;
    bool nmi_serviced = false;
    BatchCpuBus bus(buffers, env, hot);

    if (hot.nmi_pending != 0) {
        hot.nmi_pending = 0;
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
    const auto ppu_step = batch_ppu_step_env_hot(buffers, env, ppu_cycles, hot);

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
step_batch_console_instruction(BatchBuffers& buffers,
                               std::uint32_t env,
                               cpu::CpuState& state) {
    auto hot = load_ppu_hot_state(buffers, env);
    const auto result = step_batch_console_instruction_hot(buffers, env, state, hot);
    store_ppu_hot_state(buffers, env, hot);
    return result;
}

[[nodiscard]] NESLE_CUDA_BATCH_CONSOLE_HD inline BatchConsoleStepResult
step_batch_console_instruction(BatchBuffers& buffers, std::uint32_t env) {
    auto state = load_cpu_state(buffers, env);
    const auto result = step_batch_console_instruction(buffers, env, state);
    store_cpu_state(buffers, env, state);
    return result;
}

[[nodiscard]] NESLE_CUDA_BATCH_CONSOLE_HD inline BatchConsoleStepResult
fast_forward_batch_console_idle_loop_hot(BatchBuffers& buffers,
                                         std::uint32_t env,
                                         cpu::CpuState& state,
                                         PpuHotState& hot) {
    const auto ppu_cycles_to_event = batch_ppu_cycles_until_next_timing_event_hot(hot);
    const auto skipped_jumps = (ppu_cycles_to_event + 8u) / 9u;
    if (skipped_jumps == 0) {
        return {};
    }

    const auto cpu_cycles = skipped_jumps * 3u;
    const auto ppu_cycles = cpu_cycles * kPpuCyclesPerCpuCycle;
    state.cycles += cpu_cycles;
    const auto ppu_step = batch_ppu_step_env_hot(buffers, env, ppu_cycles, hot);
    return BatchConsoleStepResult{
        {},
        cpu_cycles,
        ppu_cycles,
        ppu_step.frames_completed,
        false,
        ppu_step.nmi_started,
    };
}

[[nodiscard]] NESLE_CUDA_BATCH_CONSOLE_HD inline BatchConsoleStepResult
fast_forward_batch_console_idle_loop(BatchBuffers& buffers,
                                     std::uint32_t env,
                                     cpu::CpuState& state) {
    auto hot = load_ppu_hot_state(buffers, env);
    const auto result = fast_forward_batch_console_idle_loop_hot(buffers, env, state, hot);
    store_ppu_hot_state(buffers, env, hot);
    return result;
}

}  // namespace nesle::cuda

#undef NESLE_CUDA_BATCH_CONSOLE_HD

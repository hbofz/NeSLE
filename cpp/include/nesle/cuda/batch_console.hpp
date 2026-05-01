#pragma once

#include <cstdint>

#include "nesle/cpu.hpp"
#include "nesle/cuda/batch_cpu.hpp"
#include "nesle/cuda/batch_ppu.cuh"

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

inline void clear_batch_ppu_nmi_pending(BatchBuffers& buffers, std::uint32_t env) noexcept {
    buffers.ppu.nmi_pending[env] = 0;
}

[[nodiscard]] inline BatchConsoleStepResult step_batch_console_instruction(BatchBuffers& buffers,
                                                                           std::uint32_t env) {
    auto state = load_cpu_state(buffers, env);
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
    store_cpu_state(buffers, env, state);

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

}  // namespace nesle::cuda

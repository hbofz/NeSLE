#pragma once

#include <cstdint>

#include "nesle/cpu.hpp"
#include "nesle/cuda/batch_bus.cuh"

#ifdef __CUDACC__
#define NESLE_CUDA_BATCH_CPU_HD __host__ __device__
#else
#define NESLE_CUDA_BATCH_CPU_HD
#endif

namespace nesle::cuda {

class BatchCpuBus {
public:
    NESLE_CUDA_BATCH_CPU_HD BatchCpuBus(BatchBuffers& buffers, std::uint32_t env) noexcept
        : buffers_(buffers),
          env_(env) {}

    [[nodiscard]] NESLE_CUDA_BATCH_CPU_HD std::uint8_t read(std::uint16_t address) {
        return batch_cpu_read(buffers_, env_, address);
    }

    NESLE_CUDA_BATCH_CPU_HD void write(std::uint16_t address, std::uint8_t value) {
        batch_cpu_write(buffers_, env_, address, value);
    }

private:
    BatchBuffers& buffers_;
    std::uint32_t env_ = 0;
};

[[nodiscard]] NESLE_CUDA_BATCH_CPU_HD inline cpu::CpuState load_cpu_state(
    const BatchBuffers& buffers,
    std::uint32_t env) noexcept {
    cpu::CpuState state;
    state.pc = buffers.cpu.pc[env];
    state.a = buffers.cpu.a[env];
    state.x = buffers.cpu.x[env];
    state.y = buffers.cpu.y[env];
    state.sp = buffers.cpu.sp[env];
    state.p = buffers.cpu.p[env];
    state.cycles = buffers.cpu.cycles[env];
    state.variant = cpu::CpuVariant::Ricoh2A03;
    return state;
}

NESLE_CUDA_BATCH_CPU_HD inline void store_cpu_state(BatchBuffers& buffers,
                                                    std::uint32_t env,
                                                    const cpu::CpuState& state) noexcept {
    buffers.cpu.pc[env] = state.pc;
    buffers.cpu.a[env] = state.a;
    buffers.cpu.x[env] = state.x;
    buffers.cpu.y[env] = state.y;
    buffers.cpu.sp[env] = state.sp;
    buffers.cpu.p[env] = state.p;
    buffers.cpu.cycles[env] = state.cycles;
}

NESLE_CUDA_BATCH_CPU_HD inline void reset_batch_cpu_env(BatchBuffers& buffers, std::uint32_t env) {
    BatchCpuBus bus(buffers, env);
    cpu::CpuState state;
    state.variant = cpu::CpuVariant::Ricoh2A03;
    cpu::reset(state, bus);
    store_cpu_state(buffers, env, state);
}

[[nodiscard]] NESLE_CUDA_BATCH_CPU_HD inline cpu::StepResult step_batch_cpu_env(
    BatchBuffers& buffers,
    std::uint32_t env) {
    BatchCpuBus bus(buffers, env);
    auto state = load_cpu_state(buffers, env);
    const auto result = cpu::step(state, bus);
    store_cpu_state(buffers, env, state);
    return result;
}

}  // namespace nesle::cuda

#undef NESLE_CUDA_BATCH_CPU_HD

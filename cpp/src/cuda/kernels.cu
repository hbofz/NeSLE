#include "nesle/cuda/kernels.cuh"

#ifdef __CUDACC__

#include "nesle/cuda/batch_console.hpp"
#include "nesle/cuda/batch_render.cuh"
#include "nesle/cuda/batch_step.cuh"

namespace nesle::cuda {
namespace {

__global__ void step_reward_kernel(BatchBuffers buffers, std::uint32_t num_envs) {
    const auto env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < num_envs) {
        apply_batch_reward_env(buffers, env);
    }
}

__global__ void console_step_kernel(BatchBuffers buffers,
                                    std::uint32_t num_envs,
                                    std::uint32_t frameskip,
                                    std::uint64_t max_instructions_per_frame,
                                    ConsoleStepStats stats) {
    const auto env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= num_envs) {
        return;
    }

    auto state = load_cpu_state(buffers, env);
    std::uint64_t total_instructions = 0;
    std::uint32_t total_frames_completed = 0;
    std::uint32_t budget_hits = 0;
    for (std::uint32_t frame = 0; frame < frameskip; ++frame) {
        std::uint64_t instructions = 0;
        std::uint32_t frames_completed = 0;
        while (instructions < max_instructions_per_frame && frames_completed == 0) {
            const auto step = step_batch_console_instruction(buffers, env, state);
            if (stats.opcode_counts != nullptr) {
                atomicAdd(&stats.opcode_counts[step.cpu.opcode], 1ULL);
            }
            if (stats.pc_counts != nullptr) {
                atomicAdd(&stats.pc_counts[step.cpu.pc], 1ULL);
            }
            ++instructions;
            frames_completed += step.frames_completed;
            if (step.cpu.opcode == 0x4C && state.pc == step.cpu.pc && frames_completed == 0) {
                const auto fast_forward = fast_forward_batch_console_idle_loop(buffers, env, state);
                frames_completed += fast_forward.frames_completed;
            }
        }
        total_instructions += instructions;
        total_frames_completed += frames_completed;
        if (frames_completed == 0 && instructions >= max_instructions_per_frame) {
            ++budget_hits;
        }
    }
    store_cpu_state(buffers, env, state);

    apply_batch_reward_env(buffers, env);
    if (stats.instructions != nullptr) {
        stats.instructions[env] = total_instructions;
    }
    if (stats.frames_completed != nullptr) {
        stats.frames_completed[env] = total_frames_completed;
    }
    if (stats.budget_hits != nullptr) {
        stats.budget_hits[env] = budget_hits;
    }
}

__global__ void render_rgb_kernel(BatchBuffers buffers, std::uint32_t num_envs) {
    const auto env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < num_envs) {
        render_batch_rgb_frame_env(buffers, env);
    }
}

}  // namespace

void launch_step_kernel(const BatchBuffers& buffers, StepConfig config, cudaStream_t stream) {
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((config.num_envs + kThreads - 1) / kThreads);
    step_reward_kernel<<<blocks, kThreads, 0, stream>>>(buffers, config.num_envs);
}

void launch_console_step_kernel(const BatchBuffers& buffers,
                                StepConfig config,
                                std::uint64_t max_instructions_per_frame,
                                ConsoleStepStats stats,
                                cudaStream_t stream) {
    constexpr int kThreads = 128;
    const int blocks = static_cast<int>((config.num_envs + kThreads - 1) / kThreads);
    console_step_kernel<<<blocks, kThreads, 0, stream>>>(
        buffers,
        config.num_envs,
        config.frameskip,
        max_instructions_per_frame,
        stats);
}

void launch_render_kernel(const BatchBuffers& buffers, StepConfig config, cudaStream_t stream) {
    constexpr int kThreads = 64;
    const int blocks = static_cast<int>((config.num_envs + kThreads - 1) / kThreads);
    render_rgb_kernel<<<blocks, kThreads, 0, stream>>>(buffers, config.num_envs);
}

namespace {

__global__ void reset_console_envs_kernel(BatchBuffers buffers,
                                          const std::uint8_t* mask,
                                          std::uint32_t num_envs) {
    const auto env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < num_envs && mask[env] != 0) {
        cold_reset_console_env(buffers, env);
    }
}

__global__ void reset_synthetic_envs_kernel(BatchBuffers buffers,
                                            const std::uint8_t* mask,
                                            std::uint32_t num_envs) {
    const auto env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < num_envs && mask[env] != 0) {
        cold_reset_synthetic_env(buffers, env);
    }
}

}  // namespace

void launch_reset_envs_kernel(const BatchBuffers& buffers,
                              const std::uint8_t* device_mask,
                              std::uint32_t num_envs,
                              bool console_mode,
                              cudaStream_t stream) {
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((num_envs + kThreads - 1) / kThreads);
    if (console_mode) {
        reset_console_envs_kernel<<<blocks, kThreads, 0, stream>>>(
            buffers, device_mask, num_envs);
    } else {
        reset_synthetic_envs_kernel<<<blocks, kThreads, 0, stream>>>(
            buffers, device_mask, num_envs);
    }
}

}  // namespace nesle::cuda

#endif

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
    // PPU scalars stay in registers for the whole launch, like CpuState —
    // they are otherwise ~6-10 dependent global round-trips per instruction.
    auto hot = load_ppu_hot_state(buffers, env);
    std::uint64_t total_instructions = 0;
    std::uint32_t total_frames_completed = 0;
    std::uint32_t budget_hits = 0;
    for (std::uint32_t frame = 0; frame < frameskip; ++frame) {
        std::uint64_t instructions = 0;
        std::uint32_t frames_completed = 0;
        while (instructions < max_instructions_per_frame && frames_completed == 0) {
            const auto step = step_batch_console_instruction_hot(buffers, env, state, hot);
            if (stats.opcode_counts != nullptr) {
                atomicAdd(&stats.opcode_counts[step.cpu.opcode], 1ULL);
            }
            if (stats.pc_counts != nullptr) {
                atomicAdd(&stats.pc_counts[step.cpu.pc], 1ULL);
            }
            ++instructions;
            frames_completed += step.frames_completed;
            if (step.cpu.opcode == 0x4C && state.pc == step.cpu.pc && frames_completed == 0) {
                const auto fast_forward =
                    fast_forward_batch_console_idle_loop_hot(buffers, env, state, hot);
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
    store_ppu_hot_state(buffers, env, hot);

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

// One block per env; threads stride over the frame's pixels. Each pixel is a
// pure function of PPU state (see render_batch_rgb_pixel_env), and every
// thread owns the pixels it writes, so no synchronization is needed.
__global__ void render_rgb_kernel(BatchBuffers buffers, std::uint32_t num_envs) {
    const auto env = blockIdx.x;
    if (env >= num_envs) {
        return;
    }
    // Resolving the presentation-snapshot views is a handful of loads and
    // pointer swaps; redoing it per thread is cheaper than staging it through
    // shared memory.
    auto views = resolve_render_env_views(buffers, env);
    constexpr std::uint32_t kPixels = kFrameWidth * kFrameHeight;
    for (std::uint32_t pixel = threadIdx.x; pixel < kPixels; pixel += blockDim.x) {
        render_batch_rgb_pixel_env(buffers, views.hud, views.play, env, views.split_y, pixel);
    }
}

// One thread per env, serial frame paint. The per-pixel kernel above checks
// every sprite per pixel, which costs more total arithmetic; once the env
// count alone saturates the GPU, the serial per-env renderer wins (measured
// crossover ~ several hundred envs on a GTX 1050 Ti: 2.9x faster per-pixel at
// 256 envs, 1.5x slower at 2048).
__global__ void render_rgb_kernel_serial(BatchBuffers buffers, std::uint32_t num_envs) {
    const auto env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= num_envs) {
        return;
    }
    render_batch_rgb_frame_env(buffers, env);
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
    constexpr int kThreads = 256;
    // Small batches (recording, eval, pixel obs on few envs): block-per-env,
    // threads cooperate on pixels. Large batches: the env count already
    // saturates the GPU and the serial per-env paint does less total work.
    constexpr std::uint32_t kPerPixelMaxEnvs = 512;
    if (config.num_envs <= kPerPixelMaxEnvs) {
        const int blocks = static_cast<int>(config.num_envs);
        render_rgb_kernel<<<blocks, kThreads, 0, stream>>>(buffers, config.num_envs);
    } else {
        const int blocks = static_cast<int>((config.num_envs + kThreads - 1) / kThreads);
        render_rgb_kernel_serial<<<blocks, kThreads, 0, stream>>>(buffers, config.num_envs);
    }
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

__global__ void reset_console_envs_from_snapshot_kernel(BatchBuffers buffers,
                                                        SnapshotTemplate snapshot,
                                                        const std::uint8_t* mask,
                                                        std::uint32_t num_envs) {
    const auto env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < num_envs && mask[env] != 0) {
        warm_reset_console_env(buffers, env, snapshot);
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

void launch_snapshot_reset_envs_kernel(const BatchBuffers& buffers,
                                       const SnapshotTemplate& snapshot,
                                       const std::uint8_t* device_mask,
                                       std::uint32_t num_envs,
                                       cudaStream_t stream) {
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((num_envs + kThreads - 1) / kThreads);
    reset_console_envs_from_snapshot_kernel<<<blocks, kThreads, 0, stream>>>(
        buffers, snapshot, device_mask, num_envs);
}

}  // namespace nesle::cuda

#endif

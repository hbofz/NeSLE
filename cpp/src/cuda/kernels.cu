#include "nesle/cuda/kernels.cuh"

#ifdef __CUDACC__

#include "nesle/cuda/batch_step.cuh"

namespace nesle::cuda {
namespace {

__global__ void step_reward_kernel(BatchBuffers buffers, std::uint32_t num_envs) {
    const auto env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < num_envs) {
        apply_batch_reward_env(buffers, env);
    }
}

}  // namespace

void launch_step_kernel(const BatchBuffers& buffers, StepConfig config, cudaStream_t stream) {
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((config.num_envs + kThreads - 1) / kThreads);
    step_reward_kernel<<<blocks, kThreads, 0, stream>>>(buffers, config.num_envs);
}

void launch_render_kernel(const BatchBuffers&, StepConfig, cudaStream_t) {}

}  // namespace nesle::cuda

#endif

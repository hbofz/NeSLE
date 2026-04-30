#include "nesle/cuda/kernels.cuh"

#ifdef __CUDACC__

namespace nesle::cuda {
namespace {

__global__ void mark_alive_kernel(std::uint8_t* done, std::uint32_t num_envs) {
    const auto env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < num_envs) {
        done[env] = 0;
    }
}

}  // namespace

void launch_step_kernel(const BatchBuffers& buffers, StepConfig config, cudaStream_t stream) {
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((config.num_envs + kThreads - 1) / kThreads);
    mark_alive_kernel<<<blocks, kThreads, 0, stream>>>(buffers.done, config.num_envs);
}

void launch_render_kernel(const BatchBuffers&, StepConfig, cudaStream_t) {}

}  // namespace nesle::cuda

#endif

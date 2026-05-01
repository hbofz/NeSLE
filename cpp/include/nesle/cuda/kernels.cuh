#pragma once

#include <cstdint>

#include "nesle/cuda/state.cuh"

#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#endif

namespace nesle::cuda {

struct StepConfig {
    std::uint32_t num_envs;
    std::uint32_t frameskip;
    bool render;
};

#ifdef __CUDACC__
void launch_step_kernel(const BatchBuffers& buffers, StepConfig config, cudaStream_t stream);
void launch_render_kernel(const BatchBuffers& buffers, StepConfig config, cudaStream_t stream);
#endif

}  // namespace nesle::cuda

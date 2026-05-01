#pragma once

#include <cstdint>

#include "nesle/cuda/state.cuh"

#ifdef __CUDACC__
#define NESLE_CUDA_HD __host__ __device__
#else
#define NESLE_CUDA_HD
#endif

namespace nesle::cuda {

constexpr std::uint16_t kPpuDotsPerScanline = 341;
constexpr std::uint16_t kPpuScanlinesPerFrame = 262;
constexpr std::int16_t kPpuVblankStartScanline = 241;
constexpr std::int16_t kPpuPreRenderScanline = 261;
constexpr std::uint16_t kPpuVblankFlagDot = 1;
constexpr std::int16_t kPpuCoarseSpriteZeroHitScanline = 30;
constexpr std::uint16_t kPpuCoarseSpriteZeroHitDot = 1;

struct BatchPpuStepResult {
    std::uint32_t cycles = 0;
    std::uint32_t frames_completed = 0;
    bool nmi_started = false;
};

NESLE_CUDA_HD inline bool batch_ppu_nmi_enabled(const BatchBuffers& buffers,
                                                std::uint32_t env) {
    return (buffers.ppu.ctrl[env] & 0x80) != 0;
}

NESLE_CUDA_HD inline bool batch_ppu_rendering_enabled(const BatchBuffers& buffers,
                                                      std::uint32_t env) {
    return (buffers.ppu.mask[env] & 0x18) != 0;
}

NESLE_CUDA_HD inline void batch_ppu_set_vblank(BatchBuffers& buffers,
                                               std::uint32_t env,
                                               bool enabled) {
    const auto was_in_vblank = static_cast<std::uint8_t>(buffers.ppu.status[env] & 0x80);
    if (enabled) {
        buffers.ppu.status[env] = static_cast<std::uint8_t>(buffers.ppu.status[env] | 0x80);
        if (was_in_vblank == 0 && batch_ppu_nmi_enabled(buffers, env)) {
            buffers.ppu.nmi_pending[env] = 1;
        }
        return;
    }

    buffers.ppu.status[env] = static_cast<std::uint8_t>(buffers.ppu.status[env] & 0x7F);
    buffers.ppu.nmi_pending[env] = 0;
}

NESLE_CUDA_HD inline BatchPpuStepResult batch_ppu_step_env(BatchBuffers& buffers,
                                                           std::uint32_t env,
                                                           std::uint32_t ppu_cycles) {
    BatchPpuStepResult result;
    result.cycles = ppu_cycles;

    for (std::uint32_t i = 0; i < ppu_cycles; ++i) {
        ++buffers.ppu.dot[env];
        if (buffers.ppu.dot[env] >= kPpuDotsPerScanline) {
            buffers.ppu.dot[env] = 0;
            ++buffers.ppu.scanline[env];
            if (buffers.ppu.scanline[env] >= kPpuScanlinesPerFrame) {
                buffers.ppu.scanline[env] = 0;
                ++buffers.ppu.frame[env];
                ++result.frames_completed;
            }
        }

        if (buffers.ppu.scanline[env] == kPpuVblankStartScanline &&
            buffers.ppu.dot[env] == kPpuVblankFlagDot) {
            const auto had_nmi = buffers.ppu.nmi_pending[env] != 0;
            batch_ppu_set_vblank(buffers, env, true);
            result.nmi_started = result.nmi_started ||
                                 (!had_nmi && buffers.ppu.nmi_pending[env] != 0);
        } else if (buffers.ppu.scanline[env] == kPpuCoarseSpriteZeroHitScanline &&
                   buffers.ppu.dot[env] == kPpuCoarseSpriteZeroHitDot &&
                   batch_ppu_rendering_enabled(buffers, env)) {
            buffers.ppu.status[env] = static_cast<std::uint8_t>(buffers.ppu.status[env] | 0x40);
        } else if (buffers.ppu.scanline[env] == kPpuPreRenderScanline &&
                   buffers.ppu.dot[env] == kPpuVblankFlagDot) {
            batch_ppu_set_vblank(buffers, env, false);
            buffers.ppu.status[env] = static_cast<std::uint8_t>(buffers.ppu.status[env] & 0x1F);
        }
    }

    return result;
}

}  // namespace nesle::cuda

#undef NESLE_CUDA_HD

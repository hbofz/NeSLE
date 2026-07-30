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

NESLE_CUDA_HD inline PpuHotState load_ppu_hot_state(const BatchBuffers& buffers,
                                                    std::uint32_t env) {
    PpuHotState hot;
    hot.frame_dot = buffers.ppu.frame_dot[env];
    hot.frame = buffers.ppu.frame[env];
    hot.status = buffers.ppu.status[env];
    hot.ctrl = buffers.ppu.ctrl[env];
    hot.mask = buffers.ppu.mask[env];
    hot.nmi_pending = buffers.ppu.nmi_pending[env];
    return hot;
}

NESLE_CUDA_HD inline void store_ppu_hot_state(BatchBuffers& buffers,
                                              std::uint32_t env,
                                              const PpuHotState& hot) {
    buffers.ppu.frame_dot[env] = hot.frame_dot;
    buffers.ppu.frame[env] = hot.frame;
    buffers.ppu.status[env] = hot.status;
    buffers.ppu.ctrl[env] = hot.ctrl;
    buffers.ppu.mask[env] = hot.mask;
    buffers.ppu.nmi_pending[env] = hot.nmi_pending;
}

NESLE_CUDA_HD inline bool batch_ppu_nmi_enabled(const BatchBuffers& buffers,
                                                std::uint32_t env) {
    return (buffers.ppu.ctrl[env] & 0x80) != 0;
}

NESLE_CUDA_HD inline bool batch_ppu_rendering_enabled(const BatchBuffers& buffers,
                                                      std::uint32_t env) {
    return (buffers.ppu.mask[env] & 0x18) != 0;
}

NESLE_CUDA_HD inline void batch_ppu_set_vblank_hot(PpuHotState& hot, bool enabled) {
    const auto was_in_vblank = static_cast<std::uint8_t>(hot.status & 0x80);
    if (enabled) {
        hot.status = static_cast<std::uint8_t>(hot.status | 0x80);
        if (was_in_vblank == 0 && (hot.ctrl & 0x80) != 0) {
            hot.nmi_pending = 1;
        }
        return;
    }
    hot.status = static_cast<std::uint8_t>(hot.status & 0x7F);
    hot.nmi_pending = 0;
}

NESLE_CUDA_HD inline std::uint32_t batch_ppu_frame_dot(const BatchBuffers& buffers,
                                                       std::uint32_t env) {
    return buffers.ppu.frame_dot[env];
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

NESLE_CUDA_HD inline std::uint32_t batch_ppu_cycles_until_next_timing_event(
    const BatchBuffers& buffers,
    std::uint32_t env) {
    constexpr std::uint32_t kFrameDots =
        static_cast<std::uint32_t>(kPpuDotsPerScanline) *
        static_cast<std::uint32_t>(kPpuScanlinesPerFrame);
    constexpr std::uint32_t kSpriteZeroHitDot =
        static_cast<std::uint32_t>(kPpuCoarseSpriteZeroHitScanline) *
            static_cast<std::uint32_t>(kPpuDotsPerScanline) +
        static_cast<std::uint32_t>(kPpuCoarseSpriteZeroHitDot);
    constexpr std::uint32_t kVblankDot =
        static_cast<std::uint32_t>(kPpuVblankStartScanline) *
            static_cast<std::uint32_t>(kPpuDotsPerScanline) +
        static_cast<std::uint32_t>(kPpuVblankFlagDot);
    constexpr std::uint32_t kPreRenderDot =
        static_cast<std::uint32_t>(kPpuPreRenderScanline) *
            static_cast<std::uint32_t>(kPpuDotsPerScanline) +
        static_cast<std::uint32_t>(kPpuVblankFlagDot);

    const auto current = batch_ppu_frame_dot(buffers, env);
    auto distance = [current](std::uint32_t event_dot) {
        return event_dot > current ? event_dot - current : kFrameDots - current + event_dot;
    };

    auto next = kFrameDots - current;
    const auto vblank = distance(kVblankDot);
    if (vblank < next) {
        next = vblank;
    }
    const auto prerender = distance(kPreRenderDot);
    if (prerender < next) {
        next = prerender;
    }
    if (batch_ppu_rendering_enabled(buffers, env)) {
        const auto sprite_zero = distance(kSpriteZeroHitDot);
        if (sprite_zero < next) {
            next = sprite_zero;
        }
    }
    return next;
}

NESLE_CUDA_HD inline std::uint32_t batch_ppu_cycles_until_next_timing_event_hot(
    const PpuHotState& hot) {
    constexpr std::uint32_t kFrameDots =
        static_cast<std::uint32_t>(kPpuDotsPerScanline) *
        static_cast<std::uint32_t>(kPpuScanlinesPerFrame);
    constexpr std::uint32_t kSpriteZeroHitDot =
        static_cast<std::uint32_t>(kPpuCoarseSpriteZeroHitScanline) *
            static_cast<std::uint32_t>(kPpuDotsPerScanline) +
        static_cast<std::uint32_t>(kPpuCoarseSpriteZeroHitDot);
    constexpr std::uint32_t kVblankDot =
        static_cast<std::uint32_t>(kPpuVblankStartScanline) *
            static_cast<std::uint32_t>(kPpuDotsPerScanline) +
        static_cast<std::uint32_t>(kPpuVblankFlagDot);
    constexpr std::uint32_t kPreRenderDot =
        static_cast<std::uint32_t>(kPpuPreRenderScanline) *
            static_cast<std::uint32_t>(kPpuDotsPerScanline) +
        static_cast<std::uint32_t>(kPpuVblankFlagDot);

    const auto current = hot.frame_dot;
    auto distance = [current](std::uint32_t event_dot) {
        return event_dot > current ? event_dot - current : kFrameDots - current + event_dot;
    };

    auto next = kFrameDots - current;
    const auto vblank = distance(kVblankDot);
    if (vblank < next) {
        next = vblank;
    }
    const auto prerender = distance(kPreRenderDot);
    if (prerender < next) {
        next = prerender;
    }
    if ((hot.mask & 0x18) != 0) {
        const auto sprite_zero = distance(kSpriteZeroHitDot);
        if (sprite_zero < next) {
            next = sprite_zero;
        }
    }
    return next;
}

// Hot-path variant: PPU scalars come from (and return to) `hot`, which the
// caller keeps in registers across the whole kernel. `buffers` is still needed
// for the per-frame presentation-snapshot capture (scroll/latch/snapshot
// arrays are global — they are touched per frame, not per instruction).
NESLE_CUDA_HD inline BatchPpuStepResult batch_ppu_step_env_hot(BatchBuffers& buffers,
                                                               std::uint32_t env,
                                                               std::uint32_t ppu_cycles,
                                                               PpuHotState& hot) {
    BatchPpuStepResult result;
    result.cycles = ppu_cycles;

    constexpr std::uint32_t kFrameDots =
        static_cast<std::uint32_t>(kPpuDotsPerScanline) *
        static_cast<std::uint32_t>(kPpuScanlinesPerFrame);
    constexpr std::uint32_t kSpriteZeroHitDot =
        static_cast<std::uint32_t>(kPpuCoarseSpriteZeroHitScanline) *
            static_cast<std::uint32_t>(kPpuDotsPerScanline) +
        static_cast<std::uint32_t>(kPpuCoarseSpriteZeroHitDot);
    constexpr std::uint32_t kVblankDot =
        static_cast<std::uint32_t>(kPpuVblankStartScanline) *
            static_cast<std::uint32_t>(kPpuDotsPerScanline) +
        static_cast<std::uint32_t>(kPpuVblankFlagDot);
    constexpr std::uint32_t kPreRenderDot =
        static_cast<std::uint32_t>(kPpuPreRenderScanline) *
            static_cast<std::uint32_t>(kPpuDotsPerScanline) +
        static_cast<std::uint32_t>(kPpuVblankFlagDot);

    const auto start_dot = hot.frame_dot;
    const auto end_dot = start_dot + ppu_cycles;
    result.frames_completed = end_dot / kFrameDots;

    auto crossed = [&](std::uint32_t frame_offset, std::uint32_t event_dot) {
        const auto absolute_dot = frame_offset * kFrameDots + event_dot;
        return absolute_dot > start_dot && absolute_dot <= end_dot;
    };

    for (std::uint32_t frame_offset = 0; frame_offset <= result.frames_completed; ++frame_offset) {
        if (crossed(frame_offset, kSpriteZeroHitDot) && (hot.mask & 0x18) != 0) {
            hot.status = static_cast<std::uint8_t>(hot.status | 0x40);
        }
        if (crossed(frame_offset, kVblankDot)) {
            const auto had_nmi = hot.nmi_pending != 0;
            batch_ppu_set_vblank_hot(hot, true);
            result.nmi_started = result.nmi_started || (!had_nmi && hot.nmi_pending != 0);
            if (buffers.ppu.snap_nametable != nullptr) {
                // Freeze the presentation snapshot for the frame that just
                // finished rendering: frame-start scroll/ctrl were latched at
                // the previous pre-render crossing; end values, OAM, and video
                // memory are coherent right now (the game's NMI handler that
                // mutates them for the NEXT frame has not run yet).
                buffers.ppu.snap_scroll_x_start[env] = buffers.ppu.lat_scroll_x[env];
                buffers.ppu.snap_scroll_y_start[env] = buffers.ppu.lat_scroll_y[env];
                buffers.ppu.snap_ctrl_start[env] = buffers.ppu.lat_ctrl[env];
                buffers.ppu.snap_scroll_x_end[env] = buffers.ppu.scroll_x[env];
                buffers.ppu.snap_scroll_y_end[env] = buffers.ppu.scroll_y[env];
                buffers.ppu.snap_ctrl_end[env] = hot.ctrl;
                buffers.ppu.snap_mask[env] = hot.mask;
                const auto* oam_src = buffers.ppu.oam + static_cast<std::uint64_t>(env) * kOamBytes;
                auto* oam_dst = buffers.ppu.snap_oam + static_cast<std::uint64_t>(env) * kOamBytes;
                copy_bytes_fast(oam_dst, oam_src, static_cast<std::uint32_t>(kOamBytes));
                const auto* nt_src =
                    buffers.ppu.nametable_ram + static_cast<std::uint64_t>(env) * kNametableRamBytes;
                auto* nt_dst =
                    buffers.ppu.snap_nametable + static_cast<std::uint64_t>(env) * kNametableRamBytes;
                copy_bytes_fast(nt_dst, nt_src, static_cast<std::uint32_t>(kNametableRamBytes));
                const auto* pal_src =
                    buffers.ppu.palette_ram + static_cast<std::uint64_t>(env) * kPaletteRamBytes;
                auto* pal_dst =
                    buffers.ppu.snap_palette + static_cast<std::uint64_t>(env) * kPaletteRamBytes;
                copy_bytes_fast(pal_dst, pal_src, static_cast<std::uint32_t>(kPaletteRamBytes));
            }
        }
        if (crossed(frame_offset, kPreRenderDot)) {
            batch_ppu_set_vblank_hot(hot, false);
            hot.status = static_cast<std::uint8_t>(hot.status & 0x1F);
            if (buffers.ppu.lat_ctrl != nullptr) {
                // Latch frame-start scroll/ctrl: the game's vblank handler has
                // set up the values the next visible frame begins with (SMB:
                // scroll 0 for the status bar).
                buffers.ppu.lat_scroll_x[env] = buffers.ppu.scroll_x[env];
                buffers.ppu.lat_scroll_y[env] = buffers.ppu.scroll_y[env];
                buffers.ppu.lat_ctrl[env] = hot.ctrl;
            }
        }
    }

    hot.frame_dot = end_dot % kFrameDots;
    hot.frame += result.frames_completed;

    return result;
}

NESLE_CUDA_HD inline BatchPpuStepResult batch_ppu_step_env(BatchBuffers& buffers,
                                                           std::uint32_t env,
                                                           std::uint32_t ppu_cycles) {
    auto hot = load_ppu_hot_state(buffers, env);
    const auto result = batch_ppu_step_env_hot(buffers, env, ppu_cycles, hot);
    store_ppu_hot_state(buffers, env, hot);
    return result;
}

}  // namespace nesle::cuda

#undef NESLE_CUDA_HD

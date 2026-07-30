#pragma once

#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#define NESLE_CUDA_STATE_HD __host__ __device__
#else
#define NESLE_CUDA_STATE_HD
#endif

// Portable restrict qualifier for the SoA member pointers below. Every array
// behind a live BatchBuffers is its own allocation (distinct cudaMalloc calls
// in the binding, distinct std::vector/std::array storage in the host tests),
// so no two members of one struct instance ever alias — telling the compiler
// so lets it keep loaded values in registers across stores through sibling
// pointers.
//
// The render path (batch_render.cuh, resolve_render_env_views) copies
// BatchBuffers into shadow views whose members are repointed at the snap_*
// arrays, so ACROSS instances the same allocation is reachable through
// several names (e.g. play.ppu.nametable_ram == buffers.ppu.snap_nametable,
// and hud/play share every repointed array). That is still conforming:
// restrict (C99 6.7.3.1 semantics, which the __restrict extensions follow)
// only constrains objects that are MODIFIED during the pointers' lifetime,
// and rendering writes nothing but frames_rgb — a distinct allocation only
// ever accessed through the single `target` view. Everywhere state is
// mutated (stepping, resets, snapshot capture/restore) only one BatchBuffers
// instance is live and its members point at pairwise-distinct allocations.
#if defined(__CUDACC__)
#define NESLE_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define NESLE_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define NESLE_RESTRICT __restrict__
#else
#define NESLE_RESTRICT
#endif

namespace nesle::cuda {

// Copy `n` bytes from `src` to `dst` (non-overlapping). On the CUDA device
// trajectory, uses 16-byte vector chunks when `n` and both pointers are
// 16-byte aligned (all per-env blocks are: each array is its own cudaMalloc
// allocation and the per-env strides — 2048/8192/256/32 — are multiples of
// 16); otherwise falls back to a byte loop. Host builds (including the C++
// unit tests, which compile these headers with a plain host compiler) use
// std::memcpy, which imposes no alignment requirement.
NESLE_CUDA_STATE_HD inline void copy_bytes_fast(std::uint8_t* dst,
                                                const std::uint8_t* src,
                                                std::uint32_t n) {
#if defined(__CUDA_ARCH__)
    if ((n % 16u) == 0u &&
        (reinterpret_cast<std::uintptr_t>(dst) % 16u) == 0u &&
        (reinterpret_cast<std::uintptr_t>(src) % 16u) == 0u) {
        auto* d = reinterpret_cast<uint4*>(dst);
        const auto* s = reinterpret_cast<const uint4*>(src);
        const std::uint32_t chunks = n / 16u;
        for (std::uint32_t i = 0; i < chunks; ++i) {
            d[i] = s[i];
        }
        return;
    }
    for (std::uint32_t i = 0; i < n; ++i) {
        dst[i] = src[i];
    }
#else
    std::memcpy(dst, src, n);
#endif
}

// Zero `n` bytes at `dst`. Same alignment strategy as copy_bytes_fast.
NESLE_CUDA_STATE_HD inline void zero_bytes_fast(std::uint8_t* dst, std::uint32_t n) {
#if defined(__CUDA_ARCH__)
    if ((n % 16u) == 0u && (reinterpret_cast<std::uintptr_t>(dst) % 16u) == 0u) {
        auto* d = reinterpret_cast<uint4*>(dst);
        const std::uint32_t chunks = n / 16u;
        for (std::uint32_t i = 0; i < chunks; ++i) {
            d[i] = make_uint4(0u, 0u, 0u, 0u);
        }
        return;
    }
    for (std::uint32_t i = 0; i < n; ++i) {
        dst[i] = 0;
    }
#else
    std::memset(dst, 0, n);
#endif
}

constexpr int kCpuRamBytes = 2048;
constexpr int kPrgRamBytes = 8 * 1024;
constexpr int kPaletteRamBytes = 32;
constexpr int kOamBytes = 256;
constexpr int kNametableRamBytes = 2048;
constexpr int kFrameWidth = 256;
constexpr int kFrameHeight = 240;
constexpr int kRgbChannels = 3;
constexpr std::uint8_t kNametableVertical = 0;
constexpr std::uint8_t kNametableHorizontal = 1;
constexpr std::uint8_t kNametableFourScreen = 2;

struct CpuStateSoA {
    std::uint16_t* NESLE_RESTRICT pc;
    std::uint8_t* NESLE_RESTRICT a;
    std::uint8_t* NESLE_RESTRICT x;
    std::uint8_t* NESLE_RESTRICT y;
    std::uint8_t* NESLE_RESTRICT sp;
    std::uint8_t* NESLE_RESTRICT p;
    std::uint64_t* NESLE_RESTRICT cycles;
    std::uint8_t* NESLE_RESTRICT nmi_pending;
    std::uint8_t* NESLE_RESTRICT irq_pending;
    std::uint8_t* NESLE_RESTRICT ram;
    std::uint8_t* NESLE_RESTRICT prg_ram;
    std::uint8_t* NESLE_RESTRICT controller1_shift;
    std::uint8_t* NESLE_RESTRICT controller1_shift_count;
    std::uint8_t* NESLE_RESTRICT controller1_strobe;
    std::uint32_t* NESLE_RESTRICT pending_dma_cycles;
};

struct PpuStateSoA {
    std::uint8_t* NESLE_RESTRICT ctrl;
    std::uint8_t* NESLE_RESTRICT mask;
    std::uint8_t* NESLE_RESTRICT status;
    std::uint8_t* NESLE_RESTRICT oam_addr;
    std::uint8_t* NESLE_RESTRICT nmi_pending;
    // Merged frame position: scanline * kPpuDotsPerScanline + dot. Stored as
    // one word so the per-launch hot-state load/store is a single 32-bit
    // access instead of two dependent 16-bit ones (PpuHotState composes the
    // same value in registers).
    std::uint32_t* NESLE_RESTRICT frame_dot;
    std::uint64_t* NESLE_RESTRICT frame;
    std::uint16_t* NESLE_RESTRICT v;
    std::uint16_t* NESLE_RESTRICT t;
    std::uint8_t* NESLE_RESTRICT x;
    std::uint8_t* NESLE_RESTRICT w;
    std::uint8_t* NESLE_RESTRICT open_bus;
    std::uint8_t* NESLE_RESTRICT read_buffer;
    std::uint8_t* NESLE_RESTRICT scroll_x;
    std::uint8_t* NESLE_RESTRICT scroll_y;
    std::uint8_t* NESLE_RESTRICT nametable_ram;
    std::uint8_t* NESLE_RESTRICT palette_ram;
    std::uint8_t* NESLE_RESTRICT oam;

    // Presentation snapshot — frozen at each vblank start so render() sees an
    // internally consistent picture of the just-finished frame no matter where
    // stepping paused. All nullptr (host tests, older callers) => render falls
    // back to live state, the pre-snapshot behavior. `lat_*` latch scroll/ctrl
    // at the pre-render crossing (frame start); the following vblank promotes
    // them into `snap_*_start` so start/end pairs always describe one frame.
    // Two-region rendering uses `snap_*_start` above sprite-0's bottom edge
    // (SMB's status-bar split) and `snap_*_end` below it.
    std::uint8_t* NESLE_RESTRICT lat_scroll_x;
    std::uint8_t* NESLE_RESTRICT lat_scroll_y;
    std::uint8_t* NESLE_RESTRICT lat_ctrl;
    std::uint8_t* NESLE_RESTRICT snap_scroll_x_start;
    std::uint8_t* NESLE_RESTRICT snap_scroll_y_start;
    std::uint8_t* NESLE_RESTRICT snap_ctrl_start;
    std::uint8_t* NESLE_RESTRICT snap_scroll_x_end;
    std::uint8_t* NESLE_RESTRICT snap_scroll_y_end;
    std::uint8_t* NESLE_RESTRICT snap_ctrl_end;
    std::uint8_t* NESLE_RESTRICT snap_mask;
    std::uint8_t* NESLE_RESTRICT snap_oam;        // kOamBytes per env
    std::uint8_t* NESLE_RESTRICT snap_nametable;  // kNametableRamBytes per env
    std::uint8_t* NESLE_RESTRICT snap_palette;    // kPaletteRamBytes per env
};

struct CartridgeView {
    const std::uint8_t* prg_rom;
    const std::uint8_t* chr_rom;
    std::uint32_t prg_rom_size;
    std::uint32_t chr_rom_size;
    std::uint8_t mapper;
    std::uint8_t nametable_arrangement;
};

// The PPU fields read/modified on every emulated instruction. The step kernel
// loads them into registers once per launch (like CpuState already is) and
// stores them back at exit — without this they cost ~6-10 dependent global
// round-trips per instruction. Everything else (scroll, v/t/x/w, memories,
// presentation snapshot) stays in global memory: those are touched per
// register-access or per frame, not per instruction. Load/store helpers live
// in batch_ppu.cuh next to the timing constants.
struct PpuHotState {
    std::uint32_t frame_dot = 0;  // scanline * dots-per-scanline + dot
    std::uint64_t frame = 0;
    std::uint8_t status = 0;
    std::uint8_t ctrl = 0;
    std::uint8_t mask = 0;
    std::uint8_t nmi_pending = 0;
};

struct BatchBuffers {
    CpuStateSoA cpu;
    PpuStateSoA ppu;
    CartridgeView cart;
    std::uint8_t* NESLE_RESTRICT action_masks;
    std::uint8_t* NESLE_RESTRICT done;
    float* NESLE_RESTRICT rewards;
    int* NESLE_RESTRICT previous_mario_x;
    int* NESLE_RESTRICT previous_mario_time;
    std::uint8_t* NESLE_RESTRICT frames_rgb;
};

// Read-only "bank" of N snapshot templates used by snapshot-based env resets. Each top-level
// array is num_levels copies of a single snapshot's array data, laid out contiguously.
// `env_to_level[env]` selects which slot each env restores from — letting different envs in
// the same batch start at different levels (curriculum training). For the single-level case
// num_levels is 1 and env_to_level is all zeros.
struct SnapshotTemplate {
    // Per-level array buffers (concatenated, length = num_levels * <kind>_Bytes).
    const std::uint8_t* cpu_ram = nullptr;
    const std::uint8_t* prg_ram = nullptr;
    const std::uint8_t* nametable_ram = nullptr;
    const std::uint8_t* palette_ram = nullptr;
    const std::uint8_t* oam = nullptr;

    // Per-level scalar fields (length = num_levels).
    const std::uint16_t* pc = nullptr;
    const std::uint8_t* a = nullptr;
    const std::uint8_t* x = nullptr;
    const std::uint8_t* y = nullptr;
    const std::uint8_t* sp = nullptr;
    const std::uint8_t* p = nullptr;
    const std::uint64_t* cycles = nullptr;
    const std::uint8_t* ppu_ctrl = nullptr;
    const std::uint8_t* ppu_mask = nullptr;
    const std::uint8_t* ppu_status = nullptr;
    const std::uint8_t* ppu_oam_addr = nullptr;
    const std::uint8_t* ppu_open_bus = nullptr;
    const std::uint8_t* ppu_read_buffer = nullptr;
    const std::uint8_t* ppu_x = nullptr;
    const std::uint8_t* ppu_w = nullptr;
    const std::uint16_t* ppu_v = nullptr;
    const std::uint16_t* ppu_t = nullptr;

    // Per-env level assignment (length = num_envs).
    const std::uint8_t* env_to_level = nullptr;

    std::uint32_t num_levels = 0;
};

NESLE_CUDA_STATE_HD inline const std::uint8_t* env_cpu_ram(const BatchBuffers& buffers,
                                                           std::uint32_t env) {
    return buffers.cpu.ram + static_cast<std::uint64_t>(env) * kCpuRamBytes;
}

NESLE_CUDA_STATE_HD inline std::uint8_t* env_cpu_ram(BatchBuffers& buffers, std::uint32_t env) {
    return buffers.cpu.ram + static_cast<std::uint64_t>(env) * kCpuRamBytes;
}

NESLE_CUDA_STATE_HD inline const std::uint8_t* env_oam(const BatchBuffers& buffers,
                                                       std::uint32_t env) {
    return buffers.ppu.oam + static_cast<std::uint64_t>(env) * kOamBytes;
}

NESLE_CUDA_STATE_HD inline std::uint8_t* env_oam(BatchBuffers& buffers, std::uint32_t env) {
    return buffers.ppu.oam + static_cast<std::uint64_t>(env) * kOamBytes;
}

}  // namespace nesle::cuda

#undef NESLE_CUDA_STATE_HD
#undef NESLE_RESTRICT

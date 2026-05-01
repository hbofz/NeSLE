#pragma once

#include <cstdint>

#ifdef __CUDACC__
#define NESLE_CUDA_STATE_HD __host__ __device__
#else
#define NESLE_CUDA_STATE_HD
#endif

namespace nesle::cuda {

constexpr int kCpuRamBytes = 2048;
constexpr int kPrgRamBytes = 8 * 1024;
constexpr int kPaletteRamBytes = 32;
constexpr int kOamBytes = 256;
constexpr int kNametableRamBytes = 2048;
constexpr int kFrameWidth = 256;
constexpr int kFrameHeight = 240;
constexpr int kRgbChannels = 3;

struct CpuStateSoA {
    std::uint16_t* pc;
    std::uint8_t* a;
    std::uint8_t* x;
    std::uint8_t* y;
    std::uint8_t* sp;
    std::uint8_t* p;
    std::uint64_t* cycles;
    std::uint8_t* nmi_pending;
    std::uint8_t* irq_pending;
    std::uint8_t* ram;
    std::uint8_t* prg_ram;
    std::uint8_t* controller1_shift;
    std::uint8_t* controller1_shift_count;
    std::uint8_t* controller1_strobe;
    std::uint32_t* pending_dma_cycles;
};

struct PpuStateSoA {
    std::uint8_t* ctrl;
    std::uint8_t* mask;
    std::uint8_t* status;
    std::uint8_t* oam_addr;
    std::uint8_t* nmi_pending;
    std::int16_t* scanline;
    std::uint16_t* dot;
    std::uint64_t* frame;
    std::uint16_t* v;
    std::uint16_t* t;
    std::uint8_t* x;
    std::uint8_t* w;
    std::uint8_t* nametable_ram;
    std::uint8_t* palette_ram;
    std::uint8_t* oam;
};

struct CartridgeView {
    const std::uint8_t* prg_rom;
    const std::uint8_t* chr_rom;
    std::uint32_t prg_rom_size;
    std::uint32_t chr_rom_size;
    std::uint8_t mapper;
};

struct BatchBuffers {
    CpuStateSoA cpu;
    PpuStateSoA ppu;
    CartridgeView cart;
    std::uint8_t* action_masks;
    std::uint8_t* done;
    float* rewards;
    int* previous_mario_x;
    int* previous_mario_time;
    std::uint8_t* previous_mario_dying;
    std::uint8_t* frames_rgb;
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

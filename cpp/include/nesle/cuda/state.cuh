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
constexpr std::uint8_t kNametableVertical = 0;
constexpr std::uint8_t kNametableHorizontal = 1;
constexpr std::uint8_t kNametableFourScreen = 2;

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
    std::uint8_t* open_bus;
    std::uint8_t* read_buffer;
    std::uint8_t* scroll_x;
    std::uint8_t* scroll_y;
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
    std::uint8_t nametable_arrangement;
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
    std::uint8_t* frames_rgb;
};

// Read-only template used by snapshot-based env resets. Array pointers reference device-side
// "template" buffers populated once from a parsed FCEUX FCS snapshot. The kernel copies them
// into each env's slice on reset, bypassing the cold-boot title-screen sequence entirely.
struct SnapshotTemplate {
    const std::uint8_t* cpu_ram = nullptr;        // kCpuRamBytes
    const std::uint8_t* prg_ram = nullptr;        // kPrgRamBytes
    const std::uint8_t* nametable_ram = nullptr;  // kNametableRamBytes
    const std::uint8_t* palette_ram = nullptr;    // kPaletteRamBytes
    const std::uint8_t* oam = nullptr;            // kOamBytes

    std::uint16_t pc = 0;
    std::uint8_t a = 0;
    std::uint8_t x = 0;
    std::uint8_t y = 0;
    std::uint8_t sp = 0xFD;
    std::uint8_t p = 0x24;
    std::uint64_t cycles = 0;

    std::uint8_t ppu_ctrl = 0;
    std::uint8_t ppu_mask = 0;
    std::uint8_t ppu_status = 0;
    std::uint8_t ppu_oam_addr = 0;
    std::uint8_t ppu_open_bus = 0;
    std::uint8_t ppu_read_buffer = 0;
    std::uint8_t ppu_x = 0;
    std::uint8_t ppu_w = 0;
    std::uint16_t ppu_v = 0;
    std::uint16_t ppu_t = 0;
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

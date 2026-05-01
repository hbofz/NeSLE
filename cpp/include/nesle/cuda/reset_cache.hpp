#pragma once

#include <cstdint>
#include <vector>

#include "nesle/cuda/state.cuh"

namespace nesle::cuda {

struct EnvResetSnapshot {
    std::uint16_t pc = 0;
    std::uint8_t a = 0;
    std::uint8_t x = 0;
    std::uint8_t y = 0;
    std::uint8_t sp = 0;
    std::uint8_t p = 0;
    std::uint64_t cycles = 0;
    std::uint32_t pending_dma_cycles = 0;
    std::uint8_t controller1_shift = 0;
    std::uint8_t controller1_shift_count = 8;
    std::uint8_t controller1_strobe = 0;

    std::uint8_t ppu_ctrl = 0;
    std::uint8_t ppu_mask = 0;
    std::uint8_t ppu_status = 0;
    std::uint8_t ppu_oam_addr = 0;
    std::uint8_t ppu_nmi_pending = 0;
    std::int16_t ppu_scanline = 0;
    std::uint16_t ppu_dot = 0;
    std::uint64_t ppu_frame = 0;
    std::uint16_t ppu_v = 0;
    std::uint16_t ppu_t = 0;
    std::uint8_t ppu_x = 0;
    std::uint8_t ppu_w = 0;
    std::uint8_t ppu_open_bus = 0;
    std::uint8_t ppu_read_buffer = 0;
    std::uint8_t ppu_scroll_x = 0;
    std::uint8_t ppu_scroll_y = 0;

    int previous_mario_x = 0;
    int previous_mario_time = 0;
    std::uint8_t previous_mario_dying = 0;
    float reward = 0.0F;
    std::uint8_t done = 0;

    std::vector<std::uint8_t> ram;
    std::vector<std::uint8_t> prg_ram;
    std::vector<std::uint8_t> nametable_ram;
    std::vector<std::uint8_t> palette_ram;
    std::vector<std::uint8_t> oam;
};

[[nodiscard]] inline EnvResetSnapshot capture_reset_snapshot(const BatchBuffers& buffers,
                                                             std::uint32_t env) {
    EnvResetSnapshot snapshot;
    snapshot.pc = buffers.cpu.pc[env];
    snapshot.a = buffers.cpu.a[env];
    snapshot.x = buffers.cpu.x[env];
    snapshot.y = buffers.cpu.y[env];
    snapshot.sp = buffers.cpu.sp[env];
    snapshot.p = buffers.cpu.p[env];
    snapshot.cycles = buffers.cpu.cycles[env];
    if (buffers.cpu.pending_dma_cycles != nullptr) {
        snapshot.pending_dma_cycles = buffers.cpu.pending_dma_cycles[env];
    }
    snapshot.controller1_shift = buffers.cpu.controller1_shift[env];
    snapshot.controller1_shift_count = buffers.cpu.controller1_shift_count[env];
    snapshot.controller1_strobe = buffers.cpu.controller1_strobe[env];

    snapshot.ppu_ctrl = buffers.ppu.ctrl[env];
    snapshot.ppu_mask = buffers.ppu.mask[env];
    snapshot.ppu_status = buffers.ppu.status[env];
    snapshot.ppu_oam_addr = buffers.ppu.oam_addr[env];
    snapshot.ppu_nmi_pending = buffers.ppu.nmi_pending[env];
    snapshot.ppu_scanline = buffers.ppu.scanline[env];
    snapshot.ppu_dot = buffers.ppu.dot[env];
    snapshot.ppu_frame = buffers.ppu.frame[env];
    if (buffers.ppu.v != nullptr) {
        snapshot.ppu_v = buffers.ppu.v[env];
    }
    if (buffers.ppu.t != nullptr) {
        snapshot.ppu_t = buffers.ppu.t[env];
    }
    if (buffers.ppu.x != nullptr) {
        snapshot.ppu_x = buffers.ppu.x[env];
    }
    if (buffers.ppu.w != nullptr) {
        snapshot.ppu_w = buffers.ppu.w[env];
    }
    if (buffers.ppu.open_bus != nullptr) {
        snapshot.ppu_open_bus = buffers.ppu.open_bus[env];
    }
    if (buffers.ppu.read_buffer != nullptr) {
        snapshot.ppu_read_buffer = buffers.ppu.read_buffer[env];
    }
    if (buffers.ppu.scroll_x != nullptr) {
        snapshot.ppu_scroll_x = buffers.ppu.scroll_x[env];
    }
    if (buffers.ppu.scroll_y != nullptr) {
        snapshot.ppu_scroll_y = buffers.ppu.scroll_y[env];
    }

    if (buffers.previous_mario_x != nullptr) {
        snapshot.previous_mario_x = buffers.previous_mario_x[env];
    }
    if (buffers.previous_mario_time != nullptr) {
        snapshot.previous_mario_time = buffers.previous_mario_time[env];
    }
    if (buffers.previous_mario_dying != nullptr) {
        snapshot.previous_mario_dying = buffers.previous_mario_dying[env];
    }
    if (buffers.rewards != nullptr) {
        snapshot.reward = buffers.rewards[env];
    }
    if (buffers.done != nullptr) {
        snapshot.done = buffers.done[env];
    }

    const auto* ram = env_cpu_ram(buffers, env);
    snapshot.ram.assign(ram, ram + kCpuRamBytes);
    const auto* prg_ram = buffers.cpu.prg_ram + static_cast<std::uint64_t>(env) * kPrgRamBytes;
    snapshot.prg_ram.assign(prg_ram, prg_ram + kPrgRamBytes);
    if (buffers.ppu.nametable_ram != nullptr) {
        const auto* nametable =
            buffers.ppu.nametable_ram + static_cast<std::uint64_t>(env) * kNametableRamBytes;
        snapshot.nametable_ram.assign(nametable, nametable + kNametableRamBytes);
    }
    if (buffers.ppu.palette_ram != nullptr) {
        const auto* palette =
            buffers.ppu.palette_ram + static_cast<std::uint64_t>(env) * kPaletteRamBytes;
        snapshot.palette_ram.assign(palette, palette + kPaletteRamBytes);
    }
    if (buffers.ppu.oam != nullptr) {
        const auto* oam = env_oam(buffers, env);
        snapshot.oam.assign(oam, oam + kOamBytes);
    }
    return snapshot;
}

inline void restore_reset_snapshot(BatchBuffers& buffers,
                                   std::uint32_t env,
                                   const EnvResetSnapshot& snapshot) {
    buffers.cpu.pc[env] = snapshot.pc;
    buffers.cpu.a[env] = snapshot.a;
    buffers.cpu.x[env] = snapshot.x;
    buffers.cpu.y[env] = snapshot.y;
    buffers.cpu.sp[env] = snapshot.sp;
    buffers.cpu.p[env] = snapshot.p;
    buffers.cpu.cycles[env] = snapshot.cycles;
    if (buffers.cpu.pending_dma_cycles != nullptr) {
        buffers.cpu.pending_dma_cycles[env] = snapshot.pending_dma_cycles;
    }
    buffers.cpu.controller1_shift[env] = snapshot.controller1_shift;
    buffers.cpu.controller1_shift_count[env] = snapshot.controller1_shift_count;
    buffers.cpu.controller1_strobe[env] = snapshot.controller1_strobe;

    buffers.ppu.ctrl[env] = snapshot.ppu_ctrl;
    buffers.ppu.mask[env] = snapshot.ppu_mask;
    buffers.ppu.status[env] = snapshot.ppu_status;
    buffers.ppu.oam_addr[env] = snapshot.ppu_oam_addr;
    buffers.ppu.nmi_pending[env] = snapshot.ppu_nmi_pending;
    buffers.ppu.scanline[env] = snapshot.ppu_scanline;
    buffers.ppu.dot[env] = snapshot.ppu_dot;
    buffers.ppu.frame[env] = snapshot.ppu_frame;
    if (buffers.ppu.v != nullptr) {
        buffers.ppu.v[env] = snapshot.ppu_v;
    }
    if (buffers.ppu.t != nullptr) {
        buffers.ppu.t[env] = snapshot.ppu_t;
    }
    if (buffers.ppu.x != nullptr) {
        buffers.ppu.x[env] = snapshot.ppu_x;
    }
    if (buffers.ppu.w != nullptr) {
        buffers.ppu.w[env] = snapshot.ppu_w;
    }
    if (buffers.ppu.open_bus != nullptr) {
        buffers.ppu.open_bus[env] = snapshot.ppu_open_bus;
    }
    if (buffers.ppu.read_buffer != nullptr) {
        buffers.ppu.read_buffer[env] = snapshot.ppu_read_buffer;
    }
    if (buffers.ppu.scroll_x != nullptr) {
        buffers.ppu.scroll_x[env] = snapshot.ppu_scroll_x;
    }
    if (buffers.ppu.scroll_y != nullptr) {
        buffers.ppu.scroll_y[env] = snapshot.ppu_scroll_y;
    }

    if (buffers.previous_mario_x != nullptr) {
        buffers.previous_mario_x[env] = snapshot.previous_mario_x;
    }
    if (buffers.previous_mario_time != nullptr) {
        buffers.previous_mario_time[env] = snapshot.previous_mario_time;
    }
    if (buffers.previous_mario_dying != nullptr) {
        buffers.previous_mario_dying[env] = snapshot.previous_mario_dying;
    }
    if (buffers.rewards != nullptr) {
        buffers.rewards[env] = snapshot.reward;
    }
    if (buffers.done != nullptr) {
        buffers.done[env] = snapshot.done;
    }

    auto* ram = env_cpu_ram(buffers, env);
    for (std::uint32_t i = 0; i < kCpuRamBytes; ++i) {
        ram[i] = snapshot.ram[i];
    }
    auto* prg_ram = buffers.cpu.prg_ram + static_cast<std::uint64_t>(env) * kPrgRamBytes;
    for (std::uint32_t i = 0; i < kPrgRamBytes; ++i) {
        prg_ram[i] = snapshot.prg_ram[i];
    }
    if (buffers.ppu.nametable_ram != nullptr && snapshot.nametable_ram.size() >= kNametableRamBytes) {
        auto* nametable =
            buffers.ppu.nametable_ram + static_cast<std::uint64_t>(env) * kNametableRamBytes;
        for (std::uint32_t i = 0; i < kNametableRamBytes; ++i) {
            nametable[i] = snapshot.nametable_ram[i];
        }
    }
    if (buffers.ppu.palette_ram != nullptr && snapshot.palette_ram.size() >= kPaletteRamBytes) {
        auto* palette = buffers.ppu.palette_ram + static_cast<std::uint64_t>(env) * kPaletteRamBytes;
        for (std::uint32_t i = 0; i < kPaletteRamBytes; ++i) {
            palette[i] = snapshot.palette_ram[i];
        }
    }
    if (buffers.ppu.oam != nullptr && snapshot.oam.size() >= kOamBytes) {
        auto* oam = env_oam(buffers, env);
        for (std::uint32_t i = 0; i < kOamBytes; ++i) {
            oam[i] = snapshot.oam[i];
        }
    }
}

}  // namespace nesle::cuda

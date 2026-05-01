#pragma once

#include <cstdint>

#include "nesle/cuda/state.cuh"

#ifdef __CUDACC__
#define NESLE_CUDA_DEVICE_RESET_HD __host__ __device__
#else
#define NESLE_CUDA_DEVICE_RESET_HD
#endif

namespace nesle::cuda {

struct DeviceResetSnapshots {
    std::uint16_t* pc = nullptr;
    std::uint8_t* a = nullptr;
    std::uint8_t* x = nullptr;
    std::uint8_t* y = nullptr;
    std::uint8_t* sp = nullptr;
    std::uint8_t* p = nullptr;
    std::uint64_t* cycles = nullptr;
    std::uint32_t* pending_dma_cycles = nullptr;
    std::uint8_t* controller1_shift = nullptr;
    std::uint8_t* controller1_shift_count = nullptr;
    std::uint8_t* controller1_strobe = nullptr;

    std::uint8_t* ppu_ctrl = nullptr;
    std::uint8_t* ppu_mask = nullptr;
    std::uint8_t* ppu_status = nullptr;
    std::uint8_t* ppu_oam_addr = nullptr;
    std::uint8_t* ppu_nmi_pending = nullptr;
    std::int16_t* ppu_scanline = nullptr;
    std::uint16_t* ppu_dot = nullptr;
    std::uint64_t* ppu_frame = nullptr;
    std::uint16_t* ppu_v = nullptr;
    std::uint16_t* ppu_t = nullptr;
    std::uint8_t* ppu_x = nullptr;
    std::uint8_t* ppu_w = nullptr;
    std::uint8_t* ppu_open_bus = nullptr;
    std::uint8_t* ppu_read_buffer = nullptr;
    std::uint8_t* ppu_scroll_x = nullptr;
    std::uint8_t* ppu_scroll_y = nullptr;

    int* previous_mario_x = nullptr;
    int* previous_mario_time = nullptr;
    float* rewards = nullptr;
    std::uint8_t* done = nullptr;

    std::uint8_t* ram = nullptr;
    std::uint8_t* prg_ram = nullptr;
    std::uint8_t* nametable_ram = nullptr;
    std::uint8_t* palette_ram = nullptr;
    std::uint8_t* oam = nullptr;
};

NESLE_CUDA_DEVICE_RESET_HD inline std::uint8_t* snapshot_ram(DeviceResetSnapshots snapshots,
                                                             std::uint32_t slot) {
    return snapshots.ram + static_cast<std::uint64_t>(slot) * kCpuRamBytes;
}

NESLE_CUDA_DEVICE_RESET_HD inline std::uint8_t* snapshot_prg_ram(
    DeviceResetSnapshots snapshots,
    std::uint32_t slot) {
    return snapshots.prg_ram + static_cast<std::uint64_t>(slot) * kPrgRamBytes;
}

NESLE_CUDA_DEVICE_RESET_HD inline std::uint8_t* snapshot_oam(DeviceResetSnapshots snapshots,
                                                             std::uint32_t slot) {
    return snapshots.oam + static_cast<std::uint64_t>(slot) * kOamBytes;
}

NESLE_CUDA_DEVICE_RESET_HD inline void capture_device_reset_snapshot(
    const BatchBuffers& buffers,
    DeviceResetSnapshots snapshots,
    std::uint32_t env,
    std::uint32_t slot) {
    snapshots.pc[slot] = buffers.cpu.pc[env];
    snapshots.a[slot] = buffers.cpu.a[env];
    snapshots.x[slot] = buffers.cpu.x[env];
    snapshots.y[slot] = buffers.cpu.y[env];
    snapshots.sp[slot] = buffers.cpu.sp[env];
    snapshots.p[slot] = buffers.cpu.p[env];
    snapshots.cycles[slot] = buffers.cpu.cycles[env];
    if (buffers.cpu.pending_dma_cycles != nullptr && snapshots.pending_dma_cycles != nullptr) {
        snapshots.pending_dma_cycles[slot] = buffers.cpu.pending_dma_cycles[env];
    }
    if (buffers.cpu.controller1_shift != nullptr && snapshots.controller1_shift != nullptr) {
        snapshots.controller1_shift[slot] = buffers.cpu.controller1_shift[env];
    }
    if (buffers.cpu.controller1_shift_count != nullptr &&
        snapshots.controller1_shift_count != nullptr) {
        snapshots.controller1_shift_count[slot] = buffers.cpu.controller1_shift_count[env];
    }
    if (buffers.cpu.controller1_strobe != nullptr && snapshots.controller1_strobe != nullptr) {
        snapshots.controller1_strobe[slot] = buffers.cpu.controller1_strobe[env];
    }

    snapshots.ppu_ctrl[slot] = buffers.ppu.ctrl[env];
    snapshots.ppu_mask[slot] = buffers.ppu.mask[env];
    snapshots.ppu_status[slot] = buffers.ppu.status[env];
    snapshots.ppu_oam_addr[slot] = buffers.ppu.oam_addr[env];
    snapshots.ppu_nmi_pending[slot] = buffers.ppu.nmi_pending[env];
    snapshots.ppu_scanline[slot] = buffers.ppu.scanline[env];
    snapshots.ppu_dot[slot] = buffers.ppu.dot[env];
    snapshots.ppu_frame[slot] = buffers.ppu.frame[env];
    if (buffers.ppu.v != nullptr && snapshots.ppu_v != nullptr) {
        snapshots.ppu_v[slot] = buffers.ppu.v[env];
    }
    if (buffers.ppu.t != nullptr && snapshots.ppu_t != nullptr) {
        snapshots.ppu_t[slot] = buffers.ppu.t[env];
    }
    if (buffers.ppu.x != nullptr && snapshots.ppu_x != nullptr) {
        snapshots.ppu_x[slot] = buffers.ppu.x[env];
    }
    if (buffers.ppu.w != nullptr && snapshots.ppu_w != nullptr) {
        snapshots.ppu_w[slot] = buffers.ppu.w[env];
    }
    if (buffers.ppu.open_bus != nullptr && snapshots.ppu_open_bus != nullptr) {
        snapshots.ppu_open_bus[slot] = buffers.ppu.open_bus[env];
    }
    if (buffers.ppu.read_buffer != nullptr && snapshots.ppu_read_buffer != nullptr) {
        snapshots.ppu_read_buffer[slot] = buffers.ppu.read_buffer[env];
    }
    if (buffers.ppu.scroll_x != nullptr && snapshots.ppu_scroll_x != nullptr) {
        snapshots.ppu_scroll_x[slot] = buffers.ppu.scroll_x[env];
    }
    if (buffers.ppu.scroll_y != nullptr && snapshots.ppu_scroll_y != nullptr) {
        snapshots.ppu_scroll_y[slot] = buffers.ppu.scroll_y[env];
    }

    if (buffers.previous_mario_x != nullptr && snapshots.previous_mario_x != nullptr) {
        snapshots.previous_mario_x[slot] = buffers.previous_mario_x[env];
    }
    if (buffers.previous_mario_time != nullptr && snapshots.previous_mario_time != nullptr) {
        snapshots.previous_mario_time[slot] = buffers.previous_mario_time[env];
    }
    if (buffers.rewards != nullptr && snapshots.rewards != nullptr) {
        snapshots.rewards[slot] = buffers.rewards[env];
    }
    if (buffers.done != nullptr && snapshots.done != nullptr) {
        snapshots.done[slot] = buffers.done[env];
    }

    if (buffers.cpu.ram != nullptr && snapshots.ram != nullptr) {
        const auto* ram = env_cpu_ram(buffers, env);
        auto* cached_ram = snapshot_ram(snapshots, slot);
        for (std::uint32_t i = 0; i < kCpuRamBytes; ++i) {
            cached_ram[i] = ram[i];
        }
    }
    if (buffers.cpu.prg_ram != nullptr && snapshots.prg_ram != nullptr) {
        const auto* prg_ram =
            buffers.cpu.prg_ram + static_cast<std::uint64_t>(env) * kPrgRamBytes;
        auto* cached_prg_ram = snapshot_prg_ram(snapshots, slot);
        for (std::uint32_t i = 0; i < kPrgRamBytes; ++i) {
            cached_prg_ram[i] = prg_ram[i];
        }
    }
    if (buffers.ppu.oam != nullptr && snapshots.oam != nullptr) {
        const auto* oam = env_oam(buffers, env);
        auto* cached_oam = snapshot_oam(snapshots, slot);
        for (std::uint32_t i = 0; i < kOamBytes; ++i) {
            cached_oam[i] = oam[i];
        }
    }
}

NESLE_CUDA_DEVICE_RESET_HD inline void restore_device_reset_snapshot(
    BatchBuffers& buffers,
    DeviceResetSnapshots snapshots,
    std::uint32_t env,
    std::uint32_t slot) {
    buffers.cpu.pc[env] = snapshots.pc[slot];
    buffers.cpu.a[env] = snapshots.a[slot];
    buffers.cpu.x[env] = snapshots.x[slot];
    buffers.cpu.y[env] = snapshots.y[slot];
    buffers.cpu.sp[env] = snapshots.sp[slot];
    buffers.cpu.p[env] = snapshots.p[slot];
    buffers.cpu.cycles[env] = snapshots.cycles[slot];
    if (buffers.cpu.pending_dma_cycles != nullptr && snapshots.pending_dma_cycles != nullptr) {
        buffers.cpu.pending_dma_cycles[env] = snapshots.pending_dma_cycles[slot];
    }
    if (buffers.cpu.controller1_shift != nullptr && snapshots.controller1_shift != nullptr) {
        buffers.cpu.controller1_shift[env] = snapshots.controller1_shift[slot];
    }
    if (buffers.cpu.controller1_shift_count != nullptr &&
        snapshots.controller1_shift_count != nullptr) {
        buffers.cpu.controller1_shift_count[env] = snapshots.controller1_shift_count[slot];
    }
    if (buffers.cpu.controller1_strobe != nullptr && snapshots.controller1_strobe != nullptr) {
        buffers.cpu.controller1_strobe[env] = snapshots.controller1_strobe[slot];
    }

    buffers.ppu.ctrl[env] = snapshots.ppu_ctrl[slot];
    buffers.ppu.mask[env] = snapshots.ppu_mask[slot];
    buffers.ppu.status[env] = snapshots.ppu_status[slot];
    buffers.ppu.oam_addr[env] = snapshots.ppu_oam_addr[slot];
    buffers.ppu.nmi_pending[env] = snapshots.ppu_nmi_pending[slot];
    buffers.ppu.scanline[env] = snapshots.ppu_scanline[slot];
    buffers.ppu.dot[env] = snapshots.ppu_dot[slot];
    buffers.ppu.frame[env] = snapshots.ppu_frame[slot];
    if (buffers.ppu.v != nullptr && snapshots.ppu_v != nullptr) {
        buffers.ppu.v[env] = snapshots.ppu_v[slot];
    }
    if (buffers.ppu.t != nullptr && snapshots.ppu_t != nullptr) {
        buffers.ppu.t[env] = snapshots.ppu_t[slot];
    }
    if (buffers.ppu.x != nullptr && snapshots.ppu_x != nullptr) {
        buffers.ppu.x[env] = snapshots.ppu_x[slot];
    }
    if (buffers.ppu.w != nullptr && snapshots.ppu_w != nullptr) {
        buffers.ppu.w[env] = snapshots.ppu_w[slot];
    }
    if (buffers.ppu.open_bus != nullptr && snapshots.ppu_open_bus != nullptr) {
        buffers.ppu.open_bus[env] = snapshots.ppu_open_bus[slot];
    }
    if (buffers.ppu.read_buffer != nullptr && snapshots.ppu_read_buffer != nullptr) {
        buffers.ppu.read_buffer[env] = snapshots.ppu_read_buffer[slot];
    }
    if (buffers.ppu.scroll_x != nullptr && snapshots.ppu_scroll_x != nullptr) {
        buffers.ppu.scroll_x[env] = snapshots.ppu_scroll_x[slot];
    }
    if (buffers.ppu.scroll_y != nullptr && snapshots.ppu_scroll_y != nullptr) {
        buffers.ppu.scroll_y[env] = snapshots.ppu_scroll_y[slot];
    }

    if (buffers.previous_mario_x != nullptr && snapshots.previous_mario_x != nullptr) {
        buffers.previous_mario_x[env] = snapshots.previous_mario_x[slot];
    }
    if (buffers.previous_mario_time != nullptr && snapshots.previous_mario_time != nullptr) {
        buffers.previous_mario_time[env] = snapshots.previous_mario_time[slot];
    }
    if (buffers.rewards != nullptr && snapshots.rewards != nullptr) {
        buffers.rewards[env] = snapshots.rewards[slot];
    }
    if (buffers.done != nullptr && snapshots.done != nullptr) {
        buffers.done[env] = snapshots.done[slot];
    }

    if (buffers.cpu.ram != nullptr && snapshots.ram != nullptr) {
        auto* ram = env_cpu_ram(buffers, env);
        const auto* cached_ram = snapshot_ram(snapshots, slot);
        for (std::uint32_t i = 0; i < kCpuRamBytes; ++i) {
            ram[i] = cached_ram[i];
        }
    }
    if (buffers.cpu.prg_ram != nullptr && snapshots.prg_ram != nullptr) {
        auto* prg_ram = buffers.cpu.prg_ram + static_cast<std::uint64_t>(env) * kPrgRamBytes;
        const auto* cached_prg_ram = snapshot_prg_ram(snapshots, slot);
        for (std::uint32_t i = 0; i < kPrgRamBytes; ++i) {
            prg_ram[i] = cached_prg_ram[i];
        }
    }
    if (buffers.ppu.oam != nullptr && snapshots.oam != nullptr) {
        auto* oam = env_oam(buffers, env);
        const auto* cached_oam = snapshot_oam(snapshots, slot);
        for (std::uint32_t i = 0; i < kOamBytes; ++i) {
            oam[i] = cached_oam[i];
        }
    }
}

}  // namespace nesle::cuda

#undef NESLE_CUDA_DEVICE_RESET_HD

#pragma once

#include <cstdint>

#include "nesle/cuda/state.cuh"

#ifdef __CUDACC__
#define NESLE_CUDA_HD __host__ __device__
#else
#define NESLE_CUDA_HD
#endif

namespace nesle::cuda {

constexpr std::uint32_t kMarioPlayerState = 0x000E;
constexpr std::uint32_t kMarioPlayerFloatState = 0x001D;
constexpr std::uint32_t kMarioEnemyTypeBase = 0x0016;
constexpr std::uint32_t kMarioXPage = 0x006D;
constexpr std::uint32_t kMarioXScreen = 0x0086;
constexpr std::uint32_t kMarioYViewport = 0x00B5;
constexpr std::uint32_t kMarioLives = 0x075A;
constexpr std::uint32_t kMarioGameMode = 0x0770;
constexpr std::uint32_t kMarioTimeDigits = 0x07F8;

struct MarioBatchSnapshot {
    int x_pos = 0;
    int time = 0;
    bool flag_get = false;
    bool is_dying = false;
    bool is_dead = false;
    bool is_game_over = false;
};

struct BatchReward {
    int x = 0;
    int time = 0;
    int death = 0;
    int total = 0;
};

NESLE_CUDA_HD inline int read_bcd_digits(const std::uint8_t* ram,
                                         std::uint32_t address,
                                         std::uint32_t length) {
    int value = 0;
    for (std::uint32_t i = 0; i < length; ++i) {
        value = value * 10 + static_cast<int>(ram[address + i] & 0x0F);
    }
    return value;
}

NESLE_CUDA_HD inline bool is_stage_over(const std::uint8_t* ram) {
    for (std::uint32_t i = 0; i < 5; ++i) {
        const auto enemy = ram[kMarioEnemyTypeBase + i];
        if ((enemy == 0x2D || enemy == 0x31) && ram[kMarioPlayerFloatState] == 3) {
            return true;
        }
    }
    return false;
}

NESLE_CUDA_HD inline MarioBatchSnapshot read_mario_snapshot(const std::uint8_t* ram) {
    MarioBatchSnapshot snapshot;
    snapshot.x_pos = static_cast<int>(ram[kMarioXPage]) * 0x100 +
                     static_cast<int>(ram[kMarioXScreen]);
    snapshot.time = read_bcd_digits(ram, kMarioTimeDigits, 3);
    snapshot.is_dying = ram[kMarioPlayerState] == 0x0B || ram[kMarioYViewport] > 1;
    snapshot.is_dead = ram[kMarioPlayerState] == 0x06;
    snapshot.is_game_over = ram[kMarioLives] == 0xFF;
    snapshot.flag_get = ram[kMarioGameMode] == 2 || is_stage_over(ram);
    return snapshot;
}

NESLE_CUDA_HD inline BatchReward compute_batch_reward(const MarioBatchSnapshot& previous,
                                                      const MarioBatchSnapshot& current) {
    BatchReward reward;
    reward.x = current.x_pos - previous.x_pos;
    if (reward.x < -5 || reward.x > 5) {
        reward.x = 0;
    }

    reward.time = current.time - previous.time;
    if (reward.time > 0) {
        reward.time = 0;
    }

    reward.death = (current.is_dying || current.is_dead) ? -25 : 0;
    reward.total = reward.x + reward.time + reward.death;
    return reward;
}

NESLE_CUDA_HD inline void apply_batch_reward_env(BatchBuffers& buffers, std::uint32_t env) {
    const auto* ram = env_cpu_ram(buffers, env);
    const auto current = read_mario_snapshot(ram);
    MarioBatchSnapshot previous;
    previous.x_pos = buffers.previous_mario_x[env];
    previous.time = buffers.previous_mario_time[env];
    previous.is_dying = buffers.previous_mario_dying[env] != 0;

    const auto reward = compute_batch_reward(previous, current);
    buffers.rewards[env] = static_cast<float>(reward.total);
    buffers.done[env] = (current.flag_get || current.is_dying || current.is_dead ||
                         current.is_game_over)
                            ? 1
                            : 0;
    buffers.previous_mario_x[env] = current.x_pos;
    buffers.previous_mario_time[env] = current.time;
    buffers.previous_mario_dying[env] = (current.is_dying || current.is_dead) ? 1 : 0;
}


NESLE_CUDA_HD inline void cold_reset_console_env(BatchBuffers& buffers, std::uint32_t env) {
    // Read reset vector from PRG ROM.
    std::uint16_t reset_pc = 0;
    if (buffers.cart.prg_rom != nullptr && buffers.cart.prg_rom_size > 0) {
        const auto base = buffers.cart.prg_rom_size == 16u * 1024u ? 0x3FFCu : 0x7FFCu;
        reset_pc = static_cast<std::uint16_t>(
            buffers.cart.prg_rom[base] |
            (static_cast<std::uint16_t>(buffers.cart.prg_rom[base + 1]) << 8));
    }

    // CPU registers.
    buffers.cpu.pc[env] = reset_pc;
    buffers.cpu.a[env] = 0;
    buffers.cpu.x[env] = 0;
    buffers.cpu.y[env] = 0;
    buffers.cpu.sp[env] = 0xFD;
    buffers.cpu.p[env] = 0x24;
    buffers.cpu.cycles[env] = 7;
    buffers.cpu.nmi_pending[env] = 0;
    buffers.cpu.irq_pending[env] = 0;
    buffers.cpu.controller1_shift[env] = 0;
    buffers.cpu.controller1_shift_count[env] = 8;
    buffers.cpu.controller1_strobe[env] = 0;
    buffers.cpu.pending_dma_cycles[env] = 0;

    // CPU RAM.
    auto* ram = env_cpu_ram(buffers, env);
    for (int i = 0; i < kCpuRamBytes; ++i) {
        ram[i] = 0;
    }

    // PRG RAM.
    auto* prg_ram = buffers.cpu.prg_ram + static_cast<std::uint64_t>(env) * kPrgRamBytes;
    for (int i = 0; i < kPrgRamBytes; ++i) {
        prg_ram[i] = 0;
    }

    // PPU state.
    buffers.ppu.ctrl[env] = 0;
    buffers.ppu.mask[env] = 0;
    buffers.ppu.status[env] = 0;
    buffers.ppu.oam_addr[env] = 0;
    buffers.ppu.nmi_pending[env] = 0;
    buffers.ppu.scanline[env] = 0;
    buffers.ppu.dot[env] = 0;
    buffers.ppu.frame[env] = 0;
    buffers.ppu.v[env] = 0;
    buffers.ppu.t[env] = 0;
    buffers.ppu.x[env] = 0;
    buffers.ppu.w[env] = 0;
    buffers.ppu.open_bus[env] = 0;
    buffers.ppu.read_buffer[env] = 0;
    buffers.ppu.scroll_x[env] = 0;
    buffers.ppu.scroll_y[env] = 0;

    // PPU memory.
    auto* nt = buffers.ppu.nametable_ram + static_cast<std::uint64_t>(env) * kNametableRamBytes;
    for (int i = 0; i < kNametableRamBytes; ++i) {
        nt[i] = 0;
    }
    auto* pal = buffers.ppu.palette_ram + static_cast<std::uint64_t>(env) * kPaletteRamBytes;
    for (int i = 0; i < kPaletteRamBytes; ++i) {
        pal[i] = 0;
    }
    auto* oam = env_oam(buffers, env);
    for (int i = 0; i < kOamBytes; ++i) {
        oam[i] = 0;
    }

    // Reward baselines.
    buffers.previous_mario_x[env] = 0;
    buffers.previous_mario_time[env] = 0;
    buffers.previous_mario_dying[env] = 0;
    buffers.rewards[env] = 0.0F;
    buffers.done[env] = 0;
}

NESLE_CUDA_HD inline void cold_reset_synthetic_env(BatchBuffers& buffers, std::uint32_t env) {
    auto* ram = env_cpu_ram(buffers, env);
    for (int i = 0; i < kCpuRamBytes; ++i) {
        ram[i] = 0;
    }
    ram[kMarioXPage] = 1;
    ram[kMarioXScreen] = 2;
    ram[kMarioYViewport] = 1;
    ram[kMarioLives] = 2;
    ram[kMarioPlayerState] = 0;
    ram[kMarioTimeDigits] = 4;
    ram[kMarioTimeDigits + 1] = 0;
    ram[kMarioTimeDigits + 2] = 0;

    buffers.previous_mario_x[env] = 0x100 + 2;
    buffers.previous_mario_time[env] = 400;
    buffers.previous_mario_dying[env] = 0;
    buffers.rewards[env] = 0.0F;
    buffers.done[env] = 0;
}

}  // namespace nesle::cuda

#undef NESLE_CUDA_HD

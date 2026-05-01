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

}  // namespace nesle::cuda

#undef NESLE_CUDA_HD

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace nesle::smb {

constexpr std::size_t kCpuRamBytes = 2048;

constexpr std::size_t kPlayerState = 0x000E;
constexpr std::size_t kPlayerFloatState = 0x001D;
constexpr std::size_t kEnemyTypeBase = 0x0016;
constexpr std::size_t kXPage = 0x006D;
constexpr std::size_t kXScreen = 0x0086;
constexpr std::size_t kYViewport = 0x00B5;
constexpr std::size_t kYPixel = 0x03B8;
constexpr std::size_t kLives = 0x075A;
constexpr std::size_t kStage = 0x075C;
constexpr std::size_t kWorld = 0x075F;
constexpr std::size_t kArea = 0x0760;
constexpr std::size_t kStatus = 0x0756;
constexpr std::size_t kGameMode = 0x0770;
constexpr std::size_t kCoinsDigits = 0x07ED;
constexpr std::size_t kTimeDigits = 0x07F8;
constexpr std::size_t kScoreDigits = 0x07DE;

struct MarioRamState {
    int x_pos = 0;
    int y_pos = 0;
    int time = 0;
    int coins = 0;
    int score = 0;
    int lives = 0;
    int world = 0;
    int stage = 0;
    int area = 0;
    int status_code = 0;
    int player_state = 0;
    int y_viewport = 0;
    bool flag_get = false;
    bool is_dying = false;
    bool is_dead = false;
    bool is_game_over = false;
};

struct RewardComponents {
    int x = 0;
    int time = 0;
    int death = 0;
    int total = 0;
};

[[nodiscard]] MarioRamState read_ram(std::span<const std::uint8_t> ram);
[[nodiscard]] RewardComponents compute_reward(const MarioRamState& previous,
                                              const MarioRamState& current);
[[nodiscard]] std::string status_name(int status_code);
[[nodiscard]] std::string implausible_boot_state_reason(const MarioRamState& state);
[[nodiscard]] bool is_plausible_boot_state(const MarioRamState& state);
void validate_plausible_boot_state(const MarioRamState& state);

}  // namespace nesle::smb

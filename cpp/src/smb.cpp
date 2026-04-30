#include "nesle/smb.hpp"

#include <array>
#include <stdexcept>

namespace nesle::smb {
namespace {

int read_digits(std::span<const std::uint8_t> ram, std::size_t address, std::size_t length) {
    int value = 0;
    for (std::size_t i = 0; i < length; ++i) {
        value = value * 10 + static_cast<int>(ram[address + i] & 0x0F);
    }
    return value;
}

bool is_stage_over(std::span<const std::uint8_t> ram) {
    constexpr std::array<std::uint8_t, 2> kStageOverEnemies = {0x2D, 0x31};
    for (std::size_t i = 0; i < 5; ++i) {
        const auto enemy = ram[kEnemyTypeBase + i];
        if ((enemy == kStageOverEnemies[0] || enemy == kStageOverEnemies[1]) &&
            ram[kPlayerFloatState] == 3) {
            return true;
        }
    }
    return false;
}

}  // namespace

MarioRamState read_ram(std::span<const std::uint8_t> ram) {
    if (ram.size() < kCpuRamBytes) {
        throw std::invalid_argument("Super Mario Bros. RAM view must contain 2048 bytes");
    }

    MarioRamState state;
    state.x_pos = static_cast<int>(ram[kXPage]) * 0x100 + static_cast<int>(ram[kXScreen]);
    state.y_viewport = ram[kYViewport];
    state.y_pos = state.y_viewport < 1 ? 255 + (255 - static_cast<int>(ram[kYPixel]))
                                       : 255 - static_cast<int>(ram[kYPixel]);
    state.time = read_digits(ram, kTimeDigits, 3);
    state.coins = read_digits(ram, kCoinsDigits, 2);
    state.score = read_digits(ram, kScoreDigits, 6);
    state.lives = ram[kLives];
    state.world = static_cast<int>(ram[kWorld]) + 1;
    state.stage = static_cast<int>(ram[kStage]) + 1;
    state.area = static_cast<int>(ram[kArea]) + 1;
    state.status_code = ram[kStatus];
    state.player_state = ram[kPlayerState];
    state.is_dying = state.player_state == 0x0B || state.y_viewport > 1;
    state.is_dead = state.player_state == 0x06;
    state.is_game_over = state.lives == 0xFF;
    state.flag_get = ram[kGameMode] == 2 || is_stage_over(ram);
    return state;
}

RewardComponents compute_reward(const MarioRamState& previous, const MarioRamState& current) {
    RewardComponents reward;
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

std::string status_name(int status_code) {
    if (status_code == 0) {
        return "small";
    }
    if (status_code == 1) {
        return "tall";
    }
    return "fireball";
}

std::string implausible_boot_state_reason(const MarioRamState& state) {
    if (state.world < 1 || state.world > 8) {
        return "Mario world RAM is outside the expected 1..8 range";
    }
    if (state.stage < 1 || state.stage > 4) {
        return "Mario stage RAM is outside the expected 1..4 range";
    }
    if (state.area < 1 || state.area > 32) {
        return "Mario area RAM is outside the expected 1..32 range";
    }
    if (state.time < 0 || state.time > 999) {
        return "Mario timer RAM is outside the expected 0..999 range";
    }
    if (state.coins < 0 || state.coins > 99) {
        return "Mario coin RAM is outside the expected 0..99 range";
    }
    if (state.lives > 99 && state.lives != 0xFF) {
        return "Mario lives RAM is outside the expected range";
    }
    if (state.status_code < 0 || state.status_code > 2) {
        return "Mario power-up status RAM is outside the expected 0..2 range";
    }
    if (state.player_state > 0x0D) {
        return "Mario player-state RAM is outside the expected range";
    }
    if (state.x_pos == 0 && state.time == 0 && state.lives == 0 &&
        state.world == 1 && state.stage == 1 && state.area == 1) {
        return "Mario RAM still looks like an uninitialized all-zero boot state";
    }
    return "";
}

bool is_plausible_boot_state(const MarioRamState& state) {
    return implausible_boot_state_reason(state).empty();
}

void validate_plausible_boot_state(const MarioRamState& state) {
    const auto reason = implausible_boot_state_reason(state);
    if (!reason.empty()) {
        throw std::invalid_argument(reason);
    }
}

}  // namespace nesle::smb

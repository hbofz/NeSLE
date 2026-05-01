#include <cassert>
#include <cstdint>
#include <span>
#include <vector>

#include "nesle/cuda/batch_step.cuh"
#include "nesle/smb.hpp"

namespace {

void set_digits(std::vector<std::uint8_t>& ram,
                std::size_t base,
                int hundreds,
                int tens,
                int ones) {
    ram[base] = static_cast<std::uint8_t>(hundreds);
    ram[base + 1] = static_cast<std::uint8_t>(tens);
    ram[base + 2] = static_cast<std::uint8_t>(ones);
}

void seed_mario_ram(std::vector<std::uint8_t>& ram,
                    std::size_t env,
                    int x_page,
                    int x_screen,
                    int timer,
                    std::uint8_t player_state = 8,
                    std::uint8_t y_viewport = 1) {
    const auto base = env * nesle::cuda::kCpuRamBytes;
    ram[base + nesle::smb::kXPage] = static_cast<std::uint8_t>(x_page);
    ram[base + nesle::smb::kXScreen] = static_cast<std::uint8_t>(x_screen);
    ram[base + nesle::smb::kYViewport] = y_viewport;
    ram[base + nesle::smb::kPlayerState] = player_state;
    ram[base + nesle::smb::kLives] = 2;
    set_digits(ram, base + nesle::smb::kTimeDigits, timer / 100, (timer / 10) % 10, timer % 10);
}

}  // namespace

int main() {
    constexpr std::size_t kNumEnvs = 3;
    std::vector<std::uint8_t> ram(kNumEnvs * nesle::cuda::kCpuRamBytes, 0);
    std::vector<float> rewards(kNumEnvs, 0.0F);
    std::vector<std::uint8_t> done(kNumEnvs, 0);
    std::vector<int> previous_x(kNumEnvs, 0);
    std::vector<int> previous_time(kNumEnvs, 0);

    seed_mario_ram(ram, 0, 0, 43, 399);
    previous_x[0] = 40;
    previous_time[0] = 400;

    seed_mario_ram(ram, 1, 0, 90, 398);
    previous_x[1] = 80;
    previous_time[1] = 399;

    seed_mario_ram(ram, 2, 0, 50, 399, 0x0B);
    previous_x[2] = 48;
    previous_time[2] = 400;

    nesle::cuda::BatchBuffers buffers{};
    buffers.cpu.ram = ram.data();
    buffers.done = done.data();
    buffers.rewards = rewards.data();
    buffers.previous_mario_x = previous_x.data();
    buffers.previous_mario_time = previous_time.data();

    for (std::uint32_t env = 0; env < kNumEnvs; ++env) {
        const auto base = env * nesle::cuda::kCpuRamBytes;
        const auto current = nesle::smb::read_ram(
            std::span<const std::uint8_t>(ram.data() + base, nesle::cuda::kCpuRamBytes));
        nesle::smb::MarioRamState previous{};
        previous.x_pos = previous_x[env];
        previous.time = previous_time[env];
        const auto expected = nesle::smb::compute_reward(previous, current);

        nesle::cuda::apply_batch_reward_env(buffers, env);

        assert(rewards[env] == static_cast<float>(expected.total));
        assert(previous_x[env] == current.x_pos);
        assert(previous_time[env] == current.time);
    }

    assert(rewards[0] == 2.0F);
    assert(done[0] == 0);
    assert(rewards[1] == -1.0F);
    assert(done[1] == 0);
    assert(rewards[2] == -24.0F);
    assert(done[2] == 1);

    return 0;
}

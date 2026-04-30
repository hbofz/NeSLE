#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "nesle/rom.hpp"
#include "nesle/smb.hpp"

namespace {

std::vector<std::uint8_t> make_rom(bool trainer = false) {
    std::vector<std::uint8_t> data = {'N', 'E', 'S', 0x1A, 2, 1, 0, 0};
    data.resize(16, 0);
    if (trainer) {
        data[6] |= 0x04;
        data.insert(data.end(), 512, 'T');
    }
    data.insert(data.end(), 2 * 16 * 1024, 'P');
    data.insert(data.end(), 8 * 1024, 'C');
    return data;
}

}  // namespace

int main() {
    const auto rom = nesle::parse_ines(make_rom());
    assert(rom.metadata.mapper == 0);
    assert(rom.metadata.prg_rom_banks == 2);
    assert(rom.metadata.chr_rom_banks == 1);
    assert(rom.metadata.is_nrom());
    assert(nesle::is_supported_mario_target(rom.metadata));
    assert(rom.prg_rom.size() == 2 * 16 * 1024);
    assert(rom.chr_rom.size() == 8 * 1024);

    bool threw = false;
    try {
        std::vector<std::uint8_t> bad = {'B', 'A', 'D'};
        (void)nesle::parse_ines(bad);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    std::vector<std::uint8_t> ram(nesle::smb::kCpuRamBytes, 0);
    ram[nesle::smb::kXPage] = 1;
    ram[nesle::smb::kXScreen] = 2;
    ram[nesle::smb::kYViewport] = 1;
    ram[nesle::smb::kYPixel] = 100;
    ram[nesle::smb::kLives] = 2;
    ram[nesle::smb::kTimeDigits] = 4;
    ram[nesle::smb::kTimeDigits + 1] = 0;
    ram[nesle::smb::kTimeDigits + 2] = 0;

    auto prev = nesle::smb::read_ram(ram);
    ram[nesle::smb::kXScreen] = 5;
    ram[nesle::smb::kTimeDigits] = 3;
    ram[nesle::smb::kTimeDigits + 1] = 9;
    ram[nesle::smb::kTimeDigits + 2] = 9;
    auto curr = nesle::smb::read_ram(ram);
    auto reward = nesle::smb::compute_reward(prev, curr);
    assert(prev.x_pos == 258);
    assert(curr.x_pos == 261);
    assert(reward.x == 3);
    assert(reward.time == -1);
    assert(reward.total == 2);

    return 0;
}

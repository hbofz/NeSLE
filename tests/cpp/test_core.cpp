#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "nesle/rom.hpp"
#include "nesle/smb.hpp"

namespace {

std::vector<std::uint8_t> make_rom(bool trainer = false,
                                   std::uint8_t prg_banks = 2,
                                   std::uint8_t chr_banks = 1,
                                   std::uint8_t flags6 = 0,
                                   std::uint8_t flags7 = 0) {
    std::vector<std::uint8_t> data = {'N', 'E', 'S', 0x1A, prg_banks, chr_banks, flags6, flags7};
    data.resize(16, 0);
    if (trainer) {
        data[6] |= 0x04;
        data.insert(data.end(), 512, 'T');
    }
    data.insert(data.end(), static_cast<std::size_t>(prg_banks) * 16 * 1024, 'P');
    data.insert(data.end(), static_cast<std::size_t>(chr_banks) * 8 * 1024, 'C');
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
    assert(nesle::unsupported_mario_target_reason(rom.metadata).empty());
    nesle::validate_supported_mario_target(rom.metadata);
    assert(rom.prg_rom.size() == 2 * 16 * 1024);
    assert(rom.chr_rom.size() == 8 * 1024);

    {
        const auto trainer_rom = nesle::parse_ines(make_rom(true));
        assert(!nesle::is_supported_mario_target(trainer_rom.metadata));
        bool trainer_threw = false;
        try {
            nesle::validate_supported_mario_target(trainer_rom.metadata);
        } catch (const std::invalid_argument&) {
            trainer_threw = true;
        }
        assert(trainer_threw);
    }

    {
        const auto mapper1_rom = nesle::parse_ines(make_rom(false, 2, 1, 0x10));
        assert(!nesle::is_supported_mario_target(mapper1_rom.metadata));
        assert(!nesle::unsupported_mario_target_reason(mapper1_rom.metadata).empty());
    }

    {
        const auto nes2_rom = nesle::parse_ines(make_rom(false, 2, 1, 0, 0x08));
        assert(!nesle::is_supported_mario_target(nes2_rom.metadata));
        assert(!nesle::unsupported_mario_target_reason(nes2_rom.metadata).empty());
    }

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
    assert(nesle::smb::is_plausible_boot_state(prev));
    assert(nesle::smb::implausible_boot_state_reason(prev).empty());
    nesle::smb::validate_plausible_boot_state(prev);
    assert(curr.x_pos == 261);
    assert(reward.x == 3);
    assert(reward.time == -1);
    assert(reward.total == 2);

    {
        std::vector<std::uint8_t> blank(nesle::smb::kCpuRamBytes, 0);
        const auto blank_state = nesle::smb::read_ram(blank);
        assert(!nesle::smb::is_plausible_boot_state(blank_state));
        assert(!nesle::smb::implausible_boot_state_reason(blank_state).empty());
        bool boot_threw = false;
        try {
            nesle::smb::validate_plausible_boot_state(blank_state);
        } catch (const std::invalid_argument&) {
            boot_threw = true;
        }
        assert(boot_threw);
    }

    return 0;
}

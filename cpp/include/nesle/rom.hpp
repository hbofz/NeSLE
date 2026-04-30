#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace nesle {

enum class NametableArrangement {
    Vertical,
    Horizontal,
    FourScreen,
};

struct RomMetadata {
    std::uint8_t prg_rom_banks = 0;
    std::uint8_t chr_rom_banks = 0;
    std::uint16_t mapper = 0;
    bool has_trainer = false;
    bool has_battery = false;
    bool is_nes2 = false;
    NametableArrangement nametable_arrangement = NametableArrangement::Vertical;
    std::size_t prg_rom_size = 0;
    std::size_t chr_rom_size = 0;

    [[nodiscard]] bool is_nrom() const noexcept;
};

struct RomImage {
    RomMetadata metadata;
    std::vector<std::uint8_t> prg_rom;
    std::vector<std::uint8_t> chr_rom;
};

[[nodiscard]] RomImage parse_ines(std::span<const std::uint8_t> bytes);
[[nodiscard]] std::string to_string(NametableArrangement arrangement);
[[nodiscard]] bool is_supported_mario_target(const RomMetadata& metadata) noexcept;

}  // namespace nesle

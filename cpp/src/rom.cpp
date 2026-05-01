#include "nesle/rom.hpp"

#include <algorithm>

namespace nesle {
namespace {

constexpr std::size_t kHeaderSize = 16;
constexpr std::size_t kTrainerSize = 512;
constexpr std::size_t kPrgBankSize = 16 * 1024;
constexpr std::size_t kChrBankSize = 8 * 1024;

}  // namespace

bool RomMetadata::is_nrom() const noexcept {
    return mapper == 0 && (prg_rom_banks == 1 || prg_rom_banks == 2);
}

std::string to_string(NametableArrangement arrangement) {
    switch (arrangement) {
        case NametableArrangement::Vertical:
            return "vertical";
        case NametableArrangement::Horizontal:
            return "horizontal";
        case NametableArrangement::FourScreen:
            return "four_screen";
    }
    return "unknown";
}

bool is_supported_mario_target(const RomMetadata& metadata) noexcept {
    return metadata.mapper == 0 &&
           metadata.submapper == 0 &&
           (metadata.prg_rom_banks == 1 || metadata.prg_rom_banks == 2) &&
           metadata.chr_rom_banks == 1 &&
           !metadata.has_trainer;
}

std::string unsupported_mario_target_reason(const RomMetadata& metadata) {
    if (metadata.mapper != 0) {
        return "expected mapper 0/NROM for Super Mario Bros.";
    }
    if (metadata.submapper != 0) {
        return "expected submapper 0 for Super Mario Bros.";
    }
    if (metadata.prg_rom_banks != 1 && metadata.prg_rom_banks != 2) {
        return "expected one or two 16 KB PRG ROM banks for NROM";
    }
    if (metadata.chr_rom_banks != 1) {
        return "expected one 8 KB CHR ROM bank for Super Mario Bros.";
    }
    if (metadata.has_trainer) {
        return "ROM trainers are not supported";
    }
    return "";
}

void validate_supported_mario_target(const RomMetadata& metadata) {
    const auto reason = unsupported_mario_target_reason(metadata);
    if (!reason.empty()) {
        throw std::invalid_argument(reason);
    }
}

RomImage parse_ines(std::span<const std::uint8_t> bytes) {
    if (bytes.size() < kHeaderSize) {
        throw std::invalid_argument("iNES data is shorter than the 16-byte header");
    }

    if (bytes[0] != 'N' || bytes[1] != 'E' || bytes[2] != 'S' || bytes[3] != 0x1A) {
        throw std::invalid_argument("iNES header magic must be NES<EOF>");
    }

    const auto prg_banks = bytes[4];
    const auto chr_banks = bytes[5];
    const auto flags6 = bytes[6];
    const auto flags7 = bytes[7];
    const bool is_nes2 = (flags7 & 0x0C) == 0x08;
    if (is_nes2 && bytes[9] != 0) {
        throw std::invalid_argument("NES 2.0 extended PRG/CHR ROM sizes are not supported yet");
    }

    RomMetadata metadata;
    metadata.prg_rom_banks = prg_banks;
    metadata.chr_rom_banks = chr_banks;
    metadata.mapper = static_cast<std::uint16_t>((flags6 >> 4) | (flags7 & 0xF0));
    metadata.is_nes2 = is_nes2;
    if (metadata.is_nes2) {
        metadata.mapper = static_cast<std::uint16_t>(
            metadata.mapper | (static_cast<std::uint16_t>(bytes[8] & 0x0F) << 8));
        metadata.submapper = static_cast<std::uint8_t>(bytes[8] >> 4);
    }
    metadata.has_trainer = (flags6 & 0x04) != 0;
    metadata.has_battery = (flags6 & 0x02) != 0;
    metadata.nametable_arrangement = (flags6 & 0x08) != 0
                                         ? NametableArrangement::FourScreen
                                         : ((flags6 & 0x01) != 0
                                                ? NametableArrangement::Vertical
                                                : NametableArrangement::Horizontal);
    metadata.prg_rom_size = static_cast<std::size_t>(prg_banks) * kPrgBankSize;
    metadata.chr_rom_size = static_cast<std::size_t>(chr_banks) * kChrBankSize;

    std::size_t offset = kHeaderSize;
    if (metadata.has_trainer) {
        offset += kTrainerSize;
    }

    const std::size_t required = offset + metadata.prg_rom_size + metadata.chr_rom_size;
    if (bytes.size() < required) {
        throw std::invalid_argument("iNES data is truncated for declared PRG/CHR sizes");
    }

    RomImage image;
    image.metadata = metadata;
    image.prg_rom.assign(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                         bytes.begin() + static_cast<std::ptrdiff_t>(offset + metadata.prg_rom_size));
    offset += metadata.prg_rom_size;
    image.chr_rom.assign(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                         bytes.begin() + static_cast<std::ptrdiff_t>(offset + metadata.chr_rom_size));
    return image;
}

}  // namespace nesle

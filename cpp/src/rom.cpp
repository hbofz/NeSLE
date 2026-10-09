#include "nesle/rom.hpp"

#include <algorithm>

namespace nesle {
namespace {

constexpr std::size_t kHeaderSize = 16;
constexpr std::size_t kTrainerSize = 512;
constexpr std::size_t kPrgBankSize = 16 * 1024;
constexpr std::size_t kChrBankSize = 8 * 1024;

std::size_t next_power_of_two(std::size_t value) noexcept {
    std::size_t result = 1;
    while (result < value) {
        result <<= 1u;
    }
    return result;
}

bool is_power_of_two(std::size_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

std::uint8_t page_mask(std::size_t prg_size, std::uint32_t page_bytes) noexcept {
    if (prg_size < page_bytes) {
        return 0;
    }
    // PRG is padded to a power of two before it reaches the device, so the page
    // count is always a power of two and the mask is exact.
    const std::size_t pages = next_power_of_two(prg_size) / page_bytes;
    if (pages < 2) {
        return 0;
    }
    return static_cast<std::uint8_t>(pages - 1);
}

// Boards with mirroring wired to solder pads. The iNES four-screen bit is a
// lie on these carts: there is no third and fourth CIRAM page to select, so the
// bit falls back to the horizontal/vertical choice.
bool two_bit_mirroring(std::uint16_t mapper) noexcept {
    return mapper == 2 || mapper == 11 || mapper == 30 || mapper == 94 || mapper == 180;
}

}  // namespace

bool RomMetadata::is_nrom() const noexcept {
    return mapper == 0 && (prg_rom_banks == 1 || prg_rom_banks == 2);
}

    bool RomMetadata::is_uxrom() const noexcept {
        return mapper == 2 || mapper == 11 || mapper == 30 || mapper == 94 || mapper == 180;
    }


std::string to_string(std::uint16_t mapper) {
    switch (mapper) {
        case 0:
            return "NROM";
        case 2:
            return "UxROM";
        case 11:
            return "Color Dreams (UNROM variant)";
        case 30:
            return "UNROM 512";
        case 31:
            return "NSF (mapper 31)";
        case 32:
            return "Irem G-101";
        case 34:
            return "BNROM / NINA-001";
        case 94:
            return "UN1ROM";
        case 180:
            return "UxROM (Crazy Climber variant)";
        default:
            return "mapper " + std::to_string(mapper);
    }
}

std::string to_string(NametableArrangement arrangement) {
    switch (arrangement) {
        case NametableArrangement::Vertical:
            return "vertical";
        case NametableArrangement::Horizontal:
            return "horizontal";
        case NametableArrangement::FourScreen:
            return "four_screen";
        case NametableArrangement::SingleScreenLower:
            return "single_screen_lower";
        case NametableArrangement::SingleScreenUpper:
            return "single_screen_upper";
    }
    return "unknown";
}

// Window geometry for every board the batched bus knows how to drive. The
// device side never looks at the iNES fields again; it only reads what this
// function computed, which is why adding a board is a host-side change.
MapperLayout describe_mapper(const RomMetadata& metadata) noexcept {
    MapperLayout layout;

    if (metadata.mapper == 0) {
        // NROM: 16 or 32 KB fixed at $8000, no registers.
        layout.supported = metadata.is_nrom() && !metadata.has_trainer;
        return layout;
    }

    // Bus conflicts are destructive: only bits the mask ROM already has set
    // survive a write. Every commercial UxROM title loads its bank from a
    // table of bytes that already hold the wanted value, so the permissive
    // no-conflict behavior boots them all; conflicts are enabled only when the
    // ROM declares them (NES 2.0 submapper 2), which is what FCEUX does too.
    const bool declared_conflicts = metadata.is_nes2 && metadata.submapper == 2;

    switch (metadata.mapper) {
        case 2:   // UxROM / UNROM / UOROM
        case 11:  // Color Dreams: same board, no bus conflicts
        case 30:  // UNROM 512
        case 94:  // UN1ROM: non-consecutive 16 KB pages, value >> 2
        case 180: // Crazy Climber: fixed bank 0 low, switchable window high
            layout.bank_kind = kBankKindUxrom16k;
            layout.window_bytes = 0x4000;
            layout.fixed_bytes = 0x4000;
            layout.window_at_top = metadata.mapper == 180;
            layout.bank_shift = metadata.mapper == 94 ? 2 : 0;
            layout.bus_conflicts = declared_conflicts;
            layout.bank_mask = page_mask(metadata.prg_rom_size, layout.window_bytes);
            if (metadata.mapper == 30) {
                // Bits 5-6 pick an 8 KB CHR-RAM page and bit 7 picks the CIRAM
                // page, but only an 8 KB CHR RAM is allocated, so the CHR page
                // mask stays zero: bits 5-6 are accepted and ignored, exactly
                // as a title that never sets them sees on real hardware.
                layout.chr_page_mask = 0;
                layout.runtime_mirroring = metadata.nametable_arrangement ==
                                           NametableArrangement::SingleScreenLower;
            }
            layout.supported = metadata.prg_rom_size >= 0x8000 &&
                               metadata.prg_rom_size % 0x4000 == 0;
            break;

        case 34:
            if (metadata.chr_rom_size > 0) {
                // NINA-001 (Impossible Mission II): 8 KB PRG window, upper
                // 24 KB fixed, two 4 KB CHR ROM windows, no bus conflicts.
                layout.bank_kind = kBankKindNina8k;
                layout.window_bytes = 0x2000;
                layout.fixed_bytes = 0x6000;
                layout.chr_4k_windows = 1;
                layout.bank_mask = page_mask(metadata.prg_rom_size, layout.window_bytes);
                // The device masks CHR ROM instead of taking a modulo, which
                // is only exact for power-of-two images. Impossible Mission II
                // ships 32 KB here; the board's 16 window positions address up
                // to 64 KB, and anything past the image wraps.
                layout.supported = metadata.chr_rom_size >= 0x2000 &&
                                   is_power_of_two(metadata.chr_rom_size);
            } else {
                // BNROM: one 32 KB window over the whole CPU space, CHR RAM.
                layout.bank_kind = kBankKindBnrom32k;
                layout.window_bytes = 0x8000;
                layout.fixed_bytes = 0;
                layout.bus_conflicts = declared_conflicts;
                layout.bank_mask = page_mask(metadata.prg_rom_size, layout.window_bytes);
                layout.supported = metadata.prg_rom_size >= 0x8000 &&
                                   metadata.prg_rom_size % 0x8000 == 0;
            }
            break;

        default:
            break;
    }
    return layout;
}

std::string unsupported_mapper_reason(const RomMetadata& metadata) {
    if (metadata.mapper == 0) {
        if (metadata.has_trainer) {
            return "ROM trainers are not supported";
        }
        if (!metadata.is_nrom()) {
            return "expected one or two 16 KB PRG ROM banks for NROM";
        }
        return "";
    }
    const auto layout = describe_mapper(metadata);
    if (layout.supported) {
        return "";
    }
std::string reason = "unsupported mapper: " + to_string(metadata.mapper) + " (iNES " +
                     std::to_string(metadata.mapper) + "); NeSLE emulates NROM and the " +
                     "UxROM family (0, 2, 11, 30, 34, 94, 180)";

    if (metadata.prg_rom_size < 0x8000 || metadata.prg_rom_size % 0x2000 != 0) {
        reason += "; this image also has an unsupported PRG size (" +
                  std::to_string(metadata.prg_rom_size / 1024) + " KB)";
    }
    return reason;
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
    const bool vertical = (flags6 & 0x01) != 0;
    const bool four_screen = (flags6 & 0x08) != 0;
    if (metadata.mapper == 30 && four_screen && !vertical) {
        // UNROM 512 reuses the four-screen header bit to ask for one-screen
        // mirroring, with the page selected at run time by the bank register.
        metadata.nametable_arrangement = NametableArrangement::SingleScreenLower;
    } else if (two_bit_mirroring(metadata.mapper)) {
        // UxROM boards (UNROM/UN1ROM/UOROM/UNROM 512/Crazy Climber) wire
        // mirroring to solder pads: only H and V exist, so a four-screen bit
        // cannot be honored. Some translation patches set it anyway (for
        // instance the uBAH Russian Contra), so fall back to the H/V bit
        // rather than folding the screen to vertical.
        metadata.nametable_arrangement =
            vertical ? NametableArrangement::Vertical : NametableArrangement::Horizontal;
    } else {
        metadata.nametable_arrangement = four_screen
                                             ? NametableArrangement::FourScreen
                                             : (vertical ? NametableArrangement::Vertical
                                                         : NametableArrangement::Horizontal);
    }
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

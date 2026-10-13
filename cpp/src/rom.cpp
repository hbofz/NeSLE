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
    return mapper == 2 || mapper == 94;
}

}  // namespace

bool RomMetadata::is_nrom() const noexcept {
    return mapper == 0 && (prg_rom_banks == 1 || prg_rom_banks == 2);
}

    bool RomMetadata::is_uxrom() const noexcept {
        return mapper == 2 || mapper == 94 || mapper == 34;
    }


std::string to_string(std::uint16_t mapper) {
    switch (mapper) {
        case 0:
            return "NROM";
        case 2:
            return "UxROM";
        case 34:
            return "BNROM";
        case 94:
            return "UN1ROM";
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
        case 2:  // UxROM / UNROM / UOROM
        case 94: // UN1ROM: non-consecutive 16 KB pages, value >> 2
            layout.bank_kind = kBankKindUxrom16k;
            layout.window_bytes = 0x4000;
            layout.fixed_bytes = 0x4000;
            layout.bank_shift = metadata.mapper == 94 ? 2 : 0;
            layout.bus_conflicts = declared_conflicts;
            layout.bank_mask = page_mask(metadata.prg_rom_size, layout.window_bytes);
            layout.supported = metadata.prg_rom_size >= 0x8000 &&
                               metadata.prg_rom_size % 0x4000 == 0;
            break;

        case 34:
            // BNROM: one 32 KB window over the whole CPU space, CHR RAM.
            // NINA-001 is the other board behind mapper 34; it needs its own
            // window layout and is deliberately not supported here.
            layout.bank_kind = kBankKindBnrom32k;
            layout.window_bytes = 0x8000;
            layout.fixed_bytes = 0;
            layout.bus_conflicts = declared_conflicts;
            layout.bank_mask = page_mask(metadata.prg_rom_size, layout.window_bytes);
            layout.supported = metadata.chr_rom_size == 0 &&
                               metadata.prg_rom_size >= 0x8000 &&
                               metadata.prg_rom_size % 0x8000 == 0;
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
                         std::to_string(metadata.mapper) +
                         "); NeSLE emulates NROM, UxROM, UN1ROM and BNROM (0, 2, 34, 94)";

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

std::uint32_t reset_vector_base(const MapperLayout& layout,
                                std::size_t real_prg_size) noexcept {
    if (layout.bank_kind == kBankKindBnrom32k) {
        // Bank 0's 32 KB window sits at $8000 at power-on, so the vector is the
        // last four bytes of that window - the first 32 KB of the image - and
        // not of the image.
        return layout.window_bytes - 4u;
    }
    // NROM keeps the whole image at $8000 and UxROM keeps its fixed last 16 KB
    // at $C000, so on both $FFFC is the last four bytes of the image.
    return static_cast<std::uint32_t>(real_prg_size - 4u);
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
    if (two_bit_mirroring(metadata.mapper)) {
        // UxROM boards (UNROM/UN1ROM/UOROM) wire mirroring to solder pads:
        // only H and V exist, so a four-screen bit cannot be honored. Some
        // translation patches set it anyway (for instance the uBAH Russian
        // Contra), so fall back to the H/V bit rather than folding the screen
        // to vertical.
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

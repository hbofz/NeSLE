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
    // UNROM 512 (iNES mapper 30) one-screen mode: header bit 3 with bit 0
    // clear asks for a single screen whose CIRAM page is picked by bit 7 of
    // the bank register. Power-on is always the lower page.
    SingleScreenLower,
    SingleScreenUpper,
};

struct RomMetadata {
    std::uint8_t prg_rom_banks = 0;
    std::uint8_t chr_rom_banks = 0;
    std::uint16_t mapper = 0;
    std::uint8_t submapper = 0;
    bool has_trainer = false;
    bool has_battery = false;
    bool is_nes2 = false;
    NametableArrangement nametable_arrangement = NametableArrangement::Vertical;
    std::size_t prg_rom_size = 0;
    std::size_t chr_rom_size = 0;

    [[nodiscard]] bool is_nrom() const noexcept;
    [[nodiscard]] bool is_uxrom() const noexcept;
};

// Window kinds. Mirrors nesle::cuda::BankKind so host code can switch on the
// same values describe_mapper() writes into MapperLayout::bank_kind.
constexpr std::uint8_t kBankKindNone = 0;
constexpr std::uint8_t kBankKindUxrom16k = 1;
constexpr std::uint8_t kBankKindBnrom32k = 2;

// iNES mapper numbers with a banking path. Mirrored in nesle::cuda.
constexpr std::uint16_t kMapperNrom = 0;
constexpr std::uint16_t kMapperUxrom = 2;
constexpr std::uint16_t kMapperBnrom = 34;
constexpr std::uint16_t kMapperUn1rom = 94;

// Window geometry of a banked cartridge, derived once from the iNES header.
// `window_bytes == 0` means the board has no register at all (NROM), which is
// the flag the device-side bus switches on to keep its zero-indirection path.
struct MapperLayout {
    std::uint32_t window_bytes = 0;   // switchable window at $8000
    std::uint32_t fixed_bytes = 0;    // fixed window above it
    std::uint8_t bank_shift = 0;      // register >> bank_shift before masking
    std::uint8_t bank_mask = 0;       // number of switchable pages - 1
    std::uint8_t bank_kind = 0;       // nesle::cuda::BankKind
    bool bus_conflicts = false;
    bool supported = false;
};

struct RomImage {
    RomMetadata metadata;
    std::vector<std::uint8_t> prg_rom;
    std::vector<std::uint8_t> chr_rom;
};

[[nodiscard]] RomImage parse_ines(std::span<const std::uint8_t> bytes);
[[nodiscard]] std::string to_string(NametableArrangement arrangement);
[[nodiscard]] std::string to_string(std::uint16_t mapper);
[[nodiscard]] MapperLayout describe_mapper(const RomMetadata& metadata) noexcept;
[[nodiscard]] std::string unsupported_mapper_reason(const RomMetadata& metadata);
[[nodiscard]] bool is_supported_mario_target(const RomMetadata& metadata) noexcept;
[[nodiscard]] std::string unsupported_mario_target_reason(const RomMetadata& metadata);
void validate_supported_mario_target(const RomMetadata& metadata);

// Offset of $FFFC/$FFFD in the image at power-on. NROM keeps the whole image at
// $8000 and UxROM keeps its fixed last 16 KB at $C000, so on both the vector is
// the last four bytes of the image. BNROM maps bank 0's 32 KB at $8000 at
// power-on, so its vector is the last four bytes of that window - the first
// 32 KB of the image - and not of the image.
[[nodiscard]] std::uint32_t reset_vector_base(const MapperLayout& layout,
                                              std::size_t real_prg_size) noexcept;

}  // namespace nesle

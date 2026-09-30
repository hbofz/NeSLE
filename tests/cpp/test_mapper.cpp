#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

#include "nesle/console.hpp"
#include "nesle/cuda/batch_bus.cuh"
#include "nesle/rom.hpp"

// Banking for the whole supported family, checked twice per case: the device
// path (read_prg / write_mapper_register / read_chr, i.e. what the CUDA batch
// kernel runs) and the host Console, which implements the same board
// independently. Agreement between the two is the actual assertion; the
// expected bank offsets are only there to catch both being wrong together.
namespace {

struct Fixture {
    std::vector<std::uint8_t> ram = std::vector<std::uint8_t>(nesle::cuda::kCpuRamBytes, 0);
    std::vector<std::uint8_t> prg_ram = std::vector<std::uint8_t>(nesle::cuda::kPrgRamBytes, 0);
    std::vector<std::uint8_t> chr_ram = std::vector<std::uint8_t>(nesle::cuda::kChrRamBytes, 0);
    std::vector<std::uint8_t> nametable =
        std::vector<std::uint8_t>(nesle::cuda::kNametableRamBytes, 0);
    std::vector<std::uint8_t> palette = std::vector<std::uint8_t>(nesle::cuda::kPaletteRamBytes, 0);
    std::vector<std::uint8_t> prg_bank = std::vector<std::uint8_t>(1, 0);
    std::vector<std::uint8_t> chr_bank = std::vector<std::uint8_t>(1, 0);
    std::vector<std::uint8_t> chr_bank_hi = std::vector<std::uint8_t>(1, 0);
    std::vector<std::uint8_t> arrangement = std::vector<std::uint8_t>(1, 0);
    std::vector<std::uint16_t> ppu_v = std::vector<std::uint16_t>(1, 0);
    std::vector<std::uint16_t> ppu_t = std::vector<std::uint16_t>(1, 0);
    std::vector<std::uint8_t> ppu_w = std::vector<std::uint8_t>(1, 0);
    std::vector<std::uint8_t> ppu_x = std::vector<std::uint8_t>(1, 0);
    std::vector<std::uint8_t> ppu_read_buffer = std::vector<std::uint8_t>(1, 0);
    std::vector<std::uint8_t> ppu_open_bus = std::vector<std::uint8_t>(1, 0);
    std::vector<std::uint8_t> ppu_ctrl = std::vector<std::uint8_t>(1, 0);
    std::vector<std::uint8_t> ppu_mask = std::vector<std::uint8_t>(1, 0);
    std::vector<std::uint8_t> ppu_status = std::vector<std::uint8_t>(1, 0);
    std::vector<std::uint8_t> ppu_oam_addr = std::vector<std::uint8_t>(1, 0);
    nesle::cuda::BatchBuffers buffers{};

    void attach(const nesle::RomImage& rom) {
        const auto layout = nesle::describe_mapper(rom.metadata);
        assert(layout.supported);
        assert(rom.prg_rom.size() >= 0x2000);

        // PRG padded to a power of two, exactly as the bindings do it.
        std::size_t padded = 1;
        while (padded < rom.prg_rom.size()) {
            padded <<= 1u;
        }
        padded_prg.assign(padded, 0);
        for (std::size_t i = 0; i < rom.prg_rom.size(); ++i) {
            padded_prg[i] = rom.prg_rom[i];
        }

        buffers.cpu.ram = ram.data();
        buffers.cpu.prg_ram = prg_ram.data();
        buffers.ppu.chr_ram = rom.chr_rom.empty() ? chr_ram.data() : nullptr;
        buffers.ppu.nametable_ram = nametable.data();
        buffers.ppu.palette_ram = palette.data();
        buffers.ppu.v = ppu_v.data();
        buffers.ppu.t = ppu_t.data();
        buffers.ppu.w = ppu_w.data();
        buffers.ppu.x = ppu_x.data();
        buffers.ppu.read_buffer = ppu_read_buffer.data();
        buffers.ppu.open_bus = ppu_open_bus.data();
        buffers.ppu.ctrl = ppu_ctrl.data();
        buffers.ppu.mask = ppu_mask.data();
        buffers.ppu.status = ppu_status.data();
        buffers.ppu.oam_addr = ppu_oam_addr.data();
        buffers.mapper.prg_bank = prg_bank.data();
        buffers.mapper.chr_bank = chr_bank.data();
        buffers.mapper.chr_bank_hi = chr_bank_hi.data();
        buffers.mapper.nametable_arrangement = layout.runtime_mirroring ? arrangement.data() : nullptr;
        if (buffers.mapper.nametable_arrangement != nullptr) {
            arrangement[0] = nesle::cuda::kNametableSingleScreenLower;
        }

        auto& cart = buffers.cart;
        cart.prg_rom = padded_prg.data();
        cart.prg_rom_size = static_cast<std::uint32_t>(padded_prg.size());
        cart.prg_rom_mask = static_cast<std::uint32_t>(padded_prg.size() - 1u);
        cart.chr_rom = rom.chr_rom.empty() ? nullptr : rom.chr_rom.data();
        cart.chr_rom_size = static_cast<std::uint32_t>(rom.chr_rom.size());
        cart.mapper = rom.metadata.mapper;
        cart.bank_kind = layout.bank_kind;
        cart.bus_conflicts = layout.bus_conflicts ? 1 : 0;
        cart.chr_bank_mask = layout.chr_page_mask;
        cart.mapper_mirroring = layout.runtime_mirroring ? 1 : 0;
        cart.nametable_arrangement =
            layout.runtime_mirroring
                ? nesle::cuda::kNametableSingleScreenLower
                : nesle::cuda::kNametableVertical;
        cart.prg_window_start =
            layout.window_at_top ? 0x10000u - layout.window_bytes : 0x8000u;
        cart.prg_window_mask =
            layout.window_bytes == 0 ? 0u : static_cast<std::uint32_t>(layout.window_bytes - 1u);
        cart.prg_window_shift = window_shift(layout.window_bytes);
        cart.prg_bank_mask = layout.bank_mask;
        cart.prg_bank_shift = layout.bank_shift;
        cart.prg_fixed_base =
            layout.window_at_top
                ? 0u
                : (layout.fixed_bytes == 0 || layout.fixed_bytes >= rom.prg_rom.size()
                       ? 0u
                       : static_cast<std::uint32_t>(rom.prg_rom.size() - layout.fixed_bytes));
        cart.prg_fixed_mask =
            layout.fixed_bytes == 0 ? 0u : static_cast<std::uint32_t>(layout.fixed_bytes - 1u);
    }

    static std::uint32_t window_shift(std::uint32_t bytes) noexcept {
        std::uint32_t shift = 0;
        while ((1u << shift) < bytes) {
            ++shift;
        }
        return shift;
    }

    // One write through both implementations.
    void write_both(nesle::Console& console, std::uint16_t address, std::uint8_t value) {
        nesle::cuda::batch_cpu_write(buffers, 0, address, value);
        console.write(address, value);
    }

    // Assert the switchable window holds `bank`, by comparing against the
    // image at the offset the board's geometry says it should come from.
    // `window_start` is $8000 on every board but mapper 180, whose window sits
    // at $C000.
    void assert_window(const nesle::RomImage& rom,
                       std::uint32_t bank,
                       std::uint32_t window_bytes,
                       std::uint16_t address,
                       std::uint32_t window_start = 0x8000u) const {
        const auto offset =
            static_cast<std::size_t>(bank) * window_bytes + (address - window_start);
        const auto expected = rom.prg_rom[offset];
        assert(nesle::cuda::read_prg(const_cast<nesle::cuda::BatchBuffers&>(buffers), 0, address) ==
               expected);
    }

    void assert_chr_window(const nesle::RomImage& rom,
                           std::uint32_t window_bytes,
                           std::uint16_t address) const {
        const auto offset = static_cast<std::size_t>(window_bytes) + (address & 0x0FFFu);
        assert(nesle::cuda::read_chr(const_cast<nesle::cuda::BatchBuffers&>(buffers), 0, address) ==
               rom.chr_rom[offset]);
    }

    void assert_reads_agree(nesle::Console& console) {
        for (std::uint32_t address = 0x8000; address <= 0xFFFF; address += 7) {
            const auto from_device =
                nesle::cuda::batch_cpu_read(const_cast<nesle::cuda::BatchBuffers&>(buffers), 0,
                                            static_cast<std::uint16_t>(address));
            const auto from_host = console.read(static_cast<std::uint16_t>(address));
            assert(from_device == from_host);
        }
    }

    std::vector<std::uint8_t> padded_prg;
};

nesle::RomImage make_rom(std::uint16_t mapper,
                         std::size_t prg_size,
                         std::size_t chr_size,
                         std::uint8_t flags6 = 0,
                         std::uint8_t submapper = 0) {
    nesle::RomImage rom;
    rom.metadata.mapper = mapper;
    rom.metadata.submapper = submapper;
    rom.metadata.is_nes2 = submapper != 0;
    rom.metadata.prg_rom_banks = static_cast<std::uint8_t>(prg_size / (16 * 1024));
    rom.metadata.chr_rom_banks = static_cast<std::uint8_t>(chr_size / (8 * 1024));
    rom.metadata.prg_rom_size = prg_size;
    rom.metadata.chr_rom_size = chr_size;
    rom.metadata.nametable_arrangement = (flags6 & 0x08) != 0
                                             ? nesle::NametableArrangement::FourScreen
                                             : ((flags6 & 0x01) != 0
                                                    ? nesle::NametableArrangement::Vertical
                                                    : nesle::NametableArrangement::Horizontal);
    if (mapper == nesle::kMapperUnrom512 && (flags6 & 0x08) != 0 && (flags6 & 0x01) == 0) {
        rom.metadata.nametable_arrangement = nesle::NametableArrangement::SingleScreenLower;
    }
    rom.prg_rom.resize(prg_size);
    rom.chr_rom.resize(chr_size);
    // Byte value encodes its own bank, so a read tells you which bank answered.
    for (std::size_t i = 0; i < prg_size; ++i) {
        rom.prg_rom[i] = static_cast<std::uint8_t>(((i / 0x4000) << 4) | (i & 0x0F));
    }
    for (std::size_t i = 0; i < chr_size; ++i) {
        rom.chr_rom[i] = static_cast<std::uint8_t>(((i / 0x1000) << 4) | (i & 0x0F));
    }
    return rom;
}

void check_uxrom() {
    const auto rom = make_rom(nesle::kMapperUxrom, 128 * 1024, 8 * 1024);
    const auto layout = nesle::describe_mapper(rom.metadata);
    assert(layout.bank_kind == nesle::kBankKindUxrom16k);
    assert(layout.window_bytes == 0x4000);
    assert(layout.fixed_bytes == 0x4000);
    assert(layout.bank_mask == 7);
    assert(!layout.bus_conflicts);
    assert(!layout.runtime_mirroring);

    Fixture fixture;
    fixture.attach(rom);
    nesle::Console console(rom);

    // Power-on: bank 0 in the window, the image's last 16 KB fixed.
    assert(nesle::cuda::read_prg(fixture.buffers, 0, 0x8000) == rom.prg_rom[0]);
    assert(nesle::cuda::read_prg(fixture.buffers, 0, 0xC000) == rom.prg_rom[0x1C000]);
    assert(nesle::cuda::read_prg(fixture.buffers, 0, 0xFFFF) == rom.prg_rom[0x1FFFF]);
    fixture.assert_reads_agree(console);

    for (std::uint8_t bank = 0; bank < 8; ++bank) {
        // Any address in $8000-$FFFF latches the register on a UxROM board.
        fixture.write_both(console, 0x8000, bank);
        fixture.assert_window(rom, bank, 0x4000, 0x8000);
        fixture.assert_window(rom, bank, 0x4000, 0xBFFF);
        // The fixed window does not move.
        assert(nesle::cuda::read_prg(fixture.buffers, 0, 0xC000) == rom.prg_rom[0x1C000]);
        fixture.assert_reads_agree(console);
    }

    // The register is masked, so an out-of-range value wraps like hardware.
    fixture.write_both(console, 0xFFFF, 0xFE);
    fixture.assert_window(rom, 6, 0x4000, 0x8000);
    fixture.assert_reads_agree(console);

    // A write to PRG RAM is not a bank write.
    fixture.write_both(console, 0x8000, 0x00);
    fixture.write_both(console, 0x6000, 0x03);
    assert(nesle::cuda::read_prg(fixture.buffers, 0, 0x8000) == 0x00);
    assert(nesle::cuda::batch_cpu_read(fixture.buffers, 0, 0x6000) == 0x03);
    fixture.assert_reads_agree(console);
}

void check_uxrom_180() {
    // Crazy Climber: the switchable 16 KB window sits on top of CPU space and
    // bank 0 is fixed at the bottom, the mirror image of every other UxROM.
    const auto rom = make_rom(nesle::kMapperUxrom180, 128 * 1024, 8 * 1024);
    const auto layout = nesle::describe_mapper(rom.metadata);
    assert(layout.bank_kind == nesle::kBankKindUxrom16k);
    assert(layout.window_at_top);
    assert(layout.window_bytes == 0x4000);
    assert(layout.fixed_bytes == 0x4000);
    assert(layout.bank_mask == 7);

    Fixture fixture;
    fixture.attach(rom);
    nesle::Console console(rom);

    // $8000-$BFFF is permanently bank 0; the window starts at $C000.
    assert(nesle::cuda::read_prg(fixture.buffers, 0, 0x8000) == rom.prg_rom[0x0000]);
    assert(nesle::cuda::read_prg(fixture.buffers, 0, 0xBFFF) == rom.prg_rom[0x3FFF]);
    assert(nesle::cuda::read_prg(fixture.buffers, 0, 0xC000) == rom.prg_rom[0x0000]);
    fixture.assert_reads_agree(console);

    for (std::uint8_t bank = 0; bank < 8; ++bank) {
        fixture.write_both(console, 0x8000, bank);
        fixture.assert_window(rom, bank, 0x4000, 0xC000, 0xC000);
        fixture.assert_window(rom, bank, 0x4000, 0xFFFF, 0xC000);
        // The fixed bank at the bottom does not move.
        assert(nesle::cuda::read_prg(fixture.buffers, 0, 0x8000) == rom.prg_rom[0x0000]);
        fixture.assert_reads_agree(console);
    }

    // The register is shared with the family: it masks identically.
    fixture.write_both(console, 0xC000, 0xFD);
    fixture.assert_window(rom, 5, 0x4000, 0xC000, 0xC000);
    fixture.assert_reads_agree(console);
}

void check_un1rom_shift() {
    // UN1ROM pages are non-consecutive: the register holds page >> 2.
    const auto rom = make_rom(nesle::kMapperUn1rom, 128 * 1024, 8 * 1024);
    const auto layout = nesle::describe_mapper(rom.metadata);
    assert(layout.bank_shift == 2);
    assert(layout.bank_mask == 7);

    Fixture fixture;
    fixture.attach(rom);
    nesle::Console console(rom);

    for (std::uint8_t bank = 0; bank < 8; ++bank) {
        fixture.write_both(console, 0x8000, static_cast<std::uint8_t>(bank << 2));
        fixture.assert_window(rom, bank, 0x4000, 0x8000);
        fixture.assert_reads_agree(console);
    }
}

void check_bnrom() {
    // No CHR ROM means BNROM: one 32 KB window over the whole CPU space.
    const auto rom = make_rom(nesle::kMapperBnrom, 128 * 1024, 0);
    const auto layout = nesle::describe_mapper(rom.metadata);
    assert(layout.bank_kind == nesle::kBankKindBnrom32k);
    assert(layout.window_bytes == 0x8000);
    assert(layout.fixed_bytes == 0);
    assert(layout.bank_mask == 3);

    Fixture fixture;
    fixture.attach(rom);
    nesle::Console console(rom);

    for (std::uint8_t bank = 0; bank < 4; ++bank) {
        fixture.write_both(console, 0x8000, bank);
        fixture.assert_window(rom, bank, 0x8000, 0x8000);
        fixture.assert_window(rom, bank, 0x8000, 0xFFFF);
        fixture.assert_reads_agree(console);
    }

    // CHR RAM: 8 KB fills $0000-$1FFF exactly, so the game writes tiles and
    // reads them back through the PPU.
    fixture.write_both(console, 0x2006, 0x00);
    fixture.write_both(console, 0x2006, 0x00);
    fixture.write_both(console, 0x2007, 0xAB);
    assert(nesle::cuda::read_chr(fixture.buffers, 0, 0x0000) == 0xAB);
    assert(nesle::cuda::read_chr(fixture.buffers, 0, 0x1FFF) == 0x00);
    assert(console.ppu().ppu_read(0x0000) == 0xAB);
    fixture.write_both(console, 0x2006, 0x00);
    fixture.write_both(console, 0x2006, 0x10);
    fixture.write_both(console, 0x2007, 0xCD);
    assert(nesle::cuda::read_chr(fixture.buffers, 0, 0x0010) == 0xCD);
    assert(console.ppu().ppu_read(0x0010) == 0xCD);
    assert(console.ppu().ppu_read(0x0000) == 0xAB);
}

void check_nina() {
    // CHR ROM on mapper 34 means NINA-001: 8 KB window, two 4 KB CHR windows.
    const auto rom = make_rom(nesle::kMapperBnrom, 64 * 1024, 32 * 1024);
    const auto layout = nesle::describe_mapper(rom.metadata);
    assert(layout.bank_kind == nesle::kBankKindNina8k);
    assert(layout.window_bytes == 0x2000);
    assert(layout.fixed_bytes == 0x6000);
    assert(layout.bank_mask == 7);
    assert(!layout.bus_conflicts);

    Fixture fixture;
    fixture.attach(rom);
    nesle::Console console(rom);

    for (std::uint8_t bank = 0; bank < 8; ++bank) {
        fixture.write_both(console, 0x7FFD, bank);
        fixture.assert_window(rom, bank, 0x2000, 0x8000);
        fixture.assert_window(rom, bank, 0x2000, 0x9FFF);
        // Upper 24 KB is fixed at the top of the image.
        assert(nesle::cuda::read_prg(fixture.buffers, 0, 0xA000) == rom.prg_rom[0x10000 - 0x6000]);
        fixture.assert_reads_agree(console);
    }

    // Register reads return the register, and the write also lands in the
    // PRG RAM cell at that address.
    fixture.write_both(console, 0x7FFD, 0x05);
    assert(nesle::cuda::batch_cpu_read(fixture.buffers, 0, 0x7FFD) == 0x05);
    assert(console.read(0x7FFD) == 0x05);
    // $6000 is ordinary RAM on this board and does not disturb the bank.
    assert(nesle::cuda::batch_cpu_read(fixture.buffers, 0, 0x6000) == 0x00);
    fixture.write_both(console, 0x6000, 0x77);
    assert(nesle::cuda::batch_cpu_read(fixture.buffers, 0, 0x6000) == 0x77);
    assert(nesle::cuda::batch_cpu_read(fixture.buffers, 0, 0x7FFD) == 0x05);
    fixture.assert_window(rom, 5, 0x2000, 0x8000);

    // Two independent 4 KB CHR ROM windows.
    for (std::uint8_t low = 0; low < 8; ++low) {
        for (std::uint8_t high = 0; high < 8; ++high) {
            fixture.write_both(console, 0x7FFE, low);
            fixture.write_both(console, 0x7FFF, high);
            fixture.assert_chr_window(rom, static_cast<std::uint32_t>(low) << 12, 0x0000);
            fixture.assert_chr_window(rom, static_cast<std::uint32_t>(low) << 12, 0x0FFF);
            fixture.assert_chr_window(rom, static_cast<std::uint32_t>(high) << 12, 0x1000);
            fixture.assert_chr_window(rom, static_cast<std::uint32_t>(high) << 12, 0x1FFF);
            assert(console.ppu().ppu_read(0x0000) == nesle::cuda::read_chr(fixture.buffers, 0, 0));
            assert(console.ppu().ppu_read(0x1FFF) ==
                   nesle::cuda::read_chr(fixture.buffers, 0, 0x1FFF));
        }
    }
}

void check_unrom512() {
    // Header bit 3 with bit 0 clear: one-screen mirroring, page from bit 7.
    const auto rom = make_rom(nesle::kMapperUnrom512, 128 * 1024, 0, 0x08);
    assert(rom.metadata.nametable_arrangement == nesle::NametableArrangement::SingleScreenLower);
    const auto layout = nesle::describe_mapper(rom.metadata);
    assert(layout.bank_kind == nesle::kBankKindUxrom16k);
    assert(layout.runtime_mirroring);

    Fixture fixture;
    fixture.attach(rom);
    nesle::Console console(rom);

    for (std::uint8_t bank = 0; bank < 8; ++bank) {
        fixture.write_both(console, 0x8000, bank);
        fixture.assert_window(rom, bank, 0x4000, 0x8000);
        fixture.assert_reads_agree(console);
    }

    // Bit 7 flips which CIRAM page the single screen shows, in both halves.
    assert(nesle::cuda::env_nametable_arrangement(fixture.buffers, 0) ==
           nesle::cuda::kNametableSingleScreenLower);
    // One-screen maps each of the four 1 KB halves onto the selected 1 KB
    // page, so the offset inside the page survives and the page index does not.
    assert(nesle::cuda::mirror_nametable_address(nesle::cuda::kNametableSingleScreenLower, 0x2000) ==
           0x0000);
    assert(nesle::cuda::mirror_nametable_address(nesle::cuda::kNametableSingleScreenLower, 0x2100) ==
           0x0100);
    assert(nesle::cuda::mirror_nametable_address(nesle::cuda::kNametableSingleScreenLower, 0x2800) ==
           0x0000);

    fixture.write_both(console, 0x8000, 0x80);
    assert(nesle::cuda::env_nametable_arrangement(fixture.buffers, 0) ==
           nesle::cuda::kNametableSingleScreenUpper);
    assert(nesle::cuda::mirror_nametable_address(nesle::cuda::kNametableSingleScreenUpper, 0x2000) ==
           0x0400);
    assert(nesle::cuda::mirror_nametable_address(nesle::cuda::kNametableSingleScreenUpper, 0x2100) ==
           0x0500);
    assert(nesle::cuda::mirror_nametable_address(nesle::cuda::kNametableSingleScreenUpper, 0x2800) ==
           0x0400);
    // All four nametables alias onto that one page.
    assert(nesle::cuda::mirror_nametable_address(nesle::cuda::kNametableSingleScreenUpper, 0x2000) ==
           nesle::cuda::mirror_nametable_address(nesle::cuda::kNametableSingleScreenUpper, 0x2400));
    assert(nesle::cuda::mirror_nametable_address(nesle::cuda::kNametableSingleScreenUpper, 0x2000) ==
           nesle::cuda::mirror_nametable_address(nesle::cuda::kNametableSingleScreenUpper, 0x2C00));

    // The nametable write path follows the runtime arrangement.
    fixture.write_both(console, 0x2006, 0x20);
    fixture.write_both(console, 0x2006, 0x21);
    fixture.write_both(console, 0x2007, 0x5A);
    assert(fixture.nametable[0x0421] == 0x5A);
    assert(console.ppu().ppu_read(0x2021) == 0x5A);
    assert(console.ppu().ppu_read(0x2821) == 0x5A);  // all four alias the one page
}

void check_bus_conflicts() {
    // Conflicts only when the ROM declares them (NES 2.0 submapper 2).
    const auto plain = make_rom(nesle::kMapperUxrom, 64 * 1024, 8 * 1024);
    assert(!nesle::describe_mapper(plain.metadata).bus_conflicts);

    auto conflicting = make_rom(nesle::kMapperUxrom, 64 * 1024, 8 * 1024, 0, 2);
    assert(nesle::describe_mapper(conflicting.metadata).bus_conflicts);

    Fixture fixture;
    fixture.attach(conflicting);
    nesle::Console console(conflicting);
    // Only bits the mask ROM already has set survive, so a full write to a
    // byte holding 0x13 latches 0x13, which the 4-bank mask turns into bank 3.
    const auto mask_byte = conflicting.prg_rom[0x4123];
    assert(mask_byte != 0);
    fixture.write_both(console, 0xC123, 0xFF);
    assert(nesle::cuda::read_prg(fixture.buffers, 0, 0x8000) ==
           conflicting.prg_rom[3 * 0x4000]);
    fixture.assert_reads_agree(console);
    // Writing zero latches zero whatever the mask ROM says.
    fixture.write_both(console, 0xC123, 0x00);
    assert(nesle::cuda::read_prg(fixture.buffers, 0, 0x8000) == conflicting.prg_rom[0]);
    fixture.assert_reads_agree(console);
}

void check_unsupported() {
    for (std::uint16_t mapper : {1, 3, 4, 7, 9, 66, 71, 255}) {
        const auto rom = make_rom(mapper, 64 * 1024, 8 * 1024);
        const auto layout = nesle::describe_mapper(rom.metadata);
        assert(!layout.supported);
        assert(!nesle::unsupported_mapper_reason(rom.metadata).empty());
        bool threw = false;
        try {
            nesle::Console console(rom);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    // Mapper 2 needs at least 32 KB of PRG, on a 16 KB page boundary.
    assert(!nesle::describe_mapper(make_rom(nesle::kMapperUxrom, 16 * 1024, 8 * 1024).metadata)
                .supported);
    assert(!nesle::describe_mapper(make_rom(nesle::kMapperUxrom, 40 * 1024, 8 * 1024).metadata)
                .supported);
    // NINA-001 needs power-of-two CHR ROM to mask the windows.
    assert(!nesle::describe_mapper(make_rom(nesle::kMapperBnrom, 64 * 1024, 24 * 1024).metadata)
                .supported);
    // Color Dreams and UNROM 512 differ from plain UxROM only in how the
    // register is used, so they are supported on the same PRG sizes.
    assert(nesle::describe_mapper(make_rom(nesle::kMapperColorDreams, 64 * 1024, 8 * 1024).metadata)
               .supported);
}

void check_reset() {
    {
        const auto rom = make_rom(nesle::kMapperUxrom, 128 * 1024, 8 * 1024);
        Fixture fixture;
        fixture.attach(rom);
        // No runtime mirroring on this board, so there is no per-env
        // arrangement array to reset.
        assert(fixture.buffers.mapper.nametable_arrangement == nullptr);
        fixture.prg_bank[0] = 5;
        fixture.chr_bank[0] = 3;
        nesle::cuda::reset_mapper_state(fixture.buffers, 0);
        assert(fixture.prg_bank[0] == 0);
        assert(fixture.chr_bank[0] == 0);
        assert(nesle::cuda::read_prg(fixture.buffers, 0, 0x8000) == rom.prg_rom[0]);
    }

    {
        const auto rom = make_rom(nesle::kMapperUnrom512, 128 * 1024, 0, 0x08);
        Fixture fixture;
        fixture.attach(rom);
        assert(fixture.buffers.mapper.nametable_arrangement != nullptr);
        fixture.prg_bank[0] = 5;
        fixture.arrangement[0] = nesle::cuda::kNametableSingleScreenUpper;
        nesle::cuda::reset_mapper_state(fixture.buffers, 0);
        assert(fixture.prg_bank[0] == 0);
        assert(fixture.arrangement[0] == nesle::cuda::kNametableSingleScreenLower);
    }

    // NROM leaves the register arrays alone (they are null there).
    auto nrom_buffers = nesle::cuda::BatchBuffers{};
    nesle::cuda::reset_mapper_state(nrom_buffers, 0);
}

}  // namespace

int main() {
    check_uxrom();
    check_uxrom_180();
    check_un1rom_shift();
    check_bnrom();
    check_nina();
    check_unrom512();
    check_bus_conflicts();
    check_unsupported();
    check_reset();
    return 0;
}

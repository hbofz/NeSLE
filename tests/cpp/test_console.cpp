#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "nesle/console.hpp"
#include "nesle/cpu.hpp"
#include "nesle/cpu_runner.hpp"
#include "nesle/rom.hpp"

namespace {

std::vector<std::uint8_t> make_nrom_bytes(std::uint8_t prg_banks) {
    std::vector<std::uint8_t> data = {'N', 'E', 'S', 0x1A, prg_banks, 1, 0, 0};
    data.resize(16, 0);
    const auto prg_size = static_cast<std::size_t>(prg_banks) * 16 * 1024;
    for (std::size_t i = 0; i < prg_size; ++i) {
        const auto bank = static_cast<std::uint8_t>(i / (16 * 1024));
        data.push_back(static_cast<std::uint8_t>((bank << 7) | (i & 0x7F)));
    }
    data.insert(data.end(), 8 * 1024, 0);
    return data;
}

nesle::RomImage make_nrom(std::uint8_t prg_banks) {
    return nesle::parse_ines(make_nrom_bytes(prg_banks));
}

nesle::RomImage make_program_rom() {
    auto bytes = make_nrom_bytes(2);
    constexpr std::size_t kPrgOffset = 16;
    bytes[kPrgOffset + 0x0000] = 0xA9;  // LDA #$42
    bytes[kPrgOffset + 0x0001] = 0x42;
    bytes[kPrgOffset + 0x0002] = 0x85;  // STA $02
    bytes[kPrgOffset + 0x0003] = 0x02;
    bytes[kPrgOffset + 0x0004] = 0x4C;  // JMP $8004 success trap
    bytes[kPrgOffset + 0x0005] = 0x04;
    bytes[kPrgOffset + 0x0006] = 0x80;
    bytes[kPrgOffset + 0x7FFC] = 0x00;  // RESET vector -> $8000
    bytes[kPrgOffset + 0x7FFD] = 0x80;
    return nesle::parse_ines(bytes);
}

nesle::RomImage make_nmi_rom() {
    auto bytes = make_nrom_bytes(2);
    constexpr std::size_t kPrgOffset = 16;
    bytes[kPrgOffset + 0x0000] = 0xA9;  // LDA #$80
    bytes[kPrgOffset + 0x0001] = 0x80;
    bytes[kPrgOffset + 0x0002] = 0x8D;  // STA $2000
    bytes[kPrgOffset + 0x0003] = 0x00;
    bytes[kPrgOffset + 0x0004] = 0x20;
    bytes[kPrgOffset + 0x0005] = 0x4C;  // JMP $8005
    bytes[kPrgOffset + 0x0006] = 0x05;
    bytes[kPrgOffset + 0x0007] = 0x80;
    bytes[kPrgOffset + 0x1000] = 0xE6;  // INC $00
    bytes[kPrgOffset + 0x1001] = 0x00;
    bytes[kPrgOffset + 0x1002] = 0x40;  // RTI
    bytes[kPrgOffset + 0x7FFA] = 0x00;  // NMI vector -> $9000
    bytes[kPrgOffset + 0x7FFB] = 0x90;
    bytes[kPrgOffset + 0x7FFC] = 0x00;  // RESET vector -> $8000
    bytes[kPrgOffset + 0x7FFD] = 0x80;
    return nesle::parse_ines(bytes);
}

nesle::RomImage make_dma_rom() {
    auto bytes = make_nrom_bytes(2);
    constexpr std::size_t kPrgOffset = 16;
    bytes[kPrgOffset + 0x0000] = 0xA9;  // LDA #$02
    bytes[kPrgOffset + 0x0001] = 0x02;
    bytes[kPrgOffset + 0x0002] = 0x8D;  // STA $4014
    bytes[kPrgOffset + 0x0003] = 0x14;
    bytes[kPrgOffset + 0x0004] = 0x40;
    bytes[kPrgOffset + 0x0005] = 0x4C;  // JMP $8005
    bytes[kPrgOffset + 0x0006] = 0x05;
    bytes[kPrgOffset + 0x0007] = 0x80;
    bytes[kPrgOffset + 0x7FFC] = 0x00;  // RESET vector -> $8000
    bytes[kPrgOffset + 0x7FFD] = 0x80;
    return nesle::parse_ines(bytes);
}

}  // namespace

int main() {
    {
        nesle::Console console(make_nrom(2));
        console.write(0x0001, 0xAB);
        assert(console.read(0x0801) == 0xAB);
        assert(console.read(0x1001) == 0xAB);
        assert(console.read(0x1801) == 0xAB);
    }

    {
        nesle::Console console(make_nrom(2));
        console.write(0x2008, 0x80);
        assert(console.ppu().ctrl() == 0x80);
        console.ppu().set_vblank(true);
        assert(console.ppu().nmi_pending());
        console.write(0x2005, 0x23);
        assert(console.ppu().write_latch());
        const auto status = console.read(0x2002);
        assert((status & 0x80) != 0);
        assert((console.ppu().status() & 0x80) == 0);
        assert(!console.ppu().nmi_pending());
        assert(!console.ppu().write_latch());
    }

    {
        nesle::Console console(make_nrom(2));
        console.ppu().set_vblank(true);
        assert(!console.ppu().nmi_pending());
        console.write(0x2000, 0x80);
        assert(console.ppu().nmi_pending());
    }

    {
        nesle::Console console(make_nrom(2));
        console.write(0x2006, 0x21);
        console.write(0x2006, 0x00);
        console.write(0x2007, 0x99);
        assert(console.ppu().ppu_read(0x2100) == 0x99);
        assert(console.ppu().vram_address() == 0x2101);

        console.write(0x2000, 0x04);
        console.write(0x2006, 0x22);
        console.write(0x2006, 0x00);
        console.write(0x2007, 0x55);
        assert(console.ppu().ppu_read(0x2200) == 0x55);
        assert(console.ppu().vram_address() == 0x2220);
    }

    {
        nesle::Console console(make_nrom(2));
        console.write(0x2003, 0xFE);
        for (std::uint16_t i = 0; i < 256; ++i) {
            console.write(static_cast<std::uint16_t>(0x0200 + i), static_cast<std::uint8_t>(i));
        }
        console.write(0x4014, 0x02);
        assert(console.ppu().oam()[0xFE] == 0x00);
        assert(console.ppu().oam()[0xFF] == 0x01);
        assert(console.ppu().oam()[0x00] == 0x02);
        assert(console.ppu().oam_addr() == 0xFE);
    }

    {
        nesle::Console console(make_nrom(2));
        console.controller1().set_buttons(nesle::ButtonA | nesle::ButtonRight);
        console.write(0x4016, 1);
        assert((console.read(0x4016) & 0x01) == 1);
        console.write(0x4016, 0);
        const int expected[10] = {1, 0, 0, 0, 0, 0, 0, 1, 1, 1};
        for (int bit : expected) {
            assert((console.read(0x4016) & 0x01) == bit);
        }
    }

    {
        nesle::Console nrom128(make_nrom(1));
        assert(nrom128.read(0x8000) == nrom128.read(0xC000));

        nesle::Console nrom256(make_nrom(2));
        assert(nrom256.read(0x8000) == 0x00);
        assert(nrom256.read(0xC000) == 0x80);
        nrom256.write(0x8000, 0xEE);
        assert(nrom256.read(0x8000) == 0x00);
    }

    {
        nesle::Console console(make_program_rom());
        nesle::cpu::CpuState state;
        console.reset_cpu(state);
        assert(state.pc == 0x8000);
        const auto result = nesle::cpu::run_until_trap(state, console, 0x8004, 10);
        assert(result.passed());
        assert(console.read(0x0002) == 0x42);
    }

    {
        nesle::Ppu ppu;
        assert(ppu.scanline() == 0);
        assert(ppu.dot() == 0);
        const auto before_vblank = ppu.step(
            nesle::Ppu::kVblankStartScanline * nesle::Ppu::kDotsPerScanline +
            nesle::Ppu::kVblankFlagDot);
        assert(before_vblank.frames_completed == 0);
        assert((ppu.status() & 0x80) != 0);
        assert(ppu.scanline() == nesle::Ppu::kVblankStartScanline);
        assert(ppu.dot() == nesle::Ppu::kVblankFlagDot);

        const auto to_next_frame =
            (nesle::Ppu::kScanlinesPerFrame - nesle::Ppu::kVblankStartScanline) *
                nesle::Ppu::kDotsPerScanline -
            nesle::Ppu::kVblankFlagDot;
        const auto frame = ppu.step(to_next_frame);
        assert(frame.frames_completed == 1);
        assert(ppu.frame() == 1);
        assert(ppu.scanline() == 0);
        assert(ppu.dot() == 0);
        assert((ppu.status() & 0x80) == 0);
    }

    {
        nesle::Console console(make_nmi_rom());
        nesle::cpu::CpuState state;
        console.reset_cpu(state);
        const auto frame = console.step_frame(state, 20000);
        assert(frame.frames_completed == 1);
        assert(frame.instructions > 0);
        assert(console.read(0x0000) == 1);
        assert(!console.ppu().nmi_pending());
        assert((console.ppu().status() & 0x80) == 0);
    }

    {
        nesle::Console console(make_dma_rom());
        nesle::cpu::CpuState state;
        console.reset_cpu(state);
        for (std::uint16_t i = 0; i < 256; ++i) {
            console.write(static_cast<std::uint16_t>(0x0200 + i), static_cast<std::uint8_t>(0xFF - i));
        }

        auto lda = console.step_cpu_instruction(state);
        assert(lda.cpu_cycles == 2);
        auto dma = console.step_cpu_instruction(state);
        assert(dma.cpu_cycles == 517);
        assert(dma.ppu_cycles == 1551);
        assert(console.ppu().oam()[0] == 0xFF);
        assert(console.ppu().oam()[1] == 0xFE);
    }

    return 0;
}

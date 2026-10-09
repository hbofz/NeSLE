#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "nesle/bus.hpp"
#include "nesle/cpu.hpp"
#include "nesle/rom.hpp"

namespace {

std::vector<std::uint8_t> make_nrom(std::uint8_t prg_banks) {
    std::vector<std::uint8_t> data = {'N', 'E', 'S', 0x1A, prg_banks, 1, 0, 0};
    data.resize(16, 0);
    for (std::size_t i = 0; i < static_cast<std::size_t>(prg_banks) * 16 * 1024; ++i) {
        const auto bank = static_cast<std::uint8_t>(i / (16 * 1024));
        data.push_back(static_cast<std::uint8_t>((bank << 7) | (i & 0x7F)));
    }
    data.insert(data.end(), 8 * 1024, 0);
    return data;
}

nesle::cpu::CpuState make_cpu(std::uint16_t pc) {
    nesle::cpu::CpuState state;
    state.pc = pc;
    state.sp = 0xFD;
    state.p = nesle::cpu::Unused;
    return state;
}

}  // namespace

int main() {
    {
        nesle::FlatBus bus;
        auto state = make_cpu(0x8000);
        bus.write(0x8000, 0xA9);  // LDA #$41
        bus.write(0x8001, 0x41);
        bus.write(0x8002, 0xAA);  // TAX
        bus.write(0x8003, 0xE8);  // INX
        bus.write(0x8004, 0x8D);  // STA $0200
        bus.write(0x8005, 0x00);
        bus.write(0x8006, 0x02);

        auto r1 = nesle::cpu::step(state, bus);
        auto r2 = nesle::cpu::step(state, bus);
        auto r3 = nesle::cpu::step(state, bus);
        auto r4 = nesle::cpu::step(state, bus);
        assert(r1.opcode == 0xA9 && r1.cycles == 2);
        assert(r2.opcode == 0xAA && r2.cycles == 2);
        assert(r3.opcode == 0xE8 && r3.cycles == 2);
        assert(r4.opcode == 0x8D && r4.cycles == 4);
        assert(state.a == 0x41);
        assert(state.x == 0x42);
        assert(bus.read(0x0200) == 0x41);
        assert(state.cycles == 10);
    }

    {
        nesle::FlatBus bus;
        auto state = make_cpu(0x8000);
        bus.write(0x8000, 0x20);  // JSR $9000
        bus.write(0x8001, 0x00);
        bus.write(0x8002, 0x90);
        bus.write(0x9000, 0xA9);  // LDA #$77
        bus.write(0x9001, 0x77);
        bus.write(0x9002, 0x60);  // RTS
        bus.write(0x8003, 0x85);  // STA $10
        bus.write(0x8004, 0x10);

        nesle::cpu::step(state, bus);
        assert(state.pc == 0x9000);
        assert(state.sp == 0xFB);
        nesle::cpu::step(state, bus);
        nesle::cpu::step(state, bus);
        assert(state.pc == 0x8003);
        nesle::cpu::step(state, bus);
        assert(bus.read(0x0010) == 0x77);
    }

    {
        nesle::FlatBus bus;
        auto state = make_cpu(0x80FD);
        bus.write(0x80FD, 0xD0);  // BNE +1, taken and page crossing
        bus.write(0x80FE, 0x01);
        auto result = nesle::cpu::step(state, bus);
        assert(result.cycles == 4);
        assert(state.pc == 0x8100);
    }

    {
        nesle::FlatBus bus;
        auto state = make_cpu(0x8000);
        bus.write(0x8000, 0x38);  // SEC
        bus.write(0x8001, 0xA9);  // LDA #$10
        bus.write(0x8002, 0x10);
        bus.write(0x8003, 0xE9);  // SBC #$01
        bus.write(0x8004, 0x01);
        nesle::cpu::step(state, bus);
        nesle::cpu::step(state, bus);
        nesle::cpu::step(state, bus);
        assert(state.a == 0x0F);
        assert(nesle::cpu::get_flag(state, nesle::cpu::Carry));
    }

    {
        auto image = nesle::parse_ines(make_nrom(1));
        nesle::NromBus bus(image);
        bus.write(0x0001, 0xAB);
        assert(bus.read(0x0801) == 0xAB);
        bus.write(0x2000, 0x80);
        assert(bus.read(0x2008) == 0x80);
        assert(bus.read(0x8000) == bus.read(0xC000));
    }

    {
        auto image = nesle::parse_ines(make_nrom(2));
        nesle::NromBus bus(image);
        assert(bus.read(0x8000) == 0x00);
        assert(bus.read(0xC000) == 0x80);
        assert(bus.read(0xC001) == 0x81);
    }

    // An unimplemented opcode must be a catchable error naming the opcode in hex, not
    // a crash and not a decimal number behind a "0x" prefix. 0xFC is one of the
    // unofficial NOPs a Ricoh 2A03 actually executes, and it used to be reported as
    // "0x252" - a value that does not exist, which sent the hunt for a memory-safety
    // bug into the wrong place entirely.
    {
        auto image = nesle::parse_ines(make_nrom(2));
        for (std::uint8_t opcode :
             {std::uint8_t{0xFC}, std::uint8_t{0x1A}, std::uint8_t{0xEA}}) {
            // Put the opcode at $8000 and point the PC there. reset() would otherwise
            // load the PC from the reset vector, which in make_nrom's filler image is
            // arbitrary.
            auto patched = image;
            patched.prg_rom[0] = opcode;
            nesle::NromBus rom(patched);
            nesle::cpu::CpuState state;
            nesle::cpu::reset(state, rom);
            state.pc = 0x8000;

            bool threw = false;
            std::string message;
            try {
                (void)nesle::cpu::step(state, rom);
            } catch (const std::runtime_error& e) {
                threw = true;
                message = e.what();
            }
            // 0xEA is NOP, a real opcode, so it must not throw.
            if (opcode == 0xEA) {
                assert(!threw);
                continue;
            }
            assert(threw);
            const std::string expected =
                std::string("opcode 0x") + nesle::cpu::detail::to_hex(opcode);
            assert(message.find(expected) != std::string::npos);
        }
    }

    return 0;
}

#include <cassert>
#include <cstdint>
#include <vector>

#include "nesle/bus.hpp"
#include "nesle/cpu.hpp"
#include "nesle/cpu_runner.hpp"

namespace {

nesle::cpu::CpuState state_at(std::uint16_t pc) {
    nesle::cpu::CpuState state;
    state.pc = pc;
    state.p = nesle::cpu::Unused;
    return state;
}

}  // namespace

int main() {
    {
        nesle::FlatBus bus;
        const std::vector<std::uint8_t> program = {
            0xA9, 0x01,        // LDA #$01
            0x8D, 0x00, 0x02,  // STA $0200
            0x4C, 0x05, 0x04,  // JMP $0405 success trap
        };
        bus.load(program, 0x0400);
        auto state = state_at(0x0400);
        const auto result = nesle::cpu::run_until_trap(state, bus, 0x0405, 100);
        assert(result.status == nesle::cpu::RunStatus::SuccessTrap);
        assert(result.passed());
        assert(result.pc == 0x0405);
        assert(bus.read(0x0200) == 0x01);
    }

    {
        nesle::FlatBus bus;
        const std::vector<std::uint8_t> program = {
            0x4C, 0x00, 0x04,  // JMP $0400 failure trap
        };
        bus.load(program, 0x0400);
        auto state = state_at(0x0400);
        const auto result = nesle::cpu::run_until_trap(state, bus, 0x0405, 100);
        assert(result.status == nesle::cpu::RunStatus::FailureTrap);
        assert(!result.passed());
        assert(result.pc == 0x0400);
    }

    {
        nesle::FlatBus bus;
        const std::vector<std::uint8_t> program = {
            0xEA,  // NOP
            0xEA,  // NOP
            0xEA,  // NOP
        };
        bus.load(program, 0x0400);
        auto state = state_at(0x0400);
        const auto result = nesle::cpu::run_until_trap(state, bus, 0x0405, 2);
        assert(result.status == nesle::cpu::RunStatus::Timeout);
        assert(result.instructions == 2);
    }

    {
        nesle::FlatBus bus;
        const std::vector<std::uint8_t> program = {
            0x02,  // illegal on the official-opcode core
        };
        bus.load(program, 0x0400);
        auto state = state_at(0x0400);
        const auto result = nesle::cpu::run_until_trap(state, bus, 0x0405, 100);
        assert(result.status == nesle::cpu::RunStatus::CpuException);
        assert(result.pc == 0x0400);
    }

    {
        nesle::FlatBus bus;
        const std::vector<std::uint8_t> program = {
            0xF8,        // SED
            0x18,        // CLC
            0xA9, 0x45,  // LDA #$45
            0x69, 0x55,  // ADC #$55
        };
        bus.load(program, 0x0400);

        auto ricoh = state_at(0x0400);
        ricoh.variant = nesle::cpu::CpuVariant::Ricoh2A03;
        for (int i = 0; i < 4; ++i) {
            nesle::cpu::step(ricoh, bus);
        }
        assert(ricoh.a == 0x9A);
        assert(!nesle::cpu::get_flag(ricoh, nesle::cpu::Carry));

        auto mos = state_at(0x0400);
        mos.variant = nesle::cpu::CpuVariant::Mos6502;
        for (int i = 0; i < 4; ++i) {
            nesle::cpu::step(mos, bus);
        }
        assert(mos.a == 0x00);
        assert(nesle::cpu::get_flag(mos, nesle::cpu::Carry));
    }

    {
        nesle::FlatBus bus;
        const std::vector<std::uint8_t> program = {
            0xF8,        // SED
            0x38,        // SEC
            0xA9, 0x50,  // LDA #$50
            0xE9, 0x01,  // SBC #$01
        };
        bus.load(program, 0x0400);
        auto state = state_at(0x0400);
        state.variant = nesle::cpu::CpuVariant::Mos6502;
        for (int i = 0; i < 4; ++i) {
            nesle::cpu::step(state, bus);
        }
        assert(state.a == 0x49);
        assert(nesle::cpu::get_flag(state, nesle::cpu::Carry));
    }

    return 0;
}

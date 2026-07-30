#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#ifdef __CUDACC__
#define NESLE_CPU_HD __host__ __device__
#else
#define NESLE_CPU_HD
#endif

namespace nesle::cpu {

enum class CpuVariant {
    Ricoh2A03,
    Mos6502,
};

enum StatusFlag : std::uint8_t {
    Carry = 1u << 0,
    Zero = 1u << 1,
    InterruptDisable = 1u << 2,
    Decimal = 1u << 3,
    Break = 1u << 4,
    Unused = 1u << 5,
    Overflow = 1u << 6,
    Negative = 1u << 7,
};

struct CpuState {
    std::uint16_t pc = 0;
    std::uint8_t a = 0;
    std::uint8_t x = 0;
    std::uint8_t y = 0;
    std::uint8_t sp = 0xFD;
    std::uint8_t p = Unused | InterruptDisable;
    std::uint64_t cycles = 0;
    CpuVariant variant = CpuVariant::Ricoh2A03;
};

struct StepResult {
    std::uint16_t pc = 0;
    std::uint8_t opcode = 0;
    std::uint8_t cycles = 0;
};

[[nodiscard]] NESLE_CPU_HD inline bool get_flag(const CpuState& state, StatusFlag flag) noexcept {
    return (state.p & flag) != 0;
}

NESLE_CPU_HD inline void set_flag(CpuState& state, StatusFlag flag, bool enabled) noexcept {
    if (enabled) {
        state.p |= flag;
    } else {
        state.p &= static_cast<std::uint8_t>(~flag);
    }
    state.p |= Unused;
}

NESLE_CPU_HD inline void set_zn(CpuState& state, std::uint8_t value) noexcept {
    set_flag(state, Zero, value == 0);
    set_flag(state, Negative, (value & 0x80) != 0);
}

template <typename Bus>
[[nodiscard]] NESLE_CPU_HD std::uint8_t read8(Bus& bus, std::uint16_t address) {
    return bus.read(address);
}

template <typename Bus>
NESLE_CPU_HD void write8(Bus& bus, std::uint16_t address, std::uint8_t value) {
    bus.write(address, value);
}

template <typename Bus>
[[nodiscard]] NESLE_CPU_HD std::uint16_t read16(Bus& bus, std::uint16_t address) {
    const auto low = static_cast<std::uint16_t>(read8(bus, address));
    const auto high = static_cast<std::uint16_t>(read8(bus, static_cast<std::uint16_t>(address + 1)));
    return static_cast<std::uint16_t>(low | (high << 8));
}

template <typename Bus>
NESLE_CPU_HD void reset(CpuState& state, Bus& bus) {
    state.a = 0;
    state.x = 0;
    state.y = 0;
    state.sp = 0xFD;
    state.p = Unused | InterruptDisable;
    state.pc = read16(bus, 0xFFFC);
    state.cycles = 7;
}

template <typename Bus>
NESLE_CPU_HD void irq(CpuState& state, Bus& bus) {
    if (get_flag(state, InterruptDisable)) {
        return;
    }
    auto push = [&](std::uint8_t value) {
        write8(bus, static_cast<std::uint16_t>(0x0100 | state.sp), value);
        --state.sp;
    };
    push(static_cast<std::uint8_t>((state.pc >> 8) & 0xFF));
    push(static_cast<std::uint8_t>(state.pc & 0xFF));
    push(static_cast<std::uint8_t>((state.p & ~Break) | Unused));
    set_flag(state, InterruptDisable, true);
    state.pc = read16(bus, 0xFFFE);
    state.cycles += 7;
}

template <typename Bus>
NESLE_CPU_HD void nmi(CpuState& state, Bus& bus) {
    auto push = [&](std::uint8_t value) {
        write8(bus, static_cast<std::uint16_t>(0x0100 | state.sp), value);
        --state.sp;
    };
    push(static_cast<std::uint8_t>((state.pc >> 8) & 0xFF));
    push(static_cast<std::uint8_t>(state.pc & 0xFF));
    push(static_cast<std::uint8_t>((state.p & ~Break) | Unused));
    set_flag(state, InterruptDisable, true);
    state.pc = read16(bus, 0xFFFA);
    state.cycles += 7;
}

// ---------------------------------------------------------------------------
// Table-driven decode.
//
// A single 256-entry constexpr table maps every opcode to its addressing
// mode, operation, base cycle count, and behavior flags. step() dispatches
// through two small switches (effective address, then operation) instead of
// a monolithic 151-case opcode switch. Cycle timing is encoded as:
//   total = base_cycles (table)
//         + 1 if FlagPageCrossPenalty and the indexed fetch crossed a page
//         + branch penalties (+1 taken, +1 more if the target crossed a page)
// Store and memory read-modify-write variants of the indexed modes never take
// the page-cross penalty; their fixed cost is folded into base_cycles.
// ---------------------------------------------------------------------------

enum class AddrMode : std::uint8_t {
    Implied,
    Accumulator,
    Immediate,
    ZeroPage,
    ZeroPageX,
    ZeroPageY,
    Absolute,
    AbsoluteX,
    AbsoluteY,
    IndirectX,
    IndirectY,
    Indirect,  // JMP ($addr) only; honors the 6502 page-wrap fetch bug
    Relative,  // branches; the branch operation fetches its own offset
};

enum class Op : std::uint8_t {
    Illegal,
    LDA, LDX, LDY, STA, STX, STY,
    TAX, TAY, TXA, TYA, TSX, TXS,
    PHA, PHP, PLA, PLP,
    ORA, AND, EOR, ADC, SBC,
    CMP, CPX, CPY, BIT,
    INC, DEC, INX, INY, DEX, DEY,
    ASL, LSR, ROL, ROR,
    JMP, JSR, RTS, RTI, BRK, Branch,
    CLC, SEC, CLI, SEI, CLV, CLD, SED,
    NOP,
};

enum DecodeFlags : std::uint8_t {
    FlagNone = 0,
    FlagPageCrossPenalty = 1u << 0,  // read variants of absx/absy/indy: +1 on page cross
    FlagRmw = 1u << 1,               // memory read-modify-write (shift/inc/dec on memory)
    FlagStore = 1u << 2,             // memory store; fixed cycles, no page-cross penalty
    FlagIllegal = 1u << 3,           // unofficial opcode: traps on device, throws on host
};

struct DecodeEntry {
    AddrMode mode;
    Op op;
    std::uint8_t base_cycles;
    std::uint8_t flags;
};

namespace detail {

struct DecodeTable {
    DecodeEntry entries[256];
};

constexpr DecodeTable make_decode_table() {
    DecodeTable table{};
    for (auto& entry : table.entries) {
        entry = DecodeEntry{AddrMode::Implied, Op::Illegal, 0, FlagIllegal};
    }
    const auto set = [&table](std::uint8_t opcode, AddrMode mode, Op op,
                              std::uint8_t base_cycles, std::uint8_t flags = FlagNone) {
        table.entries[opcode] = DecodeEntry{mode, op, base_cycles, flags};
    };

    set(0x00, AddrMode::Implied, Op::BRK, 7);
    set(0x01, AddrMode::IndirectX, Op::ORA, 6);
    set(0x05, AddrMode::ZeroPage, Op::ORA, 3);
    set(0x06, AddrMode::ZeroPage, Op::ASL, 5, FlagRmw);
    set(0x08, AddrMode::Implied, Op::PHP, 3);
    set(0x09, AddrMode::Immediate, Op::ORA, 2);
    set(0x0A, AddrMode::Accumulator, Op::ASL, 2);
    set(0x0D, AddrMode::Absolute, Op::ORA, 4);
    set(0x0E, AddrMode::Absolute, Op::ASL, 6, FlagRmw);
    set(0x10, AddrMode::Relative, Op::Branch, 2);
    set(0x11, AddrMode::IndirectY, Op::ORA, 5, FlagPageCrossPenalty);
    set(0x15, AddrMode::ZeroPageX, Op::ORA, 4);
    set(0x16, AddrMode::ZeroPageX, Op::ASL, 6, FlagRmw);
    set(0x18, AddrMode::Implied, Op::CLC, 2);
    set(0x19, AddrMode::AbsoluteY, Op::ORA, 4, FlagPageCrossPenalty);
    set(0x1D, AddrMode::AbsoluteX, Op::ORA, 4, FlagPageCrossPenalty);
    set(0x1E, AddrMode::AbsoluteX, Op::ASL, 7, FlagRmw);
    set(0x20, AddrMode::Absolute, Op::JSR, 6);
    set(0x21, AddrMode::IndirectX, Op::AND, 6);
    set(0x24, AddrMode::ZeroPage, Op::BIT, 3);
    set(0x25, AddrMode::ZeroPage, Op::AND, 3);
    set(0x26, AddrMode::ZeroPage, Op::ROL, 5, FlagRmw);
    set(0x28, AddrMode::Implied, Op::PLP, 4);
    set(0x29, AddrMode::Immediate, Op::AND, 2);
    set(0x2A, AddrMode::Accumulator, Op::ROL, 2);
    set(0x2C, AddrMode::Absolute, Op::BIT, 4);
    set(0x2D, AddrMode::Absolute, Op::AND, 4);
    set(0x2E, AddrMode::Absolute, Op::ROL, 6, FlagRmw);
    set(0x30, AddrMode::Relative, Op::Branch, 2);
    set(0x31, AddrMode::IndirectY, Op::AND, 5, FlagPageCrossPenalty);
    set(0x35, AddrMode::ZeroPageX, Op::AND, 4);
    set(0x36, AddrMode::ZeroPageX, Op::ROL, 6, FlagRmw);
    set(0x38, AddrMode::Implied, Op::SEC, 2);
    set(0x39, AddrMode::AbsoluteY, Op::AND, 4, FlagPageCrossPenalty);
    set(0x3D, AddrMode::AbsoluteX, Op::AND, 4, FlagPageCrossPenalty);
    set(0x3E, AddrMode::AbsoluteX, Op::ROL, 7, FlagRmw);
    set(0x40, AddrMode::Implied, Op::RTI, 6);
    set(0x41, AddrMode::IndirectX, Op::EOR, 6);
    set(0x45, AddrMode::ZeroPage, Op::EOR, 3);
    set(0x46, AddrMode::ZeroPage, Op::LSR, 5, FlagRmw);
    set(0x48, AddrMode::Implied, Op::PHA, 3);
    set(0x49, AddrMode::Immediate, Op::EOR, 2);
    set(0x4A, AddrMode::Accumulator, Op::LSR, 2);
    set(0x4C, AddrMode::Absolute, Op::JMP, 3);
    set(0x4D, AddrMode::Absolute, Op::EOR, 4);
    set(0x4E, AddrMode::Absolute, Op::LSR, 6, FlagRmw);
    set(0x50, AddrMode::Relative, Op::Branch, 2);
    set(0x51, AddrMode::IndirectY, Op::EOR, 5, FlagPageCrossPenalty);
    set(0x55, AddrMode::ZeroPageX, Op::EOR, 4);
    set(0x56, AddrMode::ZeroPageX, Op::LSR, 6, FlagRmw);
    set(0x58, AddrMode::Implied, Op::CLI, 2);
    set(0x59, AddrMode::AbsoluteY, Op::EOR, 4, FlagPageCrossPenalty);
    set(0x5D, AddrMode::AbsoluteX, Op::EOR, 4, FlagPageCrossPenalty);
    set(0x5E, AddrMode::AbsoluteX, Op::LSR, 7, FlagRmw);
    set(0x60, AddrMode::Implied, Op::RTS, 6);
    set(0x61, AddrMode::IndirectX, Op::ADC, 6);
    set(0x65, AddrMode::ZeroPage, Op::ADC, 3);
    set(0x66, AddrMode::ZeroPage, Op::ROR, 5, FlagRmw);
    set(0x68, AddrMode::Implied, Op::PLA, 4);
    set(0x69, AddrMode::Immediate, Op::ADC, 2);
    set(0x6A, AddrMode::Accumulator, Op::ROR, 2);
    set(0x6C, AddrMode::Indirect, Op::JMP, 5);
    set(0x6D, AddrMode::Absolute, Op::ADC, 4);
    set(0x6E, AddrMode::Absolute, Op::ROR, 6, FlagRmw);
    set(0x70, AddrMode::Relative, Op::Branch, 2);
    set(0x71, AddrMode::IndirectY, Op::ADC, 5, FlagPageCrossPenalty);
    set(0x75, AddrMode::ZeroPageX, Op::ADC, 4);
    set(0x76, AddrMode::ZeroPageX, Op::ROR, 6, FlagRmw);
    set(0x78, AddrMode::Implied, Op::SEI, 2);
    set(0x79, AddrMode::AbsoluteY, Op::ADC, 4, FlagPageCrossPenalty);
    set(0x7D, AddrMode::AbsoluteX, Op::ADC, 4, FlagPageCrossPenalty);
    set(0x7E, AddrMode::AbsoluteX, Op::ROR, 7, FlagRmw);
    set(0x81, AddrMode::IndirectX, Op::STA, 6, FlagStore);
    set(0x84, AddrMode::ZeroPage, Op::STY, 3, FlagStore);
    set(0x85, AddrMode::ZeroPage, Op::STA, 3, FlagStore);
    set(0x86, AddrMode::ZeroPage, Op::STX, 3, FlagStore);
    set(0x88, AddrMode::Implied, Op::DEY, 2);
    set(0x8A, AddrMode::Implied, Op::TXA, 2);
    set(0x8C, AddrMode::Absolute, Op::STY, 4, FlagStore);
    set(0x8D, AddrMode::Absolute, Op::STA, 4, FlagStore);
    set(0x8E, AddrMode::Absolute, Op::STX, 4, FlagStore);
    set(0x90, AddrMode::Relative, Op::Branch, 2);
    set(0x91, AddrMode::IndirectY, Op::STA, 6, FlagStore);
    set(0x94, AddrMode::ZeroPageX, Op::STY, 4, FlagStore);
    set(0x95, AddrMode::ZeroPageX, Op::STA, 4, FlagStore);
    set(0x96, AddrMode::ZeroPageY, Op::STX, 4, FlagStore);
    set(0x98, AddrMode::Implied, Op::TYA, 2);
    set(0x99, AddrMode::AbsoluteY, Op::STA, 5, FlagStore);
    set(0x9A, AddrMode::Implied, Op::TXS, 2);
    set(0x9D, AddrMode::AbsoluteX, Op::STA, 5, FlagStore);
    set(0xA0, AddrMode::Immediate, Op::LDY, 2);
    set(0xA1, AddrMode::IndirectX, Op::LDA, 6);
    set(0xA2, AddrMode::Immediate, Op::LDX, 2);
    set(0xA4, AddrMode::ZeroPage, Op::LDY, 3);
    set(0xA5, AddrMode::ZeroPage, Op::LDA, 3);
    set(0xA6, AddrMode::ZeroPage, Op::LDX, 3);
    set(0xA8, AddrMode::Implied, Op::TAY, 2);
    set(0xA9, AddrMode::Immediate, Op::LDA, 2);
    set(0xAA, AddrMode::Implied, Op::TAX, 2);
    set(0xAC, AddrMode::Absolute, Op::LDY, 4);
    set(0xAD, AddrMode::Absolute, Op::LDA, 4);
    set(0xAE, AddrMode::Absolute, Op::LDX, 4);
    set(0xB0, AddrMode::Relative, Op::Branch, 2);
    set(0xB1, AddrMode::IndirectY, Op::LDA, 5, FlagPageCrossPenalty);
    set(0xB4, AddrMode::ZeroPageX, Op::LDY, 4);
    set(0xB5, AddrMode::ZeroPageX, Op::LDA, 4);
    set(0xB6, AddrMode::ZeroPageY, Op::LDX, 4);
    set(0xB8, AddrMode::Implied, Op::CLV, 2);
    set(0xB9, AddrMode::AbsoluteY, Op::LDA, 4, FlagPageCrossPenalty);
    set(0xBA, AddrMode::Implied, Op::TSX, 2);
    set(0xBC, AddrMode::AbsoluteX, Op::LDY, 4, FlagPageCrossPenalty);
    set(0xBD, AddrMode::AbsoluteX, Op::LDA, 4, FlagPageCrossPenalty);
    set(0xBE, AddrMode::AbsoluteY, Op::LDX, 4, FlagPageCrossPenalty);
    set(0xC0, AddrMode::Immediate, Op::CPY, 2);
    set(0xC1, AddrMode::IndirectX, Op::CMP, 6);
    set(0xC4, AddrMode::ZeroPage, Op::CPY, 3);
    set(0xC5, AddrMode::ZeroPage, Op::CMP, 3);
    set(0xC6, AddrMode::ZeroPage, Op::DEC, 5, FlagRmw);
    set(0xC8, AddrMode::Implied, Op::INY, 2);
    set(0xC9, AddrMode::Immediate, Op::CMP, 2);
    set(0xCA, AddrMode::Implied, Op::DEX, 2);
    set(0xCC, AddrMode::Absolute, Op::CPY, 4);
    set(0xCD, AddrMode::Absolute, Op::CMP, 4);
    set(0xCE, AddrMode::Absolute, Op::DEC, 6, FlagRmw);
    set(0xD0, AddrMode::Relative, Op::Branch, 2);
    set(0xD1, AddrMode::IndirectY, Op::CMP, 5, FlagPageCrossPenalty);
    set(0xD5, AddrMode::ZeroPageX, Op::CMP, 4);
    set(0xD6, AddrMode::ZeroPageX, Op::DEC, 6, FlagRmw);
    set(0xD8, AddrMode::Implied, Op::CLD, 2);
    set(0xD9, AddrMode::AbsoluteY, Op::CMP, 4, FlagPageCrossPenalty);
    set(0xDD, AddrMode::AbsoluteX, Op::CMP, 4, FlagPageCrossPenalty);
    set(0xDE, AddrMode::AbsoluteX, Op::DEC, 7, FlagRmw);
    set(0xE0, AddrMode::Immediate, Op::CPX, 2);
    set(0xE1, AddrMode::IndirectX, Op::SBC, 6);
    set(0xE4, AddrMode::ZeroPage, Op::CPX, 3);
    set(0xE5, AddrMode::ZeroPage, Op::SBC, 3);
    set(0xE6, AddrMode::ZeroPage, Op::INC, 5, FlagRmw);
    set(0xE8, AddrMode::Implied, Op::INX, 2);
    set(0xE9, AddrMode::Immediate, Op::SBC, 2);
    set(0xEA, AddrMode::Implied, Op::NOP, 2);
    set(0xEC, AddrMode::Absolute, Op::CPX, 4);
    set(0xED, AddrMode::Absolute, Op::SBC, 4);
    set(0xEE, AddrMode::Absolute, Op::INC, 6, FlagRmw);
    set(0xF0, AddrMode::Relative, Op::Branch, 2);
    set(0xF1, AddrMode::IndirectY, Op::SBC, 5, FlagPageCrossPenalty);
    set(0xF5, AddrMode::ZeroPageX, Op::SBC, 4);
    set(0xF6, AddrMode::ZeroPageX, Op::INC, 6, FlagRmw);
    set(0xF8, AddrMode::Implied, Op::SED, 2);
    set(0xF9, AddrMode::AbsoluteY, Op::SBC, 4, FlagPageCrossPenalty);
    set(0xFD, AddrMode::AbsoluteX, Op::SBC, 4, FlagPageCrossPenalty);
    set(0xFE, AddrMode::AbsoluteX, Op::INC, 7, FlagRmw);

    return table;
}

// Plain constexpr (not __constant__) so the header stays host+device
// compilable.
inline constexpr DecodeTable kDecodeTable = make_decode_table();

#ifdef __CUDACC__
// nvcc does not allow device code to odr-use a host constexpr array, so an
// identical table is materialized in device memory. constexpr implies const,
// giving the variable internal linkage: each .cu translation unit carries its
// own ~1 KB copy, which is harmless.
__device__ constexpr DecodeTable kDecodeTableDevice = make_decode_table();
#endif

[[nodiscard]] NESLE_CPU_HD inline DecodeEntry decode(std::uint8_t opcode) noexcept {
#ifdef __CUDA_ARCH__
    return kDecodeTableDevice.entries[opcode];
#else
    return kDecodeTable.entries[opcode];
#endif
}

}  // namespace detail

template <typename Bus>
NESLE_CPU_HD StepResult step(CpuState& state, Bus& bus) {
    const std::uint16_t start_pc = state.pc;
    std::uint8_t cycles = 0;

    auto fetch8 = [&]() {
        const auto value = read8(bus, state.pc);
        state.pc = static_cast<std::uint16_t>(state.pc + 1);
        return value;
    };

    auto fetch16 = [&]() {
        const auto low = static_cast<std::uint16_t>(fetch8());
        const auto high = static_cast<std::uint16_t>(fetch8());
        return static_cast<std::uint16_t>(low | (high << 8));
    };

    auto push = [&](std::uint8_t value) {
        write8(bus, static_cast<std::uint16_t>(0x0100 | state.sp), value);
        --state.sp;
    };

    auto pull = [&]() {
        ++state.sp;
        return read8(bus, static_cast<std::uint16_t>(0x0100 | state.sp));
    };

    auto read16_zp = [&](std::uint8_t address) {
        const auto low = static_cast<std::uint16_t>(read8(bus, address));
        const auto high_address = static_cast<std::uint8_t>(address + 1);
        const auto high = static_cast<std::uint16_t>(read8(bus, high_address));
        return static_cast<std::uint16_t>(low | (high << 8));
    };

    auto read16_jmp_bug = [&](std::uint16_t address) {
        const auto low = static_cast<std::uint16_t>(read8(bus, address));
        const auto high_address =
            static_cast<std::uint16_t>((address & 0xFF00) | ((address + 1) & 0x00FF));
        const auto high = static_cast<std::uint16_t>(read8(bus, high_address));
        return static_cast<std::uint16_t>(low | (high << 8));
    };

    auto page_crossed = [](std::uint16_t a, std::uint16_t b) {
        return (a & 0xFF00) != (b & 0xFF00);
    };

    auto imm = [&]() { return state.pc++; };
    auto zp = [&]() { return static_cast<std::uint16_t>(fetch8()); };
    auto zpx = [&]() { return static_cast<std::uint16_t>(static_cast<std::uint8_t>(fetch8() + state.x)); };
    auto zpy = [&]() { return static_cast<std::uint16_t>(static_cast<std::uint8_t>(fetch8() + state.y)); };
    auto abs = [&]() { return fetch16(); };
    auto absx = [&](bool add_page_cycle) {
        const auto base = fetch16();
        const auto address = static_cast<std::uint16_t>(base + state.x);
        if (add_page_cycle && page_crossed(base, address)) {
            ++cycles;
        }
        return address;
    };
    auto absy = [&](bool add_page_cycle) {
        const auto base = fetch16();
        const auto address = static_cast<std::uint16_t>(base + state.y);
        if (add_page_cycle && page_crossed(base, address)) {
            ++cycles;
        }
        return address;
    };
    auto indx = [&]() {
        const auto pointer = static_cast<std::uint8_t>(fetch8() + state.x);
        return read16_zp(pointer);
    };
    auto indy = [&](bool add_page_cycle) {
        const auto base = read16_zp(fetch8());
        const auto address = static_cast<std::uint16_t>(base + state.y);
        if (add_page_cycle && page_crossed(base, address)) {
            ++cycles;
        }
        return address;
    };

    auto load = [&](std::uint8_t& reg, std::uint8_t value) {
        reg = value;
        set_zn(state, reg);
    };

    auto compare = [&](std::uint8_t reg, std::uint8_t value) {
        const auto result = static_cast<std::uint8_t>(reg - value);
        set_flag(state, Carry, reg >= value);
        set_zn(state, result);
    };

    auto adc = [&](std::uint8_t value) {
        const auto carry = get_flag(state, Carry) ? 1u : 0u;
        auto sum = static_cast<unsigned>(state.a) + static_cast<unsigned>(value) + carry;
        const auto binary_result = static_cast<std::uint8_t>(sum);
        set_flag(state, Overflow, ((~(state.a ^ value) & (state.a ^ binary_result)) & 0x80) != 0);
        if (state.variant == CpuVariant::Mos6502 && get_flag(state, Decimal)) {
            if (((state.a & 0x0F) + (value & 0x0F) + carry) > 9) {
                sum += 0x06;
            }
            if (sum > 0x99) {
                sum += 0x60;
            }
        }
        const auto result = static_cast<std::uint8_t>(sum);
        set_flag(state, Carry, sum > 0xFF);
        state.a = result;
        set_zn(state, state.variant == CpuVariant::Mos6502 && get_flag(state, Decimal) ? binary_result : state.a);
    };

    auto sbc = [&](std::uint8_t value) {
        if (state.variant != CpuVariant::Mos6502 || !get_flag(state, Decimal)) {
            adc(static_cast<std::uint8_t>(value ^ 0xFF));
            return;
        }

        const auto borrow = get_flag(state, Carry) ? 0 : 1;
        const auto binary_diff = static_cast<int>(state.a) - static_cast<int>(value) - borrow;
        const auto binary_result = static_cast<std::uint8_t>(binary_diff);
        int low = static_cast<int>(state.a & 0x0F) - static_cast<int>(value & 0x0F) - borrow;
        int high = static_cast<int>(state.a >> 4) - static_cast<int>(value >> 4);
        if (low < 0) {
            low -= 6;
            --high;
        }
        if (high < 0) {
            high -= 6;
        }

        set_flag(state, Carry, binary_diff >= 0);
        set_flag(state, Overflow, ((state.a ^ value) & (state.a ^ binary_result) & 0x80) != 0);
        state.a = static_cast<std::uint8_t>(((high << 4) & 0xF0) | (low & 0x0F));
        set_zn(state, binary_result);
    };

    auto logical = [&](std::uint8_t value, char op) {
        if (op == 'a') {
            state.a &= value;
        } else if (op == 'e') {
            state.a ^= value;
        } else {
            state.a |= value;
        }
        set_zn(state, state.a);
    };

    auto asl_value = [&](std::uint8_t value) {
        set_flag(state, Carry, (value & 0x80) != 0);
        value = static_cast<std::uint8_t>(value << 1);
        set_zn(state, value);
        return value;
    };

    auto lsr_value = [&](std::uint8_t value) {
        set_flag(state, Carry, (value & 0x01) != 0);
        value = static_cast<std::uint8_t>(value >> 1);
        set_zn(state, value);
        return value;
    };

    auto rol_value = [&](std::uint8_t value) {
        const bool old_carry = get_flag(state, Carry);
        set_flag(state, Carry, (value & 0x80) != 0);
        value = static_cast<std::uint8_t>((value << 1) | (old_carry ? 1 : 0));
        set_zn(state, value);
        return value;
    };

    auto ror_value = [&](std::uint8_t value) {
        const bool old_carry = get_flag(state, Carry);
        set_flag(state, Carry, (value & 0x01) != 0);
        value = static_cast<std::uint8_t>((value >> 1) | (old_carry ? 0x80 : 0));
        set_zn(state, value);
        return value;
    };

    auto branch = [&](bool condition) {
        const auto offset = static_cast<std::int8_t>(fetch8());
        // Base cost (2 cycles) comes from the decode table; only penalties here.
        if (condition) {
            const auto old_pc = state.pc;
            state.pc = static_cast<std::uint16_t>(state.pc + offset);
            ++cycles;
            if (page_crossed(old_pc, state.pc)) {
                ++cycles;
            }
        }
    };

    const auto opcode = fetch8();
    const DecodeEntry entry = detail::decode(opcode);
    const bool page_penalty = (entry.flags & FlagPageCrossPenalty) != 0;

    // Effective-address dispatch.
    std::uint16_t addr = 0;
    switch (entry.mode) {
        case AddrMode::Implied:
        case AddrMode::Accumulator:
        case AddrMode::Relative:
            break;
        case AddrMode::Immediate: addr = imm(); break;
        case AddrMode::ZeroPage: addr = zp(); break;
        case AddrMode::ZeroPageX: addr = zpx(); break;
        case AddrMode::ZeroPageY: addr = zpy(); break;
        case AddrMode::Absolute: addr = abs(); break;
        case AddrMode::AbsoluteX: addr = absx(page_penalty); break;
        case AddrMode::AbsoluteY: addr = absy(page_penalty); break;
        case AddrMode::IndirectX: addr = indx(); break;
        case AddrMode::IndirectY: addr = indy(page_penalty); break;
        case AddrMode::Indirect: addr = read16_jmp_bug(abs()); break;
    }

    // Operation dispatch.
    switch (entry.op) {
        case Op::LDA: load(state.a, read8(bus, addr)); break;
        case Op::LDX: load(state.x, read8(bus, addr)); break;
        case Op::LDY: load(state.y, read8(bus, addr)); break;
        case Op::STA: write8(bus, addr, state.a); break;
        case Op::STX: write8(bus, addr, state.x); break;
        case Op::STY: write8(bus, addr, state.y); break;
        case Op::TAX: load(state.x, state.a); break;
        case Op::TAY: load(state.y, state.a); break;
        case Op::TXA: load(state.a, state.x); break;
        case Op::TYA: load(state.a, state.y); break;
        case Op::TSX: load(state.x, state.sp); break;
        case Op::TXS: state.sp = state.x; break;
        case Op::PHA: push(state.a); break;
        case Op::PHP: push(static_cast<std::uint8_t>(state.p | Break | Unused)); break;
        case Op::PLA: state.a = pull(); set_zn(state, state.a); break;
        case Op::PLP: state.p = static_cast<std::uint8_t>((pull() | Unused) & ~Break); break;
        case Op::ORA: logical(read8(bus, addr), 'o'); break;
        case Op::AND: logical(read8(bus, addr), 'a'); break;
        case Op::EOR: logical(read8(bus, addr), 'e'); break;
        case Op::ADC: adc(read8(bus, addr)); break;
        case Op::SBC: sbc(read8(bus, addr)); break;
        case Op::CMP: compare(state.a, read8(bus, addr)); break;
        case Op::CPX: compare(state.x, read8(bus, addr)); break;
        case Op::CPY: compare(state.y, read8(bus, addr)); break;
        case Op::BIT: {
            const auto v = read8(bus, addr);
            set_flag(state, Zero, (state.a & v) == 0);
            set_flag(state, Negative, (v & 0x80) != 0);
            set_flag(state, Overflow, (v & 0x40) != 0);
            break;
        }
        case Op::INC: {
            const auto v = static_cast<std::uint8_t>(read8(bus, addr) + 1);
            write8(bus, addr, v);
            set_zn(state, v);
            break;
        }
        case Op::DEC: {
            const auto v = static_cast<std::uint8_t>(read8(bus, addr) - 1);
            write8(bus, addr, v);
            set_zn(state, v);
            break;
        }
        case Op::INX: ++state.x; set_zn(state, state.x); break;
        case Op::INY: ++state.y; set_zn(state, state.y); break;
        case Op::DEX: --state.x; set_zn(state, state.x); break;
        case Op::DEY: --state.y; set_zn(state, state.y); break;
        case Op::ASL:
            if (entry.mode == AddrMode::Accumulator) {
                state.a = asl_value(state.a);
            } else {
                write8(bus, addr, asl_value(read8(bus, addr)));
            }
            break;
        case Op::LSR:
            if (entry.mode == AddrMode::Accumulator) {
                state.a = lsr_value(state.a);
            } else {
                write8(bus, addr, lsr_value(read8(bus, addr)));
            }
            break;
        case Op::ROL:
            if (entry.mode == AddrMode::Accumulator) {
                state.a = rol_value(state.a);
            } else {
                write8(bus, addr, rol_value(read8(bus, addr)));
            }
            break;
        case Op::ROR:
            if (entry.mode == AddrMode::Accumulator) {
                state.a = ror_value(state.a);
            } else {
                write8(bus, addr, ror_value(read8(bus, addr)));
            }
            break;
        case Op::JMP: state.pc = addr; break;
        case Op::JSR: {
            const auto ret = static_cast<std::uint16_t>(state.pc - 1);
            push(static_cast<std::uint8_t>(ret >> 8));
            push(static_cast<std::uint8_t>(ret));
            state.pc = addr;
            break;
        }
        case Op::RTS: {
            const auto lo = pull();
            const auto hi = pull();
            state.pc = static_cast<std::uint16_t>((lo | (hi << 8)) + 1);
            break;
        }
        case Op::RTI: {
            state.p = static_cast<std::uint8_t>((pull() | Unused) & ~Break);
            const auto lo = pull();
            const auto hi = pull();
            state.pc = static_cast<std::uint16_t>(lo | (hi << 8));
            break;
        }
        case Op::BRK:
            ++state.pc;
            push(static_cast<std::uint8_t>((state.pc >> 8) & 0xFF));
            push(static_cast<std::uint8_t>(state.pc & 0xFF));
            push(static_cast<std::uint8_t>(state.p | Break | Unused));
            set_flag(state, InterruptDisable, true);
            state.pc = read16(bus, 0xFFFE);
            break;
        case Op::Branch: {
            // Branch opcodes are xxy10000: bits 7-6 pick the flag
            // (N, V, C, Z), bit 5 picks the value that takes the branch.
            bool flag_value = false;
            switch (opcode >> 6) {
                case 0: flag_value = get_flag(state, Negative); break;
                case 1: flag_value = get_flag(state, Overflow); break;
                case 2: flag_value = get_flag(state, Carry); break;
                default: flag_value = get_flag(state, Zero); break;
            }
            branch(flag_value == ((opcode & 0x20) != 0));
            break;
        }
        case Op::CLC: set_flag(state, Carry, false); break;
        case Op::SEC: set_flag(state, Carry, true); break;
        case Op::CLI: set_flag(state, InterruptDisable, false); break;
        case Op::SEI: set_flag(state, InterruptDisable, true); break;
        case Op::CLV: set_flag(state, Overflow, false); break;
        case Op::CLD: set_flag(state, Decimal, false); break;
        case Op::SED: set_flag(state, Decimal, true); break;
        case Op::NOP: break;
        case Op::Illegal:
        default:
#ifdef __CUDA_ARCH__
            asm("trap;");
            break;
#else
            throw std::runtime_error("unimplemented or illegal 6502 opcode 0x" + std::to_string(opcode));
#endif
    }

    cycles = static_cast<std::uint8_t>(cycles + entry.base_cycles);

    state.cycles += cycles;
    return StepResult{start_pc, opcode, cycles};
}

}  // namespace nesle::cpu

#undef NESLE_CPU_HD

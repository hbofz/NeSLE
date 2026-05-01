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
        cycles = 2;
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

    switch (opcode) {
        case 0x00: {  // BRK
            ++state.pc;
            push(static_cast<std::uint8_t>((state.pc >> 8) & 0xFF));
            push(static_cast<std::uint8_t>(state.pc & 0xFF));
            push(static_cast<std::uint8_t>(state.p | Break | Unused));
            set_flag(state, InterruptDisable, true);
            state.pc = read16(bus, 0xFFFE);
            cycles = 7;
            break;
        }
        case 0x01: logical(read8(bus, indx()), 'o'); cycles = 6; break;
        case 0x05: logical(read8(bus, zp()), 'o'); cycles = 3; break;
        case 0x06: { const auto a = zp(); write8(bus, a, asl_value(read8(bus, a))); cycles = 5; break; }
        case 0x08: push(static_cast<std::uint8_t>(state.p | Break | Unused)); cycles = 3; break;
        case 0x09: logical(read8(bus, imm()), 'o'); cycles = 2; break;
        case 0x0A: state.a = asl_value(state.a); cycles = 2; break;
        case 0x0D: logical(read8(bus, abs()), 'o'); cycles = 4; break;
        case 0x0E: { const auto a = abs(); write8(bus, a, asl_value(read8(bus, a))); cycles = 6; break; }
        case 0x10: branch(!get_flag(state, Negative)); break;
        case 0x11: logical(read8(bus, indy(true)), 'o'); cycles += 5; break;
        case 0x15: logical(read8(bus, zpx()), 'o'); cycles = 4; break;
        case 0x16: { const auto a = zpx(); write8(bus, a, asl_value(read8(bus, a))); cycles = 6; break; }
        case 0x18: set_flag(state, Carry, false); cycles = 2; break;
        case 0x19: logical(read8(bus, absy(true)), 'o'); cycles += 4; break;
        case 0x1D: logical(read8(bus, absx(true)), 'o'); cycles += 4; break;
        case 0x1E: { const auto a = absx(false); write8(bus, a, asl_value(read8(bus, a))); cycles = 7; break; }
        case 0x20: { const auto target = abs(); const auto ret = static_cast<std::uint16_t>(state.pc - 1); push(static_cast<std::uint8_t>(ret >> 8)); push(static_cast<std::uint8_t>(ret)); state.pc = target; cycles = 6; break; }
        case 0x21: logical(read8(bus, indx()), 'a'); cycles = 6; break;
        case 0x24: { const auto v = read8(bus, zp()); set_flag(state, Zero, (state.a & v) == 0); set_flag(state, Negative, (v & 0x80) != 0); set_flag(state, Overflow, (v & 0x40) != 0); cycles = 3; break; }
        case 0x25: logical(read8(bus, zp()), 'a'); cycles = 3; break;
        case 0x26: { const auto a = zp(); write8(bus, a, rol_value(read8(bus, a))); cycles = 5; break; }
        case 0x28: state.p = static_cast<std::uint8_t>((pull() | Unused) & ~Break); cycles = 4; break;
        case 0x29: logical(read8(bus, imm()), 'a'); cycles = 2; break;
        case 0x2A: state.a = rol_value(state.a); cycles = 2; break;
        case 0x2C: { const auto v = read8(bus, abs()); set_flag(state, Zero, (state.a & v) == 0); set_flag(state, Negative, (v & 0x80) != 0); set_flag(state, Overflow, (v & 0x40) != 0); cycles = 4; break; }
        case 0x2D: logical(read8(bus, abs()), 'a'); cycles = 4; break;
        case 0x2E: { const auto a = abs(); write8(bus, a, rol_value(read8(bus, a))); cycles = 6; break; }
        case 0x30: branch(get_flag(state, Negative)); break;
        case 0x31: logical(read8(bus, indy(true)), 'a'); cycles += 5; break;
        case 0x35: logical(read8(bus, zpx()), 'a'); cycles = 4; break;
        case 0x36: { const auto a = zpx(); write8(bus, a, rol_value(read8(bus, a))); cycles = 6; break; }
        case 0x38: set_flag(state, Carry, true); cycles = 2; break;
        case 0x39: logical(read8(bus, absy(true)), 'a'); cycles += 4; break;
        case 0x3D: logical(read8(bus, absx(true)), 'a'); cycles += 4; break;
        case 0x3E: { const auto a = absx(false); write8(bus, a, rol_value(read8(bus, a))); cycles = 7; break; }
        case 0x40: state.p = static_cast<std::uint8_t>((pull() | Unused) & ~Break); { const auto lo = pull(); const auto hi = pull(); state.pc = static_cast<std::uint16_t>(lo | (hi << 8)); } cycles = 6; break;
        case 0x41: logical(read8(bus, indx()), 'e'); cycles = 6; break;
        case 0x45: logical(read8(bus, zp()), 'e'); cycles = 3; break;
        case 0x46: { const auto a = zp(); write8(bus, a, lsr_value(read8(bus, a))); cycles = 5; break; }
        case 0x48: push(state.a); cycles = 3; break;
        case 0x49: logical(read8(bus, imm()), 'e'); cycles = 2; break;
        case 0x4A: state.a = lsr_value(state.a); cycles = 2; break;
        case 0x4C: state.pc = abs(); cycles = 3; break;
        case 0x4D: logical(read8(bus, abs()), 'e'); cycles = 4; break;
        case 0x4E: { const auto a = abs(); write8(bus, a, lsr_value(read8(bus, a))); cycles = 6; break; }
        case 0x50: branch(!get_flag(state, Overflow)); break;
        case 0x51: logical(read8(bus, indy(true)), 'e'); cycles += 5; break;
        case 0x55: logical(read8(bus, zpx()), 'e'); cycles = 4; break;
        case 0x56: { const auto a = zpx(); write8(bus, a, lsr_value(read8(bus, a))); cycles = 6; break; }
        case 0x58: set_flag(state, InterruptDisable, false); cycles = 2; break;
        case 0x59: logical(read8(bus, absy(true)), 'e'); cycles += 4; break;
        case 0x5D: logical(read8(bus, absx(true)), 'e'); cycles += 4; break;
        case 0x5E: { const auto a = absx(false); write8(bus, a, lsr_value(read8(bus, a))); cycles = 7; break; }
        case 0x60: { const auto lo = pull(); const auto hi = pull(); state.pc = static_cast<std::uint16_t>((lo | (hi << 8)) + 1); cycles = 6; break; }
        case 0x61: adc(read8(bus, indx())); cycles = 6; break;
        case 0x65: adc(read8(bus, zp())); cycles = 3; break;
        case 0x66: { const auto a = zp(); write8(bus, a, ror_value(read8(bus, a))); cycles = 5; break; }
        case 0x68: state.a = pull(); set_zn(state, state.a); cycles = 4; break;
        case 0x69: adc(read8(bus, imm())); cycles = 2; break;
        case 0x6A: state.a = ror_value(state.a); cycles = 2; break;
        case 0x6C: state.pc = read16_jmp_bug(abs()); cycles = 5; break;
        case 0x6D: adc(read8(bus, abs())); cycles = 4; break;
        case 0x6E: { const auto a = abs(); write8(bus, a, ror_value(read8(bus, a))); cycles = 6; break; }
        case 0x70: branch(get_flag(state, Overflow)); break;
        case 0x71: adc(read8(bus, indy(true))); cycles += 5; break;
        case 0x75: adc(read8(bus, zpx())); cycles = 4; break;
        case 0x76: { const auto a = zpx(); write8(bus, a, ror_value(read8(bus, a))); cycles = 6; break; }
        case 0x78: set_flag(state, InterruptDisable, true); cycles = 2; break;
        case 0x79: adc(read8(bus, absy(true))); cycles += 4; break;
        case 0x7D: adc(read8(bus, absx(true))); cycles += 4; break;
        case 0x7E: { const auto a = absx(false); write8(bus, a, ror_value(read8(bus, a))); cycles = 7; break; }
        case 0x81: write8(bus, indx(), state.a); cycles = 6; break;
        case 0x84: write8(bus, zp(), state.y); cycles = 3; break;
        case 0x85: write8(bus, zp(), state.a); cycles = 3; break;
        case 0x86: write8(bus, zp(), state.x); cycles = 3; break;
        case 0x88: --state.y; set_zn(state, state.y); cycles = 2; break;
        case 0x8A: load(state.a, state.x); cycles = 2; break;
        case 0x8C: write8(bus, abs(), state.y); cycles = 4; break;
        case 0x8D: write8(bus, abs(), state.a); cycles = 4; break;
        case 0x8E: write8(bus, abs(), state.x); cycles = 4; break;
        case 0x90: branch(!get_flag(state, Carry)); break;
        case 0x91: write8(bus, indy(false), state.a); cycles = 6; break;
        case 0x94: write8(bus, zpx(), state.y); cycles = 4; break;
        case 0x95: write8(bus, zpx(), state.a); cycles = 4; break;
        case 0x96: write8(bus, zpy(), state.x); cycles = 4; break;
        case 0x98: load(state.a, state.y); cycles = 2; break;
        case 0x99: write8(bus, absy(false), state.a); cycles = 5; break;
        case 0x9A: state.sp = state.x; cycles = 2; break;
        case 0x9D: write8(bus, absx(false), state.a); cycles = 5; break;
        case 0xA0: load(state.y, read8(bus, imm())); cycles = 2; break;
        case 0xA1: load(state.a, read8(bus, indx())); cycles = 6; break;
        case 0xA2: load(state.x, read8(bus, imm())); cycles = 2; break;
        case 0xA4: load(state.y, read8(bus, zp())); cycles = 3; break;
        case 0xA5: load(state.a, read8(bus, zp())); cycles = 3; break;
        case 0xA6: load(state.x, read8(bus, zp())); cycles = 3; break;
        case 0xA8: load(state.y, state.a); cycles = 2; break;
        case 0xA9: load(state.a, read8(bus, imm())); cycles = 2; break;
        case 0xAA: load(state.x, state.a); cycles = 2; break;
        case 0xAC: load(state.y, read8(bus, abs())); cycles = 4; break;
        case 0xAD: load(state.a, read8(bus, abs())); cycles = 4; break;
        case 0xAE: load(state.x, read8(bus, abs())); cycles = 4; break;
        case 0xB0: branch(get_flag(state, Carry)); break;
        case 0xB1: load(state.a, read8(bus, indy(true))); cycles += 5; break;
        case 0xB4: load(state.y, read8(bus, zpx())); cycles = 4; break;
        case 0xB5: load(state.a, read8(bus, zpx())); cycles = 4; break;
        case 0xB6: load(state.x, read8(bus, zpy())); cycles = 4; break;
        case 0xB8: set_flag(state, Overflow, false); cycles = 2; break;
        case 0xB9: load(state.a, read8(bus, absy(true))); cycles += 4; break;
        case 0xBA: load(state.x, state.sp); cycles = 2; break;
        case 0xBC: load(state.y, read8(bus, absx(true))); cycles += 4; break;
        case 0xBD: load(state.a, read8(bus, absx(true))); cycles += 4; break;
        case 0xBE: load(state.x, read8(bus, absy(true))); cycles += 4; break;
        case 0xC0: compare(state.y, read8(bus, imm())); cycles = 2; break;
        case 0xC1: compare(state.a, read8(bus, indx())); cycles = 6; break;
        case 0xC4: compare(state.y, read8(bus, zp())); cycles = 3; break;
        case 0xC5: compare(state.a, read8(bus, zp())); cycles = 3; break;
        case 0xC6: { const auto a = zp(); const auto v = static_cast<std::uint8_t>(read8(bus, a) - 1); write8(bus, a, v); set_zn(state, v); cycles = 5; break; }
        case 0xC8: ++state.y; set_zn(state, state.y); cycles = 2; break;
        case 0xC9: compare(state.a, read8(bus, imm())); cycles = 2; break;
        case 0xCA: --state.x; set_zn(state, state.x); cycles = 2; break;
        case 0xCC: compare(state.y, read8(bus, abs())); cycles = 4; break;
        case 0xCD: compare(state.a, read8(bus, abs())); cycles = 4; break;
        case 0xCE: { const auto a = abs(); const auto v = static_cast<std::uint8_t>(read8(bus, a) - 1); write8(bus, a, v); set_zn(state, v); cycles = 6; break; }
        case 0xD0: branch(!get_flag(state, Zero)); break;
        case 0xD1: compare(state.a, read8(bus, indy(true))); cycles += 5; break;
        case 0xD5: compare(state.a, read8(bus, zpx())); cycles = 4; break;
        case 0xD6: { const auto a = zpx(); const auto v = static_cast<std::uint8_t>(read8(bus, a) - 1); write8(bus, a, v); set_zn(state, v); cycles = 6; break; }
        case 0xD8: set_flag(state, Decimal, false); cycles = 2; break;
        case 0xD9: compare(state.a, read8(bus, absy(true))); cycles += 4; break;
        case 0xDD: compare(state.a, read8(bus, absx(true))); cycles += 4; break;
        case 0xDE: { const auto a = absx(false); const auto v = static_cast<std::uint8_t>(read8(bus, a) - 1); write8(bus, a, v); set_zn(state, v); cycles = 7; break; }
        case 0xE0: compare(state.x, read8(bus, imm())); cycles = 2; break;
        case 0xE1: sbc(read8(bus, indx())); cycles = 6; break;
        case 0xE4: compare(state.x, read8(bus, zp())); cycles = 3; break;
        case 0xE5: sbc(read8(bus, zp())); cycles = 3; break;
        case 0xE6: { const auto a = zp(); const auto v = static_cast<std::uint8_t>(read8(bus, a) + 1); write8(bus, a, v); set_zn(state, v); cycles = 5; break; }
        case 0xE8: ++state.x; set_zn(state, state.x); cycles = 2; break;
        case 0xE9: sbc(read8(bus, imm())); cycles = 2; break;
        case 0xEA: cycles = 2; break;
        case 0xEC: compare(state.x, read8(bus, abs())); cycles = 4; break;
        case 0xED: sbc(read8(bus, abs())); cycles = 4; break;
        case 0xEE: { const auto a = abs(); const auto v = static_cast<std::uint8_t>(read8(bus, a) + 1); write8(bus, a, v); set_zn(state, v); cycles = 6; break; }
        case 0xF0: branch(get_flag(state, Zero)); break;
        case 0xF1: sbc(read8(bus, indy(true))); cycles += 5; break;
        case 0xF5: sbc(read8(bus, zpx())); cycles = 4; break;
        case 0xF6: { const auto a = zpx(); const auto v = static_cast<std::uint8_t>(read8(bus, a) + 1); write8(bus, a, v); set_zn(state, v); cycles = 6; break; }
        case 0xF8: set_flag(state, Decimal, true); cycles = 2; break;
        case 0xF9: sbc(read8(bus, absy(true))); cycles += 4; break;
        case 0xFD: sbc(read8(bus, absx(true))); cycles += 4; break;
        case 0xFE: { const auto a = absx(false); const auto v = static_cast<std::uint8_t>(read8(bus, a) + 1); write8(bus, a, v); set_zn(state, v); cycles = 7; break; }
        default:
#ifdef __CUDA_ARCH__
            asm("trap;");
            cycles = 0;
            break;
#else
            throw std::runtime_error("unimplemented or illegal 6502 opcode 0x" + std::to_string(opcode));
#endif
    }

    state.cycles += cycles;
    return StepResult{start_pc, opcode, cycles};
}

}  // namespace nesle::cpu

#undef NESLE_CPU_HD

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "nesle/console.hpp"
#include "nesle/controller.hpp"
#include "nesle/cpu.hpp"
#include "nesle/cuda/batch_cpu.hpp"
#include "nesle/rom.hpp"

namespace {

nesle::RomImage make_step_rom() {
    nesle::RomImage rom;
    rom.metadata.mapper = 0;
    rom.metadata.prg_rom_banks = 2;
    rom.metadata.chr_rom_banks = 1;
    rom.metadata.prg_rom_size = 32 * 1024;
    rom.metadata.chr_rom_size = 8 * 1024;
    rom.prg_rom.resize(32 * 1024, 0xEA);
    rom.chr_rom.resize(8 * 1024);

    std::size_t pc = 0;
    auto emit = [&](std::uint8_t value) {
        rom.prg_rom[pc++] = value;
    };
    auto lda_imm = [&](std::uint8_t value) {
        emit(0xA9);
        emit(value);
    };
    auto sta_abs = [&](std::uint16_t address) {
        emit(0x8D);
        emit(static_cast<std::uint8_t>(address & 0x00FF));
        emit(static_cast<std::uint8_t>(address >> 8));
    };

    lda_imm(0x10);
    emit(0x85);  // STA $02
    emit(0x02);
    emit(0xE6);  // INC $02
    emit(0x02);
    lda_imm(1);
    sta_abs(0x4016);
    lda_imm(0);
    sta_abs(0x4016);
    emit(0xAD);  // LDA $4016
    emit(0x16);
    emit(0x40);
    emit(0x29);  // AND #$01
    emit(0x01);
    sta_abs(0x6000);
    emit(0x4C);  // JMP $8017
    emit(0x17);
    emit(0x80);

    rom.prg_rom[0x7FFC] = 0x00;
    rom.prg_rom[0x7FFD] = 0x80;
    return rom;
}

nesle::cuda::BatchBuffers make_buffers(std::vector<std::uint8_t>& prg_rom,
                                        std::vector<std::uint8_t>& ram,
                                        std::vector<std::uint8_t>& prg_ram,
                                        std::vector<std::uint8_t>& actions,
                                        std::vector<std::uint16_t>& pc,
                                        std::vector<std::uint8_t>& a,
                                        std::vector<std::uint8_t>& x,
                                        std::vector<std::uint8_t>& y,
                                        std::vector<std::uint8_t>& sp,
                                        std::vector<std::uint8_t>& p,
                                        std::vector<std::uint64_t>& cycles,
                                        std::vector<std::uint8_t>& controller_shift,
                                        std::vector<std::uint8_t>& controller_shift_count,
                                        std::vector<std::uint8_t>& controller_strobe) {
    nesle::cuda::BatchBuffers buffers{};
    buffers.cpu.pc = pc.data();
    buffers.cpu.a = a.data();
    buffers.cpu.x = x.data();
    buffers.cpu.y = y.data();
    buffers.cpu.sp = sp.data();
    buffers.cpu.p = p.data();
    buffers.cpu.cycles = cycles.data();
    buffers.cpu.ram = ram.data();
    buffers.cpu.prg_ram = prg_ram.data();
    buffers.cpu.controller1_shift = controller_shift.data();
    buffers.cpu.controller1_shift_count = controller_shift_count.data();
    buffers.cpu.controller1_strobe = controller_strobe.data();
    buffers.cart.prg_rom = prg_rom.data();
    buffers.cart.prg_rom_size = static_cast<std::uint32_t>(prg_rom.size());
    buffers.cart.mapper = 0;
    buffers.action_masks = actions.data();
    return buffers;
}

}  // namespace

int main() {
    constexpr std::size_t kNumEnvs = 1;
    auto rom = make_step_rom();

    std::vector<std::uint8_t> ram(kNumEnvs * nesle::cuda::kCpuRamBytes, 0);
    std::vector<std::uint8_t> prg_ram(kNumEnvs * nesle::cuda::kPrgRamBytes, 0);
    std::vector<std::uint8_t> actions(kNumEnvs, nesle::ButtonA | nesle::ButtonRight);
    std::vector<std::uint16_t> pc(kNumEnvs, 0);
    std::vector<std::uint8_t> a(kNumEnvs, 0);
    std::vector<std::uint8_t> x(kNumEnvs, 0);
    std::vector<std::uint8_t> y(kNumEnvs, 0);
    std::vector<std::uint8_t> sp(kNumEnvs, 0);
    std::vector<std::uint8_t> p(kNumEnvs, 0);
    std::vector<std::uint64_t> cycles(kNumEnvs, 0);
    std::vector<std::uint8_t> controller_shift(kNumEnvs, 0);
    std::vector<std::uint8_t> controller_shift_count(kNumEnvs, 8);
    std::vector<std::uint8_t> controller_strobe(kNumEnvs, 0);

    auto buffers = make_buffers(rom.prg_rom,
                                ram,
                                prg_ram,
                                actions,
                                pc,
                                a,
                                x,
                                y,
                                sp,
                                p,
                                cycles,
                                controller_shift,
                                controller_shift_count,
                                controller_strobe);

    nesle::Console console(make_step_rom());
    console.controller1().set_buttons(actions[0]);
    nesle::cpu::CpuState console_state;
    console.reset_cpu(console_state);
    nesle::cuda::reset_batch_cpu_env(buffers, 0);

    assert(pc[0] == console_state.pc);
    assert(sp[0] == console_state.sp);
    assert(p[0] == console_state.p);
    assert(cycles[0] == console_state.cycles);

    for (int instruction = 0; instruction < 11; ++instruction) {
        const auto console_step = nesle::cpu::step(console_state, console);
        const auto batch_step = nesle::cuda::step_batch_cpu_env(buffers, 0);

        assert(batch_step.pc == console_step.pc);
        assert(batch_step.opcode == console_step.opcode);
        assert(batch_step.cycles == console_step.cycles);
        assert(pc[0] == console_state.pc);
        assert(a[0] == console_state.a);
        assert(x[0] == console_state.x);
        assert(y[0] == console_state.y);
        assert(sp[0] == console_state.sp);
        assert(p[0] == console_state.p);
        assert(cycles[0] == console_state.cycles);
        assert(ram[0x0002] == console.read(0x0002));
        assert(prg_ram[0] == console.read(0x6000));
    }

    assert(ram[0x0002] == 0x11);
    assert(prg_ram[0] == 1);
    assert(pc[0] == 0x8017);

    return 0;
}

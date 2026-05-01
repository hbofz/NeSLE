#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "nesle/console.hpp"
#include "nesle/controller.hpp"
#include "nesle/cpu.hpp"
#include "nesle/cuda/batch_runner.hpp"
#include "nesle/rom.hpp"

namespace {

nesle::RomImage make_batch_rom() {
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

    lda_imm(0x20);
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
    emit(0xE6);  // INC $03
    emit(0x03);
    emit(0x4C);  // JMP $8019
    emit(0x19);
    emit(0x80);

    rom.prg_rom[0x7FFC] = 0x00;
    rom.prg_rom[0x7FFD] = 0x80;
    return rom;
}

nesle::cuda::BatchBuffers make_buffers(std::vector<std::uint8_t>& prg_rom,
                                        std::vector<std::uint8_t>& ram,
                                        std::vector<std::uint8_t>& prg_ram,
                                        std::vector<std::uint8_t>& actions,
                                        std::vector<std::uint8_t>& done,
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
    buffers.done = done.data();
    return buffers;
}

}  // namespace

int main() {
    constexpr std::size_t kNumEnvs = 4;
    constexpr std::uint64_t kInstructions = 12;
    auto rom = make_batch_rom();

    std::vector<std::uint8_t> ram(kNumEnvs * nesle::cuda::kCpuRamBytes, 0);
    std::vector<std::uint8_t> prg_ram(kNumEnvs * nesle::cuda::kPrgRamBytes, 0);
    std::vector<std::uint8_t> actions = {
        nesle::ButtonA,
        nesle::ButtonB,
        static_cast<std::uint8_t>(nesle::ButtonA | nesle::ButtonRight),
        0,
    };
    std::vector<std::uint8_t> done(kNumEnvs, 0);
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
                                done,
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

    std::vector<nesle::Console> consoles;
    std::vector<nesle::cpu::CpuState> console_states(kNumEnvs);
    consoles.reserve(kNumEnvs);
    for (std::size_t env = 0; env < kNumEnvs; ++env) {
        consoles.emplace_back(make_batch_rom());
        consoles[env].controller1().set_buttons(actions[env]);
        consoles[env].reset_cpu(console_states[env]);
        nesle::cuda::reset_batch_cpu_env(buffers, static_cast<std::uint32_t>(env));
    }

    const auto batch_result = nesle::cuda::run_batch_cpu(
        buffers,
        nesle::cuda::BatchRunConfig{
            static_cast<std::uint32_t>(kNumEnvs),
            kInstructions,
            false,
        });

    for (std::size_t env = 0; env < kNumEnvs; ++env) {
        for (std::uint64_t instruction = 0; instruction < kInstructions; ++instruction) {
            (void)nesle::cpu::step(console_states[env], consoles[env]);
        }

        const auto ram_base = env * nesle::cuda::kCpuRamBytes;
        const auto prg_ram_base = env * nesle::cuda::kPrgRamBytes;
        assert(batch_result.envs[env].status == nesle::cuda::BatchRunStatus::Completed);
        assert(batch_result.envs[env].instructions == kInstructions);
        assert(done[env] == 1);
        assert(pc[env] == console_states[env].pc);
        assert(a[env] == console_states[env].a);
        assert(sp[env] == console_states[env].sp);
        assert(p[env] == console_states[env].p);
        assert(cycles[env] == console_states[env].cycles);
        assert(ram[ram_base + 0x0002] == consoles[env].read(0x0002));
        assert(ram[ram_base + 0x0003] == consoles[env].read(0x0003));
        assert(prg_ram[prg_ram_base] == consoles[env].read(0x6000));
    }

    assert(prg_ram[0 * nesle::cuda::kPrgRamBytes] == 1);
    assert(prg_ram[1 * nesle::cuda::kPrgRamBytes] == 0);
    assert(prg_ram[2 * nesle::cuda::kPrgRamBytes] == 1);
    assert(prg_ram[3 * nesle::cuda::kPrgRamBytes] == 0);

    return 0;
}

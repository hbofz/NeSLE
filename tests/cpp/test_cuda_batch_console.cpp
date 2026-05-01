#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "nesle/console.hpp"
#include "nesle/cpu.hpp"
#include "nesle/cuda/batch_console.hpp"
#include "nesle/rom.hpp"

namespace {

nesle::RomImage make_base_rom() {
    nesle::RomImage rom;
    rom.metadata.mapper = 0;
    rom.metadata.prg_rom_banks = 2;
    rom.metadata.chr_rom_banks = 1;
    rom.metadata.prg_rom_size = 32 * 1024;
    rom.metadata.chr_rom_size = 8 * 1024;
    rom.prg_rom.resize(32 * 1024, 0xEA);
    rom.chr_rom.resize(8 * 1024);
    return rom;
}

nesle::RomImage make_nmi_rom() {
    auto rom = make_base_rom();
    rom.prg_rom[0x0000] = 0xA9;  // LDA #$80
    rom.prg_rom[0x0001] = 0x80;
    rom.prg_rom[0x0002] = 0x8D;  // STA $2000
    rom.prg_rom[0x0003] = 0x00;
    rom.prg_rom[0x0004] = 0x20;
    rom.prg_rom[0x0005] = 0x4C;  // JMP $8005
    rom.prg_rom[0x0006] = 0x05;
    rom.prg_rom[0x0007] = 0x80;
    rom.prg_rom[0x1000] = 0xE6;  // INC $00
    rom.prg_rom[0x1001] = 0x00;
    rom.prg_rom[0x1002] = 0x40;  // RTI
    rom.prg_rom[0x7FFA] = 0x00;  // NMI vector -> $9000
    rom.prg_rom[0x7FFB] = 0x90;
    rom.prg_rom[0x7FFC] = 0x00;  // RESET vector -> $8000
    rom.prg_rom[0x7FFD] = 0x80;
    return rom;
}

nesle::RomImage make_dma_rom() {
    auto rom = make_base_rom();
    rom.prg_rom[0x0000] = 0xA9;  // LDA #$02
    rom.prg_rom[0x0001] = 0x02;
    rom.prg_rom[0x0002] = 0x8D;  // STA $4014
    rom.prg_rom[0x0003] = 0x14;
    rom.prg_rom[0x0004] = 0x40;
    rom.prg_rom[0x0005] = 0x4C;  // JMP $8005
    rom.prg_rom[0x0006] = 0x05;
    rom.prg_rom[0x0007] = 0x80;
    rom.prg_rom[0x7FFC] = 0x00;
    rom.prg_rom[0x7FFD] = 0x80;
    return rom;
}

struct BatchStorage {
    std::vector<std::uint8_t> ram;
    std::vector<std::uint8_t> prg_ram;
    std::vector<std::uint16_t> pc;
    std::vector<std::uint8_t> a;
    std::vector<std::uint8_t> x;
    std::vector<std::uint8_t> y;
    std::vector<std::uint8_t> sp;
    std::vector<std::uint8_t> p;
    std::vector<std::uint64_t> cycles;
    std::vector<std::uint32_t> pending_dma_cycles;
    std::vector<std::uint8_t> controller_shift;
    std::vector<std::uint8_t> controller_shift_count;
    std::vector<std::uint8_t> controller_strobe;
    std::vector<std::uint8_t> ppu_ctrl;
    std::vector<std::uint8_t> ppu_mask;
    std::vector<std::uint8_t> ppu_status;
    std::vector<std::uint8_t> ppu_oam_addr;
    std::vector<std::uint8_t> ppu_nmi_pending;
    std::vector<std::int16_t> ppu_scanline;
    std::vector<std::uint16_t> ppu_dot;
    std::vector<std::uint64_t> ppu_frame;
    std::vector<std::uint8_t> ppu_w;
    std::vector<std::uint8_t> ppu_oam;
    std::vector<std::uint8_t> actions;

    explicit BatchStorage(std::size_t num_envs)
        : ram(num_envs * nesle::cuda::kCpuRamBytes, 0),
          prg_ram(num_envs * nesle::cuda::kPrgRamBytes, 0),
          pc(num_envs, 0),
          a(num_envs, 0),
          x(num_envs, 0),
          y(num_envs, 0),
          sp(num_envs, 0),
          p(num_envs, 0),
          cycles(num_envs, 0),
          pending_dma_cycles(num_envs, 0),
          controller_shift(num_envs, 0),
          controller_shift_count(num_envs, 8),
          controller_strobe(num_envs, 0),
          ppu_ctrl(num_envs, 0),
          ppu_mask(num_envs, 0),
          ppu_status(num_envs, 0),
          ppu_oam_addr(num_envs, 0),
          ppu_nmi_pending(num_envs, 0),
          ppu_scanline(num_envs, 0),
          ppu_dot(num_envs, 0),
          ppu_frame(num_envs, 0),
          ppu_w(num_envs, 0),
          ppu_oam(num_envs * nesle::cuda::kOamBytes, 0),
          actions(num_envs, 0) {}
};

nesle::cuda::BatchBuffers make_buffers(nesle::RomImage& rom, BatchStorage& storage) {
    nesle::cuda::BatchBuffers buffers{};
    buffers.cpu.pc = storage.pc.data();
    buffers.cpu.a = storage.a.data();
    buffers.cpu.x = storage.x.data();
    buffers.cpu.y = storage.y.data();
    buffers.cpu.sp = storage.sp.data();
    buffers.cpu.p = storage.p.data();
    buffers.cpu.cycles = storage.cycles.data();
    buffers.cpu.ram = storage.ram.data();
    buffers.cpu.prg_ram = storage.prg_ram.data();
    buffers.cpu.pending_dma_cycles = storage.pending_dma_cycles.data();
    buffers.cpu.controller1_shift = storage.controller_shift.data();
    buffers.cpu.controller1_shift_count = storage.controller_shift_count.data();
    buffers.cpu.controller1_strobe = storage.controller_strobe.data();
    buffers.ppu.ctrl = storage.ppu_ctrl.data();
    buffers.ppu.mask = storage.ppu_mask.data();
    buffers.ppu.status = storage.ppu_status.data();
    buffers.ppu.oam_addr = storage.ppu_oam_addr.data();
    buffers.ppu.nmi_pending = storage.ppu_nmi_pending.data();
    buffers.ppu.scanline = storage.ppu_scanline.data();
    buffers.ppu.dot = storage.ppu_dot.data();
    buffers.ppu.frame = storage.ppu_frame.data();
    buffers.ppu.w = storage.ppu_w.data();
    buffers.ppu.oam = storage.ppu_oam.data();
    buffers.cart.prg_rom = rom.prg_rom.data();
    buffers.cart.prg_rom_size = static_cast<std::uint32_t>(rom.prg_rom.size());
    buffers.cart.mapper = 0;
    buffers.action_masks = storage.actions.data();
    return buffers;
}

void assert_cpu_matches(const BatchStorage& storage,
                        const nesle::cpu::CpuState& state,
                        std::uint32_t env) {
    assert(storage.pc[env] == state.pc);
    assert(storage.a[env] == state.a);
    assert(storage.x[env] == state.x);
    assert(storage.y[env] == state.y);
    assert(storage.sp[env] == state.sp);
    assert(storage.p[env] == state.p);
    assert(storage.cycles[env] == state.cycles);
}

}  // namespace

int main() {
    {
        auto rom = make_nmi_rom();
        BatchStorage storage(1);
        auto buffers = make_buffers(rom, storage);
        nesle::Console console(make_nmi_rom());
        nesle::cpu::CpuState state;
        console.reset_cpu(state);
        nesle::cuda::reset_batch_cpu_env(buffers, 0);

        std::uint32_t frames = 0;
        for (std::uint64_t instruction = 0; instruction < 20'000 && frames == 0; ++instruction) {
            const auto console_step = console.step_cpu_instruction(state);
            const auto batch_step = nesle::cuda::step_batch_console_instruction(buffers, 0);

            assert(batch_step.cpu.pc == console_step.cpu.pc);
            assert(batch_step.cpu.opcode == console_step.cpu.opcode);
            assert(batch_step.cpu_cycles == console_step.cpu_cycles);
            assert(batch_step.ppu_cycles == console_step.ppu_cycles);
            assert(batch_step.frames_completed == console_step.frames_completed);
            assert(batch_step.nmi_serviced == console_step.nmi_serviced);
            assert(batch_step.nmi_started == console_step.nmi_started);
            assert_cpu_matches(storage, state, 0);
            frames += batch_step.frames_completed;
        }

        assert(frames == 1);
        assert(storage.ram[0] == 1);
        assert(storage.ppu_frame[0] == console.ppu().frame());
        assert(storage.ppu_scanline[0] == console.ppu().scanline());
        assert(storage.ppu_dot[0] == console.ppu().dot());
        assert((storage.ppu_status[0] & 0x80) == (console.ppu().status() & 0x80));
        assert(storage.ppu_nmi_pending[0] == 0);
    }

    {
        auto rom = make_dma_rom();
        BatchStorage storage(1);
        auto buffers = make_buffers(rom, storage);
        nesle::Console console(make_dma_rom());
        nesle::cpu::CpuState state;
        console.reset_cpu(state);
        nesle::cuda::reset_batch_cpu_env(buffers, 0);
        for (std::uint16_t i = 0; i < 256; ++i) {
            console.write(static_cast<std::uint16_t>(0x0200 + i),
                          static_cast<std::uint8_t>(0xFF - i));
            storage.ram[0x0200 + i] = static_cast<std::uint8_t>(0xFF - i);
        }

        const auto lda_console = console.step_cpu_instruction(state);
        const auto lda_batch = nesle::cuda::step_batch_console_instruction(buffers, 0);
        assert(lda_batch.cpu_cycles == lda_console.cpu_cycles);
        assert(lda_batch.ppu_cycles == lda_console.ppu_cycles);

        const auto dma_console = console.step_cpu_instruction(state);
        const auto dma_batch = nesle::cuda::step_batch_console_instruction(buffers, 0);
        assert(dma_batch.cpu_cycles == dma_console.cpu_cycles);
        assert(dma_batch.ppu_cycles == dma_console.ppu_cycles);
        assert(dma_batch.cpu_cycles == 517);
        assert(storage.pending_dma_cycles[0] == 0);
        assert(storage.ppu_oam[0] == console.ppu().oam()[0]);
        assert(storage.ppu_oam[1] == console.ppu().oam()[1]);
        assert(storage.ppu_oam[255] == console.ppu().oam()[255]);
        assert_cpu_matches(storage, state, 0);
    }

    return 0;
}

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "nesle/cuda/batch_console.hpp"
#include "nesle/cuda/reset_cache.hpp"
#include "nesle/rom.hpp"

namespace {

nesle::RomImage make_rom() {
    nesle::RomImage rom;
    rom.metadata.mapper = 0;
    rom.metadata.prg_rom_banks = 2;
    rom.metadata.chr_rom_banks = 1;
    rom.metadata.prg_rom_size = 32 * 1024;
    rom.metadata.chr_rom_size = 8 * 1024;
    rom.prg_rom.resize(32 * 1024, 0xEA);
    rom.chr_rom.resize(8 * 1024);
    rom.prg_rom[0x0000] = 0xA9;  // LDA #$05
    rom.prg_rom[0x0001] = 0x05;
    rom.prg_rom[0x0002] = 0x85;  // STA $02
    rom.prg_rom[0x0003] = 0x02;
    rom.prg_rom[0x0004] = 0x4C;  // JMP $8004
    rom.prg_rom[0x0005] = 0x04;
    rom.prg_rom[0x0006] = 0x80;
    rom.prg_rom[0x7FFC] = 0x00;
    rom.prg_rom[0x7FFD] = 0x80;
    return rom;
}

}  // namespace

int main() {
    constexpr std::size_t kNumEnvs = 2;
    auto rom = make_rom();
    std::vector<std::uint8_t> ram(kNumEnvs * nesle::cuda::kCpuRamBytes, 0);
    std::vector<std::uint8_t> prg_ram(kNumEnvs * nesle::cuda::kPrgRamBytes, 0);
    std::vector<std::uint16_t> pc(kNumEnvs, 0);
    std::vector<std::uint8_t> a(kNumEnvs, 0);
    std::vector<std::uint8_t> x(kNumEnvs, 0);
    std::vector<std::uint8_t> y(kNumEnvs, 0);
    std::vector<std::uint8_t> sp(kNumEnvs, 0);
    std::vector<std::uint8_t> p(kNumEnvs, 0);
    std::vector<std::uint64_t> cycles(kNumEnvs, 0);
    std::vector<std::uint32_t> pending_dma(kNumEnvs, 0);
    std::vector<std::uint8_t> controller_shift(kNumEnvs, 0);
    std::vector<std::uint8_t> controller_shift_count(kNumEnvs, 8);
    std::vector<std::uint8_t> controller_strobe(kNumEnvs, 0);
    std::vector<std::uint8_t> ppu_ctrl(kNumEnvs, 0);
    std::vector<std::uint8_t> ppu_mask(kNumEnvs, 0);
    std::vector<std::uint8_t> ppu_status(kNumEnvs, 0);
    std::vector<std::uint8_t> ppu_oam_addr(kNumEnvs, 0);
    std::vector<std::uint8_t> ppu_nmi_pending(kNumEnvs, 0);
    std::vector<std::int16_t> ppu_scanline(kNumEnvs, 0);
    std::vector<std::uint16_t> ppu_dot(kNumEnvs, 0);
    std::vector<std::uint64_t> ppu_frame(kNumEnvs, 0);
    std::vector<std::uint16_t> ppu_v(kNumEnvs, 0);
    std::vector<std::uint16_t> ppu_t(kNumEnvs, 0);
    std::vector<std::uint8_t> ppu_x(kNumEnvs, 0);
    std::vector<std::uint8_t> ppu_w(kNumEnvs, 0);
    std::vector<std::uint8_t> nametable(kNumEnvs * nesle::cuda::kNametableRamBytes, 0);
    std::vector<std::uint8_t> palette(kNumEnvs * nesle::cuda::kPaletteRamBytes, 0);
    std::vector<std::uint8_t> oam(kNumEnvs * nesle::cuda::kOamBytes, 0);
    std::vector<std::uint8_t> actions(kNumEnvs, 0);
    std::vector<std::uint8_t> done(kNumEnvs, 0);
    std::vector<float> rewards(kNumEnvs, 0.0F);
    std::vector<int> previous_x(kNumEnvs, 0);
    std::vector<int> previous_time(kNumEnvs, 0);
    std::vector<std::uint8_t> previous_dying(kNumEnvs, 0);

    nesle::cuda::BatchBuffers buffers{};
    buffers.cpu.pc = pc.data();
    buffers.cpu.a = a.data();
    buffers.cpu.x = x.data();
    buffers.cpu.y = y.data();
    buffers.cpu.sp = sp.data();
    buffers.cpu.p = p.data();
    buffers.cpu.cycles = cycles.data();
    buffers.cpu.pending_dma_cycles = pending_dma.data();
    buffers.cpu.ram = ram.data();
    buffers.cpu.prg_ram = prg_ram.data();
    buffers.cpu.controller1_shift = controller_shift.data();
    buffers.cpu.controller1_shift_count = controller_shift_count.data();
    buffers.cpu.controller1_strobe = controller_strobe.data();
    buffers.ppu.ctrl = ppu_ctrl.data();
    buffers.ppu.mask = ppu_mask.data();
    buffers.ppu.status = ppu_status.data();
    buffers.ppu.oam_addr = ppu_oam_addr.data();
    buffers.ppu.nmi_pending = ppu_nmi_pending.data();
    buffers.ppu.scanline = ppu_scanline.data();
    buffers.ppu.dot = ppu_dot.data();
    buffers.ppu.frame = ppu_frame.data();
    buffers.ppu.v = ppu_v.data();
    buffers.ppu.t = ppu_t.data();
    buffers.ppu.x = ppu_x.data();
    buffers.ppu.w = ppu_w.data();
    buffers.ppu.nametable_ram = nametable.data();
    buffers.ppu.palette_ram = palette.data();
    buffers.ppu.oam = oam.data();
    buffers.cart.prg_rom = rom.prg_rom.data();
    buffers.cart.prg_rom_size = static_cast<std::uint32_t>(rom.prg_rom.size());
    buffers.action_masks = actions.data();
    buffers.done = done.data();
    buffers.rewards = rewards.data();
    buffers.previous_mario_x = previous_x.data();
    buffers.previous_mario_time = previous_time.data();
    buffers.previous_mario_dying = previous_dying.data();

    nesle::cuda::reset_batch_cpu_env(buffers, 0);
    ram[0x10] = 0xAA;
    prg_ram[3] = 0xBB;
    ppu_ctrl[0] = 0x80;
    ppu_status[0] = 0x40;
    ppu_scanline[0] = 7;
    ppu_dot[0] = 9;
    ppu_frame[0] = 2;
    nametable[4] = 0xCC;
    palette[1] = 0x2A;
    oam[5] = 0xDD;
    rewards[0] = 3.0F;
    previous_x[0] = 42;
    previous_time[0] = 399;

    const auto snapshot = nesle::cuda::capture_reset_snapshot(buffers, 0);
    (void)nesle::cuda::step_batch_console_instruction(buffers, 0);
    pc[0] = 0x9999;
    ram[0x10] = 0;
    prg_ram[3] = 0;
    ppu_status[0] = 0;
    nametable[4] = 0;
    palette[1] = 0;
    oam[5] = 0;
    rewards[0] = 0.0F;
    previous_x[0] = 0;

    nesle::cuda::restore_reset_snapshot(buffers, 0, snapshot);
    assert(pc[0] == snapshot.pc);
    assert(a[0] == snapshot.a);
    assert(sp[0] == snapshot.sp);
    assert(cycles[0] == snapshot.cycles);
    assert(ram[0x10] == 0xAA);
    assert(prg_ram[3] == 0xBB);
    assert(ppu_ctrl[0] == 0x80);
    assert(ppu_status[0] == 0x40);
    assert(ppu_scanline[0] == 7);
    assert(ppu_dot[0] == 9);
    assert(ppu_frame[0] == 2);
    assert(nametable[4] == 0xCC);
    assert(palette[1] == 0x2A);
    assert(oam[5] == 0xDD);
    assert(rewards[0] == 3.0F);
    assert(previous_x[0] == 42);
    assert(previous_time[0] == 399);

    const auto first = nesle::cuda::step_batch_console_instruction(buffers, 0);
    nesle::cuda::restore_reset_snapshot(buffers, 0, snapshot);
    const auto second = nesle::cuda::step_batch_console_instruction(buffers, 0);
    assert(first.cpu.pc == second.cpu.pc);
    assert(first.cpu.opcode == second.cpu.opcode);
    assert(first.cpu_cycles == second.cpu_cycles);
    assert(pc[0] == 0x8002);

    return 0;
}

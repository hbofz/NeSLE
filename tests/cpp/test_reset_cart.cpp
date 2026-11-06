// Resuming a banked cartridge from a save state.
//
// FCEUX saves the board's bank register in the cartridge section: "LATC" for
// mappers 2 and 94, where it is the raw byte the game wrote and still needs the
// board's shift, and "REGS"[0] for mapper 34, where it is the 32 KB page number.
// It also saves 8 KB of CHR RAM ("CHRR") on every board that has one.
//
// Two failure modes this pins down, both reported in review:
//   * warm reset used to force bank 0, so a state saved with code or data in
//     $8000-$BFFF resumed on the wrong window;
//   * device_chr_ram_ was a plain allocation that no reset touched, so the first
//     episode after a restore drew whatever was there, and later episodes kept
//     the previous episode's tiles.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "nesle/cuda/batch_step.cuh"
#include "nesle/cuda/state.cuh"
#include "nesle/rom.hpp"

namespace {

constexpr std::size_t kNumEnvs = 2;
constexpr std::size_t kNumLevels = 2;

// Everything warm_reset_console_env touches, in one place so each case reads as
// "describe the board, set the state, reset, assert".
struct Fixture {
    std::vector<std::uint8_t> ram = std::vector<std::uint8_t>(kNumEnvs * nesle::cuda::kCpuRamBytes, 0);
    std::vector<std::uint8_t> prg_ram =
        std::vector<std::uint8_t>(kNumEnvs * nesle::cuda::kPrgRamBytes, 0);
    std::vector<std::uint8_t> nametable =
        std::vector<std::uint8_t>(kNumEnvs * nesle::cuda::kNametableRamBytes, 0);
    std::vector<std::uint8_t> palette =
        std::vector<std::uint8_t>(kNumEnvs * nesle::cuda::kPaletteRamBytes, 0);
    std::vector<std::uint8_t> oam = std::vector<std::uint8_t>(kNumEnvs * nesle::cuda::kOamBytes, 0);
    std::vector<std::uint8_t> chr_ram =
        std::vector<std::uint8_t>(kNumEnvs * nesle::cuda::kChrRamBytes, 0);
    std::vector<std::uint8_t> prg_bank = std::vector<std::uint8_t>(kNumEnvs, 0);

    std::vector<std::uint16_t> pc = std::vector<std::uint16_t>(kNumEnvs, 0);
    std::vector<std::uint8_t> a = std::vector<std::uint8_t>(kNumEnvs, 0);
    std::vector<std::uint8_t> x = std::vector<std::uint8_t>(kNumEnvs, 0);
    std::vector<std::uint8_t> y = std::vector<std::uint8_t>(kNumEnvs, 0);
    std::vector<std::uint8_t> sp = std::vector<std::uint8_t>(kNumEnvs, 0);
    std::vector<std::uint8_t> p = std::vector<std::uint8_t>(kNumEnvs, 0);
    std::vector<std::uint64_t> cycles = std::vector<std::uint64_t>(kNumEnvs, 0);
    std::vector<std::uint32_t> pending_dma = std::vector<std::uint32_t>(kNumEnvs, 0);
    std::vector<std::uint8_t> cpu_nmi = std::vector<std::uint8_t>(kNumEnvs, 0);
    std::vector<std::uint8_t> cpu_irq = std::vector<std::uint8_t>(kNumEnvs, 0);
    std::vector<std::uint8_t> controller_shift = std::vector<std::uint8_t>(kNumEnvs, 0);
    std::vector<std::uint8_t> controller_count = std::vector<std::uint8_t>(kNumEnvs, 8);
    std::vector<std::uint8_t> controller_strobe = std::vector<std::uint8_t>(kNumEnvs, 0);
    std::vector<std::uint8_t> ppu_ctrl = std::vector<std::uint8_t>(kNumEnvs, 0);
    std::vector<std::uint8_t> ppu_mask = std::vector<std::uint8_t>(kNumEnvs, 0);
    std::vector<std::uint8_t> ppu_status = std::vector<std::uint8_t>(kNumEnvs, 0);
    std::vector<std::uint8_t> ppu_oam_addr = std::vector<std::uint8_t>(kNumEnvs, 0);
    std::vector<std::uint8_t> ppu_nmi = std::vector<std::uint8_t>(kNumEnvs, 0);
    std::vector<std::uint32_t> ppu_frame_dot = std::vector<std::uint32_t>(kNumEnvs, 0);
    std::vector<std::uint64_t> ppu_frame = std::vector<std::uint64_t>(kNumEnvs, 0);
    std::vector<std::uint16_t> ppu_v = std::vector<std::uint16_t>(kNumEnvs, 0);
    std::vector<std::uint16_t> ppu_t = std::vector<std::uint16_t>(kNumEnvs, 0);
    std::vector<std::uint8_t> ppu_x = std::vector<std::uint8_t>(kNumEnvs, 0);
    std::vector<std::uint8_t> ppu_w = std::vector<std::uint8_t>(kNumEnvs, 0);
    std::vector<std::uint8_t> ppu_open_bus = std::vector<std::uint8_t>(kNumEnvs, 0);
    std::vector<std::uint8_t> ppu_read_buffer = std::vector<std::uint8_t>(kNumEnvs, 0);
    std::vector<std::uint8_t> ppu_scroll_x = std::vector<std::uint8_t>(kNumEnvs, 0);
    std::vector<std::uint8_t> ppu_scroll_y = std::vector<std::uint8_t>(kNumEnvs, 0);
    std::vector<std::uint8_t> actions = std::vector<std::uint8_t>(kNumEnvs, 0);
    std::vector<std::uint8_t> done = std::vector<std::uint8_t>(kNumEnvs, 0);
    std::vector<float> rewards = std::vector<float>(kNumEnvs, 0.0F);
    std::vector<int> previous_x = std::vector<int>(kNumEnvs, 0);
    std::vector<int> previous_time = std::vector<int>(kNumEnvs, 0);

    // Snapshot arrays: kNumLevels slots each.
    std::vector<std::uint8_t> snap_ram =
        std::vector<std::uint8_t>(kNumLevels * nesle::cuda::kCpuRamBytes, 0);
    std::vector<std::uint8_t> snap_prg_ram =
        std::vector<std::uint8_t>(kNumLevels * nesle::cuda::kPrgRamBytes, 0);
    std::vector<std::uint8_t> snap_nametable =
        std::vector<std::uint8_t>(kNumLevels * nesle::cuda::kNametableRamBytes, 0);
    std::vector<std::uint8_t> snap_palette =
        std::vector<std::uint8_t>(kNumLevels * nesle::cuda::kPaletteRamBytes, 0);
    std::vector<std::uint8_t> snap_oam =
        std::vector<std::uint8_t>(kNumLevels * nesle::cuda::kOamBytes, 0);
    std::vector<std::uint8_t> snap_chr_ram =
        std::vector<std::uint8_t>(kNumLevels * nesle::cuda::kChrRamBytes, 0);
    std::vector<std::uint16_t> snap_pc = std::vector<std::uint16_t>(kNumLevels, 0xC05B);
    std::vector<std::uint8_t> snap_a = std::vector<std::uint8_t>(kNumLevels, 0x3E);
    std::vector<std::uint8_t> snap_x = std::vector<std::uint8_t>(kNumLevels, 0);
    std::vector<std::uint8_t> snap_y = std::vector<std::uint8_t>(kNumLevels, 0);
    std::vector<std::uint8_t> snap_sp = std::vector<std::uint8_t>(kNumLevels, 0xFF);
    std::vector<std::uint8_t> snap_p = std::vector<std::uint8_t>(kNumLevels, 0x24);
    std::vector<std::uint64_t> snap_cycles = std::vector<std::uint64_t>(kNumLevels, 0);
    std::vector<std::uint8_t> snap_ppu_ctrl = std::vector<std::uint8_t>(kNumLevels, 0xB0);
    std::vector<std::uint8_t> snap_ppu_mask = std::vector<std::uint8_t>(kNumLevels, 0x1E);
    std::vector<std::uint8_t> snap_ppu_status = std::vector<std::uint8_t>(kNumLevels, 0);
    std::vector<std::uint8_t> snap_ppu_oam_addr = std::vector<std::uint8_t>(kNumLevels, 0);
    std::vector<std::uint8_t> snap_ppu_open_bus = std::vector<std::uint8_t>(kNumLevels, 0);
    std::vector<std::uint8_t> snap_ppu_read_buffer = std::vector<std::uint8_t>(kNumLevels, 0);
    std::vector<std::uint8_t> snap_ppu_x = std::vector<std::uint8_t>(kNumLevels, 0);
    std::vector<std::uint8_t> snap_ppu_w = std::vector<std::uint8_t>(kNumLevels, 0);
    std::vector<std::uint16_t> snap_ppu_v = std::vector<std::uint16_t>(kNumLevels, 0x0800);
    std::vector<std::uint16_t> snap_ppu_t = std::vector<std::uint16_t>(kNumLevels, 0);
    std::vector<std::uint8_t> snap_latch = std::vector<std::uint8_t>(kNumLevels, 0);
    std::vector<std::uint8_t> env_to_level = std::vector<std::uint8_t>(kNumEnvs, 0);

    nesle::cuda::BatchBuffers buffers{};
    nesle::cuda::SnapshotTemplate snap{};

    explicit Fixture(std::uint16_t mapper, std::uint32_t window_bytes, std::uint8_t shift,
                     std::uint8_t mask) {
        buffers.cpu.pc = pc.data();
        buffers.cpu.a = a.data();
        buffers.cpu.x = x.data();
        buffers.cpu.y = y.data();
        buffers.cpu.sp = sp.data();
        buffers.cpu.p = p.data();
        buffers.cpu.cycles = cycles.data();
        buffers.cpu.pending_dma_cycles = pending_dma.data();
        buffers.cpu.nmi_pending = cpu_nmi.data();
        buffers.cpu.irq_pending = cpu_irq.data();
        buffers.cpu.ram = ram.data();
        buffers.cpu.prg_ram = prg_ram.data();
        buffers.cpu.controller1_shift = controller_shift.data();
        buffers.cpu.controller1_shift_count = controller_count.data();
        buffers.cpu.controller1_strobe = controller_strobe.data();

        buffers.ppu.ctrl = ppu_ctrl.data();
        buffers.ppu.mask = ppu_mask.data();
        buffers.ppu.status = ppu_status.data();
        buffers.ppu.oam_addr = ppu_oam_addr.data();
        buffers.ppu.nmi_pending = ppu_nmi.data();
        buffers.ppu.frame_dot = ppu_frame_dot.data();
        buffers.ppu.frame = ppu_frame.data();
        buffers.ppu.v = ppu_v.data();
        buffers.ppu.t = ppu_t.data();
        buffers.ppu.x = ppu_x.data();
        buffers.ppu.w = ppu_w.data();
        buffers.ppu.open_bus = ppu_open_bus.data();
        buffers.ppu.read_buffer = ppu_read_buffer.data();
        buffers.ppu.scroll_x = ppu_scroll_x.data();
        buffers.ppu.scroll_y = ppu_scroll_y.data();
        buffers.ppu.nametable_ram = nametable.data();
        buffers.ppu.palette_ram = palette.data();
        buffers.ppu.oam = oam.data();
        buffers.ppu.chr_ram = chr_ram.data();

        buffers.mapper.prg_bank = prg_bank.data();

        // The host-side description of the board, as the binding fills it in.
        buffers.cart.mapper = mapper;
        buffers.cart.prg_window_start = 0x8000;
        buffers.cart.prg_window_shift = shift;
        buffers.cart.prg_window_mask = window_bytes - 1;
        buffers.cart.prg_bank_shift = shift;
        buffers.cart.prg_bank_mask = mask;

        buffers.action_masks = actions.data();
        buffers.done = done.data();
        buffers.rewards = rewards.data();
        buffers.previous_mario_x = previous_x.data();
        buffers.previous_mario_time = previous_time.data();

        snap.cpu_ram = snap_ram.data();
        snap.prg_ram = snap_prg_ram.data();
        snap.nametable_ram = snap_nametable.data();
        snap.palette_ram = snap_palette.data();
        snap.oam = snap_oam.data();
        snap.pc = snap_pc.data();
        snap.a = snap_a.data();
        snap.x = snap_x.data();
        snap.y = snap_y.data();
        snap.sp = snap_sp.data();
        snap.p = snap_p.data();
        snap.cycles = snap_cycles.data();
        snap.ppu_ctrl = snap_ppu_ctrl.data();
        snap.ppu_mask = snap_ppu_mask.data();
        snap.ppu_status = snap_ppu_status.data();
        snap.ppu_oam_addr = snap_ppu_oam_addr.data();
        snap.ppu_open_bus = snap_ppu_open_bus.data();
        snap.ppu_read_buffer = snap_ppu_read_buffer.data();
        snap.ppu_x = snap_ppu_x.data();
        snap.ppu_w = snap_ppu_w.data();
        snap.ppu_v = snap_ppu_v.data();
        snap.ppu_t = snap_ppu_t.data();
        snap.env_to_level = env_to_level.data();
        snap.num_levels = kNumLevels;
    }

    void fill_chr_ram_pattern(std::uint8_t seed) {
        for (std::size_t i = 0; i < snap_chr_ram.size(); ++i) {
            snap_chr_ram[i] = static_cast<std::uint8_t>((i * 31u + seed) & 0xFF);
        }
    }

    void reset_env(std::uint32_t env) { nesle::cuda::warm_reset_console_env(buffers, env, snap); }

    [[nodiscard]] std::uint8_t chr_byte(std::uint32_t env, std::size_t offset) const {
        return chr_ram[static_cast<std::size_t>(env) * nesle::cuda::kChrRamBytes + offset];
    }
};

// Mapper 2: LATC is the raw byte written, bank 0-7 over 16 KB pages.
void test_uxrom_restores_bank() {
    Fixture fixture(2, 0x4000, 0, 7);
    fixture.snap_latch[0] = 0x01;  // what every Contra save on this machine holds
    fixture.snap.cart_latch = fixture.snap_latch.data();

    fixture.reset_env(0);
    assert(fixture.prg_bank[0] == 1);
}

// Mapper 94: same tag, but the register is read as value >> 2. The retail
// Senjou no Ookami saves carry 0x18, which is bank 6.
void test_un1rom_shift_is_applied() {
    Fixture fixture(94, 0x4000, 2, 7);
    fixture.snap_latch[0] = 0x18;
    fixture.snap.cart_latch = fixture.snap_latch.data();

    fixture.reset_env(0);
    assert(fixture.prg_bank[0] == 6);
}

// Mapper 34: REGS[0] is the 32 KB page number already, no shift.
void test_bnrom_restores_page() {
    Fixture fixture(34, 0x8000, 0, 3);
    fixture.snap_latch[0] = 0x03;
    fixture.snap.cart_latch = fixture.snap_latch.data();

    fixture.reset_env(0);
    assert(fixture.prg_bank[0] == 3);
}

// Without a latch in the state the env still starts from bank 0 - that is the
// old behaviour, kept for states that genuinely carry no cart section.
void test_missing_latch_falls_back_to_bank_zero() {
    Fixture fixture(2, 0x4000, 0, 7);
    fixture.prg_bank[0] = 5;
    fixture.snap.cart_latch = nullptr;

    fixture.reset_env(0);
    assert(fixture.prg_bank[0] == 0);
}

// CHR RAM comes out of the state file, per level, and only when the board has it.
void test_chr_ram_restored_per_level() {
    Fixture fixture(2, 0x4000, 0, 7);
    fixture.fill_chr_ram_pattern(0);
    fixture.snap.chr_ram = fixture.snap_chr_ram.data();

    // Pre-fill the env with junk, the way a previous episode would leave it.
    for (auto& byte : fixture.chr_ram) {
        byte = 0xEE;
    }

    fixture.reset_env(0);
    for (std::size_t i = 0; i < nesle::cuda::kChrRamBytes; ++i) {
        assert(fixture.chr_byte(0, i) == fixture.snap_chr_ram[i]);
    }

    // Level 1 has its own pattern; env_to_level decides which one an env reads.
    fixture.fill_chr_ram_pattern(0x40);
    fixture.env_to_level[1] = 1;
    fixture.reset_env(1);
    for (std::size_t i = 0; i < nesle::cuda::kChrRamBytes; ++i) {
        assert(fixture.chr_byte(1, i) ==
               fixture.snap_chr_ram[nesle::cuda::kChrRamBytes + i]);
    }
}

// A CHR ROM board has no CHRR tag, so the pointer stays null and the per-env
// allocation must be left alone rather than zeroed behind the caller's back.
void test_no_chr_ram_leaves_allocation_untouched() {
    Fixture fixture(0, 0, 0, 0);
    for (auto& byte : fixture.chr_ram) {
        byte = 0x5A;
    }
    fixture.snap.chr_ram = nullptr;
    fixture.snap.cart_latch = nullptr;

    fixture.reset_env(0);
    for (std::size_t i = 0; i < nesle::cuda::kChrRamBytes; ++i) {
        assert(fixture.chr_byte(0, i) == 0x5A);
    }
}

// The latch is masked, so a save from a bigger board cannot index past the ROM.
void test_latch_is_masked() {
    Fixture fixture(2, 0x4000, 0, 7);
    fixture.snap_latch[0] = 0xFA;  // 250, far outside the 8 pages
    fixture.snap.cart_latch = fixture.snap_latch.data();

    fixture.reset_env(0);
    assert(fixture.prg_bank[0] == (0xFA & 7));
}

}  // namespace

int main() {
    test_uxrom_restores_bank();
    test_un1rom_shift_is_applied();
    test_bnrom_restores_page();
    test_missing_latch_falls_back_to_bank_zero();
    test_chr_ram_restored_per_level();
    test_no_chr_ram_leaves_allocation_untouched();
    test_latch_is_masked();
    return 0;
}

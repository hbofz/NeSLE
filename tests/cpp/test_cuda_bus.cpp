#include <cassert>
#include <cstdint>
#include <vector>

#include "nesle/console.hpp"
#include "nesle/controller.hpp"
#include "nesle/cuda/batch_bus.cuh"
#include "nesle/rom.hpp"

namespace {

nesle::RomImage make_nrom(std::size_t prg_size) {
    nesle::RomImage rom;
    rom.metadata.mapper = 0;
    rom.metadata.prg_rom_banks = static_cast<std::uint8_t>(prg_size / (16 * 1024));
    rom.metadata.chr_rom_banks = 1;
    rom.metadata.prg_rom_size = prg_size;
    rom.metadata.chr_rom_size = 8 * 1024;
    rom.prg_rom.resize(prg_size);
    rom.chr_rom.resize(8 * 1024);
    for (std::size_t i = 0; i < rom.prg_rom.size(); ++i) {
        rom.prg_rom[i] = static_cast<std::uint8_t>((i * 7 + 3) & 0xFF);
    }
    return rom;
}

}  // namespace

int main() {
    constexpr std::size_t kNumEnvs = 2;
    std::vector<std::uint8_t> ram(kNumEnvs * nesle::cuda::kCpuRamBytes, 0);
    std::vector<std::uint8_t> prg_ram(kNumEnvs * nesle::cuda::kPrgRamBytes, 0);
    std::vector<std::uint8_t> actions(kNumEnvs, 0);
    std::vector<std::uint8_t> controller_shift(kNumEnvs, 0);
    std::vector<std::uint8_t> controller_shift_count(kNumEnvs, 8);
    std::vector<std::uint8_t> controller_strobe(kNumEnvs, 0);
    std::vector<std::uint8_t> ppu_ctrl(kNumEnvs, 0);
    std::vector<std::uint8_t> ppu_mask(kNumEnvs, 0);
    std::vector<std::uint8_t> ppu_status(kNumEnvs, 0);
    std::vector<std::uint8_t> ppu_oam_addr(kNumEnvs, 0);

    auto rom = make_nrom(32 * 1024);

    nesle::cuda::BatchBuffers buffers{};
    buffers.cpu.ram = ram.data();
    buffers.cpu.prg_ram = prg_ram.data();
    buffers.cpu.controller1_shift = controller_shift.data();
    buffers.cpu.controller1_shift_count = controller_shift_count.data();
    buffers.cpu.controller1_strobe = controller_strobe.data();
    buffers.ppu.ctrl = ppu_ctrl.data();
    buffers.ppu.mask = ppu_mask.data();
    buffers.ppu.status = ppu_status.data();
    buffers.ppu.oam_addr = ppu_oam_addr.data();
    buffers.cart.prg_rom = rom.prg_rom.data();
    buffers.cart.prg_rom_size = static_cast<std::uint32_t>(rom.prg_rom.size());
    buffers.cart.mapper = 0;
    buffers.action_masks = actions.data();

    {
        nesle::Console console(make_nrom(32 * 1024));
        nesle::cuda::batch_cpu_write(buffers, 0, 0x0002, 0x12);
        console.write(0x0002, 0x12);
        assert(nesle::cuda::batch_cpu_read(buffers, 0, 0x0002) == console.read(0x0002));
        assert(nesle::cuda::batch_cpu_read(buffers, 0, 0x0802) == console.read(0x0802));
        assert(nesle::cuda::batch_cpu_read(buffers, 1, 0x0002) == 0);
    }

    {
        nesle::cuda::batch_cpu_write(buffers, 1, 0x6004, 0x9A);
        assert(nesle::cuda::batch_cpu_read(buffers, 1, 0x6004) == 0x9A);
        assert(nesle::cuda::batch_cpu_read(buffers, 0, 0x6004) == 0x00);
    }

    {
        nesle::Console console(make_nrom(32 * 1024));
        assert(nesle::cuda::batch_cpu_read(buffers, 0, 0x8000) == console.read(0x8000));
        assert(nesle::cuda::batch_cpu_read(buffers, 0, 0xBEEF) == console.read(0xBEEF));
        assert(nesle::cuda::batch_cpu_read(buffers, 0, 0xFFFF) == console.read(0xFFFF));
    }

    {
        auto small_rom = make_nrom(16 * 1024);
        buffers.cart.prg_rom = small_rom.prg_rom.data();
        buffers.cart.prg_rom_size = static_cast<std::uint32_t>(small_rom.prg_rom.size());
        nesle::Console console(make_nrom(16 * 1024));
        assert(nesle::cuda::batch_cpu_read(buffers, 0, 0x8000) == console.read(0x8000));
        assert(nesle::cuda::batch_cpu_read(buffers, 0, 0xC000) == console.read(0xC000));
        assert(nesle::cuda::batch_cpu_read(buffers, 0, 0xFFFF) == console.read(0xFFFF));
        buffers.cart.prg_rom = rom.prg_rom.data();
        buffers.cart.prg_rom_size = static_cast<std::uint32_t>(rom.prg_rom.size());
    }

    {
        nesle::cuda::batch_cpu_write(buffers, 0, 0x2000, 0x80);
        nesle::cuda::batch_cpu_write(buffers, 0, 0x2001, 0x1E);
        nesle::cuda::batch_cpu_write(buffers, 0, 0x2003, 0x44);
        ppu_status[0] = 0xA0;
        assert(ppu_ctrl[0] == 0x80);
        assert(ppu_mask[0] == 0x1E);
        assert(ppu_oam_addr[0] == 0x44);
        assert(nesle::cuda::batch_cpu_read(buffers, 0, 0x2002) == 0xA0);
    }

    {
        actions[0] = nesle::ButtonA | nesle::ButtonStart | nesle::ButtonRight;
        nesle::StandardController controller;
        controller.set_buttons(actions[0]);
        nesle::cuda::batch_cpu_write(buffers, 0, 0x4016, 1);
        controller.write_strobe(1);
        assert(nesle::cuda::batch_cpu_read(buffers, 0, 0x4016) == controller.read());
        nesle::cuda::batch_cpu_write(buffers, 0, 0x4016, 0);
        controller.write_strobe(0);
        for (int i = 0; i < 10; ++i) {
            assert(nesle::cuda::batch_cpu_read(buffers, 0, 0x4016) == controller.read());
        }
    }

    return 0;
}

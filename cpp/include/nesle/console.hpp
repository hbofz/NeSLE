#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "nesle/controller.hpp"
#include "nesle/cpu.hpp"
#include "nesle/ppu.hpp"
#include "nesle/rom.hpp"

namespace nesle {

class Console {
public:
    static constexpr std::size_t kCpuRamBytes = 2048;
    static constexpr std::size_t kPrgRamBytes = 8 * 1024;
    static constexpr std::size_t kApuIoBytes = 0x18;

    explicit Console(RomImage rom)
        : rom_(std::move(rom)) {
        if (!rom_.metadata.is_nrom()) {
            throw std::invalid_argument("Console currently supports only mapper 0/NROM ROMs");
        }
        if (rom_.prg_rom.empty()) {
            throw std::invalid_argument("Console requires PRG ROM bytes");
        }
    }

    [[nodiscard]] std::uint8_t read(std::uint16_t address) noexcept {
        if (address < 0x2000) {
            return cpu_ram_[address & 0x07FF];
        }
        if (address < 0x4000) {
            return ppu_.read_register(static_cast<std::uint16_t>((address - 0x2000) & 0x0007));
        }
        if (address == 0x4016) {
            return controller1_.read();
        }
        if (address == 0x4017) {
            return controller2_.read();
        }
        if (address < 0x4018) {
            return apu_io_[address - 0x4000];
        }
        if (address >= 0x6000 && address < 0x8000) {
            return prg_ram_[address - 0x6000];
        }
        if (address >= 0x8000) {
            return read_prg(address);
        }
        return 0;
    }

    void write(std::uint16_t address, std::uint8_t value) noexcept {
        if (address < 0x2000) {
            cpu_ram_[address & 0x07FF] = value;
            return;
        }
        if (address < 0x4000) {
            ppu_.write_register(static_cast<std::uint16_t>((address - 0x2000) & 0x0007), value);
            return;
        }
        if (address == 0x4014) {
            apu_io_[address - 0x4000] = value;
            run_oam_dma(value);
            return;
        }
        if (address == 0x4016) {
            apu_io_[address - 0x4000] = value;
            controller1_.write_strobe(value);
            controller2_.write_strobe(value);
            return;
        }
        if (address < 0x4018) {
            apu_io_[address - 0x4000] = value;
            return;
        }
        if (address >= 0x6000 && address < 0x8000) {
            prg_ram_[address - 0x6000] = value;
        }
    }

    void reset_cpu(cpu::CpuState& state) noexcept {
        state.variant = cpu::CpuVariant::Ricoh2A03;
        cpu::reset(state, *this);
    }

    [[nodiscard]] const RomImage& rom() const noexcept {
        return rom_;
    }

    [[nodiscard]] Ppu& ppu() noexcept {
        return ppu_;
    }

    [[nodiscard]] const Ppu& ppu() const noexcept {
        return ppu_;
    }

    [[nodiscard]] StandardController& controller1() noexcept {
        return controller1_;
    }

    [[nodiscard]] StandardController& controller2() noexcept {
        return controller2_;
    }

    [[nodiscard]] const std::array<std::uint8_t, kCpuRamBytes>& cpu_ram() const noexcept {
        return cpu_ram_;
    }

    [[nodiscard]] std::array<std::uint8_t, kCpuRamBytes>& cpu_ram() noexcept {
        return cpu_ram_;
    }

private:
    [[nodiscard]] std::uint8_t read_prg(std::uint16_t address) const noexcept {
        auto index = static_cast<std::size_t>(address - 0x8000);
        if (rom_.prg_rom.size() == 16 * 1024) {
            index &= 0x3FFF;
        } else {
            index &= 0x7FFF;
        }
        return rom_.prg_rom[index];
    }

    void run_oam_dma(std::uint8_t page) noexcept {
        const auto base = static_cast<std::uint16_t>(page << 8);
        for (std::uint16_t i = 0; i < 256; ++i) {
            ppu_.write_oam_dma(read(static_cast<std::uint16_t>(base + i)));
        }
    }

    RomImage rom_;
    std::array<std::uint8_t, kCpuRamBytes> cpu_ram_{};
    std::array<std::uint8_t, kPrgRamBytes> prg_ram_{};
    std::array<std::uint8_t, kApuIoBytes> apu_io_{};
    Ppu ppu_;
    StandardController controller1_;
    StandardController controller2_;
};

}  // namespace nesle

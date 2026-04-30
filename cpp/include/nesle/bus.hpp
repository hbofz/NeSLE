#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

#include "nesle/rom.hpp"

namespace nesle {

class FlatBus {
public:
    std::array<std::uint8_t, 65536> memory{};

    [[nodiscard]] std::uint8_t read(std::uint16_t address) const noexcept {
        return memory[address];
    }

    void write(std::uint16_t address, std::uint8_t value) noexcept {
        memory[address] = value;
    }
};

class NromBus {
public:
    static constexpr std::size_t kCpuRamBytes = 2048;
    static constexpr std::size_t kPpuRegisterBytes = 8;
    static constexpr std::size_t kApuIoRegisterBytes = 0x18;

    explicit NromBus(const RomImage& rom)
        : prg_rom_(rom.prg_rom) {
        if (!rom.metadata.is_nrom()) {
            throw std::invalid_argument("NromBus requires mapper 0 with one or two PRG banks");
        }
        if (rom.prg_rom.empty()) {
            throw std::invalid_argument("NromBus requires PRG ROM bytes");
        }
    }

    [[nodiscard]] std::uint8_t read(std::uint16_t address) const noexcept {
        if (address < 0x2000) {
            return cpu_ram_[address & 0x07FF];
        }
        if (address < 0x4000) {
            return ppu_registers_[(address - 0x2000) & 0x0007];
        }
        if (address < 0x4018) {
            return apu_io_registers_[address - 0x4000];
        }
        if (address >= 0x8000) {
            auto index = static_cast<std::size_t>(address - 0x8000);
            if (prg_rom_.size() == 16 * 1024) {
                index &= 0x3FFF;
            }
            return prg_rom_[index % prg_rom_.size()];
        }
        return 0;
    }

    void write(std::uint16_t address, std::uint8_t value) noexcept {
        if (address < 0x2000) {
            cpu_ram_[address & 0x07FF] = value;
            return;
        }
        if (address < 0x4000) {
            ppu_registers_[(address - 0x2000) & 0x0007] = value;
            return;
        }
        if (address < 0x4018) {
            apu_io_registers_[address - 0x4000] = value;
        }
    }

    [[nodiscard]] const std::array<std::uint8_t, kCpuRamBytes>& cpu_ram() const noexcept {
        return cpu_ram_;
    }

    [[nodiscard]] std::array<std::uint8_t, kCpuRamBytes>& cpu_ram() noexcept {
        return cpu_ram_;
    }

private:
    std::array<std::uint8_t, kCpuRamBytes> cpu_ram_{};
    std::array<std::uint8_t, kPpuRegisterBytes> ppu_registers_{};
    std::array<std::uint8_t, kApuIoRegisterBytes> apu_io_registers_{};
    std::span<const std::uint8_t> prg_rom_;
};

}  // namespace nesle

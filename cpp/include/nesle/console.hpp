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
    static constexpr std::uint32_t kPpuCyclesPerCpuCycle = 3;

    struct StepResult {
        cpu::StepResult cpu;
        std::uint32_t cpu_cycles = 0;
        std::uint32_t ppu_cycles = 0;
        std::uint32_t frames_completed = 0;
        bool nmi_serviced = false;
        bool nmi_started = false;
    };

    struct FrameResult {
        std::uint64_t instructions = 0;
        std::uint64_t cpu_cycles = 0;
        std::uint32_t frames_completed = 0;
    };

    explicit Console(RomImage rom)
        : rom_(std::move(rom)),
          layout_(describe_mapper(rom_.metadata)) {
        if (!layout_.supported) {
            throw std::invalid_argument(unsupported_mapper_reason(rom_.metadata));
        }
        if (rom_.prg_rom.empty()) {
            throw std::invalid_argument("Console requires PRG ROM bytes");
        }
        ppu_.configure_cartridge(rom_.chr_rom, rom_.metadata.nametable_arrangement);
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
            return read_prg_ram(address);
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
            return;
        }
        if (address >= 0x8000) {
            write_mapper_register(address, value);
        }
    }

    void reset_cpu(cpu::CpuState& state) noexcept {
        state.variant = cpu::CpuVariant::Ricoh2A03;
        cpu::reset(state, *this);
    }

    [[nodiscard]] StepResult step_cpu_instruction(cpu::CpuState& state) {
        const auto cycles_before = state.cycles;
        bool nmi_serviced = false;
        if (ppu_.nmi_pending()) {
            ppu_.clear_nmi_pending();
            cpu::nmi(state, *this);
            nmi_serviced = true;
        }

        auto cpu_step = cpu::step(state, *this);
        if (pending_dma_cycles_ != 0) {
            state.cycles += pending_dma_cycles_;
            pending_dma_cycles_ = 0;
        }

        const auto cpu_cycles = static_cast<std::uint32_t>(state.cycles - cycles_before);
        const auto ppu_cycles = cpu_cycles * kPpuCyclesPerCpuCycle;
        const auto ppu_step = ppu_.step(ppu_cycles);

        return StepResult{
            cpu_step,
            cpu_cycles,
            ppu_cycles,
            ppu_step.frames_completed,
            nmi_serviced,
            ppu_step.nmi_started,
        };
    }

    [[nodiscard]] FrameResult step_frame(cpu::CpuState& state, std::uint64_t max_instructions) {
        FrameResult result;
        const auto cycles_before = state.cycles;
        while (result.instructions < max_instructions) {
            const auto step = step_cpu_instruction(state);
            ++result.instructions;
            result.frames_completed += step.frames_completed;
            if (result.frames_completed != 0) {
                break;
            }
        }
        result.cpu_cycles = state.cycles - cycles_before;
        return result;
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
    [[nodiscard]] std::uint8_t read_prg_ram(std::uint16_t address) const noexcept {
        return prg_ram_[address - 0x6000];
    }

    void write_mapper_register(std::uint16_t address, std::uint8_t value) noexcept {
        if (layout_.window_bytes == 0) {
            return;  // NROM: no registers, the write goes nowhere
        }
        std::uint8_t latched = value;
        if (layout_.bus_conflicts) {
            latched = static_cast<std::uint8_t>(value & read_prg(address));
        }

        prg_bank_ = static_cast<std::uint8_t>((latched >> layout_.bank_shift) & layout_.bank_mask);
    }

    [[nodiscard]] std::uint8_t read_prg(std::uint16_t address) const noexcept {
        if (layout_.window_bytes == 0) {
            auto index = static_cast<std::size_t>(address - 0x8000);
            if (rom_.prg_rom.size() == 16 * 1024) {
                index &= 0x3FFF;
            } else {
                index &= 0x7FFF;
            }
            return rom_.prg_rom[index];
        }
        std::size_t index;
        const std::uint32_t offset = static_cast<std::uint32_t>(address) - 0x8000u;
        if (offset <= layout_.window_bytes - 1) {
            index = (static_cast<std::size_t>(prg_bank_ & layout_.bank_mask) * layout_.window_bytes) +
                    offset;
        } else {
            // The fixed window is the last fixed_bytes of the image, which for
            // every board here is the top 16 KB.
            const std::size_t fixed_base = rom_.prg_rom.size() - layout_.fixed_bytes;
            index = fixed_base + (address & (layout_.fixed_bytes - 1));
        }
        return rom_.prg_rom[index % rom_.prg_rom.size()];
    }

    void run_oam_dma(std::uint8_t page) noexcept {
        const auto base = static_cast<std::uint16_t>(page << 8);
        for (std::uint16_t i = 0; i < 256; ++i) {
            ppu_.write_oam_dma(read(static_cast<std::uint16_t>(base + i)));
        }
        pending_dma_cycles_ += 513;
    }

    RomImage rom_;
    MapperLayout layout_;
    std::uint8_t prg_bank_ = 0;
    std::array<std::uint8_t, kCpuRamBytes> cpu_ram_{};
    std::array<std::uint8_t, kPrgRamBytes> prg_ram_{};
    std::array<std::uint8_t, kApuIoBytes> apu_io_{};
    Ppu ppu_;
    StandardController controller1_;
    StandardController controller2_;
    std::uint32_t pending_dma_cycles_ = 0;
};

}  // namespace nesle

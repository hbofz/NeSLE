#pragma once

#include <cstdint>
#include <exception>
#include <string>

#include "nesle/console.hpp"
#include "nesle/cpu.hpp"

namespace nesle {

enum class HeadlessRunStatus {
    Completed,
    Timeout,
    CpuException,
    Trap,
};

struct HeadlessRunConfig {
    std::uint32_t frames = 1;
    std::uint64_t max_instructions = 10'000'000;
    bool stop_on_trap = true;
};

struct HeadlessRunResult {
    HeadlessRunStatus status = HeadlessRunStatus::Timeout;
    std::uint32_t frames_completed = 0;
    std::uint64_t instructions = 0;
    std::uint64_t cpu_cycles = 0;
    std::uint16_t pc = 0;
    std::uint8_t opcode = 0;
    std::uint64_t ppu_frame = 0;
    std::int16_t ppu_scanline = 0;
    std::uint16_t ppu_dot = 0;
    std::string message;

    [[nodiscard]] bool completed() const noexcept {
        return status == HeadlessRunStatus::Completed;
    }
};

[[nodiscard]] inline const char* to_string(HeadlessRunStatus status) noexcept {
    switch (status) {
        case HeadlessRunStatus::Completed:
            return "completed";
        case HeadlessRunStatus::Timeout:
            return "timeout";
        case HeadlessRunStatus::CpuException:
            return "cpu_exception";
        case HeadlessRunStatus::Trap:
            return "trap";
    }
    return "unknown";
}

[[nodiscard]] inline HeadlessRunResult run_headless(Console& console,
                                                   cpu::CpuState& state,
                                                   HeadlessRunConfig config) {
    HeadlessRunResult result;
    const auto cycles_before = state.cycles;

    while (result.instructions < config.max_instructions &&
           result.frames_completed < config.frames) {
        try {
            const auto step = console.step_cpu_instruction(state);
            ++result.instructions;
            result.frames_completed += step.frames_completed;
            result.opcode = step.cpu.opcode;

            if (config.stop_on_trap && state.pc == step.cpu.pc) {
                result.status = HeadlessRunStatus::Trap;
                result.message = "program counter is looping on itself";
                break;
            }
        } catch (const std::exception& error) {
            result.status = HeadlessRunStatus::CpuException;
            result.message = error.what();
            break;
        }
    }

    if (result.status == HeadlessRunStatus::Timeout) {
        result.status = result.frames_completed >= config.frames ? HeadlessRunStatus::Completed
                                                                 : HeadlessRunStatus::Timeout;
    }

    result.cpu_cycles = state.cycles - cycles_before;
    result.pc = state.pc;
    result.ppu_frame = console.ppu().frame();
    result.ppu_scanline = console.ppu().scanline();
    result.ppu_dot = console.ppu().dot();
    return result;
}

}  // namespace nesle

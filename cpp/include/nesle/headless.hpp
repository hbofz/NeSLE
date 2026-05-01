#pragma once

#include <cstdint>
#include <exception>
#include <cstddef>
#include <string>
#include <vector>

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
    std::size_t trace_capacity = 0;
    bool stop_on_trap = true;
    std::vector<std::uint8_t> controller1_frame_actions;
};

struct HeadlessTraceEntry {
    std::uint64_t instruction = 0;
    std::uint16_t pc = 0;
    std::uint8_t opcode = 0;
    std::uint32_t cpu_cycles = 0;
    std::uint64_t total_cpu_cycles = 0;
    std::uint64_t ppu_frame = 0;
    std::int16_t ppu_scanline = 0;
    std::uint16_t ppu_dot = 0;
    std::uint32_t frames_completed = 0;
    bool nmi_serviced = false;
    bool nmi_started = false;
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
    std::vector<HeadlessTraceEntry> trace;

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
    std::size_t trace_cursor = 0;
    if (config.trace_capacity != 0) {
        result.trace.reserve(config.trace_capacity);
    }
    if (!config.controller1_frame_actions.empty()) {
        console.controller1().set_buttons(config.controller1_frame_actions.front());
    }

    auto append_trace = [&](const Console::StepResult& step) {
        if (config.trace_capacity == 0) {
            return;
        }

        HeadlessTraceEntry entry;
        entry.instruction = result.instructions;
        entry.pc = step.cpu.pc;
        entry.opcode = step.cpu.opcode;
        entry.cpu_cycles = step.cpu_cycles;
        entry.total_cpu_cycles = state.cycles;
        entry.ppu_frame = console.ppu().frame();
        entry.ppu_scanline = console.ppu().scanline();
        entry.ppu_dot = console.ppu().dot();
        entry.frames_completed = step.frames_completed;
        entry.nmi_serviced = step.nmi_serviced;
        entry.nmi_started = step.nmi_started;

        if (result.trace.size() < config.trace_capacity) {
            result.trace.push_back(entry);
            return;
        }

        result.trace[trace_cursor] = entry;
        trace_cursor = (trace_cursor + 1) % config.trace_capacity;
    };

    auto rotate_trace = [&]() {
        if (trace_cursor == 0 || result.trace.size() < config.trace_capacity) {
            return;
        }

        std::vector<HeadlessTraceEntry> ordered;
        ordered.reserve(result.trace.size());
        for (std::size_t i = 0; i < result.trace.size(); ++i) {
            ordered.push_back(result.trace[(trace_cursor + i) % result.trace.size()]);
        }
        result.trace = std::move(ordered);
    };

    while (result.instructions < config.max_instructions &&
           result.frames_completed < config.frames) {
        try {
            const auto step = console.step_cpu_instruction(state);
            ++result.instructions;
            result.frames_completed += step.frames_completed;
            result.opcode = step.cpu.opcode;
            append_trace(step);
            if (step.frames_completed != 0 &&
                result.frames_completed < config.controller1_frame_actions.size()) {
                console.controller1().set_buttons(
                    config.controller1_frame_actions[result.frames_completed]);
            }

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
    rotate_trace();
    return result;
}

}  // namespace nesle

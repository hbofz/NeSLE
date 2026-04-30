#pragma once

#include <cstdint>
#include <exception>
#include <string>

#include "nesle/cpu.hpp"

namespace nesle::cpu {

enum class RunStatus {
    SuccessTrap,
    FailureTrap,
    Timeout,
    CpuException,
};

struct RunResult {
    RunStatus status = RunStatus::Timeout;
    std::uint16_t pc = 0;
    std::uint8_t opcode = 0;
    std::uint64_t instructions = 0;
    std::uint64_t cycles = 0;
    std::string message;

    [[nodiscard]] bool passed() const noexcept {
        return status == RunStatus::SuccessTrap;
    }
};

[[nodiscard]] inline const char* to_string(RunStatus status) noexcept {
    switch (status) {
        case RunStatus::SuccessTrap:
            return "success";
        case RunStatus::FailureTrap:
            return "failure";
        case RunStatus::Timeout:
            return "timeout";
        case RunStatus::CpuException:
            return "cpu_exception";
    }
    return "unknown";
}

template <typename Bus>
RunResult run_until_trap(CpuState& state,
                         Bus& bus,
                         std::uint16_t success_pc,
                         std::uint64_t max_instructions) {
    RunResult result;
    for (std::uint64_t instruction = 0; instruction < max_instructions; ++instruction) {
        const auto previous_pc = state.pc;
        try {
            const auto step_result = step(state, bus);
            result.opcode = step_result.opcode;
        } catch (const std::exception& error) {
            result.status = RunStatus::CpuException;
            result.pc = previous_pc;
            result.instructions = instruction;
            result.cycles = state.cycles;
            result.message = error.what();
            return result;
        }

        if (state.pc == previous_pc) {
            result.status = state.pc == success_pc ? RunStatus::SuccessTrap : RunStatus::FailureTrap;
            result.pc = state.pc;
            result.instructions = instruction + 1;
            result.cycles = state.cycles;
            return result;
        }
    }

    result.status = RunStatus::Timeout;
    result.pc = state.pc;
    result.instructions = max_instructions;
    result.cycles = state.cycles;
    return result;
}

}  // namespace nesle::cpu

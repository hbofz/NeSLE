#pragma once

#include <cstdint>
#include <exception>
#include <string>
#include <vector>

#include "nesle/cuda/batch_cpu.hpp"

namespace nesle::cuda {

enum class BatchRunStatus {
    Completed,
    Timeout,
    CpuException,
    Trap,
};

struct BatchRunConfig {
    std::uint32_t num_envs = 0;
    std::uint64_t max_instructions = 1;
    bool stop_on_trap = true;
};

struct BatchEnvResult {
    BatchRunStatus status = BatchRunStatus::Timeout;
    std::uint64_t instructions = 0;
    std::uint16_t pc = 0;
    std::uint8_t opcode = 0;
    std::uint64_t cycles = 0;
    std::string message;
};

struct BatchRunResult {
    std::vector<BatchEnvResult> envs;
};

[[nodiscard]] inline const char* to_string(BatchRunStatus status) noexcept {
    switch (status) {
        case BatchRunStatus::Completed:
            return "completed";
        case BatchRunStatus::Timeout:
            return "timeout";
        case BatchRunStatus::CpuException:
            return "cpu_exception";
        case BatchRunStatus::Trap:
            return "trap";
    }
    return "unknown";
}

[[nodiscard]] inline BatchRunResult run_batch_cpu(BatchBuffers& buffers,
                                                  BatchRunConfig config) {
    BatchRunResult result;
    result.envs.resize(config.num_envs);
    std::uint32_t active = config.num_envs;

    for (std::uint64_t instruction = 0;
         instruction < config.max_instructions && active != 0;
         ++instruction) {
        for (std::uint32_t env = 0; env < config.num_envs; ++env) {
            auto& env_result = result.envs[env];
            if (env_result.status != BatchRunStatus::Timeout) {
                continue;
            }

            const auto previous_pc = buffers.cpu.pc[env];
            try {
                const auto step = step_batch_cpu_env(buffers, env);
                ++env_result.instructions;
                env_result.pc = buffers.cpu.pc[env];
                env_result.opcode = step.opcode;
                env_result.cycles = buffers.cpu.cycles[env];

                if (config.stop_on_trap && buffers.cpu.pc[env] == previous_pc) {
                    env_result.status = BatchRunStatus::Trap;
                    env_result.message = "program counter is looping on itself";
                    buffers.done[env] = 1;
                    --active;
                }
            } catch (const std::exception& error) {
                env_result.status = BatchRunStatus::CpuException;
                env_result.pc = previous_pc;
                env_result.cycles = buffers.cpu.cycles[env];
                env_result.message = error.what();
                buffers.done[env] = 1;
                --active;
            }
        }
    }

    for (std::uint32_t env = 0; env < config.num_envs; ++env) {
        auto& env_result = result.envs[env];
        if (env_result.status == BatchRunStatus::Timeout &&
            env_result.instructions >= config.max_instructions) {
            env_result.status = BatchRunStatus::Completed;
            buffers.done[env] = 1;
        }
    }

    return result;
}

}  // namespace nesle::cuda

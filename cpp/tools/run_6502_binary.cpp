#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "nesle/bus.hpp"
#include "nesle/cpu.hpp"
#include "nesle/cpu_runner.hpp"

namespace {

struct Config {
    std::string path;
    std::uint16_t load_address = 0x0000;
    std::uint16_t start_pc = 0x0400;
    std::uint16_t success_pc = 0x3469;
    std::uint64_t max_instructions = 100'000'000;
    nesle::cpu::CpuVariant variant = nesle::cpu::CpuVariant::Ricoh2A03;
};

std::uint16_t parse_u16(const std::string& value) {
    const auto parsed = std::stoul(value, nullptr, 0);
    if (parsed > 0xFFFF) {
        throw std::invalid_argument("16-bit address is out of range: " + value);
    }
    return static_cast<std::uint16_t>(parsed);
}

std::uint64_t parse_u64(const std::string& value) {
    return std::stoull(value, nullptr, 0);
}

nesle::cpu::CpuVariant parse_variant(const std::string& value) {
    if (value == "2a03" || value == "ricoh2a03" || value == "nes") {
        return nesle::cpu::CpuVariant::Ricoh2A03;
    }
    if (value == "mos6502" || value == "6502") {
        return nesle::cpu::CpuVariant::Mos6502;
    }
    throw std::invalid_argument("variant must be one of: 2a03, mos6502");
}

Config parse_args(int argc, char** argv) {
    if (argc < 2) {
        throw std::invalid_argument(
            "usage: run_6502_binary <path> [--load 0x0000] [--start 0x0400] "
            "[--success 0x3469] [--max-instructions N] [--variant 2a03|mos6502]");
    }

    Config config;
    config.path = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i + 1 >= argc) {
            throw std::invalid_argument("missing value for " + arg);
        }
        const std::string value = argv[++i];
        if (arg == "--load") {
            config.load_address = parse_u16(value);
        } else if (arg == "--start") {
            config.start_pc = parse_u16(value);
        } else if (arg == "--success") {
            config.success_pc = parse_u16(value);
        } else if (arg == "--max-instructions") {
            config.max_instructions = parse_u64(value);
        } else if (arg == "--variant") {
            config.variant = parse_variant(value);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    return config;
}

std::vector<std::uint8_t> read_binary(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open binary: " + path);
    }
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto config = parse_args(argc, argv);
        const auto bytes = read_binary(config.path);

        nesle::FlatBus bus;
        bus.load(std::span<const std::uint8_t>(bytes.data(), bytes.size()), config.load_address);

        nesle::cpu::CpuState state;
        state.pc = config.start_pc;
        state.variant = config.variant;
        const auto result = nesle::cpu::run_until_trap(state, bus, config.success_pc, config.max_instructions);

        std::cout << nesle::cpu::to_string(result.status)
                  << " pc=0x" << std::hex << result.pc
                  << " opcode=0x" << static_cast<unsigned>(result.opcode)
                  << std::dec << " instructions=" << result.instructions
                  << " cycles=" << result.cycles;
        if (!result.message.empty()) {
            std::cout << " message=\"" << result.message << "\"";
        }
        std::cout << '\n';
        return result.passed() ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

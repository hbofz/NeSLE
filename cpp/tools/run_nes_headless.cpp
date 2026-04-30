#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include "nesle/console.hpp"
#include "nesle/cpu.hpp"
#include "nesle/file.hpp"
#include "nesle/headless.hpp"

namespace {

struct Config {
    std::string path;
    nesle::HeadlessRunConfig run;
};

std::uint32_t parse_u32(const std::string& value) {
    const auto parsed = std::stoull(value, nullptr, 0);
    if (parsed > 0xFFFF'FFFFull) {
        throw std::invalid_argument("32-bit value is out of range: " + value);
    }
    return static_cast<std::uint32_t>(parsed);
}

std::uint64_t parse_u64(const std::string& value) {
    return std::stoull(value, nullptr, 0);
}

Config parse_args(int argc, char** argv) {
    if (argc < 2) {
        throw std::invalid_argument(
            "usage: run_nes_headless <rom.nes> [--frames N] "
            "[--max-instructions N] [--allow-trap]");
    }

    Config config;
    config.path = argv[1];

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--allow-trap") {
            config.run.stop_on_trap = false;
            continue;
        }

        if (i + 1 >= argc) {
            throw std::invalid_argument("missing value for " + arg);
        }
        const std::string value = argv[++i];
        if (arg == "--frames") {
            config.run.frames = parse_u32(value);
        } else if (arg == "--max-instructions") {
            config.run.max_instructions = parse_u64(value);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    return config;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto config = parse_args(argc, argv);
        auto rom = nesle::load_ines_file(config.path);
        const auto metadata = rom.metadata;
        nesle::Console console(std::move(rom));
        nesle::cpu::CpuState state;
        console.reset_cpu(state);

        const auto result = nesle::run_headless(console, state, config.run);

        std::cout << nesle::to_string(result.status)
                  << " frames=" << result.frames_completed
                  << " instructions=" << result.instructions
                  << " cpu_cycles=" << result.cpu_cycles
                  << " pc=0x" << std::hex << result.pc
                  << " opcode=0x" << static_cast<unsigned>(result.opcode)
                  << std::dec
                  << " ppu_frame=" << result.ppu_frame
                  << " ppu_scanline=" << result.ppu_scanline
                  << " ppu_dot=" << result.ppu_dot
                  << " mapper=" << metadata.mapper
                  << " prg=" << metadata.prg_rom_size
                  << " chr=" << metadata.chr_rom_size;
        if (!result.message.empty()) {
            std::cout << " message=\"" << result.message << "\"";
        }
        std::cout << '\n';

        return result.completed() ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

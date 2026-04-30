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
#include "nesle/smb.hpp"

namespace {

struct Config {
    std::string path;
    nesle::HeadlessRunConfig run;
    bool print_trace = false;
    bool require_mario_target = false;
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
            "[--max-instructions N] [--trace N] [--allow-trap] "
            "[--require-mario-target]");
    }

    Config config;
    config.path = argv[1];

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--allow-trap") {
            config.run.stop_on_trap = false;
            continue;
        }
        if (arg == "--require-mario-target") {
            config.require_mario_target = true;
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
        } else if (arg == "--trace") {
            config.run.trace_capacity = static_cast<std::size_t>(parse_u64(value));
            config.print_trace = config.run.trace_capacity != 0;
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
        if (config.require_mario_target) {
            nesle::validate_supported_mario_target(metadata);
        }
        nesle::Console console(std::move(rom));
        nesle::cpu::CpuState state;
        console.reset_cpu(state);

        const auto result = nesle::run_headless(console, state, config.run);
        const auto mario = nesle::smb::read_ram(console.cpu_ram());

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
                  << " chr=" << metadata.chr_rom_size
                  << " mario_target=" << nesle::is_supported_mario_target(metadata)
                  << " mario_x=" << mario.x_pos
                  << " mario_y=" << mario.y_pos
                  << " mario_world=" << mario.world
                  << " mario_stage=" << mario.stage
                  << " mario_area=" << mario.area
                  << " mario_time=" << mario.time
                  << " mario_coins=" << mario.coins
                  << " mario_lives=" << mario.lives
                  << " mario_status=" << nesle::smb::status_name(mario.status_code)
                  << " mario_player_state=" << mario.player_state
                  << " mario_flag_get=" << mario.flag_get
                  << " mario_dying=" << mario.is_dying
                  << " mario_dead=" << mario.is_dead
                  << " mario_game_over=" << mario.is_game_over;
        if (!result.message.empty()) {
            std::cout << " message=\"" << result.message << "\"";
        }
        std::cout << '\n';

        if (config.print_trace) {
            for (const auto& entry : result.trace) {
                std::cout << "trace"
                          << " instruction=" << entry.instruction
                          << " pc=0x" << std::hex << entry.pc
                          << " opcode=0x" << static_cast<unsigned>(entry.opcode)
                          << std::dec
                          << " cpu_cycles=" << entry.cpu_cycles
                          << " total_cpu_cycles=" << entry.total_cpu_cycles
                          << " ppu_frame=" << entry.ppu_frame
                          << " ppu_scanline=" << entry.ppu_scanline
                          << " ppu_dot=" << entry.ppu_dot
                          << " frames_completed=" << entry.frames_completed
                          << " nmi_serviced=" << entry.nmi_serviced
                          << " nmi_started=" << entry.nmi_started
                          << '\n';
            }
        }

        return result.completed() ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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
    bool require_mario_boot = false;
    bool frames_explicit = false;
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

std::vector<std::uint8_t> parse_actions(const std::string& value) {
    std::vector<std::uint8_t> actions;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (token.empty()) {
            continue;
        }
        std::uint64_t repeat = 1;
        const auto repeat_delimiter = token.find('*');
        if (repeat_delimiter != std::string::npos) {
            repeat = parse_u64(token.substr(repeat_delimiter + 1));
            token = token.substr(0, repeat_delimiter);
            if (repeat == 0) {
                throw std::invalid_argument("controller action repeat must be non-zero");
            }
        }
        const auto parsed = parse_u64(token);
        if (parsed > 0xFF) {
            throw std::invalid_argument("controller action mask is out of range: " + token);
        }
        actions.insert(actions.end(), static_cast<std::size_t>(repeat), static_cast<std::uint8_t>(parsed));
    }
    if (actions.empty()) {
        throw std::invalid_argument("--actions must contain at least one controller mask");
    }
    return actions;
}

std::uint64_t fnv1a64(std::span<const std::uint8_t> bytes) noexcept {
    std::uint64_t hash = 14695981039346656037ull;
    for (const auto byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    return hash;
}

std::string hex64(std::uint64_t value) {
    std::ostringstream out;
    out << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}

Config parse_args(int argc, char** argv) {
    if (argc < 2) {
        throw std::invalid_argument(
            "usage: run_nes_headless <rom.nes> [--frames N] "
            "[--max-instructions N] [--trace N] [--actions masks] [--allow-trap] "
            "[--require-mario-target] [--require-mario-boot]");
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
        if (arg == "--require-mario-boot") {
            config.require_mario_boot = true;
            continue;
        }

        if (i + 1 >= argc) {
            throw std::invalid_argument("missing value for " + arg);
        }
        const std::string value = argv[++i];
        if (arg == "--frames") {
            config.run.frames = parse_u32(value);
            config.frames_explicit = true;
        } else if (arg == "--max-instructions") {
            config.run.max_instructions = parse_u64(value);
        } else if (arg == "--trace") {
            config.run.trace_capacity = static_cast<std::size_t>(parse_u64(value));
            config.print_trace = config.run.trace_capacity != 0;
        } else if (arg == "--actions") {
            config.run.controller1_frame_actions = parse_actions(value);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (!config.frames_explicit && !config.run.controller1_frame_actions.empty()) {
        config.run.frames = static_cast<std::uint32_t>(config.run.controller1_frame_actions.size());
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
        const auto initial_mario = nesle::smb::read_ram(console.cpu_ram());

        nesle::HeadlessRunResult result;
        nesle::smb::RewardComponents reward;
        if (config.run.controller1_frame_actions.empty()) {
            result = nesle::run_headless(console, state, config.run);
            reward = nesle::smb::compute_reward(initial_mario, nesle::smb::read_ram(console.cpu_ram()));
        } else {
            auto previous_mario = initial_mario;
            std::uint64_t total_instructions = 0;
            std::uint64_t total_cpu_cycles = 0;
            std::uint32_t total_frames = 0;

            for (std::uint32_t frame_index = 0; frame_index < config.run.frames; ++frame_index) {
                const auto action_index = std::min<std::size_t>(
                    frame_index,
                    config.run.controller1_frame_actions.size() - 1);
                console.controller1().set_buttons(config.run.controller1_frame_actions[action_index]);

                auto frame_config = config.run;
                frame_config.frames = 1;
                frame_config.controller1_frame_actions.clear();
                frame_config.trace_capacity =
                    frame_index + 1 == config.run.frames ? config.run.trace_capacity : 0;
                if (total_instructions >= config.run.max_instructions) {
                    frame_config.max_instructions = 0;
                } else {
                    frame_config.max_instructions = config.run.max_instructions - total_instructions;
                }

                auto frame_result = nesle::run_headless(console, state, frame_config);
                total_instructions += frame_result.instructions;
                total_cpu_cycles += frame_result.cpu_cycles;
                total_frames += frame_result.frames_completed;

                const auto current_mario = nesle::smb::read_ram(console.cpu_ram());
                const auto frame_reward = nesle::smb::compute_reward(previous_mario, current_mario);
                reward.x += frame_reward.x;
                reward.time += frame_reward.time;
                reward.death += frame_reward.death;
                reward.total += frame_reward.total;
                previous_mario = current_mario;

                result = std::move(frame_result);
                result.instructions = total_instructions;
                result.cpu_cycles = total_cpu_cycles;
                result.frames_completed = total_frames;
                if (!result.completed()) {
                    break;
                }
            }
        }
        const auto mario = nesle::smb::read_ram(console.cpu_ram());
        const auto boot_reason = nesle::smb::implausible_boot_state_reason(mario);
        const auto frame = console.ppu().render_rgb_frame();
        const auto frame_hash = fnv1a64({frame.data(), frame.size()});
        const auto ram_hash = fnv1a64({console.cpu_ram().data(), console.cpu_ram().size()});

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
                  << " mario_game_over=" << mario.is_game_over
                  << " mario_boot_plausible=" << boot_reason.empty()
                  << " reward_x=" << reward.x
                  << " reward_time=" << reward.time
                  << " reward_death=" << reward.death
                  << " reward_total=" << reward.total
                  << " ram_hash=" << hex64(ram_hash)
                  << " frame_hash=" << hex64(frame_hash);
        if (!boot_reason.empty()) {
            std::cout << " mario_boot_reason=\"" << boot_reason << "\"";
        }
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

        if (config.require_mario_boot && !boot_reason.empty()) {
            return EXIT_FAILURE;
        }
        return result.completed() ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

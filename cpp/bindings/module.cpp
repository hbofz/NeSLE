#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <string>
#include <vector>

#include "nesle/console.hpp"
#include "nesle/cpu.hpp"
#include "nesle/rom.hpp"
#include "nesle/smb.hpp"

namespace py = pybind11;

namespace {

std::vector<std::uint8_t> bytes_to_vector(const py::bytes& bytes) {
    const std::string raw = bytes;
    return {raw.begin(), raw.end()};
}

py::dict rom_metadata_to_dict(const nesle::RomMetadata& metadata) {
    py::dict out;
    out["prg_rom_banks"] = metadata.prg_rom_banks;
    out["chr_rom_banks"] = metadata.chr_rom_banks;
    out["mapper"] = metadata.mapper;
    out["submapper"] = metadata.submapper;
    out["has_trainer"] = metadata.has_trainer;
    out["has_battery"] = metadata.has_battery;
    out["is_nes2"] = metadata.is_nes2;
    out["nametable_arrangement"] = nesle::to_string(metadata.nametable_arrangement);
    out["prg_rom_size"] = metadata.prg_rom_size;
    out["chr_rom_size"] = metadata.chr_rom_size;
    out["is_nrom"] = metadata.is_nrom();
    out["is_supported_mario_target"] = nesle::is_supported_mario_target(metadata);
    return out;
}

py::dict mario_state_to_dict(const nesle::smb::MarioRamState& state) {
    py::dict out;
    out["x_pos"] = state.x_pos;
    out["y_pos"] = state.y_pos;
    out["time"] = state.time;
    out["coins"] = state.coins;
    out["score"] = state.score;
    out["life"] = state.lives;
    out["world"] = state.world;
    out["stage"] = state.stage;
    out["area"] = state.area;
    out["status"] = nesle::smb::status_name(state.status_code);
    out["status_code"] = state.status_code;
    out["player_state"] = state.player_state;
    out["flag_get"] = state.flag_get;
    out["is_dying"] = state.is_dying;
    out["is_dead"] = state.is_dead;
    out["is_game_over"] = state.is_game_over;
    return out;
}

class NativeConsoleBinding {
public:
    explicit NativeConsoleBinding(const py::bytes& bytes)
        : console_(nesle::parse_ines(bytes_to_vector(bytes))) {
        reset();
    }

    void reset() {
        state_ = nesle::cpu::CpuState{};
        console_.reset_cpu(state_);
    }

    py::dict step(std::uint8_t action_mask,
                  std::uint32_t frameskip,
                  std::uint64_t max_instructions_per_frame) {
        console_.controller1().set_buttons(action_mask);
        std::uint64_t instructions = 0;
        std::uint64_t cpu_cycles = 0;
        std::uint32_t frames_completed = 0;
        for (std::uint32_t frame = 0; frame < frameskip; ++frame) {
            const auto result = console_.step_frame(state_, max_instructions_per_frame);
            instructions += result.instructions;
            cpu_cycles += result.cpu_cycles;
            frames_completed += result.frames_completed;
        }

        py::dict out;
        out["instructions"] = instructions;
        out["cpu_cycles"] = cpu_cycles;
        out["frames_completed"] = frames_completed;
        out["pc"] = state_.pc;
        return out;
    }

    py::bytes ram() const {
        const auto& ram = console_.cpu_ram();
        return py::bytes(reinterpret_cast<const char*>(ram.data()), ram.size());
    }

    py::bytes frame() const {
        const auto frame = console_.ppu().render_rgb_frame();
        return py::bytes(reinterpret_cast<const char*>(frame.data()), frame.size());
    }

private:
    nesle::Console console_;
    nesle::cpu::CpuState state_;
};

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "Native NeSLE core helpers";

    m.def("parse_ines_metadata", [](const py::bytes& bytes) {
        const auto data = bytes_to_vector(bytes);
        return rom_metadata_to_dict(nesle::parse_ines(data).metadata);
    });

    m.def("read_mario_ram", [](const py::bytes& bytes) {
        const auto data = bytes_to_vector(bytes);
        return mario_state_to_dict(nesle::smb::read_ram(data));
    });

    py::class_<NativeConsoleBinding>(m, "NativeConsole")
        .def(py::init<const py::bytes&>())
        .def("reset", &NativeConsoleBinding::reset)
        .def("step", &NativeConsoleBinding::step)
        .def("ram", &NativeConsoleBinding::ram)
        .def("frame", &NativeConsoleBinding::frame);
}

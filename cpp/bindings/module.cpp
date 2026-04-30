#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <string>
#include <vector>

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
}

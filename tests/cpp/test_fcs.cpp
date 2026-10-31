#include <cassert>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "nesle/fcs.hpp"

namespace {

using Bytes = std::vector<std::uint8_t>;

void put_u32(Bytes& out, std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        out.push_back(static_cast<std::uint8_t>(value >> shift));
    }
}

void add_subchunk(Bytes& section, std::string_view name, std::size_t size) {
    std::string padded(name);
    padded.resize(4, '\0');
    section.insert(section.end(), padded.begin(), padded.end());
    put_u32(section, static_cast<std::uint32_t>(size));
    section.insert(section.end(), size, 0);
}

Bytes section(std::initializer_list<std::pair<std::string_view, std::size_t>> subchunks) {
    Bytes out;
    for (const auto& [name, size] : subchunks) {
        add_subchunk(out, name, size);
    }
    return out;
}

Bytes file(std::initializer_list<std::pair<std::uint8_t, Bytes>> chunks) {
    Bytes out = {'F', 'C', 'S', 0xFF};
    out.resize(16, 0);
    for (const auto& [tag, payload] : chunks) {
        out.push_back(tag);
        put_u32(out, static_cast<std::uint32_t>(payload.size()));
        out.insert(out.end(), payload.begin(), payload.end());
    }
    return out;
}

bool rejected(const Bytes& bytes) {
    try {
        (void)nesle::fcs::parse(std::span<const std::uint8_t>(bytes));
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

const Bytes kCpu = section({{"PC", 2}, {"RAM", 2048}});
const Bytes kPpu = section({{"NTAR", 2048}, {"PRAM", 32}, {"SPRA", 256}, {"PPUR", 4}});

void parses_a_real_fceux_save() {
    // Written by FCEUX 2.6.5 with no compression; the chunks start after its 16-byte header.
    std::ifstream in("docs/data/smb_level1_1_fceux.fc1", std::ios::binary);
    assert(in);
    Bytes bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    assert(bytes.size() > 16 && bytes[3] == 'X');
    bytes[3] = 0xFF;

    const auto snapshot = nesle::fcs::parse(std::span<const std::uint8_t>(bytes));
    assert(snapshot.pc == 0x8057);
    assert(snapshot.cpu_ram[0x0770] == 1);  // SMB's OperMode: in the game, past the title
    assert(snapshot.cpu_ram[0x075F] == 0);  // World 1
    assert(snapshot.palette_ram[0] != 0 || snapshot.palette_ram[1] != 0);
}

void accepts_a_state_with_every_core_section() {
    assert(!rejected(file({{1, kCpu}, {3, kPpu}})));
}

void rejects_states_that_do_not_describe_a_machine() {
    assert(rejected(file({})));
    assert(rejected(file({{1, kCpu}})));
    assert(rejected(file({{3, kPpu}})));
    assert(rejected(file({{2, Bytes(8, 0)}})));
    assert(rejected(file({{1, section({{"PC", 2}})}, {3, kPpu}})));
    assert(rejected(file({{1, section({{"RAM", 2048}})}, {3, kPpu}})));
    for (const auto missing : {"NTAR", "PRAM", "SPRA", "PPUR"}) {
        Bytes ppu;
        for (const auto& [name, size] : std::initializer_list<std::pair<std::string_view, std::size_t>>{
                 {"NTAR", 2048}, {"PRAM", 32}, {"SPRA", 256}, {"PPUR", 4}}) {
            if (name != missing) {
                add_subchunk(ppu, name, size);
            }
        }
        assert(rejected(file({{1, kCpu}, {3, ppu}})));
    }
}

}  // namespace

int main() {
    parses_a_real_fceux_save();
    accepts_a_state_with_every_core_section();
    rejects_states_that_do_not_describe_a_machine();
}

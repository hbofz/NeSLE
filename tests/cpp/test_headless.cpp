#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "nesle/console.hpp"
#include "nesle/cpu.hpp"
#include "nesle/file.hpp"
#include "nesle/headless.hpp"
#include "nesle/rom.hpp"

namespace {

std::vector<std::uint8_t> make_nrom_bytes() {
    std::vector<std::uint8_t> data = {'N', 'E', 'S', 0x1A, 2, 1, 0, 0};
    data.resize(16, 0);
    data.insert(data.end(), 32 * 1024, 0xEA);
    data.insert(data.end(), 8 * 1024, 0);

    constexpr std::size_t kPrgOffset = 16;
    data[kPrgOffset + 0x0000] = 0xA9;  // LDA #$80
    data[kPrgOffset + 0x0001] = 0x80;
    data[kPrgOffset + 0x0002] = 0x8D;  // STA $2000
    data[kPrgOffset + 0x0003] = 0x00;
    data[kPrgOffset + 0x0004] = 0x20;
    data[kPrgOffset + 0x0005] = 0xE6;  // INC $01
    data[kPrgOffset + 0x0006] = 0x01;
    data[kPrgOffset + 0x0007] = 0x4C;  // JMP $8005
    data[kPrgOffset + 0x0008] = 0x05;
    data[kPrgOffset + 0x0009] = 0x80;
    data[kPrgOffset + 0x1000] = 0xE6;  // INC $00
    data[kPrgOffset + 0x1001] = 0x00;
    data[kPrgOffset + 0x1002] = 0x40;  // RTI
    data[kPrgOffset + 0x7FFA] = 0x00;  // NMI vector -> $9000
    data[kPrgOffset + 0x7FFB] = 0x90;
    data[kPrgOffset + 0x7FFC] = 0x00;  // RESET vector -> $8000
    data[kPrgOffset + 0x7FFD] = 0x80;
    return data;
}

void write_file(const std::string& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream output(path, std::ios::binary);
    assert(output);
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    assert(output);
}

}  // namespace

int main() {
    const std::string rom_path = "/tmp/nesle_headless_test.nes";
    write_file(rom_path, make_nrom_bytes());

    {
        auto rom = nesle::load_ines_file(rom_path);
        assert(rom.metadata.is_nrom());
        assert(rom.metadata.prg_rom_size == 32 * 1024);

        nesle::Console console(std::move(rom));
        nesle::cpu::CpuState state;
        console.reset_cpu(state);

        nesle::HeadlessRunConfig config;
        config.frames = 1;
        config.max_instructions = 50'000;
        config.trace_capacity = 4;
        const auto result = nesle::run_headless(console, state, config);

        assert(result.completed());
        assert(result.frames_completed == 1);
        assert(result.instructions > 0);
        assert(result.cpu_cycles > 0);
        assert(result.ppu_frame == 1);
        assert(result.trace.size() == 4);
        assert(result.trace.front().instruction + 3 == result.trace.back().instruction);
        assert(result.trace.back().instruction == result.instructions);
        assert(result.trace.back().ppu_frame == result.ppu_frame);
        assert(console.read(0x0000) == 1);
        assert(console.read(0x0001) != 0);
    }

    {
        auto rom = nesle::load_ines_file(rom_path);
        nesle::Console console(std::move(rom));
        nesle::cpu::CpuState state;
        console.reset_cpu(state);

        nesle::HeadlessRunConfig config;
        config.frames = 2;
        config.max_instructions = 1;
        config.trace_capacity = 8;
        const auto result = nesle::run_headless(console, state, config);
        assert(result.status == nesle::HeadlessRunStatus::Timeout);
        assert(!result.completed());
        assert(result.instructions == 1);
        assert(result.trace.size() == 1);
        assert(result.trace.front().pc == 0x8000);
    }

    {
        auto rom = nesle::load_ines_file(rom_path);
        nesle::Console console(std::move(rom));
        nesle::cpu::CpuState state;
        console.reset_cpu(state);

        nesle::HeadlessRunConfig config;
        config.frames = 1;
        config.stop_on_trap = false;
        const auto result = nesle::run_headless(console, state, config);
        assert(nesle::to_string(result.status) == std::string("completed"));
    }

    return 0;
}

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
#include "nesle/smb.hpp"

namespace {

std::vector<std::uint8_t> make_nrom_bytes() {
    std::vector<std::uint8_t> data = {'N', 'E', 'S', 0x1A, 2, 1, 0, 0};
    data.resize(16, 0);
    data.insert(data.end(), 32 * 1024, 0xEA);
    data.insert(data.end(), 8 * 1024, 0);

    constexpr std::size_t kPrgOffset = 16;
    std::size_t pc = kPrgOffset;
    auto emit = [&](std::uint8_t value) {
        data[pc++] = value;
    };
    auto lda_imm = [&](std::uint8_t value) {
        emit(0xA9);
        emit(value);
    };
    auto sta_abs = [&](std::uint16_t address) {
        emit(0x8D);
        emit(static_cast<std::uint8_t>(address & 0x00FF));
        emit(static_cast<std::uint8_t>(address >> 8));
    };

    lda_imm(0x80);
    sta_abs(0x2000);
    lda_imm(1);
    sta_abs(nesle::smb::kXPage);
    lda_imm(2);
    sta_abs(nesle::smb::kXScreen);
    lda_imm(4);
    sta_abs(nesle::smb::kTimeDigits);
    lda_imm(0);
    sta_abs(nesle::smb::kTimeDigits + 1);
    sta_abs(nesle::smb::kTimeDigits + 2);
    sta_abs(nesle::smb::kWorld);
    sta_abs(nesle::smb::kStage);
    sta_abs(nesle::smb::kArea);
    lda_imm(2);
    sta_abs(nesle::smb::kLives);
    lda_imm(1);
    sta_abs(nesle::smb::kStatus);

    const auto loop_address = static_cast<std::uint16_t>(0x8000 + (pc - kPrgOffset));
    emit(0xE6);  // INC $01
    emit(0x01);
    emit(0x4C);  // JMP loop
    emit(static_cast<std::uint8_t>(loop_address & 0x00FF));
    emit(static_cast<std::uint8_t>(loop_address >> 8));

    data[kPrgOffset + 0x1000] = 0xE6;  // INC $00
    data[kPrgOffset + 0x1001] = 0x00;
    data[kPrgOffset + 0x1002] = 0x40;  // RTI
    data[kPrgOffset + 0x7FFA] = 0x00;  // NMI vector -> $9000
    data[kPrgOffset + 0x7FFB] = 0x90;
    data[kPrgOffset + 0x7FFC] = 0x00;  // RESET vector -> $8000
    data[kPrgOffset + 0x7FFD] = 0x80;
    return data;
}

std::vector<std::uint8_t> make_controller_read_rom() {
    std::vector<std::uint8_t> data = {'N', 'E', 'S', 0x1A, 2, 1, 0, 0};
    data.resize(16, 0);
    data.insert(data.end(), 32 * 1024, 0xEA);
    data.insert(data.end(), 8 * 1024, 0);

    constexpr std::size_t kPrgOffset = 16;
    std::size_t pc = kPrgOffset;
    auto emit = [&](std::uint8_t value) {
        data[pc++] = value;
    };
    auto lda_imm = [&](std::uint8_t value) {
        emit(0xA9);
        emit(value);
    };
    auto sta_abs = [&](std::uint16_t address) {
        emit(0x8D);
        emit(static_cast<std::uint8_t>(address & 0x00FF));
        emit(static_cast<std::uint8_t>(address >> 8));
    };

    lda_imm(1);
    sta_abs(0x4016);
    lda_imm(0);
    sta_abs(0x4016);
    emit(0xAD);  // LDA $4016
    emit(0x16);
    emit(0x40);
    emit(0x29);  // AND #$01
    emit(0x01);
    sta_abs(0x0002);
    emit(0x4C);  // JMP $8011
    emit(0x11);
    emit(0x80);

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
    const std::string controller_rom_path = "/tmp/nesle_controller_test.nes";
    write_file(controller_rom_path, make_controller_read_rom());

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
        const auto mario = nesle::smb::read_ram(console.cpu_ram());
        assert(mario.x_pos == 258);
        assert(mario.time == 400);
        assert(mario.lives == 2);
        assert(mario.world == 1);
        assert(mario.stage == 1);
        assert(mario.area == 1);
        assert(nesle::smb::status_name(mario.status_code) == "tall");
        assert(nesle::smb::is_plausible_boot_state(mario));
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

    {
        auto rom = nesle::load_ines_file(controller_rom_path);
        nesle::Console console(std::move(rom));
        nesle::cpu::CpuState state;
        console.reset_cpu(state);

        nesle::HeadlessRunConfig config;
        config.frames = 1;
        config.stop_on_trap = false;
        config.controller1_frame_actions = {nesle::ButtonA};
        const auto result = nesle::run_headless(console, state, config);
        assert(result.completed());
        assert(console.read(0x0002) == 1);
    }

    return 0;
}

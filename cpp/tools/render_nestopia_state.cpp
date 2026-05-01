#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "nesle/file.hpp"
#include "nesle/ppu.hpp"
#include "nesle/rom.hpp"

namespace {

struct Config {
    std::string rom_path;
    std::string state_path;
    std::string output_path;
    std::uint8_t scroll_x = 0;
    std::uint8_t scroll_y = 0;
    bool scroll_x_explicit = false;
    bool scroll_y_explicit = false;
};

struct Chunk {
    std::size_t offset = 0;
    std::size_t size = 0;
};

[[nodiscard]] std::vector<std::uint8_t> read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::invalid_argument("could not open file: " + path);
    }
    input.seekg(0, std::ios::end);
    const auto size = input.tellg();
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    input.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!input) {
        throw std::invalid_argument("could not read file: " + path);
    }
    return bytes;
}

[[nodiscard]] std::uint32_t read_le32(std::span<const std::uint8_t> bytes) {
    if (bytes.size() < 4) {
        throw std::invalid_argument("truncated little-endian u32");
    }
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8) |
           (static_cast<std::uint32_t>(bytes[2]) << 16) |
           (static_cast<std::uint32_t>(bytes[3]) << 24);
}

[[nodiscard]] std::unordered_map<std::string, Chunk> parse_chunks(std::span<const std::uint8_t> bytes,
                                                                  std::size_t base_offset) {
    std::unordered_map<std::string, Chunk> chunks;
    std::size_t offset = 0;
    while (offset + 8 <= bytes.size()) {
        std::string tag(reinterpret_cast<const char*>(bytes.data() + offset), 4);
        while (!tag.empty() && tag.back() == '\0') {
            tag.pop_back();
        }
        const auto size = read_le32(bytes.subspan(offset + 4, 4));
        offset += 8;
        if (offset + size > bytes.size()) {
            throw std::invalid_argument("Nestopia state chunk is truncated: " + tag);
        }
        chunks[tag] = Chunk{base_offset + offset, size};
        offset += size;
    }
    if (offset != bytes.size()) {
        throw std::invalid_argument("Nestopia state has trailing partial chunk bytes");
    }
    return chunks;
}

[[nodiscard]] std::span<const std::uint8_t> chunk(std::span<const std::uint8_t> file,
                                                  const std::unordered_map<std::string, Chunk>& chunks,
                                                  const std::string& tag) {
    const auto found = chunks.find(tag);
    if (found == chunks.end()) {
        throw std::invalid_argument("Nestopia state is missing chunk: " + tag);
    }
    return file.subspan(found->second.offset, found->second.size);
}

[[nodiscard]] std::uint64_t fnv1a64(std::span<const std::uint8_t> bytes) noexcept {
    std::uint64_t hash = 14695981039346656037ull;
    for (const auto byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    return hash;
}

[[nodiscard]] std::string hex64(std::uint64_t value) {
    std::ostringstream out;
    out << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}

void write_ppm(const std::string& path, const nesle::Ppu::RgbFrame& frame) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::invalid_argument("could not open output file: " + path);
    }
    output << "P6\n" << nesle::Ppu::kScreenWidth << ' ' << nesle::Ppu::kScreenHeight << "\n255\n";
    output.write(reinterpret_cast<const char*>(frame.data()),
                 static_cast<std::streamsize>(frame.size()));
    if (!output) {
        throw std::invalid_argument("could not write output file: " + path);
    }
}

[[nodiscard]] std::uint8_t parse_u8(const std::string& value) {
    const auto parsed = std::stoull(value, nullptr, 0);
    if (parsed > 0xFF) {
        throw std::invalid_argument("8-bit value is out of range: " + value);
    }
    return static_cast<std::uint8_t>(parsed);
}

[[nodiscard]] Config parse_args(int argc, char** argv) {
    if (argc < 4) {
        throw std::invalid_argument(
            "usage: render_nestopia_state <rom.nes> <State> <frame.ppm> "
            "[--scroll-x N] [--scroll-y N]");
    }

    Config config;
    config.rom_path = argv[1];
    config.state_path = argv[2];
    config.output_path = argv[3];

    for (int i = 4; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i + 1 >= argc) {
            throw std::invalid_argument("missing value for " + arg);
        }
        const std::string value = argv[++i];
        if (arg == "--scroll-x") {
            config.scroll_x = parse_u8(value);
            config.scroll_x_explicit = true;
        } else if (arg == "--scroll-y") {
            config.scroll_y = parse_u8(value);
            config.scroll_y_explicit = true;
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    return config;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        auto config = parse_args(argc, argv);
        auto rom = nesle::load_ines_file(config.rom_path);
        auto state = read_file(config.state_path);
        if (state.size() < 8 || state[0] != 'N' || state[1] != 'S' ||
            state[2] != 'T' || state[3] != 0x1A) {
            throw std::invalid_argument("expected a Nestopia NST save state");
        }

        const auto file = std::span<const std::uint8_t>(state.data(), state.size());
        const auto top = parse_chunks(file.subspan(8), 8);
        const auto ppu_span = chunk(file, top, "PPU");
        const auto ppu_chunks = parse_chunks(ppu_span, ppu_span.data() - file.data());

        const auto reg = chunk(file, ppu_chunks, "REG");
        const auto palette = chunk(file, ppu_chunks, "PAL");
        const auto oam = chunk(file, ppu_chunks, "OAM");
        const auto nametable = chunk(file, ppu_chunks, "NMT");
        if (reg.size() < 6 || palette.size() < 33 || oam.size() < 257 || nametable.size() < 2049) {
            throw std::invalid_argument("Nestopia PPU chunks are smaller than expected");
        }

        if (!config.scroll_x_explicit) {
            config.scroll_x = reg[4];
        }
        if (!config.scroll_y_explicit) {
            config.scroll_y = reg[5];
        }

        nesle::Ppu ppu;
        ppu.configure_cartridge(rom.chr_rom, rom.metadata.nametable_arrangement);
        nesle::Ppu::RenderState render_state;
        render_state.ctrl = reg[0];
        render_state.mask = reg[1];
        render_state.status = reg[2];
        render_state.scroll_x = config.scroll_x;
        render_state.scroll_y = config.scroll_y;
        render_state.palette_ram = palette.subspan(1, 32);
        render_state.oam = oam.subspan(1, 256);
        render_state.nametable_ram = nametable.subspan(1, 2048);
        ppu.load_render_state(render_state);

        const auto frame = ppu.render_rgb_frame();
        write_ppm(config.output_path, frame);
        std::cout << "rendered"
                  << " output=\"" << config.output_path << "\""
                  << " core_state=\"Nestopia NST\""
                  << " ctrl=0x" << std::hex << static_cast<unsigned>(render_state.ctrl)
                  << " mask=0x" << static_cast<unsigned>(render_state.mask)
                  << " status=0x" << static_cast<unsigned>(render_state.status)
                  << std::dec
                  << " scroll_x=" << static_cast<unsigned>(render_state.scroll_x)
                  << " scroll_y=" << static_cast<unsigned>(render_state.scroll_y)
                  << " frame_hash=" << hex64(fnv1a64({frame.data(), frame.size()}))
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

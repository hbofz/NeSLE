#pragma once

#include <cstdint>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "nesle/rom.hpp"

namespace nesle {

[[nodiscard]] inline std::vector<std::uint8_t> read_binary_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open file: " + path);
    }
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

[[nodiscard]] inline RomImage load_ines_file(const std::string& path) {
    const auto bytes = read_binary_file(path);
    return parse_ines(bytes);
}

}  // namespace nesle

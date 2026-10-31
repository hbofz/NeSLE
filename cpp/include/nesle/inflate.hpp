#pragma once

// Minimal RFC 1950 / RFC 1951 decoder.
//
// FCEUX writes its save states through zlib (src/state.cpp calls compress2 and
// uncompress), and the CUDA extension is built with nvcc against Python.h and
// pybind11 only - no zlib on the link line, and MSVC does not ship one. Rather
// than add a build dependency for one file format, this header carries the
// decoder it needs: stored, fixed-Huffman and dynamic-Huffman blocks, plus the
// zlib wrapper with its Adler-32 check.
//
// The implementation follows the structure of Mark Adler's public-domain puff.c
// (canonical Huffman tables, bit-at-a-time decoding), which is the simplest form
// that is correct rather than merely fast. States are read once per episode
// reset, not per emulated instruction, so decode speed does not matter here.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace nesle::detail {

namespace zlib_detail {

// A canonical Huffman table: count[len] is how many codes have that length,
// symbol[] holds the symbols ordered by (length, code).
struct HuffmanTable {
    std::array<std::uint16_t, 16> count{};
    std::array<std::uint16_t, 288> symbol{};
};

class BitReader {
public:
    explicit BitReader(std::span<const std::uint8_t> input) noexcept : input_(input) {}

    // Reads `n` bits, least significant first. `n` must be <= 16.
    [[nodiscard]] std::uint32_t take(int n) {
        need(n);
        const std::uint32_t mask = (1u << n) - 1u;
        const std::uint32_t value = bits_ & mask;
        bits_ >>= n;
        available_ -= n;
        return value;
    }

    // Drops the partly consumed byte so the next take() starts on a byte boundary.
    void align_to_byte() noexcept {
        const int drop = available_ & 7;
        bits_ >>= drop;
        available_ -= drop;
    }

private:
    void need(int n) {
        while (available_ < n) {
            if (position_ >= input_.size()) {
                throw std::runtime_error("inflate: stream ends mid-block");
            }
            bits_ |= static_cast<std::uint32_t>(input_[position_++]) << available_;
            available_ += 8;
        }
    }

    std::span<const std::uint8_t> input_;
    std::size_t position_ = 0;
    std::uint32_t bits_ = 0;
    int available_ = 0;
};

// Builds the canonical table for `n` symbols with the given code lengths. A
// length of zero means "unused". Over-subscribed codes are corrupt input;
// incomplete codes are legal (a block may simply not use every bit pattern).
inline void build_table(HuffmanTable& table, const std::uint8_t* lengths, int n) {
    table.count.fill(0);
    for (int i = 0; i < n; ++i) {
        ++table.count[lengths[i]];
    }
    if (table.count[0] == n) {
        return;  // no codes at all: valid, decodes nothing
    }
    int left = 1;
    for (int length = 1; length < 16; ++length) {
        left <<= 1;
        left -= table.count[length];
        if (left < 0) {
            throw std::runtime_error("inflate: over-subscribed Huffman code");
        }
    }
    std::array<std::uint16_t, 16> offsets{};
    for (int length = 1; length < 15; ++length) {
        offsets[length + 1] = static_cast<std::uint16_t>(offsets[length] + table.count[length]);
    }
    for (int i = 0; i < n; ++i) {
        if (lengths[i] != 0) {
            table.symbol[offsets[lengths[i]]++] = static_cast<std::uint16_t>(i);
        }
    }
}

inline int decode_symbol(BitReader& reader, const HuffmanTable& table) {
    int code = 0;
    int first = 0;
    int index = 0;
    for (int length = 1; length < 16; ++length) {
        code |= static_cast<int>(reader.take(1));
        const int count = table.count[length];
        if (code - count < first) {
            return table.symbol[index + (code - first)];
        }
        index += count;
        first += count;
        first <<= 1;
        code <<= 1;
    }
    throw std::runtime_error("inflate: invalid Huffman code");
}

inline const HuffmanTable& fixed_literal_table() {
    static const HuffmanTable table = [] {
        std::array<std::uint8_t, 288> lengths{};
        for (int i = 0; i < 144; ++i) lengths[i] = 8;
        for (int i = 144; i < 256; ++i) lengths[i] = 9;
        for (int i = 256; i < 280; ++i) lengths[i] = 7;
        for (int i = 280; i < 288; ++i) lengths[i] = 8;
        HuffmanTable built;
        build_table(built, lengths.data(), 288);
        return built;
    }();
    return table;
}

inline const HuffmanTable& fixed_distance_table() {
    static const HuffmanTable table = [] {
        std::array<std::uint8_t, 30> lengths{};
        lengths.fill(5);
        HuffmanTable built;
        build_table(built, lengths.data(), 30);
        return built;
    }();
    return table;
}

inline constexpr std::uint16_t kLengthBase[29] = {
    3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
    31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
inline constexpr std::uint8_t kLengthExtra[29] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
    2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
inline constexpr std::uint16_t kDistanceBase[30] = {
    1,    2,    3,    4,    5,    7,     9,     13,    17,   25,
    33,   49,   65,   97,   129,  193,   257,   385,   513,  769,
    1025, 1537, 2049, 3073, 4097, 6145,  8193,  12289, 16385, 24577};
inline constexpr std::uint8_t kDistanceExtra[30] = {
    0, 0, 0, 0, 1, 1, 2, 2,  3,  3,  4,  4,  5,  5,  6,
    6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

inline void inflate_codes(BitReader& reader,
                          const HuffmanTable& literals,
                          const HuffmanTable& distances,
                          std::vector<std::uint8_t>& out) {
    for (;;) {
        const int symbol = decode_symbol(reader, literals);
        if (symbol < 256) {
            out.push_back(static_cast<std::uint8_t>(symbol));
            continue;
        }
        if (symbol == 256) {
            return;  // end of block
        }
        const int length_index = symbol - 257;
        if (length_index >= 29) {
            throw std::runtime_error("inflate: invalid length symbol");
        }
        const std::size_t length =
            kLengthBase[length_index] + reader.take(kLengthExtra[length_index]);
        const int distance_symbol = decode_symbol(reader, distances);
        if (distance_symbol >= 30) {
            throw std::runtime_error("inflate: invalid distance symbol");
        }
        const std::size_t distance =
            kDistanceBase[distance_symbol] + reader.take(kDistanceExtra[distance_symbol]);
        if (distance > out.size()) {
            throw std::runtime_error("inflate: distance reaches before the output");
        }
        // Byte at a time: the source range may overlap the bytes being written.
        const std::size_t start = out.size() - distance;
        for (std::size_t i = 0; i < length; ++i) {
            out.push_back(out[start + i]);
        }
    }
}

inline void inflate_stored(BitReader& reader, std::vector<std::uint8_t>& out) {
    reader.align_to_byte();
    const std::uint32_t length = reader.take(16);
    const std::uint32_t complement = reader.take(16);
    if ((length ^ 0xFFFFu) != complement) {
        throw std::runtime_error("inflate: stored block length check failed");
    }
    for (std::uint32_t i = 0; i < length; ++i) {
        out.push_back(static_cast<std::uint8_t>(reader.take(8)));
    }
}

inline void inflate_dynamic(BitReader& reader, std::vector<std::uint8_t>& out) {
    static constexpr int kCodeLengthOrder[19] = {16, 17, 18, 0, 8,  7, 9,  6, 10, 5,
                                                 11, 4,  12, 3, 13, 2, 14, 1, 15};
    const int literal_count = static_cast<int>(reader.take(5)) + 257;
    const int distance_count = static_cast<int>(reader.take(5)) + 1;
    const int code_length_count = static_cast<int>(reader.take(4)) + 4;
    if (literal_count > 286 || distance_count > 30) {
        throw std::runtime_error("inflate: too many length or distance codes");
    }

    std::array<std::uint8_t, 19> code_lengths{};
    for (int i = 0; i < code_length_count; ++i) {
        code_lengths[kCodeLengthOrder[i]] = static_cast<std::uint8_t>(reader.take(3));
    }
    HuffmanTable code_length_table;
    build_table(code_length_table, code_lengths.data(), 19);

    std::array<std::uint8_t, 286 + 30> lengths{};
    int index = 0;
    while (index < literal_count + distance_count) {
        const int symbol = decode_symbol(reader, code_length_table);
        if (symbol < 16) {
            lengths[index++] = static_cast<std::uint8_t>(symbol);
            continue;
        }
        std::uint8_t value = 0;
        int repeat = 0;
        if (symbol == 16) {
            if (index == 0) {
                throw std::runtime_error("inflate: repeat with no previous length");
            }
            value = lengths[index - 1];
            repeat = 3 + static_cast<int>(reader.take(2));
        } else if (symbol == 17) {
            repeat = 3 + static_cast<int>(reader.take(3));
        } else {
            repeat = 11 + static_cast<int>(reader.take(7));
        }
        if (index + repeat > literal_count + distance_count) {
            throw std::runtime_error("inflate: code lengths overrun the symbol table");
        }
        while (repeat-- > 0) {
            lengths[index++] = value;
        }
    }
    if (lengths[256] == 0) {
        throw std::runtime_error("inflate: no end-of-block code");
    }

    HuffmanTable literal_table;
    HuffmanTable distance_table;
    build_table(literal_table, lengths.data(), literal_count);
    build_table(distance_table, lengths.data() + literal_count, distance_count);
    inflate_codes(reader, literal_table, distance_table, out);
}

}  // namespace zlib_detail

// Decodes a raw DEFLATE stream. `expected_size`, when non-zero, only reserves
// capacity - the caller is responsible for checking the final length.
[[nodiscard]] inline std::vector<std::uint8_t> inflate_raw(
    std::span<const std::uint8_t> input,
    std::size_t expected_size = 0) {
    zlib_detail::BitReader reader(input);
    std::vector<std::uint8_t> out;
    if (expected_size != 0) {
        out.reserve(expected_size);
    }
    for (;;) {
        const bool last = reader.take(1) != 0;
        const std::uint32_t type = reader.take(2);
        switch (type) {
            case 0:
                zlib_detail::inflate_stored(reader, out);
                break;
            case 1:
                zlib_detail::inflate_codes(reader,
                                           zlib_detail::fixed_literal_table(),
                                           zlib_detail::fixed_distance_table(),
                                           out);
                break;
            case 2:
                zlib_detail::inflate_dynamic(reader, out);
                break;
            default:
                throw std::runtime_error("inflate: reserved block type");
        }
        if (last) {
            return out;
        }
    }
}

namespace zlib_detail {

[[nodiscard]] inline std::uint32_t adler32(std::span<const std::uint8_t> data) noexcept {
    std::uint32_t a = 1;
    std::uint32_t b = 0;
    for (const std::uint8_t byte : data) {
        a = (a + byte) % 65521u;
        b = (b + a) % 65521u;
    }
    return (b << 16) | a;
}

}  // namespace zlib_detail

// Decodes a zlib stream (RFC 1950): two header bytes, DEFLATE payload, Adler-32.
[[nodiscard]] inline std::vector<std::uint8_t> inflate_zlib(
    std::span<const std::uint8_t> input,
    std::size_t expected_size = 0) {
    if (input.size() < 6) {
        throw std::runtime_error("inflate: zlib stream too short");
    }
    const std::uint32_t cmf = input[0];
    const std::uint32_t flg = input[1];
    if ((cmf & 0x0F) != 8) {
        throw std::runtime_error("inflate: zlib stream is not DEFLATE");
    }
    if (((cmf << 8) | flg) % 31 != 0) {
        throw std::runtime_error("inflate: zlib header check failed");
    }
    if ((flg & 0x20) != 0) {
        throw std::runtime_error("inflate: zlib preset dictionary is not supported");
    }
    // The trailer is four bytes; the payload in between is the DEFLATE stream.
    auto out = inflate_raw(input.subspan(2, input.size() - 6), expected_size);
    const auto trailer = input.subspan(input.size() - 4, 4);
    const std::uint32_t stored = (static_cast<std::uint32_t>(trailer[0]) << 24) |
                                 (static_cast<std::uint32_t>(trailer[1]) << 16) |
                                 (static_cast<std::uint32_t>(trailer[2]) << 8) |
                                 static_cast<std::uint32_t>(trailer[3]);
    if (zlib_detail::adler32(out) != stored) {
        throw std::runtime_error("inflate: Adler-32 mismatch");
    }
    return out;
}

}  // namespace nesle::detail

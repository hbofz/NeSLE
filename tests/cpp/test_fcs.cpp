// The FCS reader has to accept two containers: the original FCE Ultra
// "FCS\xff" file and the "FCSX" file FCEUX 2.6 and later write, which wraps the
// same chunk stream in zlib. Both must produce identical snapshots for identical
// content, and corrupt input must be rejected rather than read out of bounds.
//
// The FCSX fixtures below were produced with Python's zlib: kFcsxDynamic uses
// dynamic-Huffman blocks, kFcsxStoredDeflate uses stored blocks (compression
// level 0), and kFcsxUncompressed is the container's own "not compressed" form,
// signalled by a compressed size of ~0.

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "nesle/fcs.hpp"

namespace {

// FCS sub-chunk names are fixed four bytes and zero padded ("PC\0\0"). Reading
// name[2] and name[3] off a short string literal is out of bounds - MSVC happened
// to see zeroes there, g++ did not, and the chunk was silently dropped.
std::vector<std::uint8_t> sub_chunk(std::string_view name, std::span<const std::uint8_t> payload) {
    std::vector<std::uint8_t> out(4, 0);
    for (std::size_t i = 0; i < 4 && i < name.size(); ++i) {
        out[i] = static_cast<std::uint8_t>(name[i]);
    }
    const auto size = static_cast<std::uint32_t>(payload.size());
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<std::uint8_t>((size >> (8 * i)) & 0xFF));
    }
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

// Section tag 1 (CPU) holding PC, A, S and a 2 KB RAM image.
std::vector<std::uint8_t> cpu_section() {
    std::vector<std::uint8_t> payload;
    const std::array<std::uint8_t, 2> pc = {0x5B, 0xC0};  // $C05B, little-endian
    const std::array<std::uint8_t, 1> a = {0x3E};
    const std::array<std::uint8_t, 1> s = {0xFF};
    std::vector<std::uint8_t> ram(2048);
    for (std::size_t i = 0; i < ram.size(); ++i) {
        ram[i] = static_cast<std::uint8_t>((i * 37u) & 0xFFu);
    }
    for (auto part : {sub_chunk("PC", pc), sub_chunk("A", a), sub_chunk("S", s),
                      sub_chunk("RAM", ram)}) {
        payload.insert(payload.end(), part.begin(), part.end());
    }
    std::vector<std::uint8_t> section;
    section.push_back(1);
    const auto size = static_cast<std::uint32_t>(payload.size());
    for (int i = 0; i < 4; ++i) {
        section.push_back(static_cast<std::uint8_t>((size >> (8 * i)) & 0xFF));
    }
    section.insert(section.end(), payload.begin(), payload.end());
    return section;
}

// The same chunks in the original container: 16-byte header, then the stream.
std::vector<std::uint8_t> wrap_legacy(std::span<const std::uint8_t> stream) {
    std::vector<std::uint8_t> out = {'F', 'C', 'S', 0xFF, 0, 0, 0, 0,
                                     0,   0,   0,   0,    0, 0, 0, 0};
    out.insert(out.end(), stream.begin(), stream.end());
    return out;
}

inline constexpr std::array<std::uint8_t, 343> kFcsxDynamic = {
    0x46, 0x43, 0x53, 0x58, 0x29, 0x08, 0x00, 0x00, 0x7e, 0x50, 0x00, 0x00,
    0x47, 0x01, 0x00, 0x00, 0x78, 0xda, 0x63, 0x54, 0xe1, 0x60, 0x60, 0x08,
    0x70, 0x66, 0x60, 0x60, 0x62, 0x60, 0x60, 0x88, 0x3e, 0xe0, 0x08, 0x24,
    0x19, 0x81, 0xd8, 0x2e, 0x18, 0xca, 0xf8, 0x1f, 0xe4, 0xe8, 0xcb, 0xc0,
    0x00, 0x54, 0xc3, 0xa0, 0xea, 0x95, 0x3f, 0x65, 0xe7, 0x3d, 0x66, 0x0d,
    0xdf, 0xa2, 0xe9, 0x7b, 0x1e, 0xb2, 0x69, 0x07, 0x94, 0xce, 0xda, 0xff,
    0x84, 0x53, 0x2f, 0xb8, 0x62, 0xee, 0xa1, 0xe7, 0x3c, 0x86, 0x61, 0xd5,
    0x0b, 0x8e, 0xbe, 0xe2, 0x37, 0x89, 0xac, 0x5b, 0x7c, 0xe2, 0xad, 0x90,
    0x79, 0x4c, 0xe3, 0xb2, 0xd3, 0x1f, 0x44, 0xad, 0xe2, 0x5b, 0x56, 0x9e,
    0xfb, 0x2c, 0x61, 0x9b, 0xd4, 0xbe, 0xe6, 0xe2, 0x37, 0x69, 0x87, 0xd4,
    0xae, 0xf5, 0x57, 0x7e, 0xca, 0x39, 0x67, 0xf4, 0x6e, 0xba, 0xfe, 0x47,
    0xd1, 0x2d, 0x7b, 0xc2, 0xd6, 0x5b, 0xff, 0x55, 0x3c, 0xf3, 0x26, 0xef,
    0xb8, 0xcb, 0xa4, 0xee, 0x53, 0x38, 0x6d, 0xf7, 0x03, 0x56, 0x2d, 0xff,
    0x92, 0x99, 0xfb, 0x1e, 0x73, 0xe8, 0x06, 0x95, 0xcf, 0x39, 0xf8, 0x8c,
    0xdb, 0x20, 0xb4, 0x6a, 0xfe, 0x91, 0x97, 0x7c, 0xc6, 0x11, 0xb5, 0x8b,
    0x8e, 0xbf, 0x11, 0x34, 0x8b, 0x6e, 0x58, 0x7a, 0xea, 0xbd, 0x88, 0x65,
    0x5c, 0xf3, 0x8a, 0xb3, 0x9f, 0xc4, 0x6d, 0x12, 0xdb, 0x56, 0x5f, 0xf8,
    0x2a, 0x65, 0x9f, 0xd2, 0xb9, 0xee, 0xf2, 0x0f, 0x59, 0xa7, 0xf4, 0x9e,
    0x8d, 0xd7, 0x7e, 0x2b, 0xb8, 0x66, 0xf5, 0x6f, 0xb9, 0xf9, 0x4f, 0xd9,
    0x23, 0x77, 0xd2, 0xf6, 0x3b, 0x8c, 0x6a, 0xde, 0x05, 0x53, 0x77, 0xdd,
    0x67, 0xd1, 0xf4, 0x2b, 0x9e, 0xb1, 0xf7, 0x11, 0xbb, 0x4e, 0x60, 0xd9,
    0xec, 0x03, 0x4f, 0xb9, 0xf4, 0x43, 0x2a, 0xe7, 0x1d, 0x7e, 0xc1, 0x6b,
    0x14, 0x5e, 0xb3, 0xf0, 0xd8, 0x6b, 0x01, 0xd3, 0xa8, 0xfa, 0x25, 0x27,
    0xdf, 0x09, 0x5b, 0xc4, 0x36, 0x2d, 0x3f, 0xf3, 0x51, 0xcc, 0x3a, 0xa1,
    0x75, 0xd5, 0xf9, 0x2f, 0x92, 0x76, 0xc9, 0x1d, 0x6b, 0x2f, 0x7d, 0x97,
    0x71, 0x4c, 0xeb, 0xde, 0x70, 0xf5, 0x97, 0xbc, 0x4b, 0x66, 0xdf, 0xe6,
    0x1b, 0x7f, 0x95, 0xdc, 0x73, 0x26, 0x6e, 0xbb, 0x3d, 0xea, 0xff, 0x51,
    0xff, 0x8f, 0xfa, 0x7f, 0xd4, 0xff, 0xa3, 0xfe, 0x1f, 0xf5, 0xff, 0x48,
    0xf4, 0x3f, 0x00, 0x86, 0x9a, 0x00, 0xd5,
};

inline constexpr std::array<std::uint8_t, 51> kFcsxStoredDeflate = {
    0x46, 0x43, 0x53, 0x58, 0x18, 0x00, 0x00, 0x00, 0x7e, 0x50, 0x00, 0x00,
    0x23, 0x00, 0x00, 0x00, 0x78, 0x01, 0x01, 0x18, 0x00, 0xe7, 0xff, 0x01,
    0x13, 0x00, 0x00, 0x00, 0x50, 0x43, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
    0x23, 0x81, 0x41, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x5a, 0x15,
    0xdc, 0x01, 0xea,
};

inline constexpr std::array<std::uint8_t, 40> kFcsxUncompressed = {
    0x46, 0x43, 0x53, 0x58, 0x18, 0x00, 0x00, 0x00, 0x7e, 0x50, 0x00, 0x00,
    0xff, 0xff, 0xff, 0xff, 0x01, 0x13, 0x00, 0x00, 0x00, 0x50, 0x43, 0x00,
    0x00, 0x02, 0x00, 0x00, 0x00, 0x23, 0x81, 0x41, 0x00, 0x00, 0x00, 0x01,
    0x00, 0x00, 0x00, 0x5a,
};

void test_fcsx_matches_legacy() {
    const auto stream = cpu_section();
    const auto legacy = wrap_legacy(stream);
    const auto from_legacy = nesle::fcs::parse(std::span<const std::uint8_t>(legacy));
    const auto from_fcsx = nesle::fcs::parse(std::span<const std::uint8_t>(kFcsxDynamic));

    assert(from_fcsx.pc == 0xC05B);
    assert(from_fcsx.a == 0x3E);
    assert(from_fcsx.sp == 0xFF);
    assert(from_fcsx.pc == from_legacy.pc);
    assert(from_fcsx.a == from_legacy.a);
    assert(from_fcsx.sp == from_legacy.sp);
    // Every decompressed byte has to land in the same place.
    assert(from_fcsx.cpu_ram == from_legacy.cpu_ram);
    for (std::size_t i = 0; i < from_fcsx.cpu_ram.size(); ++i) {
        assert(from_fcsx.cpu_ram[i] == static_cast<std::uint8_t>((i * 37u) & 0xFFu));
    }
}

void test_fcsx_stored_deflate_blocks() {
    const auto snap = nesle::fcs::parse(std::span<const std::uint8_t>(kFcsxStoredDeflate));
    assert(snap.pc == 0x8123);
    assert(snap.a == 0x5A);
}

void test_fcsx_uncompressed_container() {
    const auto snap = nesle::fcs::parse(std::span<const std::uint8_t>(kFcsxUncompressed));
    assert(snap.pc == 0x8123);
    assert(snap.a == 0x5A);
}

void test_rejects_bad_magic() {
    auto bytes = wrap_legacy(cpu_section());
    std::memcpy(bytes.data(), "XXXX", 4);
    bool threw = false;
    try {
        (void)nesle::fcs::parse(std::span<const std::uint8_t>(bytes));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
}

void test_rejects_truncated_fcsx() {
    std::vector<std::uint8_t> bytes(kFcsxDynamic.begin(), kFcsxDynamic.begin() + 100);
    bool threw = false;
    try {
        (void)nesle::fcs::parse(std::span<const std::uint8_t>(bytes));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
}

void test_rejects_corrupt_deflate() {
    // Flip bytes inside the compressed payload: the stream stops decoding, or
    // the Adler-32 check catches it. Either way it must not be accepted.
    std::vector<std::uint8_t> bytes(kFcsxDynamic.begin(), kFcsxDynamic.end());
    for (std::size_t i = 40; i < 60; ++i) {
        bytes[i] = static_cast<std::uint8_t>(bytes[i] ^ 0xA5);
    }
    bool threw = false;
    try {
        (void)nesle::fcs::parse(std::span<const std::uint8_t>(bytes));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
}

void test_rejects_size_field_mismatch() {
    // The header claims more bytes than the payload holds.
    std::vector<std::uint8_t> bytes(kFcsxUncompressed.begin(), kFcsxUncompressed.end());
    bytes[4] = 0xFF;
    bool threw = false;
    try {
        (void)nesle::fcs::parse(std::span<const std::uint8_t>(bytes));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
}

}  // namespace

int main() {
    test_fcsx_matches_legacy();
    test_fcsx_stored_deflate_blocks();
    test_fcsx_uncompressed_container();
    test_rejects_bad_magic();
    test_rejects_truncated_fcsx();
    test_rejects_corrupt_deflate();
    test_rejects_size_field_mismatch();
    return 0;
}

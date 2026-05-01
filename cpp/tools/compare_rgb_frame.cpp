#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Image {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> rgb;
};

struct Config {
    std::string actual_path;
    std::string reference_path;
    double max_rgb_mae = 45.0;
    double min_rgb_corr = 0.65;
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

[[nodiscard]] std::string next_ppm_token(std::span<const std::uint8_t> bytes,
                                         std::size_t& offset) {
    for (;;) {
        while (offset < bytes.size() &&
               (bytes[offset] == ' ' || bytes[offset] == '\n' || bytes[offset] == '\r' ||
                bytes[offset] == '\t')) {
            ++offset;
        }
        if (offset < bytes.size() && bytes[offset] == '#') {
            while (offset < bytes.size() && bytes[offset] != '\n') {
                ++offset;
            }
            continue;
        }
        break;
    }
    const auto start = offset;
    while (offset < bytes.size() &&
           bytes[offset] != ' ' && bytes[offset] != '\n' &&
           bytes[offset] != '\r' && bytes[offset] != '\t') {
        ++offset;
    }
    if (start == offset) {
        throw std::invalid_argument("truncated PPM header");
    }
    return {reinterpret_cast<const char*>(bytes.data() + start), offset - start};
}

[[nodiscard]] Image load_ppm(const std::string& path) {
    const auto file = read_file(path);
    const auto bytes = std::span<const std::uint8_t>(file.data(), file.size());
    std::size_t offset = 0;
    if (next_ppm_token(bytes, offset) != "P6") {
        throw std::invalid_argument("expected binary PPM P6 file: " + path);
    }
    Image image;
    image.width = std::stoi(next_ppm_token(bytes, offset));
    image.height = std::stoi(next_ppm_token(bytes, offset));
    const auto max_value = std::stoi(next_ppm_token(bytes, offset));
    if (image.width <= 0 || image.height <= 0 || max_value != 255) {
        throw std::invalid_argument("unsupported PPM dimensions or max value: " + path);
    }
    if (offset >= bytes.size() ||
        !(bytes[offset] == ' ' || bytes[offset] == '\n' || bytes[offset] == '\r' ||
          bytes[offset] == '\t')) {
        throw std::invalid_argument("PPM header is not followed by pixel data: " + path);
    }
    ++offset;
    const auto expected =
        static_cast<std::size_t>(image.width) * static_cast<std::size_t>(image.height) * 3;
    if (bytes.size() - offset < expected) {
        throw std::invalid_argument("PPM pixel data is truncated: " + path);
    }
    image.rgb.assign(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                     bytes.begin() + static_cast<std::ptrdiff_t>(offset + expected));
    return image;
}

[[nodiscard]] std::uint16_t read_le16(std::span<const std::uint8_t> bytes,
                                      std::size_t offset) {
    if (offset + 2 > bytes.size()) {
        throw std::invalid_argument("truncated BMP u16");
    }
    return static_cast<std::uint16_t>(bytes[offset]) |
           (static_cast<std::uint16_t>(bytes[offset + 1]) << 8);
}

[[nodiscard]] std::uint32_t read_le32(std::span<const std::uint8_t> bytes,
                                      std::size_t offset) {
    if (offset + 4 > bytes.size()) {
        throw std::invalid_argument("truncated BMP u32");
    }
    return static_cast<std::uint32_t>(bytes[offset]) |
           (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
           (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) |
           (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

[[nodiscard]] std::int32_t read_le_i32(std::span<const std::uint8_t> bytes,
                                       std::size_t offset) {
    return static_cast<std::int32_t>(read_le32(bytes, offset));
}

[[nodiscard]] Image load_bmp(const std::string& path) {
    const auto file = read_file(path);
    const auto bytes = std::span<const std::uint8_t>(file.data(), file.size());
    if (bytes.size() < 54 || bytes[0] != 'B' || bytes[1] != 'M') {
        throw std::invalid_argument("expected BMP file: " + path);
    }

    const auto data_offset = read_le32(bytes, 10);
    const auto width = read_le_i32(bytes, 18);
    const auto stored_height = read_le_i32(bytes, 22);
    const auto planes = read_le16(bytes, 26);
    const auto bits_per_pixel = read_le16(bytes, 28);
    const auto compression = read_le32(bytes, 30);
    const auto supported_compression = compression == 0 || (compression == 3 && bits_per_pixel == 32);
    if (width <= 0 || stored_height == 0 || planes != 1 || !supported_compression ||
        (bits_per_pixel != 24 && bits_per_pixel != 32)) {
        throw std::invalid_argument("unsupported BMP format: " + path);
    }

    Image image;
    image.width = width;
    image.height = std::abs(stored_height);
    image.rgb.resize(static_cast<std::size_t>(image.width) * image.height * 3);

    const auto bytes_per_pixel = bits_per_pixel / 8;
    const auto row_stride =
        ((static_cast<std::size_t>(image.width) * bits_per_pixel + 31) / 32) * 4;
    const auto required =
        static_cast<std::size_t>(data_offset) + row_stride * static_cast<std::size_t>(image.height);
    if (required > bytes.size()) {
        throw std::invalid_argument("BMP pixel data is truncated: " + path);
    }

    for (int y = 0; y < image.height; ++y) {
        const auto source_y = stored_height > 0 ? image.height - 1 - y : y;
        const auto source_offset =
            static_cast<std::size_t>(data_offset) + static_cast<std::size_t>(source_y) * row_stride;
        for (int x = 0; x < image.width; ++x) {
            const auto source = source_offset + static_cast<std::size_t>(x) * bytes_per_pixel;
            const auto target =
                (static_cast<std::size_t>(y) * image.width + static_cast<std::size_t>(x)) * 3;
            image.rgb[target] = bytes[source + 2];
            image.rgb[target + 1] = bytes[source + 1];
            image.rgb[target + 2] = bytes[source];
        }
    }
    return image;
}

[[nodiscard]] double correlation(std::span<const std::uint8_t> actual,
                                 std::span<const std::uint8_t> reference) {
    double actual_mean = 0.0;
    double reference_mean = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        actual_mean += actual[i];
        reference_mean += reference[i];
    }
    actual_mean /= static_cast<double>(actual.size());
    reference_mean /= static_cast<double>(reference.size());

    double numerator = 0.0;
    double actual_denominator = 0.0;
    double reference_denominator = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const auto actual_delta = static_cast<double>(actual[i]) - actual_mean;
        const auto reference_delta = static_cast<double>(reference[i]) - reference_mean;
        numerator += actual_delta * reference_delta;
        actual_denominator += actual_delta * actual_delta;
        reference_denominator += reference_delta * reference_delta;
    }
    const auto denominator = std::sqrt(actual_denominator * reference_denominator);
    if (denominator <= std::numeric_limits<double>::epsilon()) {
        return actual_denominator == reference_denominator ? 1.0 : 0.0;
    }
    return numerator / denominator;
}

[[nodiscard]] Config parse_args(int argc, char** argv) {
    if (argc < 3) {
        throw std::invalid_argument(
            "usage: compare_rgb_frame <actual.ppm> <reference.bmp> "
            "[--max-rgb-mae N] [--min-rgb-corr N]");
    }

    Config config;
    config.actual_path = argv[1];
    config.reference_path = argv[2];
    for (int i = 3; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (i + 1 >= argc) {
            throw std::invalid_argument("missing value for argument");
        }
        const std::string value = argv[++i];
        if (arg == "--max-rgb-mae") {
            config.max_rgb_mae = std::stod(value);
        } else if (arg == "--min-rgb-corr") {
            config.min_rgb_corr = std::stod(value);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    return config;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto config = parse_args(argc, argv);
        const auto actual = load_ppm(config.actual_path);
        const auto reference = load_bmp(config.reference_path);
        if (actual.width != reference.width || actual.height != reference.height) {
            throw std::invalid_argument("image dimensions differ: actual=" +
                                        std::to_string(actual.width) + "x" +
                                        std::to_string(actual.height) + " reference=" +
                                        std::to_string(reference.width) + "x" +
                                        std::to_string(reference.height));
        }

        std::uint64_t absolute_delta = 0;
        std::uint8_t max_delta = 0;
        std::size_t exact_channels = 0;
        std::size_t mismatched_pixels = 0;
        const auto pixel_count =
            static_cast<std::size_t>(actual.width) * static_cast<std::size_t>(actual.height);
        for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
            bool pixel_matches = true;
            for (std::size_t channel = 0; channel < 3; ++channel) {
                const auto index = pixel * 3 + channel;
                const auto delta = static_cast<unsigned>(
                    std::max(actual.rgb[index], reference.rgb[index]) -
                    std::min(actual.rgb[index], reference.rgb[index]));
                absolute_delta += delta;
                max_delta = std::max(max_delta, static_cast<std::uint8_t>(delta));
                if (delta == 0) {
                    ++exact_channels;
                } else {
                    pixel_matches = false;
                }
            }
            if (!pixel_matches) {
                ++mismatched_pixels;
            }
        }

        const auto channel_count = static_cast<double>(actual.rgb.size());
        const auto rgb_mae = static_cast<double>(absolute_delta) / channel_count;
        const auto rgb_corr = correlation(actual.rgb, reference.rgb);
        const auto exact_channel_ratio = static_cast<double>(exact_channels) / channel_count;
        const auto pixel_mismatch_ratio =
            static_cast<double>(mismatched_pixels) / static_cast<double>(pixel_count);

        std::cout << std::fixed << std::setprecision(6)
                  << "compare_rgb_frame"
                  << " width=" << actual.width
                  << " height=" << actual.height
                  << " rgb_mae=" << rgb_mae
                  << " rgb_corr=" << rgb_corr
                  << " max_delta=" << static_cast<unsigned>(max_delta)
                  << " exact_channels=" << exact_channel_ratio
                  << " pixel_mismatch=" << pixel_mismatch_ratio
                  << " max_rgb_mae=" << config.max_rgb_mae
                  << " min_rgb_corr=" << config.min_rgb_corr
                  << '\n';

        if (rgb_mae > config.max_rgb_mae) {
            std::cerr << "RGB MAE exceeds threshold\n";
            return 1;
        }
        if (rgb_corr < config.min_rgb_corr) {
            std::cerr << "RGB correlation is below threshold\n";
            return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

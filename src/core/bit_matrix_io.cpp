#include "bmmpy/core/bit_matrix.hpp"

#include <array>
#include <fstream>
#include <limits>

namespace bmmpy {
namespace {

[[noreturn]] void throw_io_error(const char* message) { throw std::runtime_error(message); }

void write_u64(std::ostream& out, std::uint64_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!out)
        throw_io_error("failed to write matrix binary data");
}

std::uint64_t read_u64(std::istream& in) {
    std::uint64_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in)
        throw_io_error("failed to read matrix binary data");
    return value;
}

std::size_t checked_size_cast(std::uint64_t value) {
    if (value > std::numeric_limits<std::size_t>::max())
        throw MatrixError(MatrixErr::SizeTooLarge);
    return static_cast<std::size_t>(value);
}

} // namespace

void BitMatrix::save_text(std::ostream& out) const {
    out << _rows << ' ' << _cols << '\n';
    if (!out)
        throw_io_error("failed to write matrix text header");

    for (std::size_t row = 0; row < _rows; ++row) {
        for (std::size_t col = 0; col < _cols; ++col)
            out.put(get_unchecked(row, col) ? '1' : '0');

        out.put('\n');
        if (!out)
            throw_io_error("failed to write matrix text data");
    }
}

void BitMatrix::save_binary(std::ostream& out) const {
    static constexpr std::array<char, 8> k_magic = {'B', 'M', 'M', 'P', 'Y', 'B', '1', '\0'};

    out.write(k_magic.data(), static_cast<std::streamsize>(k_magic.size()));
    if (!out)
        throw_io_error("failed to write matrix binary header");

    write_u64(out, static_cast<std::uint64_t>(_rows));
    write_u64(out, static_cast<std::uint64_t>(_cols));

    const std::size_t used_words = words_per_row();
    for (std::size_t row = 0; row < _rows; ++row) {
        const std::uint64_t* row_data = row_ptr_unchecked(row);
        out.write(reinterpret_cast<const char*>(row_data),
                  static_cast<std::streamsize>(used_words * sizeof(std::uint64_t)));
        if (!out)
            throw_io_error("failed to write matrix binary payload");
    }
}

void BitMatrix::save_text(const std::filesystem::path& path) const {
    std::ofstream out(path);
    if (!out)
        throw_io_error("failed to open matrix text file for writing");
    save_text(out);
}

void BitMatrix::save_binary(const std::filesystem::path& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw_io_error("failed to open matrix binary file for writing");
    save_binary(out);
}

BitMatrix BitMatrix::load_text(std::istream& in) {
    std::size_t rows = 0;
    std::size_t cols = 0;
    if (!(in >> rows >> cols))
        throw_io_error("failed to read matrix text header");

    BitMatrix matrix(rows, cols);
    const std::size_t total_bits = checked_mul(rows, cols);
    std::size_t bit_index = 0;
    char ch = '\0';

    while (bit_index < total_bits && in.get(ch)) {
        switch (ch) {
        case '0':
        case '1': {
            const std::size_t row = bit_index / cols;
            const std::size_t col = bit_index % cols;
            matrix.set_unchecked(row, col, ch == '1');
            ++bit_index;
            break;
        }
        case ' ':
        case '\n':
        case '\r':
        case '\t':
        case '\f':
        case '\v':
            break;
        default:
            throw_io_error("matrix text contains invalid character");
        }
    }

    if (bit_index != total_bits)
        throw_io_error("failed to read matrix text payload");

    while (in.get(ch)) {
        switch (ch) {
        case ' ':
        case '\n':
        case '\r':
        case '\t':
        case '\f':
        case '\v':
            break;
        default:
            throw_io_error("matrix text contains trailing data");
        }
    }

    return matrix;
}

BitMatrix BitMatrix::load_binary(std::istream& in) {
    static constexpr std::array<char, 8> k_magic = {'B', 'M', 'M', 'P', 'Y', 'B', '1', '\0'};

    std::array<char, 8> magic{};
    in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!in)
        throw_io_error("failed to read matrix binary header");
    if (magic != k_magic)
        throw_io_error("invalid matrix binary format");

    const std::size_t rows = checked_size_cast(read_u64(in));
    const std::size_t cols = checked_size_cast(read_u64(in));

    BitMatrix matrix(rows, cols);
    const std::size_t used_words = matrix.words_per_row();

    for (std::size_t row = 0; row < rows; ++row) {
        std::uint64_t* row_data = matrix.row_ptr_unchecked(row);
        in.read(reinterpret_cast<char*>(row_data),
                static_cast<std::streamsize>(used_words * sizeof(std::uint64_t)));
        if (!in)
            throw_io_error("failed to read matrix binary payload");
    }

    return matrix;
}

BitMatrix BitMatrix::load_text(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in)
        throw_io_error("failed to open matrix text file for reading");
    return load_text(in);
}

BitMatrix BitMatrix::load_binary(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw_io_error("failed to open matrix binary file for reading");
    return load_binary(in);
}

} // namespace bmmpy
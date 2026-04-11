#include "bmmpy/core/bit_matrix.hpp"

#include "bmmpy/core/detail/bit_intrinsics.hpp"
#include "bmmpy/core/detail/bit_ops.hpp"
#include "bmmpy/core/row_window.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <string>

namespace bmmpy {
namespace {
const char* matrix_err_message(MatrixErr code) noexcept {
    switch (code) {
    case MatrixErr::SizeTooLarge:
        return "Matrix dimensions too large";
    case MatrixErr::AllocationFailed:
        return "Memory allocation failed";
    case MatrixErr::DimensionMismatch:
        return "Matrix dimensions mismatch";
    }
    return "Unknown matrix error";
}
} // namespace

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

MatrixError::MatrixError(MatrixErr code)
    : std::runtime_error(matrix_err_message(code)), _code(code) {}

std::size_t BitMatrix::checked_add(std::size_t a, std::size_t b) {
    if (a > std::numeric_limits<std::size_t>::max() - b)
        throw MatrixError(MatrixErr::SizeTooLarge);
    return a + b;
}

std::size_t BitMatrix::checked_mul(std::size_t a, std::size_t b) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        throw MatrixError(MatrixErr::SizeTooLarge);
    return a * b;
}

std::size_t BitMatrix::ceil_div(std::size_t x, std::size_t y) noexcept { return (x + y - 1) / y; }

void BitMatrix::allocate_zeroed(std::size_t total_words) {
    _total_words = total_words;
    if (_total_words == 0)
        return;

    const std::size_t bytes = checked_mul(_total_words, sizeof(std::uint64_t));
    try {
        _data = static_cast<std::uint64_t*>(::operator new[](bytes, std::align_val_t(k_alignment)));
    } catch (const std::bad_alloc&) {
        throw MatrixError(MatrixErr::AllocationFailed);
    }

    std::memset(_data, 0, bytes);
}

void BitMatrix::destroy() noexcept {
    if (_data != nullptr) {
        ::operator delete[](_data, std::align_val_t(k_alignment));
        _data = nullptr;
    }
    _rows = 0;
    _cols = 0;
    _stride = 0;
    _total_words = 0;
}

BitMatrix::BitMatrix(std::size_t rows, std::size_t cols) : _rows(rows), _cols(cols) {
    const std::size_t words_needed = ceil_div(_cols, k_word_bits);
    const std::size_t blocks_needed = ceil_div(words_needed, std::size_t(4));
    _stride = checked_mul(blocks_needed, std::size_t(4));
    allocate_zeroed(checked_mul(_rows, _stride));
}

BitMatrix::~BitMatrix() noexcept { destroy(); }

BitMatrix::BitMatrix(const BitMatrix& other)
    : _rows(other._rows), _cols(other._cols), _stride(other._stride) {
    allocate_zeroed(other._total_words);
    if (_total_words != 0)
        std::memcpy(_data, other._data, total_bytes());
}

BitMatrix& BitMatrix::operator=(const BitMatrix& other) {
    if (this == &other)
        return *this;

    BitMatrix tmp(other);
    swap(tmp);
    return *this;
}

BitMatrix::BitMatrix(BitMatrix&& other) noexcept
    : _data(other._data), _rows(other._rows), _cols(other._cols), _stride(other._stride),
      _total_words(other._total_words) {
    other._data = nullptr;
    other._rows = 0;
    other._cols = 0;
    other._stride = 0;
    other._total_words = 0;
}

void BitMatrix::copy_from_words(const std::uint64_t* src, std::size_t count) {
    if (count != _total_words)
        throw MatrixError(MatrixErr::DimensionMismatch);
    if (count != 0)
        std::memcpy(_data, src, total_bytes());
}

const std::uint64_t* BitMatrix::row_words(std::size_t row) const {
    if (row >= _rows)
        throw std::out_of_range("row out of bounds");
    return row_ptr_unchecked(row);
}

std::uint64_t* BitMatrix::row_words(std::size_t row) {
    if (row >= _rows)
        throw std::out_of_range("row out of bounds");
    return row_ptr_unchecked(row);
}

void BitMatrix::set(std::size_t row, std::size_t col, bool value) {
    if (row >= _rows || col >= _cols)
        throw std::out_of_range("index out of bounds");
    set_unchecked(row, col, value);
}

bool BitMatrix::get(std::size_t row, std::size_t col) const {
    if (row >= _rows || col >= _cols)
        throw std::out_of_range("index out of bounds");
    return get_unchecked(row, col);
}

void BitMatrix::set_unchecked(std::size_t row, std::size_t col, bool value) noexcept {
    const std::size_t word_idx = col / k_word_bits;
    const std::size_t bit_idx = col % k_word_bits;
    std::uint64_t& word = row_ptr_unchecked(row)[word_idx];
    const std::uint64_t mask = std::uint64_t(1) << bit_idx;
    if (value)
        word |= mask;
    else
        word &= ~mask;
}

bool BitMatrix::get_unchecked(std::size_t row, std::size_t col) const noexcept {
    const std::size_t word_idx = col / k_word_bits;
    const std::size_t bit_idx = col % k_word_bits;
    return (row_ptr_unchecked(row)[word_idx] & (std::uint64_t(1) << bit_idx)) != 0;
}

void BitMatrix::row_xor(std::size_t target_row, std::size_t source_row) noexcept {
    assert(target_row < _rows);
    assert(source_row < _rows);
    if (target_row == source_row)
        return;

    detail::bit_ops().row_xor(
        row_ptr_unchecked(target_row), row_ptr_unchecked(source_row), _stride);
}

void BitMatrix::row_xor_from(std::size_t target_row,
                             const BitMatrix& source,
                             std::size_t source_row) {
    if (target_row >= _rows || source_row >= source._rows)
        throw std::out_of_range("row out of bounds");
    if (_cols != source._cols || _stride != source._stride)
        throw MatrixError(MatrixErr::DimensionMismatch);

    detail::bit_ops().row_xor(
        row_ptr_unchecked(target_row), source.row_ptr_unchecked(source_row), _stride);
}

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
    const std::size_t total_bits = rows * cols;
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

BitMatrix BitMatrix::identity(std::size_t n) {
    BitMatrix result(n, n);
    for (std::size_t i = 0; i < n; ++i)
        result.set_unchecked(i, i, true);
    return result;
}

BitMatrix BitMatrix::extract_rows_by_indices(const std::vector<std::size_t>& indices) const {
    BitMatrix result(indices.size(), _cols);
    for (std::size_t dst = 0; dst < indices.size(); ++dst) {
        const std::size_t src = indices[dst];
        if (src >= _rows)
            throw std::out_of_range("source row out of bounds");
        std::memcpy(
            result.row_ptr_unchecked(dst), row_ptr_unchecked(src), _stride * sizeof(std::uint64_t));
    }
    return result;
}

RowWindow BitMatrix::row_window(const std::vector<std::size_t>& rows) {
    return RowWindow(*this, rows);
}

RowWindow BitMatrix::row_window(const std::vector<std::size_t>& rows) const {
    return RowWindow(*this, rows);
}

void BitMatrix::insert_rows_by_indices(const BitMatrix& source,
                                       const std::vector<std::size_t>& indices) {
    if (source._rows != indices.size())
        throw MatrixError(MatrixErr::DimensionMismatch);
    if (_cols != source._cols || _stride != source._stride)
        throw MatrixError(MatrixErr::DimensionMismatch);

    for (std::size_t src = 0; src < indices.size(); ++src) {
        const std::size_t dst = indices[src];
        if (dst >= _rows)
            throw std::out_of_range("target row out of bounds");
        std::memcpy(
            row_ptr_unchecked(dst), source.row_ptr_unchecked(src), _stride * sizeof(std::uint64_t));
    }
}

BitMatrix BitMatrix::add(const BitMatrix& other) const {
    if (_rows != other._rows || _cols != other._cols)
        throw MatrixError(MatrixErr::DimensionMismatch);

    BitMatrix result(*this);
    for (std::size_t r = 0; r < _rows; ++r)
        result.row_xor_from(r, other, r);
    return result;
}

BitMatrix BitMatrix::mul(const BitMatrix& other) const {
    if (_cols != other._rows)
        throw MatrixError(MatrixErr::DimensionMismatch);

    BitMatrix result(_rows, other._cols);
    const std::size_t used_words = words_per_row();

    for (std::size_t i = 0; i < _rows; ++i) {
        const std::uint64_t* lhs = row_ptr_unchecked(i);

        for (std::size_t word_idx = 0; word_idx < used_words; ++word_idx) {
            std::uint64_t bits = lhs[word_idx];
            while (bits != 0) {
                const unsigned bit = detail::ctz64(bits);
                const std::size_t k = word_idx * k_word_bits + bit;
                if (k < _cols)
                    result.row_xor_from(i, other, k);
                bits &= (bits - 1);
            }
        }
    }

    return result;
}

BitMatrix BitMatrix::power(std::uint32_t exp) const {
    if (_rows != _cols)
        throw MatrixError(MatrixErr::DimensionMismatch);

    BitMatrix result = identity(_rows);
    BitMatrix base(*this);

    while (exp != 0) {
        if ((exp & 1u) != 0)
            result = result.mul(base);
        exp >>= 1u;
        if (exp != 0)
            base = base.mul(base);
    }

    return result;
}

std::uint64_t BitMatrix::row_popcount(std::size_t row) const {
    if (row >= _rows)
        throw std::out_of_range("row out of bounds");
    return detail::bit_ops().row_popcount(row_ptr_unchecked(row), words_per_row());
}

std::uint64_t BitMatrix::weight() const {
    std::uint64_t total = 0;
    for (std::size_t r = 0; r < _rows; ++r)
        total += row_popcount(r);
    return total;
}

void BitMatrix::swap_rows(std::size_t r1, std::size_t r2) noexcept {
    assert(r1 < _rows);
    assert(r2 < _rows);
    if (r1 == r2)
        return;

    detail::bit_ops().row_swap(row_ptr_unchecked(r1), row_ptr_unchecked(r2), _stride);
}

std::size_t BitMatrix::rank() const {
    BitMatrix temp(*this);
    std::size_t rank_value = 0;

    for (std::size_t c = 0; c < _cols && rank_value < _rows; ++c) {
        std::size_t pivot = rank_value;
        while (pivot < _rows && !temp.get_unchecked(pivot, c))
            ++pivot;

        if (pivot == _rows)
            continue;

        temp.swap_rows(rank_value, pivot);

        for (std::size_t r = rank_value + 1; r < _rows; ++r) {
            if (temp.get_unchecked(r, c))
                temp.row_xor(r, rank_value);
        }

        ++rank_value;
    }

    return rank_value;
}

BitMatrix& BitMatrix::operator=(BitMatrix&& other) noexcept {
    if (this == &other)
        return *this;

    destroy();

    _data = other._data;
    _rows = other._rows;
    _cols = other._cols;
    _stride = other._stride;
    _total_words = other._total_words;

    other._data = nullptr;
    other._rows = 0;
    other._cols = 0;
    other._stride = 0;
    other._total_words = 0;
    return *this;
}

void BitMatrix::swap(BitMatrix& other) noexcept {
    using std::swap;
    swap(_data, other._data);
    swap(_rows, other._rows);
    swap(_cols, other._cols);
    swap(_stride, other._stride);
    swap(_total_words, other._total_words);
}
} // namespace bmmpy
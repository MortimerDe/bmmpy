#include "bmmpy/core/bit_matrix.hpp"

#include <cstring>
#include <limits>
#include <new>

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

} // namespace bmmpy
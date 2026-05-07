#include "bmmpy/core/bit_matrix.hpp"
#include "bmmpy/core/detail/bit_ops.hpp"
#include "bmmpy/core/row_window.hpp"

#include <cassert>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace bmmpy {

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

BitVector BitMatrix::row_bits(std::size_t row) const {
    if (row >= _rows)
        throw std::out_of_range("row out of bounds");

    const std::size_t used_words = words_per_row();
    const std::uint64_t* src = row_ptr_unchecked(row);
    return BitVector(_cols, std::vector<std::uint64_t>(src, src + used_words));
}

void BitMatrix::set_row_bits(std::size_t row, const BitVector& bits) {
    if (row >= _rows)
        throw std::out_of_range("row out of bounds");
    if (bits.bit_count() != _cols)
        throw MatrixError(MatrixErr::DimensionMismatch);

    std::uint64_t* dst = row_ptr_unchecked(row);
    std::memset(dst, 0, _stride * sizeof(std::uint64_t));

    if (!bits.words().empty()) {
        std::memcpy(dst, bits.words().data(), bits.words().size() * sizeof(std::uint64_t));
    }
}

void BitMatrix::row_or_words(std::size_t target_row,
                             std::size_t col_offset,
                             const std::uint64_t* src_words,
                             std::size_t src_bit_count) {
    if (target_row >= _rows)
        throw std::out_of_range("row out of bounds");
    if (col_offset > _cols)
        throw std::out_of_range("column offset out of bounds");
    if (src_bit_count > _cols - col_offset)
        throw std::out_of_range("source segment exceeds row width");
    if (src_bit_count == 0)
        return;
    if (src_words == nullptr)
        throw std::invalid_argument("src_words must not be null when src_bit_count > 0");

    const std::size_t used_words = words_per_row();
    const std::size_t src_word_count = ceil_div(src_bit_count, k_word_bits);
    const std::size_t word_shift = col_offset / k_word_bits;
    const unsigned bit_shift = static_cast<unsigned>(col_offset % k_word_bits);
    const std::size_t tail_bits = src_bit_count % k_word_bits;

    std::uint64_t* dst = row_ptr_unchecked(target_row);

    for (std::size_t i = 0; i < src_word_count; ++i) {
        std::uint64_t word = src_words[i];

        if (tail_bits != 0 && i + 1 == src_word_count)
            word &= ((1ull << tail_bits) - 1ull);

        const std::size_t dst_index = word_shift + i;
        dst[dst_index] |= (word << bit_shift);

        if (bit_shift != 0) {
            const std::size_t carry_index = dst_index + 1;
            if (carry_index < used_words)
                dst[carry_index] |= (word >> (k_word_bits - bit_shift));
        }
    }
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

} // namespace bmmpy
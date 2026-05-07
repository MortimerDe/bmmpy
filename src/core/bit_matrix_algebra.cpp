#include "bmmpy/core/bit_matrix.hpp"
#include "bmmpy/core/detail/bit_intrinsics.hpp"

namespace bmmpy {

BitMatrix BitMatrix::identity(std::size_t n) {
    BitMatrix result(n, n);
    for (std::size_t i = 0; i < n; ++i)
        result.set_unchecked(i, i, true);
    return result;
}

void BitMatrix::xor_assign(const BitMatrix& other) {
    if (_rows != other._rows || _cols != other._cols)
        throw MatrixError(MatrixErr::DimensionMismatch);

    for (std::size_t r = 0; r < _rows; ++r)
        row_xor_from(r, other, r);
}

BitMatrix BitMatrix::add(const BitMatrix& other) const {
    BitMatrix result(*this);
    result.xor_assign(other);
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
                bits &= (bits - 1u);
            }
        }
    }

    return result;
}

BitMatrix BitMatrix::power(std::uint32_t exp) const {
    return power(BitVector::from_u64(static_cast<std::uint64_t>(exp), 32u));
}

BitMatrix BitMatrix::power(const BitVector& exp) const {
    if (_rows != _cols)
        throw MatrixError(MatrixErr::DimensionMismatch);

    BitMatrix result = identity(_rows);
    BitMatrix base(*this);
    BitVector exponent = exp;

    while (exponent.any()) {
        if (exponent.is_odd())
            result = result.mul(base);

        exponent.shift_right_one();
        if (exponent.any())
            base = base.mul(base);
    }

    return result;
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

} // namespace bmmpy
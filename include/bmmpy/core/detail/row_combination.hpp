#pragma once

#include "bmmpy/core/bit_matrix.hpp"
#include "bmmpy/core/detail/bit_intrinsics.hpp"
#include "bmmpy/core/detail/bit_ops.hpp"
#include "bmmpy/core/detail/xor_basis.hpp"
#include "bmmpy/core/row_window.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace bmmpy::detail {

template <typename MaskWords>
void materialize_row_combination(const RowWindow& window,
                                 const MaskWords& mask_words,
                                 std::vector<std::uint64_t>& out_words) {
    std::fill(out_words.begin(), out_words.end(), std::uint64_t{0});

    const auto& ops = bit_ops();
    for (std::size_t word_idx = 0; word_idx < mask_words.size(); ++word_idx) {
        std::uint64_t bits = static_cast<std::uint64_t>(mask_words[word_idx]);
        const std::size_t base = word_idx * 64u;

        while (bits != 0) {
            const std::size_t bit = static_cast<std::size_t>(ctz64(bits));
            const std::size_t row_index = base + bit;
            if (row_index >= window.size()) {
                throw std::out_of_range(
                    "materialize_row_combination: row index exceeds window size");
            }

            ops.row_xor(out_words.data(), window.row_words(row_index), out_words.size());
            bits &= (bits - 1);
        }
    }
}

inline void materialize_row_combination(const RowWindow& window,
                                        const std::uint32_t mask,
                                        std::vector<std::uint64_t>& out_words) {
    const std::array<std::uint64_t, 1> words{static_cast<std::uint64_t>(mask)};
    materialize_row_combination(window, words, out_words);
}

template <typename MaskWords>
void write_mask_row(BitMatrix& matrix,
                    const std::size_t row,
                    const MaskWords& mask_words,
                    const std::size_t bit_limit) {
    for (std::size_t word_idx = 0; word_idx < mask_words.size(); ++word_idx) {
        std::uint64_t bits = static_cast<std::uint64_t>(mask_words[word_idx]);
        const std::size_t base = word_idx * 64u;

        while (bits != 0) {
            const std::size_t bit = static_cast<std::size_t>(ctz64(bits));
            const std::size_t col = base + bit;
            if (col >= bit_limit) {
                throw std::out_of_range("write_mask_row: mask bit exceeds transform width");
            }

            matrix.set_unchecked(row, col, true);
            bits &= (bits - 1);
        }
    }
}

inline BitMatrix build_transform_matrix(const std::size_t input_rows,
                                        const std::vector<std::vector<std::uint64_t>>& mask_rows) {
    BitMatrix transform(mask_rows.size(), input_rows);
    for (std::size_t row = 0; row < mask_rows.size(); ++row)
        write_mask_row(transform, row, mask_rows[row], input_rows);
    return transform;
}

inline BitMatrix build_transform_matrix(const std::size_t input_rows,
                                        const std::vector<std::uint32_t>& masks) {
    BitMatrix transform(masks.size(), input_rows);
    for (std::size_t row = 0; row < masks.size(); ++row) {
        const std::array<std::uint64_t, 1> words{static_cast<std::uint64_t>(masks[row])};
        write_mask_row(transform, row, words, input_rows);
    }
    return transform;
}

} // namespace bmmpy::detail
#include "bmmpy/core/bit_matrix.hpp"

#include <picosha2.h>

namespace bmmpy {
namespace {

std::uint64_t logical_last_word_mask(std::size_t cols) noexcept {
    const std::size_t tail_bits = cols % BitMatrix::k_word_bits;
    if (tail_bits == 0)
        return ~std::uint64_t{0};
    return (std::uint64_t{1} << tail_bits) - 1u;
}

void hash_u64_le(picosha2::hash256_one_by_one& hasher, std::uint64_t value) {
    unsigned char bytes[8];
    bytes[0] = static_cast<unsigned char>(value);
    bytes[1] = static_cast<unsigned char>(value >> 8u);
    bytes[2] = static_cast<unsigned char>(value >> 16u);
    bytes[3] = static_cast<unsigned char>(value >> 24u);
    bytes[4] = static_cast<unsigned char>(value >> 32u);
    bytes[5] = static_cast<unsigned char>(value >> 40u);
    bytes[6] = static_cast<unsigned char>(value >> 48u);
    bytes[7] = static_cast<unsigned char>(value >> 56u);
    hasher.process(bytes, bytes + 8);
}

} // namespace

std::string BitMatrix::hash() const {
    static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t),
                  "BitMatrix::hash requires size_t to fit into uint64_t");

    picosha2::hash256_one_by_one hasher;
    hash_u64_le(hasher, static_cast<std::uint64_t>(_rows));
    hash_u64_le(hasher, static_cast<std::uint64_t>(_cols));

    const std::size_t used_words = words_per_row();
    if (used_words == 0) {
        hasher.finish();
        return picosha2::get_hash_hex_string(hasher);
    }

    const std::uint64_t last_word_mask = logical_last_word_mask(_cols);

    for (std::size_t row = 0; row < _rows; ++row) {
        const std::uint64_t* row_data = row_ptr_unchecked(row);
        for (std::size_t word_idx = 0; word_idx < used_words; ++word_idx) {
            std::uint64_t word = row_data[word_idx];
            if (word_idx + 1 == used_words)
                word &= last_word_mask;
            hash_u64_le(hasher, word);
        }
    }

    hasher.finish();
    return picosha2::get_hash_hex_string(hasher);
}

} // namespace bmmpy
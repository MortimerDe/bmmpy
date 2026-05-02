#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace bmmpy::detail {

inline std::size_t word_count_for_bits(const std::size_t bit_count) noexcept {
    return (bit_count + 63u) / 64u;
}

inline std::uint64_t tail_mask_for_bits(const std::size_t bit_count) noexcept {
    const std::size_t tail_bits = bit_count % 64u;
    if (tail_bits == 0)
        return ~std::uint64_t{0};

    return (std::uint64_t{1} << tail_bits) - 1;
}

inline int msb_index64(const std::uint64_t value) noexcept {
#if defined(_MSC_VER) && defined(_M_X64)
    unsigned long index = 0;
    _BitScanReverse64(&index, value);
    return static_cast<int>(index);
#elif defined(_MSC_VER)
    unsigned long index = 0;
    const auto high = static_cast<unsigned long>(value >> 32);
    if (high != 0) {
        _BitScanReverse(&index, high);
        return static_cast<int>(index + 32);
    }

    _BitScanReverse(&index, static_cast<unsigned long>(value & 0xffffffffull));
    return static_cast<int>(index);
#else
    return 63 - __builtin_clzll(value);
#endif
}

inline int highest_set_bit(const std::vector<std::uint64_t>& words) noexcept {
    for (std::size_t word_idx = words.size(); word_idx > 0; --word_idx) {
        const std::uint64_t word = words[word_idx - 1];
        if (word != 0) {
            return static_cast<int>((word_idx - 1) * 64u +
                                    static_cast<std::size_t>(msb_index64(word)));
        }
    }

    return -1;
}

inline void xor_words_inplace(std::vector<std::uint64_t>& dst,
                              const std::vector<std::uint64_t>& src) noexcept {
    for (std::size_t i = 0; i < dst.size(); ++i)
        dst[i] ^= src[i];
}

inline bool mask_words_fit_bits(const std::vector<std::uint64_t>& mask_words,
                                const std::size_t bit_count) noexcept {
    const std::size_t used_words = word_count_for_bits(bit_count);

    if (used_words == 0) {
        for (const std::uint64_t word : mask_words) {
            if (word != 0)
                return false;
        }
        return true;
    }

    for (std::size_t i = used_words; i < mask_words.size(); ++i) {
        if (mask_words[i] != 0)
            return false;
    }

    if ((bit_count % 64u) != 0 && used_words <= mask_words.size()) {
        const std::uint64_t overflow_bits =
            mask_words[used_words - 1] & ~tail_mask_for_bits(bit_count);
        if (overflow_bits != 0)
            return false;
    }

    return true;
}

inline std::vector<std::uint64_t> normalize_mask_words(const std::vector<std::uint64_t>& mask_words,
                                                       const std::size_t bit_count) {
    if (!mask_words_fit_bits(mask_words, bit_count)) {
        throw std::invalid_argument("mask contains bits outside the supported row range");
    }

    std::vector<std::uint64_t> normalized(word_count_for_bits(bit_count), 0);
    const std::size_t copy_words = std::min(normalized.size(), mask_words.size());

    for (std::size_t i = 0; i < copy_words; ++i)
        normalized[i] = mask_words[i];

    return normalized;
}

inline std::vector<std::uint64_t> make_unit_mask_words(const std::size_t bit_count,
                                                       const std::size_t bit_index) {
    if (bit_index >= bit_count)
        throw std::out_of_range("unit mask bit index out of bounds");

    std::vector<std::uint64_t> words(word_count_for_bits(bit_count), 0);
    words[bit_index / 64u] = std::uint64_t{1} << (bit_index % 64u);
    return words;
}

class PivotBasis {
public:
    explicit PivotBasis(const std::size_t bit_width)
        : _pivot_rows(bit_width), _used(bit_width, false) {}

    bool try_insert(const std::vector<std::uint64_t>& candidate) {
        std::vector<std::uint64_t> reduced = candidate;
        return try_insert_reduced(std::move(reduced));
    }

    bool try_insert(std::vector<std::uint64_t>&& candidate) {
        return try_insert_reduced(std::move(candidate));
    }

    std::size_t rank() const noexcept { return _rank; }

private:
    bool try_insert_reduced(std::vector<std::uint64_t> reduced) {
        while (true) {
            const int pivot = highest_set_bit(reduced);
            if (pivot < 0)
                return false;

            const std::size_t pivot_index = static_cast<std::size_t>(pivot);
            if (!_used[pivot_index]) {
                _pivot_rows[pivot_index] = std::move(reduced);
                _used[pivot_index] = true;
                ++_rank;
                return true;
            }

            xor_words_inplace(reduced, _pivot_rows[pivot_index]);
        }
    }

    std::vector<std::vector<std::uint64_t>> _pivot_rows;
    std::vector<bool> _used;
    std::size_t _rank = 0;
};

} // namespace bmmpy::detail
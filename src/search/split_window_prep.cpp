#include "bmmpy/search/split_window_prep.hpp"

#include "bmmpy/core/detail/bit_intrinsics.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bmmpy {
namespace {

constexpr std::size_t k_word_bits = std::numeric_limits<std::uint64_t>::digits;

std::uint64_t tail_mask_for_cols(std::size_t cols) noexcept {
    const std::size_t tail_bits = cols % k_word_bits;
    if (tail_bits == 0)
        return ~std::uint64_t{0};

    return (std::uint64_t{1} << tail_bits) - 1;
}

std::uint64_t low_mask_for_bits(std::size_t low_bits) noexcept {
    if (low_bits == 0)
        return 0;

    if (low_bits >= k_word_bits)
        return ~std::uint64_t{0};

    return (std::uint64_t{1} << low_bits) - 1;
}

} // namespace

CompactSplitWindow build_compact_split_window(const RowWindow& window, std::size_t low_bits) {
    return build_compact_split_window(
        window.row_ptrs(), window.words_per_row(), window.cols(), low_bits);
}

CompactSplitWindow build_compact_split_window(const std::vector<const std::uint64_t*>& rows,
                                              std::size_t words_per_row,
                                              std::size_t cols,
                                              std::size_t low_bits) {
    CompactSplitWindow out;
    out.t = rows.size();
    out.low_bits = low_bits;

    if (low_bits > out.t) {
        throw std::invalid_argument(
            "build_compact_split_window: low_bits must be <= number of rows");
    }

    if (out.t > k_word_bits) {
        throw std::invalid_argument("build_compact_split_window: row count must be <= 64");
    }

    out.high_bits = out.t - low_bits;

    if (rows.empty() || words_per_row == 0)
        return out;

    std::unordered_map<std::uint64_t, std::int32_t> pattern_counts;
    pattern_counts.reserve(std::min<std::size_t>(cols, 1024));

    std::int64_t total_weight64 = 0;
    const std::uint64_t tail_mask = tail_mask_for_cols(cols);

    for (std::size_t word_index = 0; word_index < words_per_row; ++word_index) {
        std::uint64_t active = 0;
        for (const std::uint64_t* row_words : rows)
            active |= row_words[word_index];

        if (word_index + 1 == words_per_row)
            active &= tail_mask;

        while (active != 0) {
            const unsigned bit = detail::ctz64(active);
            const std::uint64_t bit_mask = std::uint64_t{1} << bit;

            std::uint64_t pattern = 0;
            for (std::size_t row = 0; row < out.t; ++row) {
                if ((rows[row][word_index] & bit_mask) != 0)
                    pattern |= (std::uint64_t{1} << row);
            }

            auto [it, inserted] = pattern_counts.emplace(pattern, 0);
            (void)inserted;

            if (it->second == std::numeric_limits<std::int32_t>::max()) {
                throw std::overflow_error(
                    "build_compact_split_window: column multiplicity overflow");
            }

            ++it->second;
            ++total_weight64;

            if (total_weight64 >
                static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
                throw std::overflow_error(
                    "build_compact_split_window: total weight exceeds int32 range");
            }

            active &= (active - 1);
        }
    }

    out.total_weight = static_cast<std::int32_t>(total_weight64);

    std::vector<std::pair<std::uint64_t, std::int32_t>> entries;
    entries.reserve(pattern_counts.size());

    for (const auto& entry : pattern_counts)
        entries.push_back(entry);

    std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
    });

    out.q.reserve(entries.size());
    out.r.reserve(entries.size());
    out.multiplicity.reserve(entries.size());

    const std::uint64_t low_mask = low_mask_for_bits(low_bits);

    for (const auto& [pattern, multiplicity] : entries) {
        const std::uint64_t q = low_bits == k_word_bits ? 0 : (pattern >> low_bits);
        const std::uint64_t r = pattern & low_mask;

        out.q.push_back(q);
        out.r.push_back(r);
        out.multiplicity.push_back(multiplicity);
    }

    return out;
}

} // namespace bmmpy
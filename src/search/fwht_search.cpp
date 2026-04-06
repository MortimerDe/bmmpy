#include "bmmpy/search/fwht_search.hpp"

#include "bmmpy/core/detail/bit_intrinsics.hpp"
#include "bmmpy/math/fwht.hpp"

#include <algorithm>
#include <climits>
#include <limits>
#include <stdexcept>

namespace bmmpy {
namespace {

bool candidate_less(const Candidate& lhs, const Candidate& rhs) noexcept {
    if (lhs.weight != rhs.weight)
        return lhs.weight < rhs.weight;

    const std::uint64_t lhs_mask = lhs.mask.empty() ? 0 : lhs.mask.front();
    const std::uint64_t rhs_mask = rhs.mask.empty() ? 0 : rhs.mask.front();
    return lhs_mask < rhs_mask;
}

std::uint64_t tail_mask_for_cols(std::size_t cols) noexcept {
    const std::size_t tail_bits = cols % BitMatrix::k_word_bits;
    if (tail_bits == 0)
        return ~std::uint64_t{0};

    return (std::uint64_t{1} << tail_bits) - 1;
}

} // namespace

std::vector<Candidate> FwhtSearch::search(const BitMatrix& matrix,
                                          const std::vector<std::size_t>& window_rows) {
    const std::size_t t = window_rows.size();
    if (t == 0 || t > _config.max_rows || _config.k == 0)
        return {};

    if (t > Candidate::k_word_bits) {
        throw std::invalid_argument("FwhtSearch: window_rows size must be <= 64");
    }

    if (t >= static_cast<std::size_t>(sizeof(std::size_t) * CHAR_BIT)) {
        throw std::invalid_argument(
            "FwhtSearch: window_rows size is too large for bucket indexing");
    }

    const std::size_t bucket_count = std::size_t{1} << t;
    const std::size_t words_per_row = matrix.words_per_row();
    if (words_per_row == 0)
        return {};

    std::vector<const std::uint64_t*> rows;
    rows.reserve(t);
    for (std::size_t row : window_rows) {
        if (row >= matrix.rows())
            throw std::out_of_range("window row out of bounds");
        rows.push_back(matrix.row_words(row));
    }

    _buckets.assign(bucket_count, 0);

    std::uint64_t total_weight = 0;
    const std::uint64_t tail_mask = tail_mask_for_cols(matrix.cols());

    for (std::size_t word_index = 0; word_index < words_per_row; ++word_index) {
        std::uint64_t active = 0;
        for (const std::uint64_t* row_words : rows)
            active |= row_words[word_index];

        if (word_index + 1 == words_per_row)
            active &= tail_mask;

        while (active != 0) {
            const unsigned bit = detail::ctz64(active);
            const std::uint64_t bit_mask = std::uint64_t{1} << bit;

            std::size_t pattern = 0;
            for (std::size_t r = 0; r < t; ++r) {
                if ((rows[r][word_index] & bit_mask) != 0)
                    pattern |= (std::size_t{1} << r);
            }

            if (_buckets[pattern] == std::numeric_limits<std::int32_t>::max()) {
                throw std::overflow_error("FwhtSearch: bucket overflow");
            }

            ++_buckets[pattern];
            ++total_weight;

            if (total_weight >
                static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
                throw std::overflow_error("FwhtSearch: total weight exceeds int32 range");
            }

            active &= (active - 1);
        }
    }

    fwht_inplace(_buckets.data(), _buckets.size());

    const std::size_t k = _config.k;
    std::vector<Candidate> candidates;
    candidates.reserve(std::min(k, bucket_count - 1));

    auto sort_candidates = [&candidates]() {
        std::sort(candidates.begin(), candidates.end(), candidate_less);
    };

    std::uint32_t worst_weight = std::numeric_limits<std::uint32_t>::max();

    for (std::size_t mask = 1; mask < bucket_count; ++mask) {
        const std::int64_t h = static_cast<std::int64_t>(_buckets[mask]);
        const std::int64_t w2 = static_cast<std::int64_t>(total_weight) - h;

        if (w2 < 0 || (w2 & 1) != 0)
            continue;

        const std::uint32_t weight = static_cast<std::uint32_t>(w2 / 2);

        if (candidates.size() < k) {
            candidates.push_back(Candidate::from_u64(static_cast<std::uint64_t>(mask), weight));

            if (candidates.size() == k) {
                sort_candidates();
                worst_weight = candidates.back().weight;
            }
        } else if (weight < worst_weight) {
            candidates.back() = Candidate::from_u64(static_cast<std::uint64_t>(mask), weight);
            sort_candidates();
            worst_weight = candidates.back().weight;
        }
    }

    if (candidates.size() < k)
        sort_candidates();

    return candidates;
}

} // namespace bmmpy
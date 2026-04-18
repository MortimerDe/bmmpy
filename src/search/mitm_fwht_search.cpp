#include "bmmpy/search/mitm_fwht_search.hpp"

#include "bmmpy/core/detail/bit_intrinsics.hpp"
#include "bmmpy/math/fwht.hpp"

#include <algorithm>
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

std::int64_t pack_col(std::int32_t weight, std::uint32_t high_mask) noexcept {
    const std::uint64_t upper = static_cast<std::uint64_t>(static_cast<std::uint32_t>(weight))
                                << 32;
    return static_cast<std::int64_t>(upper | static_cast<std::uint64_t>(high_mask));
}

std::int32_t unpack_weight(std::int64_t data) noexcept {
    const std::uint64_t raw = static_cast<std::uint64_t>(data);
    return static_cast<std::int32_t>(raw >> 32);
}

std::uint32_t unpack_high_mask(std::int64_t data) noexcept {
    return static_cast<std::uint32_t>(static_cast<std::uint64_t>(data));
}

} // namespace

MitmFwhtSearch::MitmFwhtSearch(MitmFwhtSearchConfig config) : _config(config) {
    const std::size_t initial_cols = std::max<std::size_t>(_config.reserve_unique_patterns, 1);
    const std::size_t max_low_bits = std::max<std::size_t>(_config.reserve_left_rows, 1);
    const std::size_t max_high_states = std::max<std::size_t>(_config.reserve_right_states, 1);

    _col_data.resize(initial_cols, 0);

    _adj_offsets.resize(max_low_bits + 1, 0);
    _adj_counts.resize(max_low_bits, 0);
    _adj_indices.resize(initial_cols * max_low_bits, 0);

    _buckets.resize(max_high_states, 0);
    _fwht_buffer.resize(max_high_states, 0);

    _candidates.reserve(_config.max_candidates);
}

std::pair<std::size_t, std::size_t> MitmFwhtSearch::get_split_info(std::size_t t) noexcept {
    const std::size_t high_bits = t <= 26 ? (t / 2) : std::min<std::size_t>(t / 2, 13);
    const std::size_t low_bits = t - high_bits;
    return {low_bits, high_bits};
}

void MitmFwhtSearch::ensure_capacity(std::size_t unique_patterns,
                                     std::size_t low_bits,
                                     std::size_t high_states) {
    if (unique_patterns > _col_data.size()) {
        const std::size_t new_len = std::max(unique_patterns, _col_data.size() * 2);
        _col_data.resize(new_len, 0);
    }

    if (low_bits > _config.reserve_left_rows) {
        _config.reserve_left_rows = low_bits + 4;
        _adj_offsets.resize(_config.reserve_left_rows + 1, 0);
        _adj_counts.resize(_config.reserve_left_rows, 0);
    }

    if (low_bits != 0 && unique_patterns > std::numeric_limits<std::size_t>::max() / low_bits) {
        throw std::overflow_error("MitmFwhtSearch: adjacency size overflow");
    }

    const std::size_t required_indices = unique_patterns * low_bits;
    if (required_indices > _adj_indices.size())
        _adj_indices.resize(required_indices + 4096, 0);

    if (high_states > _buckets.size()) {
        _config.reserve_right_states = high_states;
        _buckets.resize(high_states, 0);
        _fwht_buffer.resize(high_states, 0);
    }
}

void MitmFwhtSearch::build_adjacency(const CompactSplitWindow& prep) {
    const std::size_t unique_count = prep.multiplicity.size();
    const std::size_t low_bits = prep.low_bits;

    std::fill(_adj_counts.begin(), _adj_counts.begin() + low_bits, 0);

    for (std::size_t i = 0; i < unique_count; ++i) {
        const std::uint64_t high_mask64 = prep.q[i];
        const std::uint64_t low_mask64 = prep.r[i];

        if (high_mask64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) ||
            low_mask64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
            throw std::invalid_argument("MitmFwhtSearch: split masks must fit in u32");
        }

        _col_data[i] = pack_col(prep.multiplicity[i], static_cast<std::uint32_t>(high_mask64));

        std::uint32_t low_mask = static_cast<std::uint32_t>(low_mask64);
        while (low_mask != 0) {
            const std::size_t bit = static_cast<std::size_t>(detail::ctz64(low_mask));
            ++_adj_counts[bit];
            low_mask &= (low_mask - 1);
        }
    }

    std::size_t offset = 0;
    for (std::size_t bit = 0; bit < low_bits; ++bit) {
        _adj_offsets[bit] = offset;
        offset += static_cast<std::size_t>(_adj_counts[bit]);
        _adj_counts[bit] = 0;
    }
    _adj_offsets[low_bits] = offset;

    for (std::size_t i = 0; i < unique_count; ++i) {
        std::uint32_t low_mask = static_cast<std::uint32_t>(prep.r[i]);
        while (low_mask != 0) {
            const std::size_t bit = static_cast<std::size_t>(detail::ctz64(low_mask));
            const std::size_t pos = _adj_offsets[bit] + static_cast<std::size_t>(_adj_counts[bit]);
            _adj_indices[pos] = static_cast<std::int32_t>(i);
            ++_adj_counts[bit];
            low_mask &= (low_mask - 1);
        }
    }

    for (std::size_t bit = 0; bit < low_bits; ++bit) {
        const std::size_t start = _adj_offsets[bit];
        const std::size_t len = static_cast<std::size_t>(_adj_counts[bit]);
        if (len > 1) {
            std::sort(_adj_indices.begin() + static_cast<std::ptrdiff_t>(start),
                      _adj_indices.begin() + static_cast<std::ptrdiff_t>(start + len));
        }
    }
}

void MitmFwhtSearch::initialize_buckets(const CompactSplitWindow& prep) {
    const std::size_t n_high = std::size_t{1} << prep.high_bits;
    std::fill(_buckets.begin(), _buckets.begin() + n_high, 0);

    for (std::size_t i = 0; i < prep.multiplicity.size(); ++i) {
        const std::size_t high_mask = static_cast<std::size_t>(prep.q[i]);
        const std::int32_t weight = prep.multiplicity[i];

        const std::int64_t new_bucket = static_cast<std::int64_t>(_buckets[high_mask]) + weight;

        if (new_bucket < std::numeric_limits<std::int32_t>::min() ||
            new_bucket > std::numeric_limits<std::int32_t>::max()) {
            throw std::overflow_error("MitmFwhtSearch: bucket overflow in initialize_buckets");
        }

        _buckets[high_mask] = static_cast<std::int32_t>(new_bucket);
    }
}

void MitmFwhtSearch::apply_bit_flip(std::size_t bit) {
    const std::size_t start = _adj_offsets[bit];
    const std::size_t end = _adj_offsets[bit + 1];

    for (std::size_t j = start; j < end; ++j) {
        const std::size_t col_idx = static_cast<std::size_t>(_adj_indices[j]);

        const std::int64_t data = _col_data[col_idx];
        const std::uint32_t high_mask = unpack_high_mask(data);
        const std::int32_t old_weight = unpack_weight(data);

        const std::int64_t new_bucket = static_cast<std::int64_t>(_buckets[high_mask]) -
                                        2LL * static_cast<std::int64_t>(old_weight);

        if (new_bucket < std::numeric_limits<std::int32_t>::min() ||
            new_bucket > std::numeric_limits<std::int32_t>::max()) {
            throw std::overflow_error("MitmFwhtSearch: bucket overflow in apply_bit_flip");
        }

        _buckets[high_mask] = static_cast<std::int32_t>(new_bucket);
        _col_data[col_idx] = pack_col(-old_weight, high_mask);
    }
}

void MitmFwhtSearch::add_result(std::uint64_t mask,
                                std::uint32_t weight,
                                std::int32_t total_weight,
                                std::int32_t& min_score_threshold,
                                std::uint32_t& worst_weight) {
    const std::size_t k = _config.max_candidates;

    const Candidate incoming = Candidate::from_u64(mask, weight);

    auto sort_candidates = [this]() {
        std::sort(_candidates.begin(), _candidates.end(), candidate_less);
    };

    if (_candidates.size() < k) {
        _candidates.push_back(incoming);

        if (_candidates.size() == k) {
            sort_candidates();
            worst_weight = _candidates.back().weight;

            const std::int64_t threshold64 = static_cast<std::int64_t>(total_weight) -
                                             2LL * static_cast<std::int64_t>(worst_weight);

            min_score_threshold = threshold64 < std::numeric_limits<std::int32_t>::min()
                                      ? std::numeric_limits<std::int32_t>::min()
                                      : static_cast<std::int32_t>(threshold64);
        }

        return;
    }

    if (candidate_less(incoming, _candidates.back())) {
        _candidates.back() = incoming;
        sort_candidates();
        worst_weight = _candidates.back().weight;

        const std::int64_t threshold64 =
            static_cast<std::int64_t>(total_weight) - 2LL * static_cast<std::int64_t>(worst_weight);

        min_score_threshold = threshold64 < std::numeric_limits<std::int32_t>::min()
                                  ? std::numeric_limits<std::int32_t>::min()
                                  : static_cast<std::int32_t>(threshold64);
    }
}

void MitmFwhtSearch::process_candidate(std::uint32_t x_low,
                                       std::size_t n_high,
                                       std::size_t low_bits,
                                       std::int32_t total_weight,
                                       std::int32_t& min_score_threshold,
                                       std::uint32_t& worst_weight) {
    std::copy_n(_buckets.begin(), n_high, _fwht_buffer.begin());
    fwht_inplace(_fwht_buffer.data(), n_high);

    for (std::size_t x_high = 0; x_high < n_high; ++x_high) {
        const std::int32_t score = _fwht_buffer[x_high];
        if (score < min_score_threshold)
            continue;

        const std::int64_t w2 =
            static_cast<std::int64_t>(total_weight) - static_cast<std::int64_t>(score);

        if (w2 < 0 || (w2 & 1) != 0)
            continue;

        const std::uint64_t mask =
            (static_cast<std::uint64_t>(x_high) << low_bits) | static_cast<std::uint64_t>(x_low);

        if (mask == 0)
            continue;

        add_result(mask,
                   static_cast<std::uint32_t>(w2 / 2),
                   total_weight,
                   min_score_threshold,
                   worst_weight);
    }
}

std::vector<Candidate> MitmFwhtSearch::search(const RowWindow& window) {
    _candidates.clear();

    if (_config.max_candidates == 0)
        return {};

    const std::size_t t = window.size();
    if (t == 0)
        return {};

    if (t > Candidate::k_word_bits) {
        throw std::invalid_argument("MitmFwhtSearch: window_rows size must be <= 64");
    }

    const auto [low_bits, high_bits] = get_split_info(t);

    if (low_bits > kMaxLowBits) {
        throw std::invalid_argument("MitmFwhtSearch: low half exceeds supported width");
    }

    if (low_bits >= kMaxHalfBits || high_bits >= kMaxHalfBits) {
        throw std::invalid_argument("MitmFwhtSearch: split dimensions must be < 31");
    }

    const auto prep = build_compact_split_window(window, low_bits);
    if (prep.multiplicity.empty())
        return {};

    const std::size_t n_high = std::size_t{1} << high_bits;
    ensure_capacity(prep.multiplicity.size(), low_bits, n_high);
    build_adjacency(prep);
    initialize_buckets(prep);

    std::int32_t min_score_threshold = std::numeric_limits<std::int32_t>::min();
    std::uint32_t worst_weight = std::numeric_limits<std::uint32_t>::max();

    process_candidate(0, n_high, low_bits, prep.total_weight, min_score_threshold, worst_weight);

    std::uint32_t current_x_low = 0;
    const std::size_t n_loop = std::size_t{1} << low_bits;

    for (std::size_t i = 1; i < n_loop; ++i) {
        const std::size_t bit = static_cast<std::size_t>(detail::ctz64(i));
        apply_bit_flip(bit);
        current_x_low ^= (std::uint32_t{1} << bit);

        process_candidate(
            current_x_low, n_high, low_bits, prep.total_weight, min_score_threshold, worst_weight);
    }

    if (_candidates.size() < _config.max_candidates) {
        std::sort(_candidates.begin(), _candidates.end(), candidate_less);
    }

    return _candidates;
}

} // namespace bmmpy
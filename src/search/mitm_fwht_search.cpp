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

std::uint64_t tail_mask_for_cols(std::size_t cols) noexcept {
    const std::size_t tail_bits = cols % BitMatrix::k_word_bits;
    if (tail_bits == 0)
        return ~std::uint64_t{0};

    return (std::uint64_t{1} << tail_bits) - 1;
}

std::int64_t pack_col(std::int32_t weight, std::uint32_t right_mask) noexcept {
    const std::uint64_t upper = static_cast<std::uint64_t>(static_cast<std::uint32_t>(weight))
                                << 32;
    return static_cast<std::int64_t>(upper | static_cast<std::uint64_t>(right_mask));
}

std::int32_t unpack_weight(std::int64_t data) noexcept {
    const std::uint64_t raw = static_cast<std::uint64_t>(data);
    return static_cast<std::int32_t>(raw >> 32);
}

std::uint32_t unpack_right_mask(std::int64_t data) noexcept {
    return static_cast<std::uint32_t>(static_cast<std::uint64_t>(data));
}

} // namespace

MitmFwhtSearch::MitmFwhtSearch(MitmFwhtSearchConfig config) : _config(config) {
    const std::size_t initial_cols = std::max<std::size_t>(_config.initial_capacity_cols, 1);
    const std::size_t max_t_left = std::max<std::size_t>(_config.max_t_left, 1);
    const std::size_t max_n_right = std::max<std::size_t>(_config.max_n_right, 1);

    _col_data.resize(initial_cols, 0);
    _left_masks.resize(initial_cols, 0);

    _adj_offsets.resize(max_t_left + 1, 0);
    _adj_counts.resize(max_t_left, 0);
    _adj_indices.resize(initial_cols * max_t_left, 0);

    _buckets.resize(max_n_right, 0);
    _fwht_buffer.resize(max_n_right, 0);

    _col_map.reserve(initial_cols);
    _candidates.reserve(_config.k_limit);
}

std::pair<std::size_t, std::size_t> MitmFwhtSearch::get_split_info(std::size_t t) noexcept {
    const std::size_t t_right = t <= 26 ? (t / 2) : std::min<std::size_t>(t / 2, 13);
    const std::size_t t_left = t - t_right;
    return {t_left, t_right};
}

void MitmFwhtSearch::ensure_capacity(std::size_t cols, std::size_t t_left, std::size_t n_right) {
    if (cols > _col_data.size()) {
        const std::size_t new_len = std::max(cols, _col_data.size() * 2);
        _col_data.resize(new_len, 0);
        _left_masks.resize(new_len, 0);
        _col_map.reserve(new_len);
    }

    if (t_left > _config.max_t_left) {
        _config.max_t_left = t_left + 4;
        _adj_offsets.resize(_config.max_t_left + 1, 0);
        _adj_counts.resize(_config.max_t_left, 0);
    }

    if (t_left != 0 && cols > std::numeric_limits<std::size_t>::max() / t_left) {
        throw std::overflow_error("MitmFwhtSearch: adjacency size overflow");
    }

    const std::size_t required_indices = cols * t_left;
    if (required_indices > _adj_indices.size())
        _adj_indices.resize(required_indices + 4096, 0);

    if (n_right > _buckets.size()) {
        _config.max_n_right = n_right;
        _buckets.resize(n_right, 0);
        _fwht_buffer.resize(n_right, 0);
    }
}

std::pair<std::size_t, std::int32_t>
MitmFwhtSearch::prepare_columns(const std::vector<const std::uint64_t*>& rows,
                                std::size_t words_per_row,
                                std::size_t cols,
                                std::size_t t_left) {
    const std::size_t t = rows.size();
    const std::size_t t_right = t - t_left;
    const std::size_t n_right = std::size_t{1} << t_right;

    _col_map.clear();

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

            std::uint32_t s_left = 0;
            std::uint32_t s_right = 0;

            for (std::size_t r = 0; r < t_left; ++r) {
                if ((rows[r][word_index] & bit_mask) != 0)
                    s_left |= (std::uint32_t{1} << r);
            }

            for (std::size_t r = 0; r < t_right; ++r) {
                if ((rows[t_left + r][word_index] & bit_mask) != 0)
                    s_right |= (std::uint32_t{1} << r);
            }

            const std::uint64_t key =
                (static_cast<std::uint64_t>(s_left) << 32) | static_cast<std::uint64_t>(s_right);

            auto [it, inserted] = _col_map.emplace(key, 0);
            (void)inserted;

            if (it->second == std::numeric_limits<std::int32_t>::max())
                throw std::overflow_error("MitmFwhtSearch: column multiplicity overflow");

            ++it->second;
            ++total_weight64;

            if (total_weight64 >
                static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
                throw std::overflow_error("MitmFwhtSearch: total weight exceeds int32 range");
            }

            active &= (active - 1);
        }
    }

    const std::size_t unique_count = _col_map.size();
    if (unique_count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("MitmFwhtSearch: unique column count exceeds int32 range");
    }

    ensure_capacity(unique_count, t_left, n_right);

    std::fill(_adj_counts.begin(), _adj_counts.begin() + t_left, 0);

    std::size_t idx = 0;
    for (const auto& entry : _col_map) {
        const std::uint32_t s_left = static_cast<std::uint32_t>(entry.first >> 32);
        const std::uint32_t s_right = static_cast<std::uint32_t>(entry.first);
        const std::int32_t weight = entry.second;

        _col_data[idx] = pack_col(weight, s_right);
        _left_masks[idx] = s_left;

        std::uint32_t temp = s_left;
        while (temp != 0) {
            const std::size_t b = static_cast<std::size_t>(detail::ctz64(temp));
            ++_adj_counts[b];
            temp &= (temp - 1);
        }

        ++idx;
    }

    std::size_t offset = 0;
    for (std::size_t b = 0; b < t_left; ++b) {
        _adj_offsets[b] = offset;
        offset += static_cast<std::size_t>(_adj_counts[b]);
        _adj_counts[b] = 0;
    }
    _adj_offsets[t_left] = offset;

    for (std::size_t i = 0; i < unique_count; ++i) {
        std::uint32_t s_left = _left_masks[i];
        while (s_left != 0) {
            const std::size_t b = static_cast<std::size_t>(detail::ctz64(s_left));
            const std::size_t pos = _adj_offsets[b] + static_cast<std::size_t>(_adj_counts[b]);
            _adj_indices[pos] = static_cast<std::int32_t>(i);
            ++_adj_counts[b];
            s_left &= (s_left - 1);
        }
    }

    for (std::size_t b = 0; b < t_left; ++b) {
        const std::size_t start = _adj_offsets[b];
        const std::size_t len = static_cast<std::size_t>(_adj_counts[b]);
        if (len > 1) {
            std::sort(_adj_indices.begin() + static_cast<std::ptrdiff_t>(start),
                      _adj_indices.begin() + static_cast<std::ptrdiff_t>(start + len));
        }
    }

    return {unique_count, static_cast<std::int32_t>(total_weight64)};
}

void MitmFwhtSearch::initialize_buckets(std::size_t unique_count, std::size_t n_right) {
    std::fill(_buckets.begin(), _buckets.begin() + n_right, 0);

    for (std::size_t i = 0; i < unique_count; ++i) {
        const std::int64_t data = _col_data[i];
        const std::int32_t weight = unpack_weight(data);
        const std::uint32_t right_mask = unpack_right_mask(data);

        const std::int64_t new_bucket = static_cast<std::int64_t>(_buckets[right_mask]) + weight;

        if (new_bucket < std::numeric_limits<std::int32_t>::min() ||
            new_bucket > std::numeric_limits<std::int32_t>::max()) {
            throw std::overflow_error("MitmFwhtSearch: bucket overflow in initialize_buckets");
        }

        _buckets[right_mask] = static_cast<std::int32_t>(new_bucket);
    }
}

void MitmFwhtSearch::apply_bit_flip(std::size_t bit) {
    const std::size_t start = _adj_offsets[bit];
    const std::size_t end = _adj_offsets[bit + 1];

    for (std::size_t j = start; j < end; ++j) {
        const std::size_t col_idx = static_cast<std::size_t>(_adj_indices[j]);

        const std::int64_t data = _col_data[col_idx];
        const std::uint32_t right_mask = unpack_right_mask(data);
        const std::int32_t old_weight = unpack_weight(data);

        const std::int64_t new_bucket = static_cast<std::int64_t>(_buckets[right_mask]) -
                                        2LL * static_cast<std::int64_t>(old_weight);

        if (new_bucket < std::numeric_limits<std::int32_t>::min() ||
            new_bucket > std::numeric_limits<std::int32_t>::max()) {
            throw std::overflow_error("MitmFwhtSearch: bucket overflow in apply_bit_flip");
        }

        _buckets[right_mask] = static_cast<std::int32_t>(new_bucket);
        _col_data[col_idx] = pack_col(-old_weight, right_mask);
    }
}

void MitmFwhtSearch::add_result(std::uint64_t mask,
                                std::uint32_t weight,
                                std::int32_t total_weight,
                                std::int32_t& min_score_threshold,
                                std::uint32_t& worst_weight) {
    const std::size_t k = _config.k_limit;

    auto sort_candidates = [this]() {
        std::sort(_candidates.begin(), _candidates.end(), candidate_less);
    };

    if (_candidates.size() < k) {
        _candidates.push_back(Candidate::from_u64(mask, weight));

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

    if (weight < worst_weight) {
        _candidates.back() = Candidate::from_u64(mask, weight);
        sort_candidates();
        worst_weight = _candidates.back().weight;

        const std::int64_t threshold64 =
            static_cast<std::int64_t>(total_weight) - 2LL * static_cast<std::int64_t>(worst_weight);

        min_score_threshold = threshold64 < std::numeric_limits<std::int32_t>::min()
                                  ? std::numeric_limits<std::int32_t>::min()
                                  : static_cast<std::int32_t>(threshold64);
    }
}

void MitmFwhtSearch::process_candidate(std::uint32_t x_left,
                                       std::size_t n_right,
                                       std::size_t t_left,
                                       std::int32_t total_weight,
                                       std::int32_t& min_score_threshold,
                                       std::uint32_t& worst_weight) {
    std::copy_n(_buckets.begin(), n_right, _fwht_buffer.begin());
    fwht_inplace(_fwht_buffer.data(), n_right);

    for (std::size_t x_right = 0; x_right < n_right; ++x_right) {
        const std::int32_t score = _fwht_buffer[x_right];
        if (score <= min_score_threshold)
            continue;

        const std::int64_t w2 =
            static_cast<std::int64_t>(total_weight) - static_cast<std::int64_t>(score);

        if (w2 < 0 || (w2 & 1) != 0)
            continue;

        const std::uint64_t mask =
            (static_cast<std::uint64_t>(x_right) << t_left) | static_cast<std::uint64_t>(x_left);

        if (mask == 0)
            continue;

        add_result(mask,
                   static_cast<std::uint32_t>(w2 / 2),
                   total_weight,
                   min_score_threshold,
                   worst_weight);
    }
}

std::vector<Candidate> MitmFwhtSearch::search(const BitMatrix& matrix,
                                              const std::vector<std::size_t>& window_rows) {
    _candidates.clear();

    if (_config.k_limit == 0)
        return {};

    const std::size_t t = window_rows.size();
    if (t == 0)
        return {};

    if (t > Candidate::k_word_bits) {
        throw std::invalid_argument("MitmFwhtSearch: window_rows size must be <= 64");
    }

    const auto [t_left, t_right] = get_split_info(t);

    if (t_left > kMaxTLeft) {
        throw std::invalid_argument("MitmFwhtSearch: left half exceeds supported width");
    }

    if (t_left >= kMaxTHalf || t_right >= kMaxTHalf) {
        throw std::invalid_argument("MitmFwhtSearch: split dimensions must be < 31");
    }

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

    const auto [unique_cols, total_weight] =
        prepare_columns(rows, words_per_row, matrix.cols(), t_left);

    if (unique_cols == 0)
        return {};

    const std::size_t n_right = std::size_t{1} << t_right;
    initialize_buckets(unique_cols, n_right);

    std::int32_t min_score_threshold = std::numeric_limits<std::int32_t>::min();
    std::uint32_t worst_weight = std::numeric_limits<std::uint32_t>::max();

    process_candidate(0, n_right, t_left, total_weight, min_score_threshold, worst_weight);

    std::uint32_t current_x_left = 0;
    const std::size_t n_loop = std::size_t{1} << t_left;

    for (std::size_t i = 1; i < n_loop; ++i) {
        const std::size_t bit = static_cast<std::size_t>(detail::ctz64(i));
        apply_bit_flip(bit);
        current_x_left ^= (std::uint32_t{1} << bit);

        process_candidate(
            current_x_left, n_right, t_left, total_weight, min_score_threshold, worst_weight);
    }

    if (_candidates.size() < _config.k_limit) {
        std::sort(_candidates.begin(), _candidates.end(), candidate_less);
    }

    return _candidates;
}

} // namespace bmmpy
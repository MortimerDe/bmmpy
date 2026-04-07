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

} // namespace bmmpy
#include "bmmpy/apply/global_greedy_applier.hpp"

#include "bmmpy/core/detail/row_combination.hpp"
#include "bmmpy/core/detail/xor_basis.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bmmpy {
namespace {

struct PoolEntry {
    std::vector<std::uint64_t> mask_words;
    std::uint32_t weight = 0;
};

std::uint32_t checked_u32_weight(const std::uint64_t weight) {
    if (weight > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::invalid_argument(
            "GlobalGreedyApplier: row weight exceeds supported u32 candidate range");
    }

    return static_cast<std::uint32_t>(weight);
}

std::vector<PoolEntry> build_pool(const RowWindow& window,
                                  const std::vector<Candidate>& candidates) {
    std::vector<PoolEntry> pool;
    pool.reserve(candidates.size() + window.size());

    for (const Candidate& candidate : candidates) {
        pool.push_back(PoolEntry{
            detail::normalize_mask_words(candidate.mask, window.size()),
            candidate.weight,
        });
    }

    for (std::size_t row = 0; row < window.size(); ++row) {
        pool.push_back(PoolEntry{
            detail::make_unit_mask_words(window.size(), row),
            checked_u32_weight(window.row_popcount(row)),
        });
    }

    std::stable_sort(pool.begin(), pool.end(), [](const PoolEntry& lhs, const PoolEntry& rhs) {
        return lhs.weight < rhs.weight;
    });

    return pool;
}

} // namespace

ApplyResult GlobalGreedyApplier::apply(RowWindow& window,
                                       const std::vector<Candidate>& candidates) const {
    ApplyResult result;

    if (window.size() == 0)
        return result;

    const BitMatrix before = window.materialize();
    const std::uint64_t old_weight = before.weight();

    detail::PivotBasis transform_basis(window.size());
    std::vector<std::vector<std::uint64_t>> chosen_masks;
    chosen_masks.reserve(window.size());

    const std::vector<PoolEntry> pool = build_pool(window, candidates);

    for (const PoolEntry& entry : pool) {
        if (!transform_basis.try_insert(entry.mask_words))
            continue;

        chosen_masks.push_back(entry.mask_words);

        if (chosen_masks.size() == window.size())
            break;
    }

    if (chosen_masks.size() != window.size()) {
        throw std::runtime_error(
            "GlobalGreedyApplier: failed to assemble a full invertible transform");
    }

    const BitMatrix transform = detail::build_transform_matrix(window.size(), chosen_masks);
    const BitMatrix after = transform.mul(before);
    const std::uint64_t new_weight = after.weight();

    if (new_weight > old_weight)
        return result;

    if (_require_improvement && new_weight == old_weight)
        return result;

    window.assign_materialized(after);

    result.applied_count = 1;
    result.weight_improvement = old_weight - new_weight;
    return result;
}

} // namespace bmmpy
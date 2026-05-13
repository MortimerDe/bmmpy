#include "bmmpy/algebra/transform_utils.hpp"
#include "bmmpy/core/detail/row_combination.hpp"
#include "bmmpy/core/detail/xor_basis.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace bmmpy {
namespace {

std::vector<std::vector<std::uint64_t>>
normalize_candidate_masks(const std::size_t input_rows, const std::vector<Candidate>& candidates) {
    std::vector<std::vector<std::uint64_t>> mask_rows;
    mask_rows.reserve(candidates.size());

    for (const Candidate& candidate : candidates) {
        mask_rows.push_back(detail::normalize_mask_words(candidate.mask, input_rows));
    }

    return mask_rows;
}

BitMatrix materialize_candidate_rows(const RowWindow& window,
                                     const std::vector<std::vector<std::uint64_t>>& mask_rows) {
    BitMatrix basis_rows(mask_rows.size(), window.cols());
    std::vector<std::uint64_t> materialized(window.words_per_row(), std::uint64_t{0});

    for (std::size_t row = 0; row < mask_rows.size(); ++row) {
        detail::materialize_row_combination(window, mask_rows[row], materialized);
        std::copy(materialized.begin(), materialized.end(), basis_rows.row_words(row));
    }

    return basis_rows;
}

} // namespace

BitMatrix build_transform_matrix(const std::size_t input_rows,
                                 const std::vector<Candidate>& candidates) {
    const auto mask_rows = normalize_candidate_masks(input_rows, candidates);
    return detail::build_transform_matrix(input_rows, mask_rows);
}

BitMatrix build_transform_matrix_checked(const RowWindow& window,
                                         const std::vector<Candidate>& candidates) {
    const std::size_t target_rank = window.materialize().rank();

    if (candidates.size() != target_rank) {
        throw std::invalid_argument(
            "build_transform_matrix_checked: candidate count must equal window rank");
    }

    const auto mask_rows = normalize_candidate_masks(window.size(), candidates);
    const BitMatrix basis_rows = materialize_candidate_rows(window, mask_rows);

    if (basis_rows.rank() != target_rank) {
        throw std::invalid_argument(
            "build_transform_matrix_checked: candidates do not materialize to a full-rank basis");
    }

    return detail::build_transform_matrix(window.size(), mask_rows);
}

} // namespace bmmpy
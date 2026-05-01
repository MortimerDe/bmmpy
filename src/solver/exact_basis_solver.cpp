#include "bmmpy/solver/exact_basis_solver.hpp"

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

std::uint64_t state_count_for_rows(const std::size_t rows) {
    if (rows == 0)
        return 0;

    if (rows > 32) {
        throw std::invalid_argument(
            "ExactBasisSolver: this implementation supports at most 32 rows");
    }

    return (std::uint64_t{1} << rows) - 1;
}

template <typename Visitor> void enumerate_gray(const RowWindow& window, Visitor&& visit) {
    const std::size_t rows = window.size();
    const std::size_t word_count = window.words_per_row();
    const std::uint64_t states = state_count_for_rows(rows);

    if (states == 0)
        return;

    const auto& ops = detail::bit_ops();
    std::vector<std::uint64_t> current(word_count, std::uint64_t{0});
    std::uint32_t previous_gray = 0;

    for (std::uint64_t index = 1; index <= states; ++index) {
        const std::uint32_t gray = static_cast<std::uint32_t>(index ^ (index >> 1));
        const std::uint32_t diff = gray ^ previous_gray;
        const std::size_t toggled_row = static_cast<std::size_t>(detail::ctz64(diff));

        ops.row_xor(current.data(), window.row_words(toggled_row), word_count);

        const std::size_t weight =
            static_cast<std::size_t>(ops.row_popcount(current.data(), word_count));
        visit(gray, weight, current);

        previous_gray = gray;
    }
}

} // namespace

ExactBasisResult ExactBasisSolver::solve(const RowWindow& window) const {
    ExactBasisResult result;
    result.input_rows = window.size();
    result.cols = window.cols();

    const std::size_t rows = window.size();
    if (rows == 0) {
        result.transform_matrix = BitMatrix(0, 0);
        result.basis_matrix = BitMatrix(0, window.cols());
        return result;
    }

    if (rows > _config.max_rows) {
        throw std::invalid_argument(
            "ExactBasisSolver: window has more rows than the configured max_rows cap");
    }

    const std::uint64_t states = state_count_for_rows(rows);
    if (states > _config.max_states) {
        throw std::invalid_argument(
            "ExactBasisSolver: state count exceeds the configured max_states cap");
    }

    const std::uint64_t required_bytes = states * static_cast<std::uint64_t>(sizeof(std::uint32_t));
    if (required_bytes > _config.max_storage_bytes) {
        throw std::invalid_argument(
            "ExactBasisSolver: required packed-mask storage exceeds max_storage_bytes");
    }

    BitMatrix materialized = window.materialize();
    const std::size_t target_rank = materialized.rank();

    result.rank = target_rank;
    result.enumerated_states = states;

    if (target_rank == 0) {
        result.transform_matrix = BitMatrix(0, rows);
        result.basis_matrix = BitMatrix(0, window.cols());
        return result;
    }

    const std::size_t cols = window.cols();
    std::vector<std::uint32_t> counts(cols + 1, 0);

    enumerate_gray(
        window,
        [&](const std::uint32_t, const std::size_t weight, const std::vector<std::uint64_t>&) {
            if (weight > cols) {
                throw std::logic_error("ExactBasisSolver: encountered weight above column count");
            }
            ++counts[weight];
        });

    std::vector<std::size_t> offsets(cols + 2, 0);
    for (std::size_t weight = 0; weight <= cols; ++weight)
        offsets[weight + 1] = offsets[weight] + counts[weight];

    std::vector<std::uint32_t> packed_masks(offsets.back(), 0);
    std::vector<std::size_t> next_offsets = offsets;

    enumerate_gray(
        window,
        [&](const std::uint32_t mask, const std::size_t weight, const std::vector<std::uint64_t>&) {
            packed_masks[next_offsets[weight]++] = mask;
        });

    detail::PivotBasis basis(cols);
    std::vector<std::uint64_t> candidate_words(window.words_per_row(), std::uint64_t{0});

    result.basis_masks.reserve(target_rank);
    result.basis_weights.reserve(target_rank);

    for (std::size_t weight = 0; weight <= cols && result.basis_masks.size() < target_rank;
         ++weight) {
        for (std::size_t idx = offsets[weight]; idx < offsets[weight + 1]; ++idx) {
            const std::uint32_t mask = packed_masks[idx];
            detail::materialize_row_combination(window, mask, candidate_words);

            if (!basis.try_insert(candidate_words))
                continue;

            result.basis_masks.push_back(mask);
            result.basis_weights.push_back(static_cast<std::uint32_t>(weight));
            result.total_weight += weight;

            if (result.basis_masks.size() == target_rank)
                break;
        }
    }

    if (result.basis_masks.size() != target_rank) {
        throw std::runtime_error(
            "ExactBasisSolver: failed to assemble a full basis from enumerated candidates");
    }

    result.transform_matrix = detail::build_transform_matrix(rows, result.basis_masks);
    result.basis_matrix = result.transform_matrix.mul(materialized);

    return result;
}

} // namespace bmmpy
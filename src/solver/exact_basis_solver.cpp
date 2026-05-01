#include "bmmpy/solver/exact_basis_solver.hpp"

#include "bmmpy/core/detail/bit_intrinsics.hpp"
#include "bmmpy/core/detail/bit_ops.hpp"

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

int clz64(const std::uint64_t value) noexcept {
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

int highest_set_bit(const std::vector<std::uint64_t>& words) noexcept {
    for (std::size_t word_idx = words.size(); word_idx > 0; --word_idx) {
        const std::uint64_t word = words[word_idx - 1];
        if (word != 0) {
            return static_cast<int>((word_idx - 1) * 64 + static_cast<std::size_t>(clz64(word)));
        }
    }

    return -1;
}

void xor_words_inplace(std::vector<std::uint64_t>& dst, const std::vector<std::uint64_t>& src) {
    for (std::size_t i = 0; i < dst.size(); ++i)
        dst[i] ^= src[i];
}

void materialize_mask_words(const RowWindow& window,
                            const std::uint32_t mask,
                            std::vector<std::uint64_t>& out_words) {
    std::fill(out_words.begin(), out_words.end(), std::uint64_t{0});

    const auto& bit_ops = detail::bit_ops();
    std::uint32_t bits = mask;

    while (bits != 0) {
        const std::uint32_t low_bit = bits & (0u - bits);
        const std::size_t row_index = static_cast<std::size_t>(detail::ctz64(low_bit));
        bit_ops.row_xor(out_words.data(), window.row_words(row_index), out_words.size());
        bits ^= low_bit;
    }
}

template <typename Visitor> void enumerate_gray(const RowWindow& window, Visitor&& visit) {
    const std::size_t rows = window.size();
    const std::size_t word_count = window.words_per_row();
    const std::uint64_t states = state_count_for_rows(rows);

    if (states == 0)
        return;

    const auto& bit_ops = detail::bit_ops();
    std::vector<std::uint64_t> current(word_count, std::uint64_t{0});
    std::uint32_t previous_gray = 0;

    for (std::uint64_t index = 1; index <= states; ++index) {
        const std::uint32_t gray = static_cast<std::uint32_t>(index ^ (index >> 1));
        const std::uint32_t diff = gray ^ previous_gray;
        const std::size_t toggled_row = static_cast<std::size_t>(detail::ctz64(diff));

        bit_ops.row_xor(current.data(), window.row_words(toggled_row), word_count);

        const std::size_t weight =
            static_cast<std::size_t>(bit_ops.row_popcount(current.data(), word_count));
        visit(gray, weight, current);

        previous_gray = gray;
    }
}

BitMatrix build_transform_matrix(const std::size_t input_rows,
                                 const std::vector<std::uint32_t>& masks) {
    BitMatrix transform(masks.size(), input_rows);

    for (std::size_t row = 0; row < masks.size(); ++row) {
        std::uint32_t bits = masks[row];
        while (bits != 0) {
            const std::uint32_t low_bit = bits & (0u - bits);
            const std::size_t col = static_cast<std::size_t>(detail::ctz64(low_bit));
            transform.set(row, col, true);
            bits ^= low_bit;
        }
    }

    return transform;
}

class PivotBasis {
public:
    explicit PivotBasis(const std::size_t cols) : _pivot_rows(cols), _used(cols, false) {}

    bool try_insert(const std::vector<std::uint64_t>& candidate) {
        std::vector<std::uint64_t> reduced = candidate;

        while (true) {
            const int pivot = highest_set_bit(reduced);
            if (pivot < 0)
                return false;

            if (!_used[static_cast<std::size_t>(pivot)]) {
                _pivot_rows[static_cast<std::size_t>(pivot)] = std::move(reduced);
                _used[static_cast<std::size_t>(pivot)] = true;
                return true;
            }

            xor_words_inplace(reduced, _pivot_rows[static_cast<std::size_t>(pivot)]);
        }
    }

private:
    std::vector<std::vector<std::uint64_t>> _pivot_rows;
    std::vector<bool> _used;
};

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

    PivotBasis basis(cols);
    std::vector<std::uint64_t> candidate_words(window.words_per_row(), std::uint64_t{0});

    result.basis_masks.reserve(target_rank);
    result.basis_weights.reserve(target_rank);

    for (std::size_t weight = 0; weight <= cols && result.basis_masks.size() < target_rank;
         ++weight) {
        for (std::size_t idx = offsets[weight]; idx < offsets[weight + 1]; ++idx) {
            const std::uint32_t mask = packed_masks[idx];
            materialize_mask_words(window, mask, candidate_words);

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

    result.transform_matrix = build_transform_matrix(rows, result.basis_masks);
    result.basis_matrix = result.transform_matrix.mul(materialized);

    return result;
}

} // namespace bmmpy
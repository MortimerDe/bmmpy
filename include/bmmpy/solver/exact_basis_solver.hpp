#pragma once

#include "bmmpy/core/bit_matrix.hpp"
#include "bmmpy/core/row_window.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bmmpy {

struct ExactBasisSolverConfig {
    std::size_t max_rows = 24;
    std::uint64_t max_states = (std::uint64_t{1} << 24) - 1;
    std::size_t max_storage_bytes = 128u * 1024u * 1024u;
};

struct ExactBasisResult {
    std::size_t input_rows = 0;
    std::size_t cols = 0;
    std::size_t rank = 0;
    std::uint64_t enumerated_states = 0;
    std::uint64_t total_weight = 0;
    std::vector<std::uint32_t> basis_masks;
    std::vector<std::uint32_t> basis_weights;
    BitMatrix transform_matrix;
    BitMatrix basis_matrix;
};

class ExactBasisSolver final {
public:
    explicit ExactBasisSolver(ExactBasisSolverConfig config = {}) : _config(config) {}

    const char* name() const noexcept { return "exact_basis"; }

    ExactBasisResult solve(const RowWindow& window) const;

private:
    ExactBasisSolverConfig _config;
};

} // namespace bmmpy
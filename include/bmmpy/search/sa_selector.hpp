#pragma once

#include "bmmpy/core/bit_matrix.hpp"
#include "bmmpy/core/row_window.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bmmpy {

enum class WindowScorePolicyKind {
    PairwiseSynergy = 0,
    HigherOrderSynergy = 1,
};

enum class CoolingPolicyKind {
    AdaptiveGeometric = 0,
};

struct SASelectorConfig {
    std::size_t iterations = 10000;
    std::size_t restarts = 8;
    std::uint64_t seed = 0;

    WindowScorePolicyKind score_policy = WindowScorePolicyKind::PairwiseSynergy;
    CoolingPolicyKind cooling_policy = CoolingPolicyKind::AdaptiveGeometric;

    std::size_t temperature_probe_samples = 64;
    double initial_acceptance_probability = 0.8;
    double cooling_rate = 0.99;
    double min_temperature = 1e-6;
};

struct SASelectionResult {
    std::vector<std::size_t> rows;
    std::int64_t score = 0;
    std::size_t accepted_moves = 0;
    std::size_t iterations_run = 0;
    std::size_t best_iteration = 0;
    std::size_t restart_index = 0;
    std::uint64_t seed = 0;
};

class SASelector {
public:
    explicit SASelector(SASelectorConfig config = {}) : _config(config) {}

    SASelectionResult select(const BitMatrix& matrix, std::size_t window_size) const;

    RowWindow select_window(BitMatrix& matrix, std::size_t window_size) const;
    RowWindow select_window(const BitMatrix& matrix, std::size_t window_size) const;

    const char* name() const noexcept { return "sa"; }

private:
    SASelectorConfig _config;
};

} // namespace bmmpy
#include "bmmpy/search/sa_selector.hpp"

#include "bmmpy/core/detail/xorshift64.hpp"
#include "bmmpy/search/sa_selector_internal.hpp"

#include <stdexcept>

namespace bmmpy {

SASelectionResult SASelector::select(const BitMatrix& matrix, std::size_t window_size) const {
    if (_config.restarts == 0)
        throw std::invalid_argument("SASelector: restarts must be >= 1");

    if (!sa_detail::is_valid_probability(_config.initial_acceptance_probability)) {
        throw std::invalid_argument("SASelector: initial_acceptance_probability must be in (0, 1)");
    }

    if (_config.cooling_rate <= 0.0 || _config.cooling_rate > 1.0) {
        throw std::invalid_argument("SASelector: cooling_rate must be in (0, 1]");
    }

    if (_config.min_temperature <= 0.0) {
        throw std::invalid_argument("SASelector: min_temperature must be > 0");
    }

    if (window_size > matrix.rows()) {
        throw std::invalid_argument("SASelector: window_size exceeds matrix row count");
    }

    SASelectionResult out;
    out.seed = (_config.seed == 0 ? detail::k_default_xorshift64_seed : _config.seed);

    if (window_size == 0)
        return out;

    switch (_config.cooling_policy) {
    case CoolingPolicyKind::AdaptiveGeometric:
        break;
    default:
        throw std::invalid_argument("SASelector: unsupported cooling policy");
    }

    detail::XorShift64 rng(out.seed);
    bool has_best = false;

    auto consider_candidate = [&](const sa_detail::RestartResult& candidate, std::size_t restart) {
        if (!has_best ||
            sa_detail::is_better_result(candidate.score, candidate.rows, out.score, out.rows)) {
            out.rows = candidate.rows;
            out.score = candidate.score;
            out.accepted_moves = candidate.accepted_moves;
            out.iterations_run = candidate.iterations_run;
            out.best_iteration = candidate.best_iteration;
            out.restart_index = restart;
            has_best = true;
        }
    };

    for (std::size_t restart = 0; restart < _config.restarts; ++restart) {
        switch (_config.score_policy) {
        case WindowScorePolicyKind::PairwiseSynergy:
            consider_candidate(sa_detail::run_pairwise_restart(matrix, window_size, _config, rng),
                               restart);
            break;
        case WindowScorePolicyKind::HigherOrderSynergy:
            consider_candidate(
                sa_detail::run_higher_order_restart(matrix, window_size, _config, rng), restart);
            break;
        default:
            throw std::invalid_argument("SASelector: unsupported score policy");
        }
    }

    return out;
}

RowWindow SASelector::select_window(BitMatrix& matrix, std::size_t window_size) const {
    return matrix.row_window(select(matrix, window_size).rows);
}

RowWindow SASelector::select_window(const BitMatrix& matrix, std::size_t window_size) const {
    return matrix.row_window(select(matrix, window_size).rows);
}

} // namespace bmmpy
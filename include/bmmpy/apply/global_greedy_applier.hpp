#pragma once

#include "bmmpy/apply/apply_result.hpp"
#include "bmmpy/core/row_window.hpp"
#include "bmmpy/types/candidate.hpp"

#include <vector>

namespace bmmpy {

class GlobalGreedyApplier final {
public:
    explicit GlobalGreedyApplier(bool require_improvement = true) noexcept
        : _require_improvement(require_improvement) {}

    const char* name() const noexcept { return "global_greedy"; }

    ApplyResult apply(RowWindow& window, const std::vector<Candidate>& candidates) const;

private:
    bool _require_improvement = true;
};

} // namespace bmmpy
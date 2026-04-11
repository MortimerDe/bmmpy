#pragma once

#include "bmmpy/core/row_window.hpp"
#include "bmmpy/types/candidate.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace bmmpy {
class Searcher {
public:
    virtual ~Searcher() = default;

    virtual std::vector<Candidate> search(const RowWindow& window) = 0;
    virtual const char* name() const noexcept = 0;

private:
};

} // namespace bmmpy
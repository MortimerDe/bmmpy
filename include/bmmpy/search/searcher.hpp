#pragma once

#include "bmmpy/core/bit_matrix.hpp"
#include "bmmpy/types/candidate.hpp"

#include <cstddef>
#include <vector>

namespace bmmpy {
class Searcher {
public:
    virtual ~Searcher() = default;

    virtual std::vector<Candidate> search(const BitMatrix& matrix,
                                          const std::vector<std::size_t>& window_rows) = 0;

private:
};

} // namespace bmmpy
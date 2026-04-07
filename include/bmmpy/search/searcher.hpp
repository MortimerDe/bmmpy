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
    virtual const char* name() const noexcept = 0;

    virtual std::string describe(std::size_t window_size) const {
        return std::string(name()) + "" + std::to_string(window_size);
    }

private:
};

} // namespace bmmpy
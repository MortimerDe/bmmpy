#pragma once

#include "bmmpy/core/bit_matrix.hpp"
#include "bmmpy/types/candidate.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bmmpy {

struct FwhtSearchConfig {
    std::size_t max_rows = 16;
    std::size_t k = 64;
};

class FwhtSearch {
public:
    explicit FwhtSearch(FwhtSearchConfig config = {}) : _config(config) {}

    std::vector<Candidate> search(const BitMatrix& matrix,
                                  const std::vector<std::size_t>& window_rows);

private:
    FwhtSearchConfig _config;
    std::vector<std::int32_t> _buckets;
};

} // namespace bmmpy
#pragma once

#include "bmmpy/core/bit_matrix.hpp"
#include "bmmpy/search/searcher.hpp"
#include "bmmpy/types/candidate.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bmmpy {

struct FwhtSearchConfig {
    std::size_t max_rows = 16;
    std::size_t max_candidates = 64;
};

class FwhtSearch final : public Searcher {
public:
    explicit FwhtSearch(FwhtSearchConfig config = {}) : _config(config) {}

    std::vector<Candidate> search(const RowWindow& window) override;
    const char* name() const noexcept override { return "fwht"; }

private:
    FwhtSearchConfig _config;
    std::vector<std::int32_t> _buckets;
};

} // namespace bmmpy
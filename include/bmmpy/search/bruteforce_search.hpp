#pragma once

#include "bmmpy/search/searcher.hpp"

#include <cstddef>

namespace bmmpy {

struct BruteforceSearchConfig {
    std::size_t max_candidates = 128;
    std::size_t chunk_bits = 0; // 0 = auto
};

class BruteforceSearch final : public Searcher {
public:
    explicit BruteforceSearch(BruteforceSearchConfig config = {}) : _config(config) {}

    std::vector<Candidate> search(const RowWindow& window) override;
    const char* name() const noexcept override { return "bruteforce"; }

private:
    BruteforceSearchConfig _config;
};

} // namespace bmmpy
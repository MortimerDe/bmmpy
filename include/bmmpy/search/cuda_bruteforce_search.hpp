#pragma once

#include "bmmpy/search/searcher.hpp"

#include <cstddef>

namespace bmmpy {

struct CudaBruteforceSearchConfig {
    std::size_t max_candidates = 128;
    std::size_t chunk_bits = 0; // 0 = auto, resolved later by the CUDA backend
};

class CudaBruteforceSearch final : public Searcher {
public:
    explicit CudaBruteforceSearch(CudaBruteforceSearchConfig config = {}) : _config(config) {}

    std::vector<Candidate> search(const RowWindow& window) override;

    const char* name() const noexcept override { return "cuda_bruteforce"; }

private:
    static constexpr std::size_t k_min_rows = 16;
    static constexpr std::size_t k_max_rows = 64;

    CudaBruteforceSearchConfig _config;
};

} // namespace bmmpy
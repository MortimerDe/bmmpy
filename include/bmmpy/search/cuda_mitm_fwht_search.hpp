#pragma once

#include "bmmpy/search/searcher.hpp"

#include <cstddef>

namespace bmmpy {

struct CudaMitmFwhtSearchConfig {
    std::size_t max_candidates = 64;
    std::size_t low_bits = 0; // auto, resolved later by the CUDA backend
};

class CudaMitmFwhtSearch final : public Searcher {
public:
    explicit CudaMitmFwhtSearch(CudaMitmFwhtSearchConfig config = {}) : _config(config) {}

    std::vector<Candidate> search(const RowWindow& window) override;

    const char* name() const noexcept override { return "cuda_mitm_fwht"; }

private:
    static constexpr std::size_t k_min_rows = 16;
    static constexpr std::size_t k_max_rows = 64;

    CudaMitmFwhtSearchConfig _config;
};

} // namespace bmmpy
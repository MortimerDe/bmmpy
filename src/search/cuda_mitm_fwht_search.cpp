#include "bmmpy/search/cuda_mitm_fwht_search.hpp"

#include "bmmpy/stub.hpp"

#include <stdexcept>

namespace bmmpy {

std::vector<Candidate> CudaMitmFwhtSearch::search(const RowWindow& window) {
    if (_config.max_candidates == 0)
        return {};
    const std::size_t t = window.size();
    if (t == 0)
        return {};

    if (t > Candidate::k_word_bits)
        throw std::invalid_argument(
            "CudaMitmFwhtSearch: too many rows in the window (must be <= 64)");

    if (_config.low_bits != 0 && _config.low_bits >= t)
        throw std::invalid_argument(
            "CudaMitmFwhtSearch: low_bits must be in [1, t - 1] or 0 for auto");

    if (window.words_per_row() == 0)
        return {};

#if !defined(BMMPY_HAS_CUDA_BACKEND)
    throw std::runtime_error("CudaMitmFwhtSearch: CUDA backend is not compiled into this build");
#else
    const RuntimeFeatures features = get_runtime_features();
    if (!features.cuda_available)
        throw std::runtime_error("CudaMitmFwhtSearch: no CUDA device is available");

    throw std::runtime_error("CudaMitmFwhtSearch: CUDA search path is not implemented yet");
#endif
}

} // namespace bmmpy
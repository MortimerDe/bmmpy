#include "bmmpy/stub.hpp"

#include "bmmpy/core/detail/bit_intrinsics.hpp"

#if defined(BMMPY_HAS_OPENMP)
#include <omp.h>
#endif

namespace bmmpy {

std::string get_version() { return BMMPY_VERSION; }

RuntimeFeatures get_runtime_features() {
    RuntimeFeatures out;

#if defined(BMMPY_HAS_AVX2_BACKEND)
    out.avx2_compiled = true;
#endif

    out.avx2_available = detail::runtime_supports_avx2();

#if defined(BMMPY_HAS_OPENMP)
    out.parallel_compiled = true;
    out.max_threads = omp_get_max_threads();
    out.parallel_enabled = out.max_threads > 1;
#endif

    out.bit_ops_backend = out.avx2_available ? "avx2" : "scalar";
    out.fwht_backend = out.avx2_available ? "avx2" : "scalar";

    return out;
}

int add(int a, int b) { return a + b; }

} // namespace bmmpy
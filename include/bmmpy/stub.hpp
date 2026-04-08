#pragma once

#include <string>

namespace bmmpy {

struct RuntimeFeatures {
    bool avx2_compiled = false;
    bool avx2_available = false;
    bool parallel_compiled = false;
    bool parallel_enabled = false;
    int max_threads = 1;
    std::string bit_ops_backend = "scalar";
    std::string fwht_backend = "scalar";
};

std::string get_version();
RuntimeFeatures get_runtime_features();
int add(int a, int b);

} // namespace bmmpy
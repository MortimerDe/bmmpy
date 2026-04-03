#include "bmmpy/stub.hpp"

namespace bmmpy {

std::string get_version() {
    return BMMPY_VERSION;
}

int add(int a, int b) {
    return a + b;
}

} // namespace bmmpy

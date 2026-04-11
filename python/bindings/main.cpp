#include "bindings.hpp"

namespace pyb = bmmpy::bindings;

NB_MODULE(_bmmpy, m) {
    m.doc() = "Python bindings for bmmpy";

    pyb::bind_window(m);
    pyb::bind_bit_matrix(m);
    pyb::bind_candidate(m);
    pyb::bind_search(m);
    pyb::bind_apply(m);
    pyb::bind_runtime(m);
    pyb::bind_math(m);
}
#include "bindings.hpp"

namespace pyb = bmmpy::bindings;

NB_MODULE(_bmmpy, m) {
    m.doc() = R"(
        Low-level nanobind bindings for bmmpy.

        This module exposes the native core types and algorithms used by the public bmmpy package. Most users should import from bmmpy instead of bmmpy._bmmpy.
    )";

    pyb::bind_window(m);
    pyb::bind_bit_matrix(m);
    pyb::bind_algebra(m);
    pyb::bind_candidate(m);
    pyb::bind_search(m);
    pyb::bind_apply(m);
    pyb::bind_runtime(m);
    pyb::bind_math(m);
    pyb::bind_solver(m);
}
#pragma once

#include <nanobind/nanobind.h>

namespace bmmpy::bindings {

namespace nb = nanobind;

void bind_window(nb::module_& m);
void bind_bit_matrix(nb::module_& m);
void bind_candidate(nb::module_& m);
void bind_search(nb::module_& m);
void bind_apply(nb::module_& m);
void bind_runtime(nb::module_& m);
void bind_math(nb::module_& m);
void bind_solver(nb::module_& m);
void bind_algebra(nb::module_& m);

} // namespace bmmpy::bindings
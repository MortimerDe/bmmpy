#pragma once

#include <nanobind/nanobind.h>

namespace bmmpy::bindings {

void bind_bit_matrix(nanobind::module_& m);
void bind_candidate(nanobind::module_& m);
void bind_search(nanobind::module_& m);
void bind_apply(nanobind::module_& m);
void bind_runtime(nanobind::module_& m);
void bind_math(nanobind::module_& m);

}
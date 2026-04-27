#include "bindings.hpp"
#include "bmmpy/apply/greedy_applier.hpp"
#include "bmmpy/core/row_window.hpp"

#include <cstdint>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace bmmpy::bindings {

void bind_apply(nb::module_& m) {
    nb::class_<::bmmpy::ApplyResult>(m, "ApplyResult")
        .def(nb::init<>())
        .def_rw("applied_count", &::bmmpy::ApplyResult::applied_count)
        .def_rw("weight_improvement", &::bmmpy::ApplyResult::weight_improvement);

    nb::class_<::bmmpy::GreedyApplier>(m, "GreedyApplier")
        .def(nb::init<std::uint64_t, bool, std::uint64_t>(),
             nb::arg("min_gain"),
             nb::arg("stochastic") = false,
             nb::arg("seed") = 0)
        .def("apply", &::bmmpy::GreedyApplier::apply, nb::arg("window"), nb::arg("candidates"));
}

} // namespace bmmpy::bindings
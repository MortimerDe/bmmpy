#include "bindings.hpp"
#include "bmmpy/core/row_window.hpp"
#include "bmmpy/search/fwht_search.hpp"
#include "bmmpy/search/mitm_fwht_search.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace bmmpy::bindings {

void bind_search(nb::module_& m) {
    nb::class_<::bmmpy::FwhtSearchConfig>(m, "FwhtSearchConfig")
        .def(nb::init<>())
        .def_rw("max_rows", &::bmmpy::FwhtSearchConfig::max_rows)
        .def_rw("k", &::bmmpy::FwhtSearchConfig::k);

    nb::class_<::bmmpy::MitmFwhtSearchConfig>(m, "MitmFwhtSearchConfig")
        .def(nb::init<>())
        .def_rw("initial_capacity_cols", &::bmmpy::MitmFwhtSearchConfig::initial_capacity_cols)
        .def_rw("max_t_left", &::bmmpy::MitmFwhtSearchConfig::max_t_left)
        .def_rw("max_n_right", &::bmmpy::MitmFwhtSearchConfig::max_n_right)
        .def_rw("k", &::bmmpy::MitmFwhtSearchConfig::k);

    nb::class_<::bmmpy::FwhtSearch>(m, "FwhtSearch")
        .def(nb::init<::bmmpy::FwhtSearchConfig>(), nb::arg("config") = ::bmmpy::FwhtSearchConfig{})
        .def("name", &::bmmpy::FwhtSearch::name)
        .def("describe", &::bmmpy::FwhtSearch::describe, nb::arg("window_size"))
        .def("search", &::bmmpy::FwhtSearch::search, nb::arg("window"));

    nb::class_<::bmmpy::MitmFwhtSearch>(m, "MitmFwhtSearch")
        .def(nb::init<::bmmpy::MitmFwhtSearchConfig>(),
             nb::arg("config") = ::bmmpy::MitmFwhtSearchConfig{})
        .def("name", &::bmmpy::MitmFwhtSearch::name)
        .def("describe", &::bmmpy::MitmFwhtSearch::describe, nb::arg("window_size"))
        .def("search", &::bmmpy::MitmFwhtSearch::search, nb::arg("window"));
}

} // namespace bmmpy::bindings
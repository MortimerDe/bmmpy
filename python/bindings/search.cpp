#include "bindings.hpp"
#include "bmmpy/core/row_window.hpp"
#include "bmmpy/search/bruteforce_search.hpp"
#include "bmmpy/search/cuda_mitm_fwht_search.hpp"
#include "bmmpy/search/fwht_search.hpp"
#include "bmmpy/search/mitm_fwht_search.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace bmmpy::bindings {

void bind_search(nb::module_& m) {
    nb::class_<::bmmpy::BruteforceSearchConfig>(m, "BruteforceSearchConfig")
        .def(nb::init<>())
        .def_rw("max_candidates", &::bmmpy::BruteforceSearchConfig::max_candidates)
        .def_rw("chunk_bits", &::bmmpy::BruteforceSearchConfig::chunk_bits);

    nb::class_<::bmmpy::FwhtSearchConfig>(m, "FwhtSearchConfig")
        .def(nb::init<>())
        .def_rw("max_rows", &::bmmpy::FwhtSearchConfig::max_rows)
        .def_rw("max_candidates", &::bmmpy::FwhtSearchConfig::max_candidates);

    nb::class_<::bmmpy::MitmFwhtSearchConfig>(m, "MitmFwhtSearchConfig")
        .def(nb::init<>())
        .def_rw("reserve_unique_patterns", &::bmmpy::MitmFwhtSearchConfig::reserve_unique_patterns)
        .def_rw("reserve_left_rows", &::bmmpy::MitmFwhtSearchConfig::reserve_left_rows)
        .def_rw("reserve_right_states", &::bmmpy::MitmFwhtSearchConfig::reserve_right_states)
        .def_rw("max_candidates", &::bmmpy::MitmFwhtSearchConfig::max_candidates);

    nb::class_<::bmmpy::CudaMitmFwhtSearchConfig>(m, "CudaMitmFwhtSearchConfig")
        .def(nb::init<>())
        .def_rw("max_candidates", &::bmmpy::CudaMitmFwhtSearchConfig::max_candidates)
        .def_rw("low_bits", &::bmmpy::CudaMitmFwhtSearchConfig::low_bits);

    nb::class_<::bmmpy::BruteforceSearch>(m, "BruteforceSearch")
        .def(nb::init<::bmmpy::BruteforceSearchConfig>(),
             nb::arg("config") = ::bmmpy::BruteforceSearchConfig{})
        .def("name", &::bmmpy::BruteforceSearch::name)
        .def("search", &::bmmpy::BruteforceSearch::search, nb::arg("window"));

    nb::class_<::bmmpy::FwhtSearch>(m, "FwhtSearch")
        .def(nb::init<::bmmpy::FwhtSearchConfig>(), nb::arg("config") = ::bmmpy::FwhtSearchConfig{})
        .def("name", &::bmmpy::FwhtSearch::name)
        .def("search", &::bmmpy::FwhtSearch::search, nb::arg("window"));

    nb::class_<::bmmpy::MitmFwhtSearch>(m, "MitmFwhtSearch")
        .def(nb::init<::bmmpy::MitmFwhtSearchConfig>(),
             nb::arg("config") = ::bmmpy::MitmFwhtSearchConfig{})
        .def("name", &::bmmpy::MitmFwhtSearch::name)
        .def("search", &::bmmpy::MitmFwhtSearch::search, nb::arg("window"));

    nb::class_<::bmmpy::CudaMitmFwhtSearch>(m, "CudaMitmFwhtSearch")
        .def(nb::init<::bmmpy::CudaMitmFwhtSearchConfig>(),
             nb::arg("config") = ::bmmpy::CudaMitmFwhtSearchConfig{})
        .def("name", &::bmmpy::CudaMitmFwhtSearch::name)
        .def("search", &::bmmpy::CudaMitmFwhtSearch::search, nb::arg("window"));
}

} // namespace bmmpy::bindings
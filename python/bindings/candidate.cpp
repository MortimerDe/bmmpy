#include "bmmpy/types/candidate.hpp"

#include "bindings.hpp"

#include <cstdint>
#include <nanobind/make_iterator.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace {

std::vector<std::size_t> candidate_selected_rows(const bmmpy::Candidate& candidate) {
    std::vector<std::size_t> result;
    result.reserve(candidate.mask_popcount());

    for (std::size_t row : candidate.selected_rows()) {
        result.push_back(row);
    }

    return result;
}

std::string candidate_repr(const bmmpy::Candidate& candidate) {
    return "Candidate(mask_words=" + std::to_string(candidate.mask.size()) +
           ", weight=" + std::to_string(candidate.weight) + ")";
}

} // namespace

namespace bmmpy::bindings {

void bind_candidate(nb::module_& m) {
    nb::class_<::bmmpy::Candidate>(m, "Candidate")
        .def(nb::init<>())
        .def(nb::init<::bmmpy::Candidate::mask_type, std::uint32_t>(),
             nb::arg("mask"),
             nb::arg("weight"))
        .def("__len__", &::bmmpy::Candidate::mask_popcount)
        .def_rw("mask", &::bmmpy::Candidate::mask)
        .def_rw("weight", &::bmmpy::Candidate::weight)
        .def("__repr__", &candidate_repr)
        .def("__contains__", &::bmmpy::Candidate::has_row, nb::arg("row"))
        .def(
            "__iter__",
            [](const ::bmmpy::Candidate& candidate) {
                auto rows = candidate.selected_rows();
                return nb::make_iterator(nb::type<::bmmpy::Candidate>(),
                                         "selected_row_iterator",
                                         rows.begin(),
                                         rows.end());
            },
            nb::keep_alive<0, 1>())
        .def("has_row", &::bmmpy::Candidate::has_row, nb::arg("row"))
        .def_prop_ro("rows", &candidate_selected_rows)
        .def("mask_popcount", &::bmmpy::Candidate::mask_popcount)
        .def("mask_u64", &::bmmpy::Candidate::mask_u64)
        .def("selected_rows", &candidate_selected_rows)
        .def_static("from_u64", &::bmmpy::Candidate::from_u64, nb::arg("mask"), nb::arg("weight"))
        .def_static(
            "from_words",
            [](const std::vector<std::uint64_t>& mask, std::uint32_t weight) {
                return ::bmmpy::Candidate(mask, weight);
            },
            nb::arg("mask"),
            nb::arg("weight"),
            nb::rv_policy::move);
}

} // namespace bmmpy::bindings
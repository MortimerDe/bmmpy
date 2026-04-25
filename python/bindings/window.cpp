#include "bindings.hpp"
#include "bmmpy/core/bit_matrix.hpp"
#include "bmmpy/core/row_window.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <string>

namespace nb = nanobind;

namespace {

std::string row_window_repr(const bmmpy::RowWindow& window) {
    return "RowWindow(size=" + std::to_string(window.size()) +
           ", cols=" + std::to_string(window.cols()) + ")";
}

} // namespace

namespace bmmpy::bindings {

void bind_window(nb::module_& m) {
    nb::class_<::bmmpy::RowWindow>(m, "RowWindow")
        .def("__len__", &::bmmpy::RowWindow::size)
        .def("__repr__", &row_window_repr)
        .def_prop_ro("size", &::bmmpy::RowWindow::size)
        .def_prop_ro("cols", &::bmmpy::RowWindow::cols)
        .def_prop_ro("shape",
                     [](const ::bmmpy::RowWindow& window) {
                         return nb::make_tuple(window.size(), window.cols());
                     })
        .def_prop_ro("rows", &::bmmpy::RowWindow::global_rows)
        .def("row_popcount", &::bmmpy::RowWindow::row_popcount, nb::arg("local_row"))
        .def_prop_ro("total_weight", &::bmmpy::RowWindow::total_weight)
        .def("materialize", &::bmmpy::RowWindow::materialize, nb::rv_policy::move);
}

} // namespace bmmpy::bindings
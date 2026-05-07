#include "bindings.hpp"
#include "bmmpy/algebra/mastrovito.hpp"
#include "bmmpy/algebra/transforms.hpp"
#include "bmmpy/core/bit_vector.hpp"

#include <cstdint>
#include <nanobind/stl/vector.h>
#include <vector>

namespace nb = nanobind;

namespace {
std::vector<std::uint64_t> bitvector_to_words(const ::bmmpy::BitVector& bits) {
    return bits.words();
}
} // namespace

namespace bmmpy::bindings {

void bind_algebra(nb::module_& m) {
    nb::exception<::bmmpy::algebra::RowTransformError>(m, "RowTransformError", PyExc_ValueError);

    nb::class_<::bmmpy::BitVector>(m, "BitVector")
        .def(nb::init<>())
        .def(nb::init<std::size_t>(), nb::arg("bit_count"))
        .def_prop_ro("bit_count", &::bmmpy::BitVector::bit_count)
        .def_prop_ro("word_count", &::bmmpy::BitVector::word_count)
        .def("__len__", &::bmmpy::BitVector::bit_count)
        .def("any", &::bmmpy::BitVector::any)
        .def("none", &::bmmpy::BitVector::none)
        .def("get", &::bmmpy::BitVector::get, nb::arg("bit_index"))
        .def("set", &::bmmpy::BitVector::set, nb::arg("bit_index"), nb::arg("value") = true)
        .def("flip", &::bmmpy::BitVector::flip, nb::arg("bit_index"))
        .def("to_words", &bitvector_to_words)
        .def_static("from_u64",
                    &::bmmpy::BitVector::from_u64,
                    nb::arg("value"),
                    nb::arg("bit_count") = 0,
                    nb::rv_policy::move)
        .def_static(
            "from_words",
            [](std::size_t bit_count, const std::vector<std::uint64_t>& words) {
                return ::bmmpy::BitVector::from_words(bit_count, words.data(), words.size());
            },
            nb::arg("bit_count"),
            nb::arg("words"),
            nb::rv_policy::move)
        .def_static("from_positions",
                    &::bmmpy::BitVector::from_positions,
                    nb::arg("bit_count"),
                    nb::arg("positions"),
                    nb::rv_policy::move);

    m.def("find_row_transform",
          &::bmmpy::algebra::find_row_transform,
          nb::arg("source"),
          nb::arg("target"),
          nb::rv_policy::move);

    m.def(
        "invert_matrix", &::bmmpy::algebra::invert_matrix, nb::arg("matrix"), nb::rv_policy::move);

    m.def("powers_to_field_element",
          &::bmmpy::algebra::powers_to_field_element,
          nb::arg("powers"),
          nb::arg("degree"),
          nb::arg("poly_powers"),
          nb::rv_policy::move);

    m.def("build_basis_change_matrix",
          &::bmmpy::algebra::build_basis_change_matrix,
          nb::arg("basis_elements"),
          nb::arg("degree"),
          nb::rv_policy::move);

    m.def("basis_mask_to_field_element",
          &::bmmpy::algebra::basis_mask_to_field_element,
          nb::arg("mask"),
          nb::arg("basis_elements"),
          nb::arg("degree"),
          nb::rv_policy::move);

    m.def("field_element_to_basis_mask",
          &::bmmpy::algebra::field_element_to_basis_mask,
          nb::arg("element_bits"),
          nb::arg("change_inv"),
          nb::rv_policy::move);

    nb::class_<::bmmpy::algebra::MastrovitoCore>(m, "MastrovitoCore")
        .def(nb::init<std::size_t,
                      std::vector<std::size_t>,
                      ::bmmpy::BitVector,
                      std::vector<::bmmpy::BitVector>>(),
             nb::arg("degree"),
             nb::arg("powers"),
             nb::arg("element_standard_bits"),
             nb::arg("basis_elements") = std::vector<::bmmpy::BitVector>{})
        .def_prop_ro("degree", &::bmmpy::algebra::MastrovitoCore::degree)
        .def_prop_ro("powers", &::bmmpy::algebra::MastrovitoCore::powers)
        .def("get_basis_multiplication_matrices",
             &::bmmpy::algebra::MastrovitoCore::get_basis_multiplication_matrices,
             nb::rv_policy::move)
        .def("get_mastrovito_matrix",
             &::bmmpy::algebra::MastrovitoCore::get_mastrovito_matrix,
             nb::arg("power_bits"),
             nb::rv_policy::move)
        .def(
            "build_check_matrix",
            [](::bmmpy::algebra::MastrovitoCore& core,
               std::size_t c,
               std::size_t k,
               const ::bmmpy::BitVector& start_bits) {
                return core.build_check_matrix(c, k, start_bits);
            },
            nb::arg("c"),
            nb::arg("k"),
            nb::arg("start_bits"),
            nb::rv_policy::move);
}

} // namespace bmmpy::bindings
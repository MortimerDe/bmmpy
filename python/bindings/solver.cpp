#include "bindings.hpp"
#include "bmmpy/solver/exact_basis_solver.hpp"

#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace bmmpy::bindings {

void bind_solver(nb::module_& m) {
    nb::class_<::bmmpy::ExactBasisSolverConfig>(m, "ExactBasisSolverConfig")
        .def(nb::init<>())
        .def_rw("max_rows", &::bmmpy::ExactBasisSolverConfig::max_rows)
        .def_rw("max_states", &::bmmpy::ExactBasisSolverConfig::max_states)
        .def_rw("max_storage_bytes", &::bmmpy::ExactBasisSolverConfig::max_storage_bytes);

    nb::class_<::bmmpy::ExactBasisResult>(m, "ExactBasisResult")
        .def(nb::init<>())
        .def_rw("input_rows", &::bmmpy::ExactBasisResult::input_rows)
        .def_rw("cols", &::bmmpy::ExactBasisResult::cols)
        .def_rw("rank", &::bmmpy::ExactBasisResult::rank)
        .def_rw("enumerated_states", &::bmmpy::ExactBasisResult::enumerated_states)
        .def_rw("total_weight", &::bmmpy::ExactBasisResult::total_weight)
        .def_rw("basis_masks", &::bmmpy::ExactBasisResult::basis_masks)
        .def_rw("basis_weights", &::bmmpy::ExactBasisResult::basis_weights)
        .def_rw("transform_matrix", &::bmmpy::ExactBasisResult::transform_matrix)
        .def_rw("basis_matrix", &::bmmpy::ExactBasisResult::basis_matrix);

    nb::class_<::bmmpy::ExactBasisSolver>(m, "ExactBasisSolver")
        .def(nb::init<::bmmpy::ExactBasisSolverConfig>(),
             nb::arg("config") = ::bmmpy::ExactBasisSolverConfig{})
        .def("name", &::bmmpy::ExactBasisSolver::name)
        .def("solve", &::bmmpy::ExactBasisSolver::solve, nb::arg("window"));
}

} // namespace bmmpy::bindings
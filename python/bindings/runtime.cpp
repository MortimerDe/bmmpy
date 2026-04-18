#include "bindings.hpp"
#include "bmmpy/stub.hpp"

#include <nanobind/stl/string.h>

namespace nb = nanobind;

namespace bmmpy::bindings {

void bind_runtime(nb::module_& m) {
    nb::class_<::bmmpy::RuntimeFeatures>(m, "RuntimeFeatures")
        .def(nb::init<>())
        .def_rw("avx2_compiled", &::bmmpy::RuntimeFeatures::avx2_compiled)
        .def_rw("avx2_available", &::bmmpy::RuntimeFeatures::avx2_available)
        .def_rw("cuda_compiled", &::bmmpy::RuntimeFeatures::cuda_compiled)
        .def_rw("cuda_available", &::bmmpy::RuntimeFeatures::cuda_available)
        .def_rw("parallel_compiled", &::bmmpy::RuntimeFeatures::parallel_compiled)
        .def_rw("parallel_enabled", &::bmmpy::RuntimeFeatures::parallel_enabled)
        .def_rw("max_threads", &::bmmpy::RuntimeFeatures::max_threads)
        .def_rw("bit_ops_backend", &::bmmpy::RuntimeFeatures::bit_ops_backend)
        .def_rw("fwht_backend", &::bmmpy::RuntimeFeatures::fwht_backend)
        .def("__repr__", [](const ::bmmpy::RuntimeFeatures& value) {
            return "RuntimeFeatures("
                   "avx2_compiled=" +
                   std::string(value.avx2_compiled ? "True" : "False") +
                   ", avx2_available=" + std::string(value.avx2_available ? "True" : "False") +
                   ", cuda_compiled=" + std::string(value.cuda_compiled ? "True" : "False") +
                   ", cuda_available=" + std::string(value.cuda_available ? "True" : "False") +
                   ", parallel_compiled=" +
                   std::string(value.parallel_compiled ? "True" : "False") +
                   ", parallel_enabled=" + std::string(value.parallel_enabled ? "True" : "False") +
                   ", max_threads=" + std::to_string(value.max_threads) + ", bit_ops_backend='" +
                   value.bit_ops_backend + "', fwht_backend='" + value.fwht_backend + "')";
        });

    m.def("get_version", &::bmmpy::get_version, "Get the version of the library");
    m.def("add", &::bmmpy::add, nb::arg("a"), nb::arg("b"), "Add two integers");
    m.def("get_runtime_features",
          &::bmmpy::get_runtime_features,
          "Return runtime feature flags for this build and host");
}

} // namespace bmmpy::bindings
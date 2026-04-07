#include "bmmpy/math/comb.hpp"
#include "bmmpy/math/fwht.hpp"
#include "bmmpy/stub.hpp"

#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <vector>

namespace nb = nanobind;

NB_MODULE(_bmmpy, m) {
    m.doc() = "Python bindings for bmmpy";

    m.def("get_version", &bmmpy::get_version, "Get the version of the library");
    m.def("add", &bmmpy::add, nb::arg("a"), nb::arg("b"), "Add two integers");

    m.def(
        "fixed_weight_masks_u32",
        [](std::uint32_t n, std::uint32_t k) {
            std::vector<std::uint32_t> out;
            bmmpy::fixed_weight_masks_u32(n, k, out);
            return out;
        },
        nb::arg("n"),
        nb::arg("k"),
        nb::rv_policy::move);

    m.def(
        "fixed_weight_masks_u64",
        [](std::uint32_t n, std::uint32_t k) {
            std::vector<std::uint64_t> out;
            bmmpy::fixed_weight_masks_u64(n, k, out);
            return out;
        },
        nb::arg("n"),
        nb::arg("k"),
        nb::rv_policy::move);

    m.def(
        "fwht_i16",
        [](std::vector<std::int16_t> data) {
            bmmpy::fwht_inplace(data.data(), data.size());
            return data;
        },
        nb::arg("data"),
        nb::rv_policy::move);

    m.def(
        "fwht_i32",
        [](std::vector<std::int32_t> data) {
            bmmpy::fwht_inplace(data.data(), data.size());
            return data;
        },
        nb::arg("data"),
        nb::rv_policy::move);
}
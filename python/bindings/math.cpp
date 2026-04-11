#include "bindings.hpp"
#include "bmmpy/math/comb.hpp"
#include "bmmpy/math/fwht.hpp"

#include <cstdint>
#include <nanobind/stl/vector.h>
#include <vector>

namespace nb = nanobind;

namespace bmmpy::bindings {

void bind_math(nb::module_& m) {
    m.def(
        "fixed_weight_masks_u32",
        [](std::uint32_t n, std::uint32_t k) {
            std::vector<std::uint32_t> out;
            ::bmmpy::fixed_weight_masks_u32(n, k, out);
            return out;
        },
        nb::arg("n"),
        nb::arg("k"),
        nb::rv_policy::move);

    m.def(
        "fixed_weight_masks_u64",
        [](std::uint32_t n, std::uint32_t k) {
            std::vector<std::uint64_t> out;
            ::bmmpy::fixed_weight_masks_u64(n, k, out);
            return out;
        },
        nb::arg("n"),
        nb::arg("k"),
        nb::rv_policy::move);

    m.def(
        "fwht_i16",
        [](std::vector<std::int16_t> data) {
            ::bmmpy::fwht_inplace(data.data(), data.size());
            return data;
        },
        nb::arg("data"),
        nb::rv_policy::move);

    m.def(
        "fwht_i32",
        [](std::vector<std::int32_t> data) {
            ::bmmpy::fwht_inplace(data.data(), data.size());
            return data;
        },
        nb::arg("data"),
        nb::rv_policy::move);
}

} // namespace bmmpy::bindings
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "bmmpy/stub.hpp"

namespace nb = nanobind;

NB_MODULE(bmmpy, m)
{
    m.def("get_version", &bmmpy::get_version, "Get the version of the library");
    m.def("add", &bmmpy::add, "Add two integers");
}
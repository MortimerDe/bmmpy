#include "bmmpy/core/bit_matrix.hpp"
#include "bmmpy/math/comb.hpp"
#include "bmmpy/math/fwht.hpp"
#include "bmmpy/stub.hpp"

#include <cstdint>
#include <filesystem>
#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace {

bmmpy::BitMatrix matrix_from_rows(const std::vector<std::string>& rows) {
    const std::size_t row_count = rows.size();
    const std::size_t col_count = row_count == 0 ? 0 : rows.front().size();

    bmmpy::BitMatrix matrix(row_count, col_count);

    for (std::size_t row = 0; row < row_count; ++row) {
        if (rows[row].size() != col_count) {
            throw std::invalid_argument("BitMatrix.from_rows: all rows must have equal width");
        }

        for (std::size_t col = 0; col < col_count; ++col) {
            const char ch = rows[row][col];
            if (ch == '1') {
                matrix.set(row, col, true);
            } else if (ch != '0') {
                throw std::invalid_argument(
                    "BitMatrix.from_rows: rows must contain only '0' and '1'");
            }
        }
    }

    return matrix;
}

std::vector<std::string> matrix_to_rows(const bmmpy::BitMatrix& matrix) {
    std::vector<std::string> rows(matrix.rows(), std::string(matrix.cols(), '0'));

    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t col = 0; col < matrix.cols(); ++col) {
            if (matrix.get(row, col)) {
                rows[row][col] = '1';
            }
        }
    }

    return rows;
}

std::string matrix_to_text(const bmmpy::BitMatrix& matrix) {
    std::ostringstream out;
    matrix.save_text(out);
    return out.str();
}

bmmpy::BitMatrix matrix_from_text(const std::string& text) {
    std::istringstream in(text);
    return bmmpy::BitMatrix::load_text(in);
}

std::string matrix_repr(const bmmpy::BitMatrix& matrix) {
    return "BitMatrix(rows=" + std::to_string(matrix.rows()) +
           ", cols=" + std::to_string(matrix.cols()) +
           ", weight=" + std::to_string(matrix.weight()) + ")";
}

} // namespace

NB_MODULE(_bmmpy, m) {
    m.doc() = "Python bindings for bmmpy";

    m.def("get_version", &bmmpy::get_version, "Get the version of the library");

    nb::exception<bmmpy::MatrixError>(m, "MatrixError", PyExc_RuntimeError);

    nb::enum_<bmmpy::MatrixErr>(m, "MatrixErr")
        .value("SizeTooLarge", bmmpy::MatrixErr::SizeTooLarge)
        .value("AllocationFailed", bmmpy::MatrixErr::AllocationFailed)
        .value("DimensionMismatch", bmmpy::MatrixErr::DimensionMismatch);

    nb::class_<bmmpy::BitMatrix>(m, "BitMatrix")
        .def(nb::init<>())
        .def(nb::init<std::size_t, std::size_t>(), nb::arg("rows"), nb::arg("cols"))
        .def_prop_ro("rows", &bmmpy::BitMatrix::rows)
        .def_prop_ro("cols", &bmmpy::BitMatrix::cols)
        .def_prop_ro("shape",
                     [](const bmmpy::BitMatrix& matrix) {
                         return nb::make_tuple(matrix.rows(), matrix.cols());
                     })
        .def_prop_ro("stride_words", &bmmpy::BitMatrix::stride_words)
        .def_prop_ro("words_per_row", &bmmpy::BitMatrix::words_per_row)
        .def_prop_ro("total_words", &bmmpy::BitMatrix::total_words)
        .def_prop_ro("total_bytes", &bmmpy::BitMatrix::total_bytes)
        .def("__len__", &bmmpy::BitMatrix::rows)
        .def("__repr__", &matrix_repr)
        .def("get", &bmmpy::BitMatrix::get, nb::arg("row"), nb::arg("col"))
        .def("set", &bmmpy::BitMatrix::set, nb::arg("row"), nb::arg("col"), nb::arg("value"))
        .def(
            "copy_from_words",
            [](bmmpy::BitMatrix& matrix, const std::vector<std::uint64_t>& words) {
                matrix.copy_from_words(words.data(), words.size());
            },
            nb::arg("words"))
        .def("row_xor", &bmmpy::BitMatrix::row_xor, nb::arg("target_row"), nb::arg("source_row"))
        .def("row_xor_from",
             &bmmpy::BitMatrix::row_xor_from,
             nb::arg("target_row"),
             nb::arg("source"),
             nb::arg("source_row"))
        .def("add", &bmmpy::BitMatrix::add, nb::arg("other"), nb::rv_policy::move)
        .def("mul", &bmmpy::BitMatrix::mul, nb::arg("other"), nb::rv_policy::move)
        .def("__matmul__", &bmmpy::BitMatrix::mul, nb::arg("other"), nb::rv_policy::move)
        .def("power", &bmmpy::BitMatrix::power, nb::arg("exp"), nb::rv_policy::move)
        .def("row_popcount", &bmmpy::BitMatrix::row_popcount, nb::arg("row"))
        .def("weight", &bmmpy::BitMatrix::weight)
        .def("swap_rows", &bmmpy::BitMatrix::swap_rows, nb::arg("r1"), nb::arg("r2"))
        .def("rank", &bmmpy::BitMatrix::rank)
        .def("extract_rows_by_indices",
             &bmmpy::BitMatrix::extract_rows_by_indices,
             nb::arg("indices"),
             nb::rv_policy::move)
        .def("insert_rows_by_indices",
             &bmmpy::BitMatrix::insert_rows_by_indices,
             nb::arg("source"),
             nb::arg("indices"))
        .def("to_rows", &matrix_to_rows)
        .def("to_text", &matrix_to_text)
        .def(
            "save_text_file",
            [](const bmmpy::BitMatrix& matrix, const std::filesystem::path& path) {
                matrix.save_text(path);
            },
            nb::arg("path"))
        .def(
            "save_binary_file",
            [](const bmmpy::BitMatrix& matrix, const std::filesystem::path& path) {
                matrix.save_binary(path);
            },
            nb::arg("path"))
        .def_static("from_rows", &matrix_from_rows, nb::arg("rows"), nb::rv_policy::move)
        .def_static("from_text", &matrix_from_text, nb::arg("text"), nb::rv_policy::move)
        .def_static(
            "load_text_file",
            [](const std::filesystem::path& path) { return bmmpy::BitMatrix::load_text(path); },
            nb::arg("path"),
            nb::rv_policy::move)
        .def_static(
            "load_binary_file",
            [](const std::filesystem::path& path) { return bmmpy::BitMatrix::load_binary(path); },
            nb::arg("path"),
            nb::rv_policy::move)
        .def_static("identity", &bmmpy::BitMatrix::identity, nb::arg("n"), nb::rv_policy::move);

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
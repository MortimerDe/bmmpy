#include "bmmpy/apply/greedy_selection.hpp"
#include "bmmpy/core/bit_matrix.hpp"
#include "bmmpy/math/comb.hpp"
#include "bmmpy/math/fwht.hpp"
#include "bmmpy/search/fwht_search.hpp"
#include "bmmpy/search/mitm_fwht_search.hpp"
#include "bmmpy/stub.hpp"
#include "bmmpy/types/candidate.hpp"

#include <cstdint>
#include <filesystem>
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
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

std::vector<std::size_t> candidate_selected_rows(const bmmpy::Candidate& candidate) {
    std::vector<std::size_t> result;
    result.reserve(candidate.mask_popcount());

    for (std::size_t row : candidate.selected_rows()) {
        result.push_back(row);
    }

    return result;
}

std::string matrix_to_text(const bmmpy::BitMatrix& matrix) {
    std::ostringstream out;
    matrix.save_text(out);
    return out.str();
}

std::pair<std::size_t, std::size_t> matrix_index(const nb::tuple& index) {
    if (index.size() != 2) {
        throw std::invalid_argument("BitMatrix indices must be a (row, col) tuple");
    }
    return {nb::cast<std::size_t>(index[0]), nb::cast<std::size_t>(index[1])};
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

std::string candidate_repr(const bmmpy::Candidate& candidate) {
    return "Candidate(mask_words=" + std::to_string(candidate.mask.size()) +
           ", weight=" + std::to_string(candidate.weight) + ")";
}

} // namespace

NB_MODULE(_bmmpy, m) {
    m.doc() = "Python bindings for bmmpy";

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
        .def("__str__", &matrix_to_text)
        .def("__repr__", &matrix_repr)
        .def(
            "copy",
            [](const bmmpy::BitMatrix& matrix) { return bmmpy::BitMatrix(matrix); },
            nb::rv_policy::move)
        .def(
            "__getitem__",
            [](const bmmpy::BitMatrix& matrix, const nb::tuple& index) {
                const auto [row, col] = matrix_index(index);
                return matrix.get(row, col);
            },
            nb::arg("index"))
        .def(
            "__setitem__",
            [](bmmpy::BitMatrix& matrix, const nb::tuple& index, bool value) {
                const auto [row, col] = matrix_index(index);
                matrix.set(row, col, value);
            },
            nb::arg("index"),
            nb::arg("value"))
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

    nb::class_<bmmpy::ApplyResult>(m, "ApplyResult")
        .def(nb::init<>())
        .def_rw("applied_count", &bmmpy::ApplyResult::applied_count)
        .def_rw("weight_improvement", &bmmpy::ApplyResult::weight_improvement);

    nb::class_<bmmpy::Candidate>(m, "Candidate")
        .def(nb::init<>())
        .def(nb::init<bmmpy::Candidate::mask_type, std::uint32_t>(),
             nb::arg("mask"),
             nb::arg("weight"))
        .def("__len__", &bmmpy::Candidate::mask_popcount)
        .def_rw("mask", &bmmpy::Candidate::mask)
        .def_rw("weight", &bmmpy::Candidate::weight)
        .def("__repr__", &candidate_repr)
        .def("__contains__", &bmmpy::Candidate::has_row, nb::arg("row"))
        .def(
            "__iter__",
            [](const bmmpy::Candidate& candidate) {
                auto rows = candidate.selected_rows();
                return nb::make_iterator(nb::type<bmmpy::Candidate>(),
                                         "selected_row_iterator",
                                         rows.begin(),
                                         rows.end());
            },
            nb::keep_alive<0, 1>())
        .def("has_row", &bmmpy::Candidate::has_row, nb::arg("row"))
        .def_prop_ro("rows", &candidate_selected_rows)
        .def("mask_popcount", &bmmpy::Candidate::mask_popcount)
        .def("mask_u64", &bmmpy::Candidate::mask_u64)
        .def("selected_rows", &candidate_selected_rows)
        .def_static("from_u64", &bmmpy::Candidate::from_u64, nb::arg("mask"), nb::arg("weight"))
        .def_static(
            "from_words",
            [](const std::vector<std::uint64_t>& mask, std::uint32_t weight) {
                return bmmpy::Candidate(mask, weight);
            },
            nb::arg("mask"),
            nb::arg("weight"),
            nb::rv_policy::move);

    nb::class_<bmmpy::FwhtSearchConfig>(m, "FwhtSearchConfig")
        .def(nb::init<>())
        .def_rw("max_rows", &bmmpy::FwhtSearchConfig::max_rows)
        .def_rw("k", &bmmpy::FwhtSearchConfig::k);

    nb::class_<bmmpy::MitmFwhtSearchConfig>(m, "MitmFwhtSearchConfig")
        .def(nb::init<>())
        .def_rw("initial_capacity_cols", &bmmpy::MitmFwhtSearchConfig::initial_capacity_cols)
        .def_rw("max_t_left", &bmmpy::MitmFwhtSearchConfig::max_t_left)
        .def_rw("max_n_right", &bmmpy::MitmFwhtSearchConfig::max_n_right)
        .def_rw("k_limit", &bmmpy::MitmFwhtSearchConfig::k_limit);

    nb::class_<bmmpy::FwhtSearch>(m, "FwhtSearch")
        .def(nb::init<bmmpy::FwhtSearchConfig>(), nb::arg("config") = bmmpy::FwhtSearchConfig{})
        .def("name", &bmmpy::FwhtSearch::name)
        .def("describe", &bmmpy::FwhtSearch::describe, nb::arg("window_size"))
        .def("search", &bmmpy::FwhtSearch::search, nb::arg("matrix"), nb::arg("window_rows"));

    nb::class_<bmmpy::MitmFwhtSearch>(m, "MitmFwhtSearch")
        .def(nb::init<bmmpy::MitmFwhtSearchConfig>(),
             nb::arg("config") = bmmpy::MitmFwhtSearchConfig{})
        .def("name", &bmmpy::MitmFwhtSearch::name)
        .def("describe", &bmmpy::MitmFwhtSearch::describe, nb::arg("window_size"))
        .def("search", &bmmpy::MitmFwhtSearch::search, nb::arg("matrix"), nb::arg("window_rows"));

    nb::class_<bmmpy::GreedySelection>(m, "GreedySelection")
        .def(nb::init<std::uint64_t, bool, std::uint64_t>(),
             nb::arg("min_gain"),
             nb::arg("stochastic") = false,
             nb::arg("seed") = 0)
        .def("apply",
             &bmmpy::GreedySelection::apply,
             nb::arg("matrix"),
             nb::arg("window_rows"),
             nb::arg("candidates"));

    nb::class_<bmmpy::RuntimeFeatures>(m, "RuntimeFeatures")
        .def(nb::init<>())
        .def_rw("avx2_compiled", &bmmpy::RuntimeFeatures::avx2_compiled)
        .def_rw("avx2_available", &bmmpy::RuntimeFeatures::avx2_available)
        .def_rw("parallel_compiled", &bmmpy::RuntimeFeatures::parallel_compiled)
        .def_rw("parallel_enabled", &bmmpy::RuntimeFeatures::parallel_enabled)
        .def_rw("max_threads", &bmmpy::RuntimeFeatures::max_threads)
        .def_rw("bit_ops_backend", &bmmpy::RuntimeFeatures::bit_ops_backend)
        .def_rw("fwht_backend", &bmmpy::RuntimeFeatures::fwht_backend)
        .def("__repr__", [](const bmmpy::RuntimeFeatures& value) {
            return "RuntimeFeatures("
                   "avx2_compiled=" +
                   std::string(value.avx2_compiled ? "True" : "False") +
                   ", avx2_available=" + std::string(value.avx2_available ? "True" : "False") +
                   ", parallel_compiled=" +
                   std::string(value.parallel_compiled ? "True" : "False") +
                   ", parallel_enabled=" + std::string(value.parallel_enabled ? "True" : "False") +
                   ", max_threads=" + std::to_string(value.max_threads) + ", bit_ops_backend='" +
                   value.bit_ops_backend + "', fwht_backend='" + value.fwht_backend + "')";
        });

    m.def("get_version", &bmmpy::get_version, "Get the version of the library");
    m.def("add", &bmmpy::add, nb::arg("a"), nb::arg("b"), "Add two integers");
    m.def("get_runtime_features",
          &bmmpy::get_runtime_features,
          "Return runtime feature flags for this build and host");

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
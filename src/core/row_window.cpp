#include "bmmpy/core/row_window.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace bmmpy {

RowWindow::RowWindow(BitMatrix& matrix, std::vector<std::size_t> rows)
    : _matrix(&matrix), _mutable_matrix(&matrix), _global_rows(std::move(rows)) {
    initialize();
}

RowWindow::RowWindow(const BitMatrix& matrix, std::vector<std::size_t> rows)
    : _matrix(&matrix), _global_rows(std::move(rows)) {
    initialize();
}

const BitMatrix& RowWindow::matrix() const noexcept { return *_matrix; }

std::size_t RowWindow::cols() const noexcept { return _matrix->cols(); }

std::size_t RowWindow::words_per_row() const noexcept { return _matrix->words_per_row(); }

std::size_t RowWindow::stride_words() const noexcept { return _matrix->stride_words(); }

std::size_t RowWindow::global_row(std::size_t local_row) const {
    if (local_row >= _global_rows.size())
        throw std::out_of_range("row window local row out of bounds");
    return _global_rows[local_row];
}

const std::uint64_t* RowWindow::row_words(std::size_t local_row) const {
    if (local_row >= _row_ptrs.size())
        throw std::out_of_range("row window local row out of bounds");
    return _row_ptrs[local_row];
}

std::uint64_t RowWindow::row_popcount(std::size_t local_row) const {
    return _matrix->row_popcount(global_row(local_row));
}

bool RowWindow::get(std::size_t local_row, std::size_t col) const {
    return _matrix->get(global_row(local_row), col);
}

void RowWindow::set(std::size_t local_row, std::size_t col, bool value) {
    require_mutable_matrix().set(global_row(local_row), col, value);
}

void RowWindow::row_xor(std::size_t target_local_row, std::size_t source_local_row) {
    require_mutable_matrix().row_xor(global_row(target_local_row), global_row(source_local_row));
}

void RowWindow::row_xor_from(std::size_t target_local_row,
                             const RowWindow& source,
                             std::size_t source_local_row) {
    require_mutable_matrix().row_xor_from(
        global_row(target_local_row), source.matrix(), source.global_row(source_local_row));
}

BitMatrix RowWindow::materialize() const { return _matrix->extract_rows_by_indices(_global_rows); }

void RowWindow::initialize() {
    if (_matrix == nullptr)
        throw std::invalid_argument("RowWindow requires a source matrix");

    std::vector<std::size_t> sorted_rows = _global_rows;
    std::sort(sorted_rows.begin(), sorted_rows.end());
    if (std::adjacent_find(sorted_rows.begin(), sorted_rows.end()) != sorted_rows.end()) {
        throw std::invalid_argument("RowWindow rows must be unique");
    }

    _row_ptrs.clear();
    _row_ptrs.reserve(_global_rows.size());

    for (std::size_t row : _global_rows) {
        if (row >= _matrix->rows())
            throw std::out_of_range("row window row out of bounds");
        _row_ptrs.push_back(_matrix->row_words(row));
    }
}

BitMatrix& RowWindow::require_mutable_matrix() {
    if (_mutable_matrix == nullptr)
        throw std::logic_error("RowWindow is not mutable");
    return *_mutable_matrix;
}

} // namespace bmmpy
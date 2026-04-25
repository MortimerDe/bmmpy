#include "bmmpy/core/row_window.hpp"

#include "bmmpy/core/bit_matrix.hpp"
#include "bmmpy/core/detail/bit_ops.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace bmmpy {
namespace {

constexpr std::size_t kWordBits = std::numeric_limits<std::uint64_t>::digits;

std::uint64_t tail_mask_for_cols(std::size_t cols) noexcept {
    const std::size_t tail_bits = cols % kWordBits;
    if (tail_bits == 0)
        return ~std::uint64_t{0};

    return (std::uint64_t{1} << tail_bits) - 1;
}

} // namespace

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

std::uint64_t RowWindow::total_weight() const {
    const std::size_t word_count = words_per_row();
    if (_row_ptrs.empty() || word_count == 0)
        return 0;

    const auto& bit_ops = detail::bit_ops();
    const std::uint64_t tail_mask = tail_mask_for_cols(cols());

    std::uint64_t total = 0;
    for (std::size_t word_index = 0; word_index < word_count; ++word_index) {
        std::uint64_t active = 0;
        for (const std::uint64_t* row_words : _row_ptrs)
            active |= row_words[word_index];

        if (word_index + 1 == word_count)
            active &= tail_mask;

        total += bit_ops.row_popcount(&active, 1);
    }

    return total;
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
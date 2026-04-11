#pragma once

#include "bmmpy/core/bit_matrix.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bmmpy {

class RowWindow {
public:
    RowWindow(BitMatrix& matrix, std::vector<std::size_t> rows);
    RowWindow(const BitMatrix& matrix, std::vector<std::size_t> rows);

    const BitMatrix& matrix() const noexcept;
    bool is_mutable() const noexcept { return _mutable_matrix != nullptr; }

    std::size_t size() const noexcept { return _global_rows.size(); }
    std::size_t cols() const noexcept;
    std::size_t words_per_row() const noexcept;
    std::size_t stride_words() const noexcept;

    const std::vector<std::size_t>& global_rows() const noexcept { return _global_rows; }
    std::size_t global_row(std::size_t local_row) const;

    const std::vector<const std::uint64_t*>& row_ptrs() const noexcept { return _row_ptrs; }
    const std::uint64_t* row_words(std::size_t local_row) const;
    std::uint64_t row_popcount(std::size_t local_row) const;
    bool get(std::size_t local_row, std::size_t col) const;

    void set(std::size_t local_row, std::size_t col, bool value);
    void row_xor(std::size_t target_local_row, std::size_t source_local_row);
    void row_xor_from(std::size_t target_local_row,
                      const RowWindow& source,
                      std::size_t source_local_row);

    BitMatrix materialize() const;

private:
    void initialize();
    BitMatrix& require_mutable_matrix();

    const BitMatrix* _matrix = nullptr;
    BitMatrix* _mutable_matrix = nullptr;
    std::vector<std::size_t> _global_rows;
    std::vector<const std::uint64_t*> _row_ptrs;
};

} // namespace bmmpy
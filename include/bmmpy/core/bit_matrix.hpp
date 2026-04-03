#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace bmmpy {

enum class MatrixErr { SizeTooLarge, AllocationFailed, DimensionMismatch };

class MatrixError : public std::runtime_error {
public:
    explicit MatrixError(MatrixErr code);
    MatrixErr code() const noexcept { return _code; }

private:
    MatrixErr _code;
};

class BitMatrix {
public:
    static constexpr std::size_t k_word_bits = 64;
    static constexpr std::size_t k_alignment = 32;

    BitMatrix() noexcept = default;
    BitMatrix(std::size_t rows, std::size_t cols);
    ~BitMatrix() noexcept;

    BitMatrix(const BitMatrix& other);
    BitMatrix& operator=(const BitMatrix& other);

    BitMatrix(BitMatrix&& other) noexcept;
    BitMatrix& operator=(BitMatrix&& other) noexcept;

    void swap(BitMatrix& other) noexcept;

    std::size_t rows() const noexcept { return _rows; }
    std::size_t cols() const noexcept { return _cols; }
    std::size_t stride_words() const noexcept { return _stride; }
    std::size_t words_per_row() const noexcept {
        return (_cols + k_word_bits - 1) / k_word_bits;
    }
    std::size_t total_words() const noexcept { return _total_words; }
    std::size_t total_bytes() const noexcept {
        return _total_words * sizeof(std::uint64_t);
    }

    const std::uint64_t* data() const noexcept { return _data; }
    std::uint64_t* data() noexcept { return _data; }

    void copy_from_words(const std::uint64_t* src, std::size_t count);

    const std::uint64_t* row_words(std::size_t row) const;
    std::uint64_t* row_words(std::size_t row);

    void set(std::size_t row, std::size_t col, bool value);
    bool get(std::size_t row, std::size_t col) const;

    void set_unchecked(std::size_t row, std::size_t col, bool value) noexcept;
    bool get_unchecked(std::size_t row, std::size_t col) const noexcept;

    void row_xor(std::size_t target_row, std::size_t source_row) noexcept;
    void row_xor_from(std::size_t target_row,
                      const BitMatrix& source,
                      std::size_t source_row);

    BitMatrix add(const BitMatrix& other) const;
    BitMatrix mul(const BitMatrix& other) const;
    BitMatrix power(std::uint32_t exp) const;

    std::uint64_t row_popcount(std::size_t row) const;
    std::uint64_t weight() const;

    void swap_rows(std::size_t r1, std::size_t r2) noexcept;
    std::size_t rank() const;

    BitMatrix
    extract_rows_by_indices(const std::vector<std::size_t>& indices) const;
    void insert_rows_by_indices(const BitMatrix& source,
                                const std::vector<std::size_t>& indices);

    static BitMatrix identity(std::size_t n);

private:
    static std::size_t checked_add(std::size_t a, std::size_t b);
    static std::size_t checked_mul(std::size_t a, std::size_t b);
    static std::size_t ceil_div(std::size_t a, std::size_t b) noexcept;

    void allocate_zeroed(std::size_t total_words);
    void destroy() noexcept;

    std::uint64_t* row_ptr_unchecked(std::size_t row) noexcept {
        return _data + row * _stride;
    }
    const std::uint64_t* row_ptr_unchecked(std::size_t row) const noexcept {
        return _data + row * _stride;
    }

    static void row_xor_scalar(std::uint64_t* dst,
                               const std::uint64_t* src,
                               std::size_t len) noexcept;
    static std::uint64_t row_popcount_scalar(const std::uint64_t* src,
                                             std::size_t len) noexcept;
    static unsigned ctz64(std::uint64_t value) noexcept; // Count Trailing Zeros

    // fields
    std::uint64_t* _data = nullptr;
    std::size_t _rows = 0;
    std::size_t _cols = 0;
    std::size_t _stride = 0;
    std::size_t _total_words = 0;
};

} // namespace bmmpy
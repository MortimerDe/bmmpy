#pragma once

#include "bmmpy/core/bit_matrix.hpp"

#include <stdexcept>

namespace bmmpy::algebra {

class RowTransformError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

BitMatrix find_row_transform(const BitMatrix& source, const BitMatrix& target);

BitMatrix invert_matrix(const BitMatrix& matrix);

} // namespace bmmpy::algebra
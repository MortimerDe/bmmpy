#pragma once

#include "bmmpy/core/bit_matrix.hpp"
#include "bmmpy/core/row_window.hpp"
#include "bmmpy/types/candidate.hpp"

#include <cstddef>
#include <vector>

namespace bmmpy {

BitMatrix build_transform_matrix(std::size_t input_rows, const std::vector<Candidate>& candidates);

BitMatrix build_transform_matrix_checked(const RowWindow& window,
                                         const std::vector<Candidate>& candidates);

} // namespace bmmpy
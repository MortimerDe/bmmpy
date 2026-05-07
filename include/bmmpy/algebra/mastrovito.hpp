#pragma once

#include "bmmpy/core/bit_matrix.hpp"
#include "bmmpy/core/bit_vector.hpp"

#include <cstddef>
#include <vector>

namespace bmmpy::algebra {

BitVector powers_to_field_element(const std::vector<std::size_t>& powers,
                                  std::size_t degree,
                                  const std::vector<std::size_t>& poly_powers);

BitMatrix build_basis_change_matrix(const std::vector<BitVector>& basis_elements,
                                    std::size_t degree);

BitVector basis_mask_to_field_element(const BitVector& mask,
                                      const std::vector<BitVector>& basis_elements,
                                      std::size_t degree);

BitVector field_element_to_basis_mask(const BitVector& element_bits, const BitMatrix& change_inv);

class MastrovitoCore {
public:
    MastrovitoCore(std::size_t degree,
                   std::vector<std::size_t> poly_powers,
                   BitVector element_standard_bits,
                   std::vector<BitVector> basis_elements = {});

    std::size_t degree() const noexcept { return _degree; }

    const std::vector<std::size_t>& powers() const noexcept { return _poly_powers; }

    const BitVector& element_standard_bits() const noexcept { return _element_standard_bits; }

    std::vector<BitMatrix> get_basis_multiplication_matrices() const;

    BitMatrix get_mastrovito_matrix(const BitVector& power) const;

    BitMatrix build_check_matrix(std::size_t c, std::size_t k) const;

    BitMatrix build_check_matrix(std::size_t c, std::size_t k, const BitVector& start_i) const;

private:
    std::size_t _degree = 0;
    std::vector<std::size_t> _poly_powers;
    BitVector _element_standard_bits;
    std::vector<BitVector> _basis_elements;
    std::vector<BitMatrix> _power_basis_matrices;
    BitMatrix _m_alpha;
};

} // namespace bmmpy::algebra
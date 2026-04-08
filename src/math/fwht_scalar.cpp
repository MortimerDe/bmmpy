#include "bmmpy/math/detail/fwht_ops.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace bmmpy::detail {
namespace {

template <typename T> using UnsignedOf = typename std::make_unsigned<T>::type;

template <typename T> T from_unsigned_bits(UnsignedOf<T> bits) noexcept {
    T value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

template <typename T> T wrapping_add(T lhs, T rhs) noexcept {
    const auto sum =
        static_cast<UnsignedOf<T>>(lhs) + static_cast<UnsignedOf<T>>(rhs);
    return from_unsigned_bits<T>(sum);
}

template <typename T> T wrapping_sub(T lhs, T rhs) noexcept {
    const auto diff =
        static_cast<UnsignedOf<T>>(lhs) - static_cast<UnsignedOf<T>>(rhs);
    return from_unsigned_bits<T>(diff);
}

template <typename T>
void comb_scalar_impl(T* left, T* right, std::size_t len) noexcept {
    for (std::size_t i = 0; i < len; ++i) {
        const T u = left[i];
        const T v = right[i];
        left[i] = wrapping_add(u, v);
        right[i] = wrapping_sub(u, v);
    }
}

} // namespace

void comb_i16_scalar(std::int16_t* left,
                     std::int16_t* right,
                     std::size_t len) noexcept {
    comb_scalar_impl(left, right, len);
}

void comb_i32_scalar(std::int32_t* left,
                     std::int32_t* right,
                     std::size_t len) noexcept {
    comb_scalar_impl(left, right, len);
}

} // namespace bmmpy::detail
#include "bmmpy/math/fwht.hpp"

#include "bmmpy/math/detail/fwht_ops.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace bmmpy {
namespace {

constexpr std::size_t kParallelThreshold = std::size_t(1) << 17;

bool is_power_of_two(std::size_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

template <typename T>
using CombFn = void (*)(T* left, T* right, std::size_t len) noexcept;

template <typename T> CombFn<T> select_comb();

template <> CombFn<std::int16_t> select_comb<std::int16_t>() {
    return detail::fwht_ops().comb_i16;
}

template <> CombFn<std::int32_t> select_comb<std::int32_t>() {
    return detail::fwht_ops().comb_i32;
}

template <typename T> void fwht_inplace_impl(T* data, std::size_t size) {
    if (size <= 1)
        return;

    if (!is_power_of_two(size))
        throw std::invalid_argument(
            "fwht_inplace: size must be a power of two");

    const auto comb = select_comb<T>();

    std::size_t len = 1;
    while (len < size) {
        const std::size_t step = len * 2;
        const std::size_t blocks = size / step;

        if (size >= kParallelThreshold) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (std::ptrdiff_t block = 0;
                 block < static_cast<std::ptrdiff_t>(blocks);
                 ++block) {
                T* chunk = data + static_cast<std::size_t>(block) * step;
                comb(chunk, chunk + len, len);
            }
        } else {
            for (std::size_t base = 0; base < size; base += step)
                comb(data + base, data + base + len, len);
        }

        len = step;
    }
}

template <typename ScoreT>
void calc_scores_and_order_impl(const ScoreT* h,
                                std::size_t m,
                                std::int32_t n,
                                ScoreT* s_by_mask,
                                std::int32_t* order,
                                std::int32_t* cnt,
                                std::int32_t* off) {
    if (n < 0)
        throw std::invalid_argument(
            "calc_scores_and_order: n must be non-negative");

    const std::size_t bucket_count = static_cast<std::size_t>(n) + 1;
    std::fill(cnt, cnt + bucket_count, 0);
    std::fill(off, off + bucket_count, 0);

    if (m == 0)
        return;

    s_by_mask[0] = static_cast<ScoreT>(0);

    for (std::size_t u = 1; u < m; ++u) {
        const std::int32_t hu = static_cast<std::int32_t>(h[u]);
        std::int32_t su = (n - hu) >> 1;
        if (su < 0)
            su = 0;
        else if (su > n)
            su = n;

        s_by_mask[u] = static_cast<ScoreT>(su);
        cnt[static_cast<std::size_t>(su)] += 1;
    }

    for (std::size_t v = 0; v < static_cast<std::size_t>(n); ++v)
        off[v + 1] = off[v] + cnt[v];

    for (std::size_t u = 1; u < m; ++u) {
        const auto su =
            static_cast<std::size_t>(static_cast<std::int32_t>(s_by_mask[u]));
        const std::int32_t pos = off[su];
        off[su] = pos + 1;
        order[static_cast<std::size_t>(pos)] = static_cast<std::int32_t>(u);
    }
}

} // namespace

void fwht_inplace(std::int16_t* data, std::size_t size) {
    fwht_inplace_impl(data, size);
}

void fwht_inplace(std::int32_t* data, std::size_t size) {
    fwht_inplace_impl(data, size);
}

void calc_scores_and_order(const std::int16_t* h,
                           std::size_t m,
                           std::int32_t n,
                           std::int16_t* s_by_mask,
                           std::int32_t* order,
                           std::int32_t* cnt,
                           std::int32_t* off) {
    calc_scores_and_order_impl(h, m, n, s_by_mask, order, cnt, off);
}

void calc_scores_and_order(const std::int32_t* h,
                           std::size_t m,
                           std::int32_t n,
                           std::int32_t* s_by_mask,
                           std::int32_t* order,
                           std::int32_t* cnt,
                           std::int32_t* off) {
    calc_scores_and_order_impl(h, m, n, s_by_mask, order, cnt, off);
}

} // namespace bmmpy
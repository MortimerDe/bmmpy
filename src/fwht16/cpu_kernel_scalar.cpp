#include "bmmpy/fwht16/cpu_kernel.hpp"

#include <algorithm>

namespace bmmpy::fwht16 {
namespace {
bool topk_item_less(const Fwht16TopKItem& lhs, const Fwht16TopKItem& rhs) {
    if (lhs.weight != rhs.weight) {
        return lhs.weight < rhs.weight;
    }
    return lhs.mask < rhs.mask;
}

} // namespace

void build_histogram_16(const ColumnMasks16& sample, std::int16_t* histogram) noexcept {
    std::fill_n(histogram, Fwht16Constants::k_spectrum_size, std::int16_t{0});

    for (std::uint16_t mask : sample.masks)
        ++histogram[mask];
}

void fwht16_scalar(std::int16_t* data) noexcept {
    for (std::size_t len = 1; len < Fwht16Constants::k_spectrum_size; len <<= 1) {
        const std::size_t step = len << 1;

        for (std::size_t base = 0; base < Fwht16Constants::k_spectrum_size; base += step) {
            std::int16_t* left = data + base;
            std::int16_t* right = left + len;

            for (std::size_t i = 0; i < len; ++i) {
                const std::int16_t u = left[i];
                const std::int16_t v = right[i];
                left[i] = static_cast<std::int16_t>(u + v);
                right[i] = static_cast<std::int16_t>(u - v);
            }
        }
    }
}

void extract_topk_16(const std::int16_t* spectrum, std::size_t k, Fwht16TopKItem* out) noexcept {
    std::size_t used = 0;

    for (std::size_t mask = 1; mask < Fwht16Constants::k_spectrum_size; ++mask) {
        const int w2 =
            static_cast<int>(Fwht16Constants::k_max_weight) - static_cast<int>(spectrum[mask]);

        if (w2 < 0 || (w2 & 1) != 0)
            continue;

        const Fwht16TopKItem candidate{
            static_cast<std::uint16_t>(mask),
            static_cast<std::uint16_t>(w2 / 2),
        };

        if (used < k) {
            out[used++] = candidate;

            if (used == k)
                std::make_heap(out, out + k, topk_item_less);

            continue;
        }

        if (topk_item_less(candidate, out[0])) {
            std::pop_heap(out, out + k, topk_item_less);
            out[k - 1] = candidate;
            std::push_heap(out, out + k, topk_item_less);
        }
    }

    if (used < k) {
        std::sort(out, out + used, topk_item_less);
        for (std::size_t i = used; i < k; ++i)
            out[i] = {};
        return;
    }

    std::sort_heap(out, out + k, topk_item_less);
}

} // namespace bmmpy::fwht16
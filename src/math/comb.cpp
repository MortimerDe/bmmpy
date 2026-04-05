#include "bmmpy/math/comb.hpp"

#include <stdexcept>

namespace bmmpy {
namespace {

constexpr std::uint32_t kMaxNU32 = 32;
constexpr std::uint32_t kMaxNU64 = 64;

} // namespace

void fixed_weight_masks_u32(std::uint32_t n,
                            std::uint32_t k,
                            std::vector<std::uint32_t>& out) {
    out.clear();

    if (k == 0) {
        out.push_back(0);
        return;
    }

    if (n > kMaxNU32)
        throw std::invalid_argument(
            "fixed_weight_masks_u32: n exceeds 32-bit mask width");

    if (k > n)
        return;

    std::uint64_t comb = (std::uint64_t(1) << k) - 1;
    const std::uint64_t limit = std::uint64_t(1) << n;

    while (comb < limit) {
        out.push_back(static_cast<std::uint32_t>(comb));

        const std::uint64_t c = comb & (~comb + 1);
        const std::uint64_t r = comb + c;
        if (r == 0)
            break;

        comb = (((r ^ comb) >> 2) / c) | r;
    }
}

void fixed_weight_masks_u64(std::uint32_t n,
                            std::uint32_t k,
                            std::vector<std::uint64_t>& out) {
    out.clear();

    if (k == 0) {
        out.push_back(0);
        return;
    }

    if (n > kMaxNU64)
        throw std::invalid_argument(
            "fixed_weight_masks_u64: n exceeds 64-bit mask width");

    if (k > n)
        return;

    std::uint64_t comb = (std::uint64_t(1) << k) - 1;
    const std::uint64_t limit =
        n == 64 ? std::uint64_t(0) : (std::uint64_t(1) << n);

    if (n == 64) {
        for (;;) {
            out.push_back(comb);

            const std::uint64_t c = comb & (~comb + 1);
            const std::uint64_t r = comb + c;
            if (r == 0)
                break;

            comb = (((r ^ comb) >> 2) / c) | r;
        }
        return;
    }

    while (comb < limit) {
        out.push_back(comb);

        const std::uint64_t c = comb & (~comb + 1);
        const std::uint64_t r = comb + c;
        if (r == 0)
            break;

        comb = (((r ^ comb) >> 2) / c) | r;
    }
}

} // namespace bmmpy
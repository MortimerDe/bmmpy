#include "bmmpy/core/bit_matrix.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#if defined(BMMPY_HAS_OPENMP)
#include <omp.h>
#endif
#include <limits>

namespace bmmpy {
namespace {
constexpr std::size_t kAutoChunkBits = 12;
struct TopKEntry {
    std::uint64_t mask = 0;
    std::uint32_t weight = 0;
};

bool topk_less(const TopKEntry& lhs, const TopKEntry& rhs) noexcept {
    if (lhs.weight != rhs.weight)
        return lhs.weight < rhs.weight;
    return lhs.mask < rhs.mask;
}

void insert_topk(std::vector<TopKEntry>& best, std::size_t k, const TopKEntry incoming) {
    if (incoming.mask == 0 || k == 0)
        return;

    const auto pos = std::lower_bound(best.begin(), best.end(), incoming, topk_less);
    const std::size_t idx = static_cast<std::size_t>(pos - best.begin());

    if (best.size() < k) {
        best.insert(best.begin() + static_cast<std::ptrdiff_t>(idx), incoming);
        return;
    }

    if (idx >= k)
        return;

    for (std::size_t i = k - 1; i > idx; --i)
        best[i] = best[i - 1];

    best[idx] = incoming;
}

std::uint32_t checked_weight(const std::uint64_t raw_weight) {
    if (raw_weight > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::overflow_error("BruteforceSearch: candidate weight exceeds uint32 range");
    }

    return static_cast<std::uint32_t>(raw_weight);
}

class AlignedWordBuffer {
public:
    explicit AlignedWordBuffer(std::size_t words) : _size(words) {
        if (_size == 0)
            return;

        _data = static_cast<std::uint64_t*>(::operator new[](
            _size * sizeof(std::uint64_t), std::align_val_t(BitMatrix::k_alignment)));
        std::fill_n(_data, _size, std::uint64_t{0});
    }

    ~AlignedWordBuffer() noexcept { reset(); }

    AlignedWordBuffer(const AlignedWordBuffer&) = delete;
    AlignedWordBuffer& operator=(const AlignedWordBuffer&) = delete;

    AlignedWordBuffer(AlignedWordBuffer&& other) noexcept : _data(other._data), _size(other._size) {
        other._data = nullptr;
        other._size = 0;
    }

    AlignedWordBuffer& operator=(AlignedWordBuffer&& other) noexcept {
        if (this == &other)
            return *this;

        reset();
        _data = other._data;
        _size = other._size;
        other._data = nullptr;
        other._size = 0;
        return *this;
    }

    std::uint64_t* data() noexcept { return _data; }
    const std::uint64_t* data() const noexcept { return _data; }

private:
    void reset() noexcept {
        if (_data != nullptr) {
            ::operator delete[](_data, std::align_val_t(BitMatrix::k_alignment));
            _data = nullptr;
        }
        _size = 0;
    }
    std::uint64_t* _data = nullptr;
    std::size_t _size = 0;
};

} // namespace
} // namespace bmmpy
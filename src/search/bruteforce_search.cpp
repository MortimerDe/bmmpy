#include "bmmpy/search/bruteforce_search.hpp"

#include "bmmpy/core/bit_matrix.hpp"
#include "bmmpy/core/detail/bit_intrinsics.hpp"
#include "bmmpy/core/detail/bit_ops.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

#if defined(BMMPY_HAS_OPENMP)
#include <omp.h>
#endif

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

std::size_t resolve_chunk_bits(const std::size_t t, const std::size_t configured_chunk_bits) {
    if (configured_chunk_bits == 0)
        return std::min<std::size_t>(t, kAutoChunkBits);

    if (configured_chunk_bits > t || configured_chunk_bits >= Candidate::k_word_bits) {
        throw std::invalid_argument(
            "BruteforceSearch: chunk_bits must be in [1, min(t, 63)] or 0 for auto");
    }

    return configured_chunk_bits;
}

template <typename Buffer>
void sweep_prefix(const std::vector<const std::uint64_t*>& rows,
                  const std::size_t t,
                  const std::size_t chunk_bits,
                  const std::uint64_t prefix,
                  Buffer& current,
                  const std::size_t word_count,
                  std::vector<TopKEntry>& best,
                  const std::size_t k) {
    auto* current_words = current.data();
    std::fill_n(current_words, word_count, std::uint64_t{0});

    const auto& bit_ops = detail::bit_ops();
    const std::size_t high_bits = t - chunk_bits;

    for (std::size_t high = 0; high < high_bits; ++high) {
        if (((prefix >> high) & 1ull) == 0)
            continue;

        bit_ops.row_xor(current_words, rows[chunk_bits + high], word_count);
    }

    std::uint64_t current_mask = prefix << chunk_bits;
    if (current_mask != 0) {
        insert_topk(best,
                    k,
                    TopKEntry{
                        current_mask,
                        checked_weight(bit_ops.row_popcount(current_words, word_count)),
                    });
    }

    const std::uint64_t low_states = std::uint64_t{1} << chunk_bits;
    for (std::uint64_t step = 1; step < low_states; ++step) {
        const std::size_t bit = static_cast<std::size_t>(detail::ctz64(step));
        bit_ops.row_xor(current_words, rows[bit], word_count);
        current_mask ^= (std::uint64_t{1} << bit);

        insert_topk(best,
                    k,
                    TopKEntry{
                        current_mask,
                        checked_weight(bit_ops.row_popcount(current_words, word_count)),
                    });
    }
}

std::vector<Candidate>
merge_thread_results(const std::vector<std::vector<TopKEntry>>& thread_results,
                     const std::size_t k) {
    std::vector<TopKEntry> merged;
    merged.reserve(k);

    for (const auto& local : thread_results) {
        for (const TopKEntry& entry : local)
            insert_topk(merged, k, entry);
    }

    std::vector<Candidate> out;
    out.reserve(merged.size());

    for (const TopKEntry& entry : merged)
        out.push_back(Candidate::from_u64(entry.mask, entry.weight));

    return out;
}

template <typename WorkerFactory>
std::vector<Candidate>
run_prefix_parallel(const std::size_t high_bits, const std::size_t k, WorkerFactory&& make_worker) {
    const std::uint64_t prefix_count =
        high_bits == 0 ? std::uint64_t{1} : (std::uint64_t{1} << high_bits);

    std::size_t thread_slots = 1;
#if defined(BMMPY_HAS_OPENMP)
    thread_slots = static_cast<std::size_t>(std::max(omp_get_max_threads(), 1));
#endif

    std::vector<std::vector<TopKEntry>> thread_results(thread_slots);
    for (auto& local : thread_results)
        local.reserve(k);

#if defined(_OPENMP)
    if (high_bits < 63) {
#pragma omp parallel if (prefix_count > 1)
        {
            auto worker = make_worker();

            int tid = 0;
#if defined(BMMPY_HAS_OPENMP)
            tid = omp_get_thread_num();
#endif

            auto& local = thread_results[static_cast<std::size_t>(tid)];
            local.clear();

#pragma omp for schedule(static)
            for (long long prefix_index = 0; prefix_index < static_cast<long long>(prefix_count);
                 ++prefix_index) {
                worker(static_cast<std::uint64_t>(prefix_index), local);
            }
        }

        return merge_thread_results(thread_results, k);
    }
#endif

    auto worker = make_worker();
    auto& local = thread_results[0];
    local.clear();

    for (std::uint64_t prefix = 0; prefix < prefix_count; ++prefix)
        worker(prefix, local);

    return merge_thread_results(thread_results, k);
}

template <std::size_t WordCount>
std::vector<Candidate>
run_fixed_word_search(const RowWindow& window, const std::size_t chunk_bits, const std::size_t k) {
    const auto& rows = window.row_ptrs();
    const std::size_t t = window.size();

    return run_prefix_parallel(t - chunk_bits, k, [&rows, t, chunk_bits, k]() {
        struct Worker {
            const std::vector<const std::uint64_t*>& rows;
            std::size_t t;
            std::size_t chunk_bits;
            std::size_t k;
            alignas(BitMatrix::k_alignment) std::array<std::uint64_t, WordCount> current{};

            void operator()(const std::uint64_t prefix, std::vector<TopKEntry>& local) {
                sweep_prefix(rows, t, chunk_bits, prefix, current, WordCount, local, k);
            }
        };

        return Worker{rows, t, chunk_bits, k};
    });
}

std::vector<Candidate> run_dynamic_word_search(const RowWindow& window,
                                               const std::size_t chunk_bits,
                                               const std::size_t k) {
    const auto& rows = window.row_ptrs();
    const std::size_t t = window.size();
    const std::size_t word_count = window.words_per_row();

    return run_prefix_parallel(t - chunk_bits, k, [&rows, t, chunk_bits, word_count, k]() {
        struct Worker {
            const std::vector<const std::uint64_t*>& rows;
            std::size_t t;
            std::size_t chunk_bits;
            std::size_t word_count;
            std::size_t k;
            AlignedWordBuffer current;

            Worker(const std::vector<const std::uint64_t*>& rows_value,
                   const std::size_t t_value,
                   const std::size_t chunk_bits_value,
                   const std::size_t word_count_value,
                   const std::size_t k_value)
                : rows(rows_value), t(t_value), chunk_bits(chunk_bits_value),
                  word_count(word_count_value), k(k_value), current(word_count_value) {}

            void operator()(const std::uint64_t prefix, std::vector<TopKEntry>& local) {
                sweep_prefix(rows, t, chunk_bits, prefix, current, word_count, local, k);
            }
        };

        return Worker(rows, t, chunk_bits, word_count, k);
    });
}

} // namespace

std::vector<Candidate> BruteforceSearch::search(const RowWindow& window) {
    if (_config.max_candidates == 0)
        return {};

    const std::size_t t = window.size();
    if (t == 0)
        return {};

    if (t > Candidate::k_word_bits) {
        throw std::invalid_argument("BruteforceSearch: window_rows size must be <= 64");
    }

    if (window.words_per_row() == 0)
        return {};

    if (window.cols() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::overflow_error("BruteforceSearch: window width exceeds candidate weight range");
    }

    const std::size_t chunk_bits = resolve_chunk_bits(t, _config.chunk_bits);

    switch (window.words_per_row()) {
    case 8:
        return run_fixed_word_search<8>(window, chunk_bits, _config.max_candidates);
    case 16:
        return run_fixed_word_search<16>(window, chunk_bits, _config.max_candidates);
    default:
        return run_dynamic_word_search(window, chunk_bits, _config.max_candidates);
    }
}

} // namespace bmmpy
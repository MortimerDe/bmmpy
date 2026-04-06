#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace bmmpy {

struct Candidate {
    using word_type = std::uint64_t;
    using mask_type = std::vector<word_type>;

    static constexpr std::size_t k_word_bits = 64;

    struct SelectedRows {
        class iterator {
        public:
            using iterator_category = std::input_iterator_tag;
            using value_type = std::size_t;
            using difference_type = std::ptrdiff_t;
            using reference = value_type;
            using pointer = void;

            iterator() = default;

            iterator(const mask_type* words, std::size_t word_idx) noexcept
                : _words(words), _word_idx(word_idx) {
                if (_words == nullptr || _word_idx >= _words->size()) {
                    if (_words != nullptr)
                        _word_idx = _words->size();
                    return;
                }

                _current = (*_words)[_word_idx];
                _base = _word_idx * k_word_bits;
                skip_empty_words();
            }

            value_type operator*() const noexcept {
                return _base + static_cast<value_type>(ctz64(_current));
            }

            iterator& operator++() noexcept {
                if (_words == nullptr || _word_idx >= _words->size())
                    return *this;

                _current &= (_current - 1);
                skip_empty_words();
                return *this;
            }

            iterator operator++(int) noexcept {
                iterator tmp(*this);
                ++(*this);
                return tmp;
            }

            friend bool operator==(const iterator& lhs,
                                   const iterator& rhs) noexcept {
                return lhs._words == rhs._words &&
                       lhs._word_idx == rhs._word_idx &&
                       lhs._current == rhs._current;
            }

            friend bool operator!=(const iterator& lhs,
                                   const iterator& rhs) noexcept {
                return !(lhs == rhs);
            }

        private:
            static unsigned ctz64(word_type value) noexcept {
#if defined(_MSC_VER)
                unsigned long index = 0;
                _BitScanForward64(&index, value);
                return static_cast<unsigned>(index);
#else
                return static_cast<unsigned>(__builtin_ctzll(value));
#endif
            }

            void skip_empty_words() noexcept {
                while (_words != nullptr && _word_idx < _words->size() &&
                       _current == 0) {
                    ++_word_idx;
                    if (_word_idx >= _words->size())
                        return;

                    _current = (*_words)[_word_idx];
                    _base = _word_idx * k_word_bits;
                }
            }

            const mask_type* _words = nullptr;
            std::size_t _word_idx = 0;
            word_type _current = 0;
            std::size_t _base = 0;
        };

        explicit SelectedRows(const mask_type& words) noexcept
            : _words(&words) {}

        iterator begin() const noexcept { return iterator(_words, 0); }

        iterator end() const noexcept {
            return iterator(_words, _words == nullptr ? 0 : _words->size());
        }

    private:
        const mask_type* _words = nullptr;
    };

    mask_type mask;
    std::uint32_t weight = 0;

    Candidate() = default;

    Candidate(mask_type mask_value, std::uint32_t weight_value)
        : mask(std::move(mask_value)), weight(weight_value) {}

    static Candidate make(mask_type mask, std::uint32_t weight) {
        return Candidate(std::move(mask), weight);
    }

    static Candidate from_u64(word_type mask, std::uint32_t weight) {
        return Candidate(mask_type{mask}, weight);
    }

    static Candidate from_words(const mask_type& mask, std::uint32_t weight) {
        return Candidate(mask, weight);
    }

    static Candidate from_words(mask_type mask, std::uint32_t weight) {
        return Candidate(std::move(mask), weight);
    }

    static Candidate from_words(const word_type* words,
                                std::size_t count,
                                std::uint32_t weight) {
        return Candidate(mask_type(words, words + count), weight);
    }

    bool has_row(std::size_t i) const noexcept {
        const std::size_t word_idx = i / k_word_bits;
        if (word_idx >= mask.size())
            return false;

        return ((mask[word_idx] >> (i % k_word_bits)) & word_type{1}) != 0;
    }

    std::uint32_t mask_popcount() const noexcept {
        std::uint32_t total = 0;
        for (word_type word : mask) {
#if defined(_MSC_VER)
            total += static_cast<std::uint32_t>(__popcnt64(word));
#else
            total += static_cast<std::uint32_t>(__builtin_popcountll(word));
#endif
        }
        return total;
    }

    SelectedRows selected_rows() const noexcept { return SelectedRows(mask); }

    word_type mask_u64() const {
        if (mask.size() > 1 &&
            !std::all_of(mask.begin() + 1, mask.end(), [](word_type word) {
                return word == 0;
            })) {
            throw std::logic_error(
                "Candidate::mask_u64: mask does not fit in u64");
        }

        return mask.empty() ? 0 : mask.front();
    }
};

} // namespace bmmpy
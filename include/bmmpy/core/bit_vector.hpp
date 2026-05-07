#pragma once

#include "bmmpy/core/detail/bit_intrinsics.hpp"
#include "bmmpy/core/detail/xor_basis.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bmmpy {

class BitVector {
public:
    BitVector() = default;

    explicit BitVector(std::size_t bit_count)
        : _bit_count(bit_count), _words(word_count_for_bits(bit_count), 0) {}

    BitVector(std::size_t bit_count, std::vector<std::uint64_t> words)
        : _bit_count(bit_count), _words(std::move(words)) {
        _words.resize(word_count_for_bits(bit_count), 0);
        trim();
    }

    static constexpr std::size_t word_count_for_bits(std::size_t bit_count) noexcept {
        return (bit_count + 63u) / 64u;
    }

    static BitVector from_u64(std::uint64_t value, std::size_t bit_count = 0) {
        if (bit_count == 0)
            bit_count = value == 0 ? 1u : static_cast<std::size_t>(detail::msb_index64(value)) + 1u;

        if (bit_count < 64u && (value >> bit_count) != 0)
            throw std::invalid_argument("BitVector::from_u64: value does not fit into bit_count");

        BitVector bits(bit_count);
        if (!bits._words.empty())
            bits._words[0] = value;
        bits.trim();
        return bits;
    }

    static BitVector
    from_words(std::size_t bit_count, const std::uint64_t* words, std::size_t word_count) {
        const std::size_t required = word_count_for_bits(bit_count);
        if (word_count < required)
            throw std::invalid_argument("BitVector::from_words: not enough source words");

        std::vector<std::uint64_t> storage(required, 0);
        for (std::size_t i = 0; i < required; ++i)
            storage[i] = words[i];

        return BitVector(bit_count, std::move(storage));
    }

    static BitVector from_positions(std::size_t bit_count,
                                    const std::vector<std::size_t>& positions) {
        BitVector bits(bit_count);
        for (std::size_t bit_index : positions)
            bits.set(bit_index, true);
        return bits;
    }

    static BitVector unit(std::size_t bit_count, std::size_t bit_index) {
        BitVector bits(bit_count);
        bits.set(bit_index, true);
        return bits;
    }

    std::size_t bit_count() const noexcept { return _bit_count; }

    std::size_t word_count() const noexcept { return _words.size(); }

    const std::vector<std::uint64_t>& words() const noexcept { return _words; }

    std::vector<std::uint64_t>& words() noexcept { return _words; }

    const std::uint64_t* data() const noexcept { return _words.data(); }

    std::uint64_t* data() noexcept { return _words.data(); }

    bool any() const noexcept {
        for (std::uint64_t word : _words) {
            if (word != 0)
                return true;
        }
        return false;
    }

    bool none() const noexcept { return !any(); }

    bool is_odd() const noexcept { return !_words.empty() && ((_words[0] & 1ull) != 0); }

    bool get(std::size_t bit_index) const {
        if (bit_index >= _bit_count)
            throw std::out_of_range("BitVector::get: bit index out of range");

        const std::size_t word_index = bit_index / 64u;
        const std::size_t offset = bit_index % 64u;
        return ((_words[word_index] >> offset) & 1ull) != 0;
    }

    void set(std::size_t bit_index, bool value) {
        if (bit_index >= _bit_count)
            throw std::out_of_range("BitVector::set: bit index out of range");

        const std::size_t word_index = bit_index / 64u;
        const std::size_t offset = bit_index % 64u;
        const std::uint64_t mask = 1ull << offset;

        if (value)
            _words[word_index] |= mask;
        else
            _words[word_index] &= ~mask;
    }

    void flip(std::size_t bit_index) {
        if (bit_index >= _bit_count)
            throw std::out_of_range("BitVector::flip: bit index out of range");

        const std::size_t word_index = bit_index / 64u;
        const std::size_t offset = bit_index % 64u;
        _words[word_index] ^= (1ull << offset);
    }

    void xor_assign(const BitVector& other) {
        if (_bit_count != other._bit_count)
            throw std::invalid_argument("BitVector::xor_assign: bit_count mismatch");

        for (std::size_t i = 0; i < _words.size(); ++i)
            _words[i] ^= other._words[i];

        trim();
    }

    void shift_right_one() noexcept {
        std::uint64_t carry = 0;

        for (std::size_t i = _words.size(); i > 0; --i) {
            std::uint64_t& word = _words[i - 1];
            const std::uint64_t next_carry = (word & 1ull) << 63u;
            word = (word >> 1u) | carry;
            carry = next_carry;
        }

        trim();
    }

    std::size_t bit_length() const noexcept {
        for (std::size_t i = _words.size(); i > 0; --i) {
            const std::uint64_t word = _words[i - 1];
            if (word != 0)
                return (i - 1u) * 64u + static_cast<std::size_t>(detail::msb_index64(word)) + 1u;
        }

        return 0;
    }

    std::size_t highest_set_bit() const {
        const std::size_t length = bit_length();
        if (length == 0)
            throw std::out_of_range("BitVector::highest_set_bit: no bits are set");
        return length - 1u;
    }

    template <typename Fn> void for_each_set_bit(Fn&& fn) const {
        for (std::size_t word_index = 0; word_index < _words.size(); ++word_index) {
            std::uint64_t bits = _words[word_index];

            while (bits != 0) {
                const unsigned bit = bmmpy::detail::ctz64(bits);
                const std::size_t bit_index = word_index * 64u + static_cast<std::size_t>(bit);
                if (bit_index < _bit_count)
                    fn(bit_index);
                bits &= (bits - 1u);
            }
        }
    }

    bool operator==(const BitVector& other) const noexcept {
        return _bit_count == other._bit_count && _words == other._words;
    }

    bool operator!=(const BitVector& other) const noexcept { return !(*this == other); }

private:
    void trim() noexcept {
        if (_words.empty() || _bit_count == 0)
            return;

        const std::size_t tail_bits = _bit_count % 64u;
        if (tail_bits == 0)
            return;

        _words.back() &= ((1ull << tail_bits) - 1ull);
    }

    std::size_t _bit_count = 0;
    std::vector<std::uint64_t> _words;
};

} // namespace bmmpy
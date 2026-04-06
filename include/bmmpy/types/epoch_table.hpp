#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace bmmpy {

template <typename T> class EpochTable {
    static_assert(std::is_copy_constructible<T>::value,
                  "EpochTable<T> requires copy-constructible T");
    static_assert(std::is_copy_assignable<T>::value,
                  "EpochTable<T> requires copy-assignable T");

public:
    EpochTable() = default;

    EpochTable(std::size_t size, T empty_value)
        : _values(size, empty_value), _stamps(size, 0), _epoch(1),
          _empty(empty_value) {}

    std::size_t size() const noexcept { return _values.size(); }
    std::size_t len() const noexcept { return size(); }

    void clear() noexcept {
        _epoch = static_cast<std::uint32_t>(_epoch + 1);
        if (_epoch == 0) {
            _stamps.assign(_stamps.size(), 0);
            _epoch = 1;
        }
    }

    T get(std::size_t key) const noexcept {
        return _stamps[key] == _epoch ? _values[key] : _empty;
    }

    void set(std::size_t key, T value) noexcept {
        _stamps[key] = _epoch;
        _values[key] = value;
    }

    T replace(std::size_t key, T value) noexcept {
        if (_stamps[key] == _epoch) {
            T old = _values[key];
            _values[key] = value;
            return old;
        }

        _stamps[key] = _epoch;
        _values[key] = value;
        return _empty;
    }

    bool is_set(std::size_t key) const noexcept {
        return _stamps[key] == _epoch;
    }

    bool contains(std::size_t key) const noexcept { return is_set(key); }

    T empty_value() const noexcept { return _empty; }

private:
    std::vector<T> _values;
    std::vector<std::uint32_t> _stamps;
    std::uint32_t _epoch = 1;
    T _empty{};
};

} // namespace bmmpy
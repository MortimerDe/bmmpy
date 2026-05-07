#include "test_common.hpp"

#include <array>
#include <vector>

using bmmpy::BitMatrix;
using bmmpy::test::TestCase;

namespace {

void test_hash_is_canonical_and_data_sensitive() {
    BitMatrix canonical(1, 65);
    canonical.set(0, 0, true);
    canonical.set(0, 64, true);

    BitMatrix noisy(1, 65);
    std::array<std::uint64_t, 4> words{};
    words[0] = 0x1ull;
    words[1] = 0x1ull | (std::uint64_t{1} << 17u);
    words[2] = 0xaaaaaaaaaaaaaaaaull;
    words[3] = 0x5555555555555555ull;
    noisy.copy_from_words(words.data(), words.size());

    bmmpy::test::require(canonical.hash() == noisy.hash(),
                         "hash should ignore stride padding and bits beyond logical cols");

    BitMatrix changed = noisy;
    changed.set(0, 1, true);

    bmmpy::test::require(changed.hash() != noisy.hash(),
                         "hash should change when a logical matrix bit changes");
}

} // namespace

void append_bit_matrix_hash_tests(std::vector<TestCase>& tests) {
    tests.push_back(
        {"hash_is_canonical_and_data_sensitive", &test_hash_is_canonical_and_data_sensitive});
}
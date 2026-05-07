#include "test_common.hpp"

#include <sstream>
#include <vector>

using bmmpy::BitMatrix;
using bmmpy::test::TestCase;

namespace {

void test_text_io_roundtrip() {
    BitMatrix original = bmmpy::test::matrix_from_rows({
        "10101",
        "01010",
        "11100",
    });

    std::stringstream buffer;
    original.save_text(buffer);

    BitMatrix loaded = BitMatrix::load_text(buffer);
    bmmpy::test::expect_rows(loaded, {"10101", "01010", "11100"});
}

void test_binary_io_roundtrip_with_unpadded_payload() {
    BitMatrix original(3, 130);
    original.set(0, 0, true);
    original.set(0, 64, true);
    original.set(1, 63, true);
    original.set(1, 129, true);
    original.set(2, 5, true);

    std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
    original.save_binary(buffer);
    buffer.seekg(0);

    BitMatrix loaded = BitMatrix::load_binary(buffer);

    bmmpy::test::require(loaded.rows() == 3, "binary io rows mismatch");
    bmmpy::test::require(loaded.cols() == 130, "binary io cols mismatch");
    bmmpy::test::require(loaded.words_per_row() == 3, "binary io words_per_row mismatch");
    bmmpy::test::expect_rows(
        loaded,
        {
            "100000000000000000000000000000000000000000000000000000000000000010000000000000"
            "0000000000000000000000000000000000000000000000000000",
            "000000000000000000000000000000000000000000000000000000000000000100000000000000"
            "0000000000000000000000000000000000000000000000000001",
            "000001000000000000000000000000000000000000000000000000000000000000000000000000"
            "0000000000000000000000000000000000000000000000000000",
        });
}

void test_text_io_exceptions() {
    {
        std::stringstream buffer("2 3\n010\n01x\n");
        bmmpy::test::expect_runtime_error([&] { (void)BitMatrix::load_text(buffer); },
                                          "text io invalid char");
    }

    {
        std::stringstream buffer("2 3\n010\n01\n");
        bmmpy::test::expect_runtime_error([&] { (void)BitMatrix::load_text(buffer); },
                                          "text io truncated payload");
    }
}

void test_binary_io_exceptions() {
    {
        std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
        buffer.write("BADMAGC", 7);
        buffer.seekg(0);

        bmmpy::test::expect_runtime_error([&] { (void)BitMatrix::load_binary(buffer); },
                                          "binary io invalid magic");
    }

    {
        std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
        BitMatrix matrix = bmmpy::test::matrix_from_rows({"101", "010"});
        matrix.save_binary(buffer);
        std::string data = buffer.str();
        data.resize(data.size() - 1);

        std::stringstream truncated(data, std::ios::in | std::ios::out | std::ios::binary);
        bmmpy::test::expect_runtime_error([&] { (void)BitMatrix::load_binary(truncated); },
                                          "binary io truncated payload");
    }
}

} // namespace

void append_bit_matrix_io_tests(std::vector<TestCase>& tests) {
    tests.push_back({"text_io_roundtrip", &test_text_io_roundtrip});
    tests.push_back({"binary_io_roundtrip_with_unpadded_payload",
                     &test_binary_io_roundtrip_with_unpadded_payload});
    tests.push_back({"text_io_exceptions", &test_text_io_exceptions});
    tests.push_back({"binary_io_exceptions", &test_binary_io_exceptions});
}
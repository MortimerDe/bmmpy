#include "bmmpy/fwht16/constants.hpp"
#include "bmmpy/fwht16/engine.hpp"
#include "bmmpy/fwht16/types.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

void require(bool condition, std::string_view message) {
    if (!condition)
        fail(std::string(message));
}

template <typename Exception, typename Fn> void expect_throw(Fn&& fn, std::string_view context) {
    try {
        fn();
    } catch (const Exception&) {
        return;
    } catch (const std::exception& ex) {
        fail(std::string(context) + ": unexpected exception type: " + ex.what());
    }

    fail(std::string(context) + ": expected exception");
}

void test_constants_are_stable() {
    require(bmmpy::fwht16::Fwht16Constants::k_rows == 16, "k_rows");
    require(bmmpy::fwht16::Fwht16Constants::k_cols == 512, "k_cols");
    require(bmmpy::fwht16::Fwht16Constants::k_spectrum_size == 65536, "k_spectrum_size");
    require(bmmpy::fwht16::Fwht16Constants::k_max_weight == 512, "k_max_weight");
}

void test_cpu_stub_returns_empty_topk_layout() {
    bmmpy::fwht16::ColumnMasks16 sample{};
    bmmpy::fwht16::Fwht16Engine engine;

    bmmpy::fwht16::Fwht16BatchRequest request;
    request.samples = &sample;
    request.batch_size = 1;
    request.backend = bmmpy::fwht16::Fwht16Backend::cpu;
    request.mode = bmmpy::fwht16::Fwht16ResultMode::topk;
    request.topk = 8;

    const auto response = engine.run(request);

    require(response.actual_backend == bmmpy::fwht16::Fwht16Backend::cpu, "actual_backend");
    require(response.mode == bmmpy::fwht16::Fwht16ResultMode::topk, "mode");
    require(response.batch_size == 1, "batch_size");
    require(response.topk_offsets.size() == 2, "topk_offsets size");
    require(response.topk_offsets[0] == 0, "topk_offsets[0]");
    require(response.topk_offsets[1] == 0, "topk_offsets[1]");
    require(response.topk_results.empty(), "topk_results empty");
    require(response.spectra.empty(), "spectra empty");
}

void test_cpu_stub_returns_zeroed_spectrum_layout() {
    bmmpy::fwht16::ColumnMasks16 samples[2]{};
    bmmpy::fwht16::Fwht16Engine engine;

    bmmpy::fwht16::Fwht16BatchRequest request;
    request.samples = samples;
    request.batch_size = 2;
    request.backend = bmmpy::fwht16::Fwht16Backend::cpu;
    request.mode = bmmpy::fwht16::Fwht16ResultMode::spectrum;

    const auto response = engine.run(request);

    require(response.actual_backend == bmmpy::fwht16::Fwht16Backend::cpu,
            "spectrum actual_backend");
    require(response.mode == bmmpy::fwht16::Fwht16ResultMode::spectrum, "spectrum mode");
    require(response.batch_size == 2, "spectrum batch_size");
    require(response.topk_offsets.empty(), "spectrum topk_offsets empty");
    require(response.topk_results.empty(), "spectrum topk_results empty");
    require(response.spectra.size() == 2 * bmmpy::fwht16::Fwht16Constants::k_spectrum_size,
            "spectra size");

    for (std::int16_t value : response.spectra)
        require(value == 0, "spectra zero initialized");
}

void test_engine_rejects_null_samples_for_nonzero_batch() {
    bmmpy::fwht16::Fwht16Engine engine;
    bmmpy::fwht16::Fwht16BatchRequest request;
    request.samples = nullptr;
    request.batch_size = 1;

    expect_throw<std::invalid_argument>([&] { (void)engine.run(request); }, "null samples");
}

void test_engine_rejects_zero_topk_in_topk_mode() {
    bmmpy::fwht16::ColumnMasks16 sample{};
    bmmpy::fwht16::Fwht16Engine engine;
    bmmpy::fwht16::Fwht16BatchRequest request;
    request.samples = &sample;
    request.batch_size = 1;
    request.mode = bmmpy::fwht16::Fwht16ResultMode::topk;
    request.topk = 0;

    expect_throw<std::invalid_argument>([&] { (void)engine.run(request); }, "zero topk");
}

void test_engine_cpu_route_uses_cpu_backend() {
    bmmpy::fwht16::ColumnMasks16 sample{};
    bmmpy::fwht16::Fwht16Engine engine;
    bmmpy::fwht16::Fwht16BatchRequest request;
    request.samples = &sample;
    request.batch_size = 1;
    request.backend = bmmpy::fwht16::Fwht16Backend::cpu;

    const auto response = engine.run(request);
    require(response.actual_backend == bmmpy::fwht16::Fwht16Backend::cpu, "cpu route");
}

void test_engine_auto_route_falls_back_to_cpu() {
    bmmpy::fwht16::ColumnMasks16 sample{};
    bmmpy::fwht16::Fwht16Engine engine;
    bmmpy::fwht16::Fwht16BatchRequest request;
    request.samples = &sample;
    request.batch_size = 1;
    request.backend = bmmpy::fwht16::Fwht16Backend::auto_select;

    const auto response = engine.run(request);
    require(response.actual_backend == bmmpy::fwht16::Fwht16Backend::cpu, "auto route");
}

void test_engine_gpu_route_throws_when_unavailable() {
    bmmpy::fwht16::ColumnMasks16 sample{};
    bmmpy::fwht16::Fwht16Engine engine;
    bmmpy::fwht16::Fwht16BatchRequest request;
    request.samples = &sample;
    request.batch_size = 1;
    request.backend = bmmpy::fwht16::Fwht16Backend::gpu;

    expect_throw<std::runtime_error>([&] { (void)engine.run(request); }, "gpu unavailable");
}

struct TestCase {
    const char* name;
    void (*fn)();
};

} // namespace

int main() {
    const TestCase tests[] = {
        {"constants_are_stable", &test_constants_are_stable},
        {"cpu_stub_returns_empty_topk_layout", &test_cpu_stub_returns_empty_topk_layout},
        {"cpu_stub_returns_zeroed_spectrum_layout", &test_cpu_stub_returns_zeroed_spectrum_layout},
        {"engine_rejects_null_samples_for_nonzero_batch",
         &test_engine_rejects_null_samples_for_nonzero_batch},
        {"engine_rejects_zero_topk_in_topk_mode", &test_engine_rejects_zero_topk_in_topk_mode},
        {"engine_cpu_route_uses_cpu_backend", &test_engine_cpu_route_uses_cpu_backend},
        {"engine_auto_route_falls_back_to_cpu", &test_engine_auto_route_falls_back_to_cpu},
        {"engine_gpu_route_throws_when_unavailable",
         &test_engine_gpu_route_throws_when_unavailable},
    };

    for (const TestCase& test : tests) {
        try {
            test.fn();
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& ex) {
            std::cerr << "[FAIL] " << test.name << ": " << ex.what() << '\n';
            return 1;
        }
    }

    std::cout << "All fwht16 tests passed\n";
    return 0;
}
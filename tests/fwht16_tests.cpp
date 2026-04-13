#include "bmmpy/fwht16/constants.hpp"
#include "bmmpy/fwht16/cpu_dispatch.hpp"
#include "bmmpy/fwht16/engine.hpp"
#include "bmmpy/fwht16/types.hpp"
#include "bmmpy/math/fwht.hpp"

#include <algorithm>
#include <cstddef>
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

bool topk_item_less(const bmmpy::fwht16::Fwht16TopKItem& lhs,
                    const bmmpy::fwht16::Fwht16TopKItem& rhs) noexcept {
    if (lhs.weight != rhs.weight)
        return lhs.weight < rhs.weight;

    return lhs.mask < rhs.mask;
}

bmmpy::fwht16::Fwht16CpuBackend expected_auto_cpu_backend() {
    return bmmpy::fwht16::resolve_cpu_dispatch(bmmpy::fwht16::Fwht16CpuBackend::auto_select)
        .backend;
}

bool avx2_available() noexcept {
    try {
        return bmmpy::fwht16::resolve_cpu_dispatch(bmmpy::fwht16::Fwht16CpuBackend::avx2).backend ==
               bmmpy::fwht16::Fwht16CpuBackend::avx2;
    } catch (const std::exception&) {
        return false;
    }
}

bmmpy::fwht16::ColumnMasks16 make_sample(std::uint32_t seed) {
    bmmpy::fwht16::ColumnMasks16 sample{};
    for (std::size_t i = 0; i < bmmpy::fwht16::Fwht16Constants::k_cols; ++i) {
        sample.masks[i] =
            static_cast<std::uint16_t>((seed + i * 73u + (i % 11u) * 4099u) & 0xFFFFu);
    }
    return sample;
}

bmmpy::fwht16::ColumnMasks16 make_repeated_sample(std::uint16_t mask) {
    bmmpy::fwht16::ColumnMasks16 sample{};
    sample.masks.fill(mask);
    return sample;
}

std::vector<std::int16_t> make_reference_spectrum(const bmmpy::fwht16::ColumnMasks16& sample) {
    std::vector<std::int16_t> spectrum(bmmpy::fwht16::Fwht16Constants::k_spectrum_size, 0);

    for (std::uint16_t mask : sample.masks)
        ++spectrum[mask];

    bmmpy::fwht_inplace(spectrum.data(), spectrum.size());
    return spectrum;
}

std::vector<bmmpy::fwht16::Fwht16TopKItem>
make_reference_topk(const std::vector<std::int16_t>& spectrum, std::size_t k) {
    std::vector<bmmpy::fwht16::Fwht16TopKItem> items;
    items.reserve(bmmpy::fwht16::Fwht16Constants::k_spectrum_size - 1);

    for (std::size_t mask = 1; mask < bmmpy::fwht16::Fwht16Constants::k_spectrum_size; ++mask) {
        const int w2 = static_cast<int>(bmmpy::fwht16::Fwht16Constants::k_max_weight) -
                       static_cast<int>(spectrum[mask]);

        require(w2 >= 0 && (w2 & 1) == 0, "reference topk weight parity");

        items.push_back({
            static_cast<std::uint16_t>(mask),
            static_cast<std::uint16_t>(w2 / 2),
        });
    }

    std::sort(items.begin(), items.end(), topk_item_less);
    items.resize(k);
    return items;
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

void test_cpu_auto_select_uses_best_available_backend() {
    bmmpy::fwht16::ColumnMasks16 sample{};
    bmmpy::fwht16::Fwht16Engine engine;

    bmmpy::fwht16::Fwht16BatchRequest request;
    request.samples = &sample;
    request.batch_size = 1;
    request.backend = bmmpy::fwht16::Fwht16Backend::cpu;
    request.cpu_backend = bmmpy::fwht16::Fwht16CpuBackend::auto_select;
    request.mode = bmmpy::fwht16::Fwht16ResultMode::topk;
    request.topk = 4;

    const auto response = engine.run(request);

    require(response.actual_backend == bmmpy::fwht16::Fwht16Backend::cpu, "auto actual_backend");
    require(response.actual_cpu_backend == expected_auto_cpu_backend(), "auto actual_cpu_backend");
    require(response.topk_results.size() == 4, "auto topk_results size");
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

void test_engine_rejects_topk_too_large() {
    bmmpy::fwht16::ColumnMasks16 sample{};
    bmmpy::fwht16::Fwht16Engine engine;
    bmmpy::fwht16::Fwht16BatchRequest request;
    request.samples = &sample;
    request.batch_size = 1;
    request.mode = bmmpy::fwht16::Fwht16ResultMode::topk;
    request.topk = bmmpy::fwht16::Fwht16Constants::k_spectrum_size;

    expect_throw<std::invalid_argument>([&] { (void)engine.run(request); }, "topk too large");
}

void test_engine_cpu_route_uses_cpu_backend() {
    bmmpy::fwht16::ColumnMasks16 sample{};
    bmmpy::fwht16::Fwht16Engine engine;
    bmmpy::fwht16::Fwht16BatchRequest request;
    request.samples = &sample;
    request.batch_size = 1;
    request.backend = bmmpy::fwht16::Fwht16Backend::cpu;
    request.cpu_backend = bmmpy::fwht16::Fwht16CpuBackend::scalar;

    const auto response = engine.run(request);
    require(response.actual_backend == bmmpy::fwht16::Fwht16Backend::cpu, "cpu route");
    require(response.actual_cpu_backend == bmmpy::fwht16::Fwht16CpuBackend::scalar,
            "cpu route backend");
}

void test_engine_auto_route_uses_best_available_cpu_backend() {
    bmmpy::fwht16::ColumnMasks16 sample{};
    bmmpy::fwht16::Fwht16Engine engine;
    bmmpy::fwht16::Fwht16BatchRequest request;
    request.samples = &sample;
    request.batch_size = 1;
    request.backend = bmmpy::fwht16::Fwht16Backend::auto_select;
    request.cpu_backend = bmmpy::fwht16::Fwht16CpuBackend::auto_select;

    const auto response = engine.run(request);
    require(response.actual_backend == bmmpy::fwht16::Fwht16Backend::cpu, "auto route");
    require(response.actual_cpu_backend == expected_auto_cpu_backend(), "auto cpu route");
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

void test_cpu_avx2_route_matches_availability() {
    bmmpy::fwht16::ColumnMasks16 sample{};
    bmmpy::fwht16::Fwht16Engine engine;
    bmmpy::fwht16::Fwht16BatchRequest request;
    request.samples = &sample;
    request.batch_size = 1;
    request.backend = bmmpy::fwht16::Fwht16Backend::cpu;
    request.cpu_backend = bmmpy::fwht16::Fwht16CpuBackend::avx2;
    request.mode = bmmpy::fwht16::Fwht16ResultMode::topk;
    request.topk = 4;

    if (avx2_available()) {
        const auto response = engine.run(request);
        require(response.actual_backend == bmmpy::fwht16::Fwht16Backend::cpu,
                "avx2 actual_backend");
        require(response.actual_cpu_backend == bmmpy::fwht16::Fwht16CpuBackend::avx2,
                "avx2 actual_cpu_backend");
        require(response.topk_results.size() == 4, "avx2 topk size");
        return;
    }

    expect_throw<std::runtime_error>([&] { (void)engine.run(request); }, "avx2 unavailable");
}

void test_cpu_avx512_route_throws_when_unimplemented() {
    bmmpy::fwht16::ColumnMasks16 sample{};
    bmmpy::fwht16::Fwht16Engine engine;
    bmmpy::fwht16::Fwht16BatchRequest request;
    request.samples = &sample;
    request.batch_size = 1;
    request.backend = bmmpy::fwht16::Fwht16Backend::cpu;
    request.cpu_backend = bmmpy::fwht16::Fwht16CpuBackend::avx512;

    expect_throw<std::runtime_error>([&] { (void)engine.run(request); }, "avx512 unavailable");
}

void test_cpu_scalar_spectrum_matches_reference_fwht() {
    const auto sample = make_sample(17u);

    bmmpy::fwht16::Fwht16Engine engine;
    bmmpy::fwht16::Fwht16BatchRequest request;
    request.samples = &sample;
    request.batch_size = 1;
    request.backend = bmmpy::fwht16::Fwht16Backend::cpu;
    request.cpu_backend = bmmpy::fwht16::Fwht16CpuBackend::scalar;
    request.mode = bmmpy::fwht16::Fwht16ResultMode::spectrum;

    const auto response = engine.run(request);
    const auto expected = make_reference_spectrum(sample);

    require(response.spectra.size() == expected.size(), "spectrum size");
    for (std::size_t i = 0; i < expected.size(); ++i)
        require(response.spectra[i] == expected[i], "spectrum value");
}

void test_cpu_scalar_batched_spectrum_matches_reference_fwht() {
    const bmmpy::fwht16::ColumnMasks16 samples[2] = {
        make_sample(5u),
        make_sample(123u),
    };

    bmmpy::fwht16::Fwht16Engine engine;
    bmmpy::fwht16::Fwht16BatchRequest request;
    request.samples = samples;
    request.batch_size = 2;
    request.backend = bmmpy::fwht16::Fwht16Backend::cpu;
    request.cpu_backend = bmmpy::fwht16::Fwht16CpuBackend::scalar;
    request.mode = bmmpy::fwht16::Fwht16ResultMode::spectrum;

    const auto response = engine.run(request);

    for (std::size_t sample_index = 0; sample_index < 2; ++sample_index) {
        const auto expected = make_reference_spectrum(samples[sample_index]);
        const std::size_t base = sample_index * bmmpy::fwht16::Fwht16Constants::k_spectrum_size;

        for (std::size_t i = 0; i < expected.size(); ++i)
            require(response.spectra[base + i] == expected[i], "batched spectrum value");
    }
}

void test_cpu_scalar_topk_for_zero_sample() {
    bmmpy::fwht16::ColumnMasks16 sample{};
    bmmpy::fwht16::Fwht16Engine engine;

    bmmpy::fwht16::Fwht16BatchRequest request;
    request.samples = &sample;
    request.batch_size = 1;
    request.backend = bmmpy::fwht16::Fwht16Backend::cpu;
    request.cpu_backend = bmmpy::fwht16::Fwht16CpuBackend::scalar;
    request.mode = bmmpy::fwht16::Fwht16ResultMode::topk;
    request.topk = 8;

    const auto response = engine.run(request);
    const auto expected = make_reference_topk(make_reference_spectrum(sample), request.topk);

    require(response.topk_results.size() == expected.size(), "topk_results size");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        require(response.topk_results[i].mask == expected[i].mask, "topk mask");
        require(response.topk_results[i].weight == expected[i].weight, "topk weight");
    }
}

void test_cpu_scalar_spectrum_for_zero_sample() {
    bmmpy::fwht16::ColumnMasks16 samples[2]{};
    bmmpy::fwht16::Fwht16Engine engine;

    bmmpy::fwht16::Fwht16BatchRequest request;
    request.samples = samples;
    request.batch_size = 2;
    request.backend = bmmpy::fwht16::Fwht16Backend::cpu;
    request.cpu_backend = bmmpy::fwht16::Fwht16CpuBackend::scalar;
    request.mode = bmmpy::fwht16::Fwht16ResultMode::spectrum;

    const auto response = engine.run(request);

    for (std::size_t sample_index = 0; sample_index < 2; ++sample_index) {
        const auto expected = make_reference_spectrum(samples[sample_index]);
        const std::size_t base = sample_index * bmmpy::fwht16::Fwht16Constants::k_spectrum_size;

        for (std::size_t i = 0; i < expected.size(); ++i)
            require(response.spectra[base + i] == expected[i], "zero-sample spectrum value");
    }
}

void test_cpu_scalar_topk_matches_reference_scan_general() {
    const auto sample = make_sample(91u);

    bmmpy::fwht16::Fwht16Engine engine;
    bmmpy::fwht16::Fwht16BatchRequest request;
    request.samples = &sample;
    request.batch_size = 1;
    request.backend = bmmpy::fwht16::Fwht16Backend::cpu;
    request.cpu_backend = bmmpy::fwht16::Fwht16CpuBackend::scalar;
    request.mode = bmmpy::fwht16::Fwht16ResultMode::topk;
    request.topk = 16;

    const auto response = engine.run(request);
    const auto expected = make_reference_topk(make_reference_spectrum(sample), request.topk);

    require(response.topk_results.size() == expected.size(), "general topk size");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        require(response.topk_results[i].mask == expected[i].mask, "general topk mask");
        require(response.topk_results[i].weight == expected[i].weight, "general topk weight");
    }
}

void test_cpu_scalar_batched_topk_matches_reference_scan() {
    const bmmpy::fwht16::ColumnMasks16 samples[2] = {
        make_sample(5u),
        make_sample(123u),
    };

    bmmpy::fwht16::Fwht16Engine engine;
    bmmpy::fwht16::Fwht16BatchRequest request;
    request.samples = samples;
    request.batch_size = 2;
    request.backend = bmmpy::fwht16::Fwht16Backend::cpu;
    request.cpu_backend = bmmpy::fwht16::Fwht16CpuBackend::scalar;
    request.mode = bmmpy::fwht16::Fwht16ResultMode::topk;
    request.topk = 8;

    const auto response = engine.run(request);

    require(response.topk_results.size() == 2 * request.topk, "batched topk size");

    for (std::size_t sample_index = 0; sample_index < 2; ++sample_index) {
        const auto expected =
            make_reference_topk(make_reference_spectrum(samples[sample_index]), request.topk);
        const std::size_t base = sample_index * request.topk;

        for (std::size_t i = 0; i < request.topk; ++i) {
            require(response.topk_results[base + i].mask == expected[i].mask, "batched topk mask");
            require(response.topk_results[base + i].weight == expected[i].weight,
                    "batched topk weight");
        }
    }
}

void test_cpu_scalar_topk_one_matches_reference() {
    const auto sample = make_sample(777u);

    bmmpy::fwht16::Fwht16Engine engine;
    bmmpy::fwht16::Fwht16BatchRequest request;
    request.samples = &sample;
    request.batch_size = 1;
    request.backend = bmmpy::fwht16::Fwht16Backend::cpu;
    request.cpu_backend = bmmpy::fwht16::Fwht16CpuBackend::scalar;
    request.mode = bmmpy::fwht16::Fwht16ResultMode::topk;
    request.topk = 1;

    const auto response = engine.run(request);
    const auto expected = make_reference_topk(make_reference_spectrum(sample), 1);

    require(response.topk_results.size() == 1, "topk=1 size");
    require(response.topk_results[0].mask == expected[0].mask, "topk=1 mask");
    require(response.topk_results[0].weight == expected[0].weight, "topk=1 weight");
}

void test_cpu_scalar_repeated_mask_best_has_zero_weight() {
    const auto sample = make_repeated_sample(0x1234u);

    bmmpy::fwht16::Fwht16Engine engine;
    bmmpy::fwht16::Fwht16BatchRequest request;
    request.samples = &sample;
    request.batch_size = 1;
    request.backend = bmmpy::fwht16::Fwht16Backend::cpu;
    request.cpu_backend = bmmpy::fwht16::Fwht16CpuBackend::scalar;
    request.mode = bmmpy::fwht16::Fwht16ResultMode::topk;
    request.topk = 4;

    const auto response = engine.run(request);
    const auto expected = make_reference_topk(make_reference_spectrum(sample), request.topk);

    require(response.topk_results.size() == request.topk, "repeated-mask topk size");
    require(response.topk_results[0].weight == 0, "repeated-mask best weight");

    for (std::size_t i = 0; i < request.topk; ++i) {
        require(response.topk_results[i].mask == expected[i].mask, "repeated-mask topk mask");
        require(response.topk_results[i].weight == expected[i].weight, "repeated-mask topk weight");
    }
}

void test_cpu_scalar_repeated_mask_spectrum_matches_reference() {
    const auto sample = make_repeated_sample(0x00A5u);

    bmmpy::fwht16::Fwht16Engine engine;
    bmmpy::fwht16::Fwht16BatchRequest request;
    request.samples = &sample;
    request.batch_size = 1;
    request.backend = bmmpy::fwht16::Fwht16Backend::cpu;
    request.cpu_backend = bmmpy::fwht16::Fwht16CpuBackend::scalar;
    request.mode = bmmpy::fwht16::Fwht16ResultMode::spectrum;

    const auto response = engine.run(request);
    const auto expected = make_reference_spectrum(sample);

    require(response.spectra.size() == expected.size(), "repeated-mask spectrum size");
    for (std::size_t i = 0; i < expected.size(); ++i)
        require(response.spectra[i] == expected[i], "repeated-mask spectrum value");
}

void test_cpu_avx2_spectrum_matches_scalar() {
    if (!avx2_available())
        return;

    const bmmpy::fwht16::ColumnMasks16 samples[2] = {
        make_sample(41u),
        make_sample(203u),
    };

    bmmpy::fwht16::Fwht16Engine engine;

    bmmpy::fwht16::Fwht16BatchRequest scalar_request;
    scalar_request.samples = samples;
    scalar_request.batch_size = 2;
    scalar_request.backend = bmmpy::fwht16::Fwht16Backend::cpu;
    scalar_request.cpu_backend = bmmpy::fwht16::Fwht16CpuBackend::scalar;
    scalar_request.mode = bmmpy::fwht16::Fwht16ResultMode::spectrum;

    bmmpy::fwht16::Fwht16BatchRequest avx2_request = scalar_request;
    avx2_request.cpu_backend = bmmpy::fwht16::Fwht16CpuBackend::avx2;

    const auto scalar_response = engine.run(scalar_request);
    const auto avx2_response = engine.run(avx2_request);

    require(scalar_response.spectra.size() == avx2_response.spectra.size(), "avx2 spectrum size");
    for (std::size_t i = 0; i < scalar_response.spectra.size(); ++i) {
        require(scalar_response.spectra[i] == avx2_response.spectra[i], "avx2 spectrum value");
    }
}

void test_cpu_avx2_topk_matches_scalar() {
    if (!avx2_available())
        return;

    const bmmpy::fwht16::ColumnMasks16 samples[2] = {
        make_sample(7u),
        make_sample(311u),
    };

    bmmpy::fwht16::Fwht16Engine engine;

    bmmpy::fwht16::Fwht16BatchRequest scalar_request;
    scalar_request.samples = samples;
    scalar_request.batch_size = 2;
    scalar_request.backend = bmmpy::fwht16::Fwht16Backend::cpu;
    scalar_request.cpu_backend = bmmpy::fwht16::Fwht16CpuBackend::scalar;
    scalar_request.mode = bmmpy::fwht16::Fwht16ResultMode::topk;
    scalar_request.topk = 16;

    bmmpy::fwht16::Fwht16BatchRequest avx2_request = scalar_request;
    avx2_request.cpu_backend = bmmpy::fwht16::Fwht16CpuBackend::avx2;

    const auto scalar_response = engine.run(scalar_request);
    const auto avx2_response = engine.run(avx2_request);

    require(scalar_response.topk_results.size() == avx2_response.topk_results.size(),
            "avx2 topk size");

    for (std::size_t i = 0; i < scalar_response.topk_results.size(); ++i) {
        require(scalar_response.topk_results[i].mask == avx2_response.topk_results[i].mask,
                "avx2 topk mask");
        require(scalar_response.topk_results[i].weight == avx2_response.topk_results[i].weight,
                "avx2 topk weight");
    }
}

struct TestCase {
    const char* name;
    void (*fn)();
};

} // namespace

int main() {
    const TestCase tests[] = {
        {"constants_are_stable", &test_constants_are_stable},
        {"cpu_auto_select_uses_best_available_backend",
         &test_cpu_auto_select_uses_best_available_backend},
        {"engine_rejects_null_samples_for_nonzero_batch",
         &test_engine_rejects_null_samples_for_nonzero_batch},
        {"engine_rejects_zero_topk_in_topk_mode", &test_engine_rejects_zero_topk_in_topk_mode},
        {"engine_cpu_route_uses_cpu_backend", &test_engine_cpu_route_uses_cpu_backend},
        {"engine_auto_route_uses_best_available_cpu_backend",
         &test_engine_auto_route_uses_best_available_cpu_backend},
        {"engine_gpu_route_throws_when_unavailable",
         &test_engine_gpu_route_throws_when_unavailable},
        {"cpu_avx2_route_matches_availability", &test_cpu_avx2_route_matches_availability},
        {"cpu_avx512_route_throws_when_unimplemented",
         &test_cpu_avx512_route_throws_when_unimplemented},
        {"engine_rejects_topk_too_large", &test_engine_rejects_topk_too_large},
        {"cpu_scalar_spectrum_matches_reference_fwht",
         &test_cpu_scalar_spectrum_matches_reference_fwht},
        {"cpu_scalar_batched_spectrum_matches_reference_fwht",
         &test_cpu_scalar_batched_spectrum_matches_reference_fwht},
        {"cpu_scalar_topk_for_zero_sample", &test_cpu_scalar_topk_for_zero_sample},
        {"cpu_scalar_spectrum_for_zero_sample", &test_cpu_scalar_spectrum_for_zero_sample},
        {"cpu_scalar_topk_matches_reference_scan_general",
         &test_cpu_scalar_topk_matches_reference_scan_general},
        {"cpu_scalar_batched_topk_matches_reference_scan",
         &test_cpu_scalar_batched_topk_matches_reference_scan},
        {"cpu_scalar_topk_one_matches_reference", &test_cpu_scalar_topk_one_matches_reference},
        {"cpu_scalar_repeated_mask_best_has_zero_weight",
         &test_cpu_scalar_repeated_mask_best_has_zero_weight},
        {"cpu_scalar_repeated_mask_spectrum_matches_reference",
         &test_cpu_scalar_repeated_mask_spectrum_matches_reference},
        {"cpu_avx2_spectrum_matches_scalar", &test_cpu_avx2_spectrum_matches_scalar},
        {"cpu_avx2_topk_matches_scalar", &test_cpu_avx2_topk_matches_scalar},
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
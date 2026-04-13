#include "bmmpy/fwht16/engine.hpp"
#include "bmmpy/fwht16/types.hpp"

#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

constexpr std::size_t k_topk = 16;
constexpr std::uint64_t k_fnv_off = 1469598103934665603ull;
constexpr std::uint64_t k_fnv_prime = 1099511628211ull;

enum class DatasetKind {
    random,
    repeated,
};

struct RunOutcome {
    bmmpy::fwht16::Fwht16CpuBackend backend = bmmpy::fwht16::Fwht16CpuBackend::scalar;
    std::uint64_t checksum = 0;
};

const char* cpu_backend_name(bmmpy::fwht16::Fwht16CpuBackend backend) noexcept {
    switch (backend) {
    case bmmpy::fwht16::Fwht16CpuBackend::auto_select:
        return "auto";
    case bmmpy::fwht16::Fwht16CpuBackend::scalar:
        return "scalar";
    case bmmpy::fwht16::Fwht16CpuBackend::avx2:
        return "avx2";
    case bmmpy::fwht16::Fwht16CpuBackend::avx512:
        return "avx512";
    }
    return "unknown";
}

bmmpy::fwht16::ColumnMasks16 make_random_sample(std::uint32_t seed) {
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

std::vector<bmmpy::fwht16::ColumnMasks16> make_dataset(std::size_t batch_size,
                                                       DatasetKind dataset_kind) {
    std::vector<bmmpy::fwht16::ColumnMasks16> samples;
    samples.reserve(batch_size);

    for (std::size_t i = 0; i < batch_size; ++i) {
        if (dataset_kind == DatasetKind::random) {
            samples.push_back(make_random_sample(static_cast<std::uint32_t>(17u + i * 101u)));
        } else {
            samples.push_back(
                make_repeated_sample(static_cast<std::uint16_t>((0x1234u + i * 257u) & 0xFFFFu)));
        }
    }

    return samples;
}

std::uint64_t checksum_response(const bmmpy::fwht16::Fwht16BatchResponse& response) noexcept {
    std::uint64_t hash = k_fnv_off;

    for (const auto& item : response.topk_results) {
        const std::uint64_t packed =
            (static_cast<std::uint64_t>(item.mask) << 32) | static_cast<std::uint64_t>(item.weight);
        hash ^= packed;
        hash *= k_fnv_prime;
    }

    hash ^= static_cast<std::uint64_t>(response.actual_cpu_backend);
    hash *= k_fnv_prime;
    return hash;
}

RunOutcome run_batch_once(bmmpy::fwht16::Fwht16Engine& engine,
                          const std::vector<bmmpy::fwht16::ColumnMasks16>& samples,
                          bmmpy::fwht16::Fwht16CpuBackend cpu_backend) {
    bmmpy::fwht16::Fwht16BatchRequest request;
    request.samples = samples.data();
    request.batch_size = samples.size();
    request.backend = bmmpy::fwht16::Fwht16Backend::cpu;
    request.cpu_backend = cpu_backend;
    request.mode = bmmpy::fwht16::Fwht16ResultMode::topk;
    request.topk = k_topk;

    const auto response = engine.run(request);
    return RunOutcome{response.actual_cpu_backend, checksum_response(response)};
}

RunOutcome run_single_once(bmmpy::fwht16::Fwht16Engine& engine,
                           const std::vector<bmmpy::fwht16::ColumnMasks16>& samples,
                           bmmpy::fwht16::Fwht16CpuBackend cpu_backend) {
    RunOutcome outcome;
    outcome.checksum = k_fnv_off;

    for (std::size_t i = 0; i < samples.size(); ++i) {
        bmmpy::fwht16::Fwht16BatchRequest request;
        request.samples = &samples[i];
        request.batch_size = 1;
        request.backend = bmmpy::fwht16::Fwht16Backend::cpu;
        request.cpu_backend = cpu_backend;
        request.mode = bmmpy::fwht16::Fwht16ResultMode::topk;
        request.topk = k_topk;

        const auto response = engine.run(request);
        if (i == 0)
            outcome.backend = response.actual_cpu_backend;

        outcome.checksum ^= checksum_response(response);
        outcome.checksum *= k_fnv_prime;
    }

    return outcome;
}

template <typename Fn>
void run_benchmark(benchmark::State& state, std::size_t samples_per_iteration, Fn&& fn) {
    RunOutcome warmup{};
    try {
        warmup = fn();
    } catch (const std::exception& ex) {
        state.SkipWithError(ex.what());
        return;
    }

    state.SetLabel(cpu_backend_name(warmup.backend));

    std::uint64_t checksum = warmup.checksum;
    benchmark::DoNotOptimize(checksum);

    for (auto _ : state) {
        const RunOutcome outcome = fn();
        checksum ^= outcome.checksum;
        benchmark::DoNotOptimize(checksum);
    }

    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) *
                            static_cast<std::int64_t>(samples_per_iteration));
    state.counters["samples/s"] = benchmark::Counter(static_cast<double>(samples_per_iteration),
                                                     benchmark::Counter::kIsIterationInvariantRate);
    benchmark::DoNotOptimize(checksum);
}

void bm_fwht16_scalar_single_random(benchmark::State& state) {
    const auto batch_size = static_cast<std::size_t>(state.range(0));
    auto samples = make_dataset(batch_size, DatasetKind::random);
    bmmpy::fwht16::Fwht16Engine engine;

    run_benchmark(state, batch_size, [&] {
        return run_single_once(engine, samples, bmmpy::fwht16::Fwht16CpuBackend::scalar);
    });
}

void bm_fwht16_scalar_batch_random(benchmark::State& state) {
    const auto batch_size = static_cast<std::size_t>(state.range(0));
    auto samples = make_dataset(batch_size, DatasetKind::random);
    bmmpy::fwht16::Fwht16Engine engine;

    run_benchmark(state, batch_size, [&] {
        return run_batch_once(engine, samples, bmmpy::fwht16::Fwht16CpuBackend::scalar);
    });
}

void bm_fwht16_avx2_batch_random(benchmark::State& state) {
    const auto batch_size = static_cast<std::size_t>(state.range(0));
    auto samples = make_dataset(batch_size, DatasetKind::random);
    bmmpy::fwht16::Fwht16Engine engine;

    run_benchmark(state, batch_size, [&] {
        return run_batch_once(engine, samples, bmmpy::fwht16::Fwht16CpuBackend::avx2);
    });
}

void bm_fwht16_scalar_batch_repeated(benchmark::State& state) {
    const auto batch_size = static_cast<std::size_t>(state.range(0));
    auto samples = make_dataset(batch_size, DatasetKind::repeated);
    bmmpy::fwht16::Fwht16Engine engine;

    run_benchmark(state, batch_size, [&] {
        return run_batch_once(engine, samples, bmmpy::fwht16::Fwht16CpuBackend::scalar);
    });
}

void bm_fwht16_avx2_batch_repeated(benchmark::State& state) {
    const auto batch_size = static_cast<std::size_t>(state.range(0));
    auto samples = make_dataset(batch_size, DatasetKind::repeated);
    bmmpy::fwht16::Fwht16Engine engine;

    run_benchmark(state, batch_size, [&] {
        return run_batch_once(engine, samples, bmmpy::fwht16::Fwht16CpuBackend::avx2);
    });
}

} // namespace

BENCHMARK(bm_fwht16_scalar_single_random)->Arg(1)->Arg(32)->Arg(2048);
BENCHMARK(bm_fwht16_scalar_batch_random)->Arg(1)->Arg(32)->Arg(2048);
BENCHMARK(bm_fwht16_avx2_batch_random)->Arg(1)->Arg(32)->Arg(2048);

BENCHMARK(bm_fwht16_scalar_batch_repeated)->Arg(1)->Arg(32)->Arg(2048);
BENCHMARK(bm_fwht16_avx2_batch_repeated)->Arg(1)->Arg(32)->Arg(2048);

BENCHMARK_MAIN();
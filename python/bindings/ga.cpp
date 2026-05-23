#include "bindings.hpp"
#include "bmmpy/ga/genetic_algorithm.hpp"
#include "bmmpy/ga/island_model.hpp"
#include "bmmpy/ga/migration/bounded_channel.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace {
namespace ga = ::bmmpy::ga;
namespace migration = ::bmmpy::ga::migration;

ga::IslandModel make_island_model(ga::IslandModelConfig config,
                                  std::vector<ga::GeneticAlgorithmConfig> ga_configs,
                                  std::size_t channel_capacity,
                                  migration::OverflowPolicy overflow_policy) {
    if (config.islands.empty()) {
        throw std::invalid_argument("IslandModel: islands must not be empty");
    }

    if (ga_configs.size() != config.islands.size()) {
        throw std::invalid_argument("IslandModel: ga_configs size must match config.islands size");
    }

    if (channel_capacity == 0) {
        throw std::invalid_argument("IslandModel: channel_capacity must be greater than zero");
    }

    std::vector<std::pair<std::size_t, ga::GeneticAlgorithmConfig>> configs_by_island;
    configs_by_island.reserve(config.islands.size());

    for (std::size_t index = 0; index < config.islands.size(); ++index) {
        const std::size_t island_id = config.islands[index].island_id;
        const bool duplicate_id =
            std::any_of(configs_by_island.begin(),
                        configs_by_island.end(),
                        [island_id](const auto& entry) { return entry.first == island_id; });

        if (duplicate_id) {
            throw std::invalid_argument("IslandModel: island_id values must be unique");
        }

        configs_by_island.emplace_back(island_id, ga_configs[index]);
    }

    ga::AlgorithmFactory factory =
        [configs_by_island = std::move(configs_by_island)](
            const ga::IslandSpec& spec) -> std::unique_ptr<ga::Algorithm> {
        const auto it =
            std::find_if(configs_by_island.begin(),
                         configs_by_island.end(),
                         [&spec](const auto& entry) { return entry.first == spec.island_id; });

        if (it == configs_by_island.end()) {
            throw std::invalid_argument(
                "IslandModel: missing GeneticAlgorithmConfig for island_id=" +
                std::to_string(spec.island_id));
        }

        return std::make_unique<ga::GeneticAlgorithm>(it->second);
    };

    auto channel = std::make_unique<migration::BoundedChannel>(channel_capacity, overflow_policy);

    return ga::IslandModel(std::move(config), std::move(factory), std::move(channel));
}

} // namespace

namespace bmmpy::bindings {

void bind_ga(nb::module_& m) {
    nb::enum_<migration::OverflowPolicy>(m, "OverflowPolicy")
        .value("DropOldest", migration::OverflowPolicy::DropOldest)
        .value("DropNewest", migration::OverflowPolicy::DropNewest)
        .value("RejectPublish", migration::OverflowPolicy::RejectPublish);

    nb::class_<ga::StopCriteria>(m, "StopCriteria")
        .def(nb::init<>())
        .def_rw("max_generations", &ga::StopCriteria::max_generations)
        .def_rw("max_stale_generations", &ga::StopCriteria::max_stale_generations)
        .def_rw("target_total_weight", &ga::StopCriteria::target_total_weight);

    nb::class_<ga::RunStats>(m, "RunStats")
        .def(nb::init<>())
        .def_rw("generations", &ga::RunStats::generations)
        .def_rw("evaluations", &ga::RunStats::evaluations)
        .def_rw("stale_generations", &ga::RunStats::stale_generations)
        .def_rw("seed", &ga::RunStats::seed);

    nb::class_<ga::MigrationPolicy>(m, "MigrationPolicy")
        .def(nb::init<>())
        .def_rw("interval_generations", &ga::MigrationPolicy::interval_generations)
        .def_rw("export_count", &ga::MigrationPolicy::export_count)
        .def_rw("import_count", &ga::MigrationPolicy::import_count)
        .def_rw("shared_pool_capacity", &ga::MigrationPolicy::shared_pool_capacity);

    nb::class_<ga::IslandSpec>(m, "IslandSpec")
        .def(nb::init<>())
        .def_rw("island_id", &ga::IslandSpec::island_id)
        .def_rw("migration", &ga::IslandSpec::migration);

    nb::class_<ga::IslandSnapshot>(m, "IslandSnapshot")
        .def(nb::init<>())
        .def_rw("island_id", &ga::IslandSnapshot::island_id)
        .def_rw("running", &ga::IslandSnapshot::running)
        .def_rw("stop_requested", &ga::IslandSnapshot::stop_requested)
        .def_rw("finished", &ga::IslandSnapshot::finished)
        .def_rw("stats", &ga::IslandSnapshot::stats)
        .def_rw("best_score", &ga::IslandSnapshot::best_score)
        .def_rw("best_individual", &ga::IslandSnapshot::best_individual);

    nb::class_<ga::IslandModelSnapshot>(m, "IslandModelSnapshot")
        .def(nb::init<>())
        .def_rw("running", &ga::IslandModelSnapshot::running)
        .def_rw("stop_requested", &ga::IslandModelSnapshot::stop_requested)
        .def_rw("island_count", &ga::IslandModelSnapshot::island_count)
        .def_rw("total_generations", &ga::IslandModelSnapshot::total_generations)
        .def_rw("best_score", &ga::IslandModelSnapshot::best_score)
        .def_rw("best_individual", &ga::IslandModelSnapshot::best_individual)
        .def_rw("islands", &ga::IslandModelSnapshot::islands);

    nb::class_<ga::GeneticAlgorithmConfig>(m, "GeneticAlgorithmConfig")
        .def(nb::init<>())
        .def_rw("population_size", &ga::GeneticAlgorithmConfig::population_size)
        .def_rw("elite_count", &ga::GeneticAlgorithmConfig::elite_count)
        .def_rw("tournament_size", &ga::GeneticAlgorithmConfig::tournament_size)
        .def_rw("num_parents", &ga::GeneticAlgorithmConfig::num_parents)
        .def_rw("num_offspring", &ga::GeneticAlgorithmConfig::num_offspring)
        .def_rw("enable_catastrophe", &ga::GeneticAlgorithmConfig::enable_catastrophe)
        .def_rw("catastrophe_threshold", &ga::GeneticAlgorithmConfig::catastrophe_threshold)
        .def_rw("catastrophe_survival_rate", &ga::GeneticAlgorithmConfig::catastrophe_survival_rate)
        .def_rw("mutation_rate", &ga::GeneticAlgorithmConfig::mutation_rate)
        .def_rw("stop", &ga::GeneticAlgorithmConfig::stop)
        .def_rw("seed", &ga::GeneticAlgorithmConfig::seed);

    nb::class_<ga::IslandModelConfig>(m, "IslandModelConfig")
        .def(nb::init<>())
        .def_rw("islands", &ga::IslandModelConfig::islands);

    nb::class_<ga::GeneticAlgorithm>(m, "GeneticAlgorithm")
        .def(nb::init<ga::GeneticAlgorithmConfig>(),
             nb::arg("config") = ga::GeneticAlgorithmConfig{})
        .def("name", &ga::GeneticAlgorithm::name)
        .def("initialize",
             &ga::GeneticAlgorithm::initialize,
             nb::arg("window"),
             nb::keep_alive<1, 2>())
        .def("step", &ga::GeneticAlgorithm::step, nb::call_guard<nb::gil_scoped_release>())
        .def("done", &ga::GeneticAlgorithm::done)
        .def("generation", &ga::GeneticAlgorithm::generation)
        .def("best_individual", &ga::GeneticAlgorithm::best_individual)
        .def("stats", &ga::GeneticAlgorithm::stats)
        .def("best_score", &ga::GeneticAlgorithm::best_score)
        .def("optimize",
             &ga::GeneticAlgorithm::optimize,
             nb::arg("window"),
             nb::call_guard<nb::gil_scoped_release>());

    nb::class_<ga::IslandModel>(m, "IslandModel")
        .def(
            "__init__",
            [](ga::IslandModel* self,
               ga::IslandModelConfig config,
               std::vector<ga::GeneticAlgorithmConfig> ga_configs,
               std::size_t channel_capacity,
               migration::OverflowPolicy overflow_policy) {
                new (self) ga::IslandModel(make_island_model(
                    std::move(config), std::move(ga_configs), channel_capacity, overflow_policy));
            },
            nb::arg("config"),
            nb::arg("ga_configs"),
            nb::arg("channel_capacity") = std::size_t{256},
            nb::arg("overflow_policy") = migration::OverflowPolicy::DropOldest)
        .def("name", &ga::IslandModel::name)
        .def("initialize", &ga::IslandModel::initialize, nb::arg("window"))
        .def("start", &ga::IslandModel::start)
        .def("request_stop", &ga::IslandModel::request_stop)
        .def("wait", &ga::IslandModel::wait, nb::call_guard<nb::gil_scoped_release>())
        .def("running", &ga::IslandModel::running)
        .def("stop_requested", &ga::IslandModel::stop_requested)
        .def("best_individual", &ga::IslandModel::best_individual)
        .def("snapshot", &ga::IslandModel::snapshot)
        .def("run_to_completion",
             &ga::IslandModel::run_to_completion,
             nb::arg("window"),
             nb::call_guard<nb::gil_scoped_release>());
}

} // namespace bmmpy::bindings
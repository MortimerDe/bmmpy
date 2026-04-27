#include "bindings.hpp"
#include "bmmpy/core/row_window.hpp"
#include "bmmpy/search/bruteforce_search.hpp"
#include "bmmpy/search/cuda_bruteforce_search.hpp"
#include "bmmpy/search/cuda_mitm_fwht_search.hpp"
#include "bmmpy/search/fwht_search.hpp"
#include "bmmpy/search/mitm_fwht_search.hpp"
#include "bmmpy/search/sa_selector.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace bmmpy::bindings {

void bind_search(nb::module_& m) {
    nb::class_<::bmmpy::BruteforceSearchConfig>(m, "BruteforceSearchConfig")
        .def(nb::init<>())
        .def_rw("max_candidates", &::bmmpy::BruteforceSearchConfig::max_candidates)
        .def_rw("chunk_bits", &::bmmpy::BruteforceSearchConfig::chunk_bits);

    nb::class_<::bmmpy::FwhtSearchConfig>(m, "FwhtSearchConfig")
        .def(nb::init<>())
        .def_rw("max_rows", &::bmmpy::FwhtSearchConfig::max_rows)
        .def_rw("max_candidates", &::bmmpy::FwhtSearchConfig::max_candidates);

    nb::class_<::bmmpy::MitmFwhtSearchConfig>(m, "MitmFwhtSearchConfig")
        .def(nb::init<>())
        .def_rw("reserve_unique_patterns", &::bmmpy::MitmFwhtSearchConfig::reserve_unique_patterns)
        .def_rw("reserve_left_rows", &::bmmpy::MitmFwhtSearchConfig::reserve_left_rows)
        .def_rw("reserve_right_states", &::bmmpy::MitmFwhtSearchConfig::reserve_right_states)
        .def_rw("max_candidates", &::bmmpy::MitmFwhtSearchConfig::max_candidates);

    nb::class_<::bmmpy::CudaMitmFwhtSearchConfig>(m, "CudaMitmFwhtSearchConfig")
        .def(nb::init<>())
        .def_rw("max_candidates", &::bmmpy::CudaMitmFwhtSearchConfig::max_candidates)
        .def_rw("low_bits", &::bmmpy::CudaMitmFwhtSearchConfig::low_bits);

    nb::class_<::bmmpy::BruteforceSearch>(m, "BruteforceSearch")
        .def(nb::init<::bmmpy::BruteforceSearchConfig>(),
             nb::arg("config") = ::bmmpy::BruteforceSearchConfig{})
        .def("name", &::bmmpy::BruteforceSearch::name)
        .def("search", &::bmmpy::BruteforceSearch::search, nb::arg("window"));

    nb::class_<::bmmpy::FwhtSearch>(m, "FwhtSearch")
        .def(nb::init<::bmmpy::FwhtSearchConfig>(), nb::arg("config") = ::bmmpy::FwhtSearchConfig{})
        .def("name", &::bmmpy::FwhtSearch::name)
        .def("search", &::bmmpy::FwhtSearch::search, nb::arg("window"));

    nb::class_<::bmmpy::MitmFwhtSearch>(m, "MitmFwhtSearch")
        .def(nb::init<::bmmpy::MitmFwhtSearchConfig>(),
             nb::arg("config") = ::bmmpy::MitmFwhtSearchConfig{})
        .def("name", &::bmmpy::MitmFwhtSearch::name)
        .def("search", &::bmmpy::MitmFwhtSearch::search, nb::arg("window"));

    nb::class_<::bmmpy::CudaMitmFwhtSearch>(m, "CudaMitmFwhtSearch")
        .def(nb::init<::bmmpy::CudaMitmFwhtSearchConfig>(),
             nb::arg("config") = ::bmmpy::CudaMitmFwhtSearchConfig{})
        .def("name", &::bmmpy::CudaMitmFwhtSearch::name)
        .def("search", &::bmmpy::CudaMitmFwhtSearch::search, nb::arg("window"));

    nb::class_<::bmmpy::CudaBruteforceSearchConfig>(m, "CudaBruteforceSearchConfig")
        .def(nb::init<>())
        .def_rw("max_candidates", &::bmmpy::CudaBruteforceSearchConfig::max_candidates)
        .def_rw("chunk_bits", &::bmmpy::CudaBruteforceSearchConfig::chunk_bits);
    nb::class_<::bmmpy::CudaBruteforceSearch>(m, "CudaBruteforceSearch")
        .def(nb::init<::bmmpy::CudaBruteforceSearchConfig>(),
             nb::arg("config") = ::bmmpy::CudaBruteforceSearchConfig{})
        .def("name", &::bmmpy::CudaBruteforceSearch::name)
        .def("search", &::bmmpy::CudaBruteforceSearch::search, nb::arg("window"));

    nb::enum_<::bmmpy::WindowScorePolicyKind>(m, "WindowScorePolicyKind")
        .value("PairwiseSynergy", ::bmmpy::WindowScorePolicyKind::PairwiseSynergy);

    nb::enum_<::bmmpy::CoolingPolicyKind>(m, "CoolingPolicyKind")
        .value("AdaptiveGeometric", ::bmmpy::CoolingPolicyKind::AdaptiveGeometric);

    nb::class_<::bmmpy::SASelectorConfig>(m, "SASelectorConfig")
        .def(nb::init<>())
        .def_rw("iterations", &::bmmpy::SASelectorConfig::iterations)
        .def_rw("restarts", &::bmmpy::SASelectorConfig::restarts)
        .def_rw("seed", &::bmmpy::SASelectorConfig::seed)
        .def_rw("score_policy", &::bmmpy::SASelectorConfig::score_policy)
        .def_rw("cooling_policy", &::bmmpy::SASelectorConfig::cooling_policy)
        .def_rw("temperature_probe_samples", &::bmmpy::SASelectorConfig::temperature_probe_samples)
        .def_rw("initial_acceptance_probability",
                &::bmmpy::SASelectorConfig::initial_acceptance_probability)
        .def_rw("cooling_rate", &::bmmpy::SASelectorConfig::cooling_rate)
        .def_rw("min_temperature", &::bmmpy::SASelectorConfig::min_temperature);

    nb::class_<::bmmpy::SASelectionResult>(m, "SASelectionResult")
        .def(nb::init<>())
        .def_rw("rows", &::bmmpy::SASelectionResult::rows)
        .def_rw("score", &::bmmpy::SASelectionResult::score)
        .def_rw("accepted_moves", &::bmmpy::SASelectionResult::accepted_moves)
        .def_rw("iterations_run", &::bmmpy::SASelectionResult::iterations_run)
        .def_rw("best_iteration", &::bmmpy::SASelectionResult::best_iteration)
        .def_rw("restart_index", &::bmmpy::SASelectionResult::restart_index)
        .def_rw("seed", &::bmmpy::SASelectionResult::seed);

    nb::class_<::bmmpy::SASelector>(m, "SASelector")
        .def(nb::init<::bmmpy::SASelectorConfig>(), nb::arg("config") = ::bmmpy::SASelectorConfig{})
        .def("name", &::bmmpy::SASelector::name)
        .def("select", &::bmmpy::SASelector::select, nb::arg("matrix"), nb::arg("window_size"))
        .def(
            "select_window",
            [](::bmmpy::SASelector& selector, ::bmmpy::BitMatrix& matrix, std::size_t window_size) {
                return selector.select_window(matrix, window_size);
            },
            nb::arg("matrix"),
            nb::arg("window_size"),
            nb::keep_alive<0, 2>(),
            nb::rv_policy::move);
}

} // namespace bmmpy::bindings
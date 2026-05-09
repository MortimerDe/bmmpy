#pragma once

#include "bmmpy/core/row_window.hpp"
#include "bmmpy/ga/types.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace bmmpy::ga {

class Algorithm {
public:
    virtual ~Algorithm() = default;

    virtual void initialize(const ::bmmpy::RowWindow& window) = 0;
    virtual void step_generation() = 0;

    virtual bool done() const noexcept = 0;
    virtual std::size_t generation() const noexcept = 0;

    virtual Individual best_individual() const = 0;
    virtual RunStats stats() const = 0;

    virtual std::vector<Individual> export_migrants(std::size_t max_count) = 0;
    virtual void import_migrants(std::vector<Individual> migrants) = 0;

    virtual const char* name() const noexcept = 0;
};

using AlgorithmFactory = std::function<std::unique_ptr<Algorithm>(std::size_t island_id)>;

} // namespace bmmpy::ga
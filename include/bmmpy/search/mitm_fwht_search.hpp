#pragma once

#include "bmmpy/search/searcher.hpp"
#include "bmmpy/search/split_window_prep.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace bmmpy {

struct MitmFwhtSearchConfig {
    std::size_t reserve_unique_patterns = 1024;
    std::size_t reserve_left_rows = 20;
    std::size_t reserve_right_states = std::size_t{1} << 16;
    std::size_t max_candidates = 64;
};

class MitmFwhtSearch final : public Searcher {
public:
    explicit MitmFwhtSearch(MitmFwhtSearchConfig config = {});

    std::vector<Candidate> search(const RowWindow& window) override;

    const char* name() const noexcept override { return "mitm_fwht"; }

private:
    static constexpr std::size_t k_max_low_bits = 30;
    static constexpr std::size_t k_max_half_bits = 31;

    static std::pair<std::size_t, std::size_t> get_split_info(std::size_t t) noexcept;

    void
    ensure_capacity(std::size_t unique_patterns, std::size_t low_bits, std::size_t high_states);

    void build_adjacency(const CompactSplitWindow& prep);

    void initialize_buckets(const CompactSplitWindow& prep);

    void apply_bit_flip(std::size_t bit);

    void process_candidate(std::uint32_t x_low,
                           std::size_t n_high,
                           std::size_t low_bits,
                           std::int32_t total_weight,
                           std::int32_t& min_score_threshold,
                           std::uint32_t& worst_weight);

    void add_result(std::uint64_t mask,
                    std::uint32_t weight,
                    std::int32_t total_weight,
                    std::int32_t& min_score_threshold,
                    std::uint32_t& worst_weight);

    std::vector<std::int64_t> _col_data;

    std::vector<std::size_t> _adj_offsets;
    std::vector<std::int32_t> _adj_indices;
    std::vector<std::int32_t> _adj_counts;

    std::vector<std::int32_t> _buckets;
    std::vector<std::int32_t> _fwht_buffer;

    std::vector<Candidate> _candidates;

    MitmFwhtSearchConfig _config;
};

} // namespace bmmpy
#pragma once

#include "bmmpy/search/searcher.hpp"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
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
    static constexpr std::size_t kMaxTLeft = 30;
    static constexpr std::size_t kMaxTHalf = 31;

    static std::pair<std::size_t, std::size_t> get_split_info(std::size_t t) noexcept;

    void ensure_capacity(std::size_t cols, std::size_t t_left, std::size_t n_right);

    std::pair<std::size_t, std::int32_t>
    prepare_columns(const std::vector<const std::uint64_t*>& rows,
                    std::size_t words_per_row,
                    std::size_t cols,
                    std::size_t t_left);

    void initialize_buckets(std::size_t unique_count, std::size_t n_right);

    void apply_bit_flip(std::size_t bit);

    void process_candidate(std::uint32_t x_left,
                           std::size_t n_right,
                           std::size_t t_left,
                           std::int32_t total_weight,
                           std::int32_t& min_score_threshold,
                           std::uint32_t& worst_weight);

    void add_result(std::uint64_t mask,
                    std::uint32_t weight,
                    std::int32_t total_weight,
                    std::int32_t& min_score_threshold,
                    std::uint32_t& worst_weight);

    std::vector<std::int64_t> _col_data;
    std::vector<std::uint32_t> _left_masks;

    std::vector<std::size_t> _adj_offsets;
    std::vector<std::int32_t> _adj_indices;
    std::vector<std::int32_t> _adj_counts;

    std::vector<std::int32_t> _buckets;
    std::vector<std::int32_t> _fwht_buffer;

    std::unordered_map<std::uint64_t, std::int32_t> _col_map;
    std::vector<Candidate> _candidates;

    MitmFwhtSearchConfig _config;
};

} // namespace bmmpy
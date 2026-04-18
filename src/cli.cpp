#include "bmmpy/apply/greedy_selection.hpp"
#include "bmmpy/core/bit_matrix.hpp"
#include "bmmpy/search/cuda_mitm_fwht_search.hpp"
#include "bmmpy/search/mitm_fwht_search.hpp"
#include "bmmpy/stub.hpp"

#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

std::vector<std::size_t> getRandomIndices(std::size_t N, std::size_t count = 28) {
    std::vector<std::size_t> all_indices(N);
    std::iota(all_indices.begin(), all_indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());

    if (count < N) {
        for (std::size_t i = 0; i < count; ++i) {
            std::uniform_int_distribution<std::size_t> dist(i, N - 1);
            std::swap(all_indices[i], all_indices[dist(g)]);
        }
        all_indices.resize(count);
    } else {
        std::shuffle(all_indices.begin(), all_indices.end(), g);
    }

    return all_indices;
}

int main() {
    std::cout << "Version: " << bmmpy::get_version() << std::endl;

    bmmpy::BitMatrix bm = bmmpy::BitMatrix::load_text("___matrix.txt");

    std::vector<std::size_t> rows = getRandomIndices(bm.rows(), 32);
    auto window = bm.row_window(rows);
    std::cout << "window rows: " << window.materialize().weight() << std::endl;
    bmmpy::CudaMitmFwhtSearch searcher({64, 0});
    // bmmpy::MitmFwhtSearch searcher({});

    bmmpy::GreedySelection greedy({1, false, 0x12345678});

    int32_t weight_before = bm.weight();

    try {
        std::cout << "weight: " << bm.weight() << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        const auto candidates = searcher.search(window);

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> duration = end - start;

        std::cout << "candidates found: " << candidates.size() << std::endl;
        greedy.apply(window, candidates);

        std::cout << "Time taken: " << duration.count() << " ms" << std::endl;
        std::cout << "diff: " << weight_before - bm.weight() << std::endl;

    } catch (const std::exception& ex) {
        std::cout << "cuda search error: " << ex.what() << '\n';
    }

    return 0;
}
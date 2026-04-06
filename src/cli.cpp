#include "bmmpy/stub.hpp"
#include "bmmpy/types/candidate.hpp"

#include <iostream>

int main() {
    std::cout << "Version: " << bmmpy::get_version() << std::endl;
    std::cout << "Add: " << bmmpy::add(2, 3) << std::endl;

    using bmmpy::Candidate;

    Candidate cand =
        Candidate::from_u64((1ULL << 0) | (1ULL << 2) | (1ULL << 5), 7);

    std::cout << "weight = " << cand.weight << "\n";
    std::cout << "mask popcount = " << cand.mask_popcount() << "\n";

    std::cout << "has row 2: " << cand.has_row(2) << "\n";
    std::cout << "has row 3: " << cand.has_row(3) << "\n";

    std::cout << "selected rows: ";
    for (std::size_t row : cand.selected_rows()) {
        std::cout << row << " ";
    }
    std::cout << "\n";

    std::uint64_t raw = cand.mask_u64();
    std::cout << "mask_u64 = " << raw << "\n";

    return 0;
}
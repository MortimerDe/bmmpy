#include "bmmpy/search/fwht_search.hpp"
#include "bmmpy/search/searcher.hpp"
#include "bmmpy/stub.hpp"
#include "bmmpy/types/candidate.hpp"

#include <iostream>

int main() {
    std::cout << "Version: " << bmmpy::get_version() << std::endl;

    bmmpy::BitMatrix bm = bmmpy::BitMatrix::load_text("___matrix.txt");

    std::cout << bm.weight() << std::endl;

    return 0;
}
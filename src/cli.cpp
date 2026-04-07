#include "bmmpy/search/fwht_search.hpp"
#include "bmmpy/search/searcher.hpp"
#include "bmmpy/stub.hpp"
#include "bmmpy/types/candidate.hpp"

#include <iostream>

int main() {
    std::cout << "Version: " << bmmpy::get_version() << std::endl;

    bmmpy::Searcher* s = new bmmpy::FwhtSearch();

    std::cout << (*s).describe(32) << std::endl;

    delete s;

    return 0;
}
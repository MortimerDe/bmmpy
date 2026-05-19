#include "bmmpy.hpp"

int main() {
    if (bmmpy::get_version().empty())
        return 1;

    return bmmpy::add(1, 2) == 3 ? 0 : 1;
}
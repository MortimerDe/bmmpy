#include <iostream>
#include "bmmpy/stub.hpp"

int main()
{
    std::cout << "Version: " << bmmpy::get_version() << std::endl;
    std::cout << "Add: " << bmmpy::add(2, 3) << std::endl;

    return 0;
}
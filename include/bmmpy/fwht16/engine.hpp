#pragma once

#include "bmmpy/fwht16/types.hpp"

namespace bmmp::fwht16 {
class Fwht16Engine {
public:
    Fwht16BatchResponse run(const Fwht16BatchRequest& request) const;
};
} // namespace bmmp::fwht16
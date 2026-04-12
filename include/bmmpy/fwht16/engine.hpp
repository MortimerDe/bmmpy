#pragma once

#include "bmmpy/fwht16/types.hpp"

namespace bmmpy {
class Fwht16Engine {
public:
    Fwht16BatchResponse run(const Fwht16BatchRequest& request) const;
};
} // namespace bmmpy
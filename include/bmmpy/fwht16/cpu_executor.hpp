#pragma once

#include "bmmpy/fwht16/types.hpp"

namespace bmmpy {
class CpuFwht16Executor {
public:
    Fwht16BatchResponse run(const Fwht16BatchRequest& request) const;
};
} // namespace bmmpy
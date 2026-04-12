#pragma once

#include "bmmpy/fwht16/types.hpp"

namespace bmmp::fwht16 {
class CpuFwht16Executor {
public:
    Fwht16BatchResponse run(const Fwht16BatchRequest& request) const;
};
} // namespace bmmp::fwht16
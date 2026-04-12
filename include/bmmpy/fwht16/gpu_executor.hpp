#pragma once

#include "bmmpy/fwht16/types.hpp"

namespace bmmpy {

class GpuFwht16Executor {
public:
    static bool available() noexcept;
    Fwht16BatchResponse run(const Fwht16BatchRequest& request) const;
};

} // namespace bmmpy
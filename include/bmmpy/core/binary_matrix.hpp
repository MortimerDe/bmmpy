#pragma once

#include <stdexcept>

namespace bmmpy
{
    enum class MatrixErr
    {
        SizeTooLarge,
        AllocationFailed,
        DimensionMismatch
    };

    class MatrixError : public std::runtime_error
    {
    public:
        explicit MatrixError(MatrixErr code);
        MatrixErr code() const noexcept { return code_; }

    private:
        MatrixErr code_;
    };
}
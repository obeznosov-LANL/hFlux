#pragma once

#include <Kokkos_Core.hpp>

using Real = double;
using Vector = std::vector<Real>;
using Dim2 = Kokkos::Array<Real, 2>;    // R, Z
using Dim3 = Kokkos::Array<Real, 3>;    // R, phi, Z
using Dim5 = Kokkos::Array<Real, 5>;    // p, xi, R, phi, Z
using Dim6 = Kokkos::Array<Real, 6>;    // R, phi, Z (start) and R, phi, Z (end)
using IntDim2 = Kokkos::Array<int, 2>;

KOKKOS_INLINE_FUNCTION
void cross_product(const Dim3& A, const Dim3& B, Dim3& result) {
    result[0] = A[1] * B[2] - A[2] * B[1];
    result[1] = A[2] * B[0] - A[0] * B[2];
    result[2] = A[0] * B[1] - A[1] * B[0];
}

template <typename T>
[[nodiscard]] KOKKOS_INLINE_FUNCTION
typename T::value_type dot_product(const T& A, const T& B) {
    typename T::value_type ret = 0.0;
    for (std::size_t i = 0; i < A.size(); ++i) {
        ret += A[i] * B[i];
    }
    return ret;
}

enum class ErrorCode {
    Success = 0,
    OutOfBounds,
    NoParticle,
    WallImpact,
    MomentumCutoff,
    StillBorn,
    Defrag,
    TimeIntervalViolation,
    BeyondFirstWall,
    DimensionIsZero,
    HBelowMin
};

// Backwards compatibility alias
using ERROR_CODE = ErrorCode;

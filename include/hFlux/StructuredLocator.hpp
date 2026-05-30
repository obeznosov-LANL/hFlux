#pragma once

#include <cmath>
#include "common.hpp"

struct StructuredLocator {
  Real R0;
  Real Z0;
  Real dR;
  Real dZ;
  Real R1;
  Real Z1;
  int nR;
  int nZ;

  KOKKOS_INLINE_FUNCTION
  StructuredLocator(const Real R0, const Real Z0,
                    const Real dR, const Real dZ,
                    const int nR, const int nZ)
      : R0(R0), Z0(Z0), dR(dR), dZ(dZ), nR(nR), nZ(nZ),
        R1(R0 + nR * dR),
        Z1(Z0 + nZ * dZ) {};

  KOKKOS_INLINE_FUNCTION
  ErrorCode locate(const Real R, const Real Z,
                   int& iR, int& iZ,
                   Real& xiR, Real& xiZ) const {

    if (R < R0 || R > R1 || Z < Z0 || Z > Z1) {
      return ErrorCode::OutOfBounds;
    }

    locate_1d(R, R0, dR, nR, iR, xiR);
    locate_1d(Z, Z0, dZ, nZ, iZ, xiZ);

    return ErrorCode::Success;
  }

private:
  KOKKOS_INLINE_FUNCTION
  static void locate_1d(const Real x, const Real x0, const Real dx,
                        const int n, int& i, Real& xi) {
    const Real s = (x - x0) / dx;
    i = static_cast<int>(floor(s));
    xi = s - i - 0.5;
  };
};

template<int swidth>
StructuredLocator makeHermiteLocator(const StructuredLocator& fd) {
  static_assert(swidth >= 3);
  static_assert((swidth - 1) % 2 == 0);

  constexpr int stride = swidth - 1;
  constexpr int margin = stride / 2;

  return StructuredLocator{
    fd.R0 + margin * fd.dR,
    fd.Z0 + margin * fd.dZ,
    fd.dR * stride,
    fd.dZ * stride,
    fd.nR / stride - 1,
    fd.nZ / stride - 1
  };
};

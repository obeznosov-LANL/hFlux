//========================================================================================
// (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

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
  ErrorCode checkBounds(const Real R, const Real Z) const {

    if (R < R0 || R > R1 || Z < Z0 || Z > Z1) {
      return ErrorCode::OutOfBounds;
    }
    return ErrorCode::Success;
  }

  KOKKOS_INLINE_FUNCTION
  void locateCell(const Real R, const Real Z,
                   int& iR, int& iZ) const {

    locateCell_1d(R, R0, dR, iR);
    locateCell_1d(Z, Z0, dZ, iZ);
  }


  KOKKOS_INLINE_FUNCTION
  void locate(const Real R, const Real Z,
                   int& iR, int& iZ,
                   Real& xiR, Real& xiZ) const {

    locate_1d(R, R0, dR, iR, xiR);
    locate_1d(Z, Z0, dZ, iZ, xiZ);
  }

private:
  KOKKOS_INLINE_FUNCTION
  static void locateCell_1d(const Real x, const Real x0, const Real dx, int& i) {
    const Real s = (x - x0) / dx;
    i = static_cast<int>(floor(s));
  };

  KOKKOS_INLINE_FUNCTION
  static void locate_1d(const Real x, const Real x0, const Real dx, int& i, Real& xi) {
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

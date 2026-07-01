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
#include "common.hpp"
#include "Taylor.hpp"
#include "StructuredLocator.hpp"

struct Evaluator {
  StructuredLocator locator;

  template<class FieldView>
  KOKKOS_INLINE_FUNCTION
  void evalField(Dim3& F, Real R, Real Z, FieldView hermite_data) const {
    int iR = 0;
    int iZ = 0;
    Real xiR = 0.0;
    Real xiZ = 0.0;

    locator.locate(R, Z, iR, iZ, xiR, xiZ);

    auto pcofs = Kokkos::subview(
        hermite_data, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, iR, iZ);
    evalTaylor3(F, xiR, xiZ, pcofs);
  }

  template<class PsiView>
  KOKKOS_INLINE_FUNCTION
  void evalPsi(Real& psi, Real R, Real Z, PsiView psi_data) const {
    int iR = 0;
    int iZ = 0;
    Real xiR = 0.0;
    Real xiZ = 0.0;

    locator.locate(R, Z, iR, iZ, xiR, xiZ);

    auto pcofs = Kokkos::subview(psi_data, Kokkos::ALL, Kokkos::ALL, iR, iZ);
    evalTaylor(psi, xiR, xiZ, pcofs);
  }
};

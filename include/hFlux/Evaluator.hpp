#pragma once
#include "common.hpp"
#include "Taylor.hpp"
#include "StructuredLocator.hpp"

struct Evaluator {
  StructuredLocator locator;

  template<class FieldView>
  KOKKOS_INLINE_FUNCTION
  ErrorCode evalField(Dim3& F, Real R, Real Z, FieldView hermite_data) const {
    int iR = 0;
    int iZ = 0;
    Real xiR = 0.0;
    Real xiZ = 0.0;

    const ErrorCode status = locator.locate(R, Z, iR, iZ, xiR, xiZ);
    if (status != ErrorCode::Success) {
      return status;
    }

    auto pcofs = Kokkos::subview(
        hermite_data, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, iR, iZ);
    evalTaylor3(F, xiR, xiZ, pcofs);

    return ErrorCode::Success;
  }

  template<class PsiView>
  KOKKOS_INLINE_FUNCTION
  ErrorCode evalPsi(Real& psi, Real R, Real Z, PsiView psi_data) const {
    int iR = 0;
    int iZ = 0;
    Real xiR = 0.0;
    Real xiZ = 0.0;

    const ErrorCode status = locator.locate(R, Z, iR, iZ, xiR, xiZ);
    if (status != ErrorCode::Success) {
      return status;
    }

    auto pcofs = Kokkos::subview(psi_data, Kokkos::ALL, Kokkos::ALL, iR, iZ);
    evalTaylor(psi, xiR, xiZ, pcofs);

    return ErrorCode::Success;
  }
};

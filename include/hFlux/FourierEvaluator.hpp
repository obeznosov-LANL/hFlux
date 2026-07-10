#pragma once
#include "common.hpp"
#include "Taylor.hpp"
#include "StructuredLocator.hpp"

struct FourierEvaluator {
  StructuredLocator locator;

  template<class FieldView>
  KOKKOS_INLINE_FUNCTION
  void evalField(Dim3& F, Real R, Real Z, Real phi, FieldView hermite_data) const {
    int iR = 0;
    int iZ = 0;
    Real xiR = 0.0;
    Real xiZ = 0.0;

    locator.locate(R, Z, iR, iZ, xiR, xiZ);

    auto pcofs = Kokkos::subview(
        hermite_data, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, iR, iZ);

    Real correction = 0.0;
    evalTaylorFourier4(F, correction, xiR, xiZ, phi, pcofs);
    F[2] -= correction / R;
  }

  // Phi-dependent poloidal flux, evaluated from per-channel Fourier psi
  // coefficients (channel axis contiguous / stride 1), analogous to how the
  // field's B_phi channels are stored and summed over phi.
  template<class PsiView>
  KOKKOS_INLINE_FUNCTION
  void evalPsi(Real& psi, Real R, Real Z, Real phi, PsiView psi_data) const {
    int iR = 0;
    int iZ = 0;
    Real xiR = 0.0;
    Real xiZ = 0.0;

    locator.locate(R, Z, iR, iZ, xiR, xiZ);

    auto pcofs = Kokkos::subview(
        psi_data, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, iR, iZ);
    evalTaylorFourier(psi, xiR, xiZ, phi, pcofs);
  }
};

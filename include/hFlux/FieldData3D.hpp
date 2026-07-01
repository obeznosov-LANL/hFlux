#pragma once

#include <Kokkos_DualView.hpp>
#include "common.hpp"
#include "StructuredLocator.hpp"


// Hermite coefficients layout for best point evaluation:
// hermite_data(idR, idZ, packed_component, iR, iZ)
// packed_component = (ndims+1) * fueier_channel + quantity
// quantity 0: R B_R
// quantity 1: R B_phi
// quantity 2: R B_Z from 2D cleaning on each fouier channel
// quantity 3: R B_Z correction for phi dependency


template<int m, int swidth,  typename ExecSpace, int ndims=3>
struct FieldData3D {
  using DataView =
      Kokkos::DualView<Real***, Kokkos::LayoutRight, ExecSpace>;
  using HermiteView =
      Kokkos::DualView<Real*****, Kokkos::LayoutLeft, ExecSpace>;
  using PsiView =
      Kokkos::DualView<Real****, Kokkos::LayoutLeft, ExecSpace>;

  StructuredLocator fd_locator;
  StructuredLocator hermite_locator;

  DataView data;
  HermiteView hermite_data;
  PsiView psi_data;

  const int nphi;

  FieldData3D(int nR_data, int nZ_data, int nphi_data,
                      Real R0, Real Z0, Real dR, Real dZ, Real dphi)
      : fd_locator(R0, Z0, dR, dZ, nR_data - 1, nZ_data - 1),
        nphi(nphi_data), dphi(2.0 * M_PI / nphi),
        hermite_locator(makeHermiteLocator<swidth>(fd_locator)),
        data("data", nR_data, nZ_data, ndims * nphi_data),
        hermite_data("hermite_data",
                      2 * m + 3, 2 * m + 3, (ndims+1) * nphi_data,
                      hermite_locator.nR,
                      hermite_locator.nZ),
        psi_data("psi_data",
                  2 * m + 3, 2 * m + 3,
                  hermite_locator.nR,
                  hermite_locator.nZ) {}

  KOKKOS_INLINE_FUNCTION
  static int sample_component(int iphi, int d) {
    return ndims * iphi + d;
  }

  KOKKOS_INLINE_FUNCTION
  static int fourier_component(int channel, int d) {
    return (ndims+1) * channel + d;
  }

  KOKKOS_INLINE_FUNCTION
  static int correction_component(int channel) {
    return (ndims+1) * channel + ndims;
  }
};

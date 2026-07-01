#pragma once

#include <Kokkos_DualView.hpp>
#include "common.hpp"
#include "StructuredLocator.hpp"


// Hermite coefficients layout for best point evaluation:
// hermite_data(idR, idZ, packed_component, iR, iZ)
// packed_component = (ndims+1) * fourier_channel + quantity
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
  const Real dphi;

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
    return ndims * channel + d;
  }

  KOKKOS_INLINE_FUNCTION
  static int fourier_component_with_correction(int channel, int d) {
    return (ndims+1) * channel + d;
  }

  KOKKOS_INLINE_FUNCTION
  static int correction_component(int channel) {
    return (ndims+1) * channel + ndims;
  }

  template<class SampleDataView, class FourierDataView>
  void sampleToFourier(SampleDataView sample_data,
      FourierDataView fourier_data) {
    const int nR = sample_data.extent_int(0);
    const int nZ = sample_data.extent_int(1);

    KOKKOS_ASSERT(nR <= fourier_data.extent_int(0));
    KOKKOS_ASSERT(nZ <= fourier_data.extent_int(1));
    KOKKOS_ASSERT(sample_data.extent_int(2)  >= ndims * nphi);
    KOKKOS_ASSERT(fourier_data.extent_int(2) >= ndims * nphi);

    using exec_space = typename FourierDataView::execution_space;
    using policy_t = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>;

    const int ncos = nphi / 2;
    const int nsin = (nphi - 1) / 2;
    const int sin_offset = 1 + ncos;
    const Real inv_nphi = 1.0 / static_cast<Real>(nphi);

    Kokkos::parallel_for(
        "sampleToFourier",
        policy_t({0,0,0}, {nR, nZ, ndims}),
        KOKKOS_LAMBDA(int iR, int iZ, int d) {
          Real sum = 0.0;
          for (int iphi = 0; iphi < nphi; ++iphi) {
            sum += sample_data(iR, iZ, sample_component(iphi, d));
          }
          fourier_data(iR, iZ, fourier_component(0, d)) = sum * inv_nphi;

          for (int k = 1; k <= ncos; ++k) {
            Real cos_sum = 0.0;
            for (int iphi = 0; iphi < nphi; ++iphi) {
              const Real phi = static_cast<Real>(iphi) * dphi;
              cos_sum += sample_data(iR, iZ, sample_component(iphi, d)) *
                         Kokkos::cos(static_cast<Real>(k) * phi);
            }

            const bool is_nyquist = (nphi % 2 == 0) && (k == ncos);
            const Real scale = is_nyquist ? inv_nphi : 2.0 * inv_nphi;
            fourier_data(iR, iZ, fourier_component(k, d)) = scale * cos_sum;
          }

          for (int k = 1; k <= nsin; ++k) {
            Real sin_sum = 0.0;
            for (int iphi = 0; iphi < nphi; ++iphi) {
              const Real phi = static_cast<Real>(iphi) * dphi;
              sin_sum += sample_data(iR, iZ, sample_component(iphi, d)) *
                         Kokkos::sin(static_cast<Real>(k) * phi);
            }

            fourier_data(iR, iZ, fourier_component(sin_offset + k - 1, d)) =
                2.0 * inv_nphi * sin_sum;
          }
        });
  }
};

#include <cmath>
#include <format>
#include <iostream>

#include "hFlux/FieldData3D.hpp"
#include "hFlux/FourierEvaluator.hpp"
#include "hFlux/Interpolator.hpp"
#include "AnalyticField.hpp"

namespace {


void run(const int nR_data, const int nZ_data, Real& hR,
         Kokkos::Array<Real, 3>& l2err) {
  static constexpr int m = 2;
  static constexpr int swidth = 7;
  static constexpr int nphi = 8;

  const int nphi_data = 20;

  using exec_space = Kokkos::DefaultExecutionSpace;
  using policy3D = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>;

  const Real R0 = 1.525;
  const Real Z0 = -2.975;
  const Real R1 = R0 + 99.0 * 0.0345;
  const Real Z1 = Z0 + 199.0 * 0.02975;
  const Real dR = (R1 - R0) / static_cast<Real>(nR_data - 1);
  const Real dZ = (Z1 - Z0) / static_cast<Real>(nZ_data - 1);
  const Real dphi = 2.0 * M_PI / static_cast<Real>(nphi);

  FieldData3D<m, swidth, exec_space> data(nR_data, nZ_data, nphi,
                                           R0, Z0, dR, dZ, dphi);
  Kokkos::DualView<Real***, Kokkos::LayoutRight, exec_space> fourier_data(
      "fourier_data", nR_data, nZ_data, (3 + 1) * nphi);

  Real q0 = 2.1;
  Real q2 = 2.0;
  Real R_a = 3.0;
  Real E_0 = 70.0;
  Real perturb_amp = 0.05;

  AnalyticField af(q0, q2, R_a, E_0, perturb_amp);
  auto sample_data = data.data;
  Kokkos::parallel_for(
      "set_non_axisymmetric_field",
      policy3D({0, 0, 0}, {nR_data, nZ_data, nphi}),
      KOKKOS_LAMBDA(const int i, const int j, const int iphi) {
        const Real R = R0 + dR * static_cast<Real>(i);
        const Real Z = Z0 + dZ * static_cast<Real>(j);
        const Real phi = dphi * static_cast<Real>(iphi);

        Dim3 B = {};
        af.eval(B, R, Z, phi);
        for (int d = 0; d < 3; ++d) {
          sample_data.view_device()(i, j, FieldData3D<m, swidth, exec_space>::sample_component(iphi, d)) =
              R * B[d];
        }
      });
  sample_data.modify_device();

  data.sampleToFourier(data.data.view_device(), fourier_data.view_device());
  fourier_data.modify_device();

  Interpolator<m, swidth> itrp;
  itrp.interpolate(
      data.fd_locator, data.hermite_locator,
      fourier_data.view_device(),
      data.hermite_data.view_device());
  data.hermite_data.modify_device();

  // Make each Fourier channel divergence free.
  // Layout is stride-4 per channel: quantities 0=R B_R, 1=R B_phi,
  // 2=R B_Z, 3=phi-correction. nphi channels total.
  // Build the phi-correction (from R B_phi) first, then cleanDivergence
  // anchors R B_Z consistently with that correction.
  constexpr int component_stride = 4;
  itrp.computeChi(data.hermite_locator,
      data.hermite_data.view_device(),
      /*component0=*/0, /*nfields=*/nphi, component_stride);

  data.DifferentiatePhiCorrection(data.hermite_data.view_device());

  itrp.cleanDivergence(data.hermite_locator,
      data.hermite_data.view_device(),
      /*component0=*/0, /*nfields=*/nphi, component_stride);

  data.hermite_data.modify_device();

  l2err = {};

  const int nR_pl = 160;
  const int nZ_pl = 320;
  const int nphi_pl = 7;
  const Real eps = 1e-8;
  const Real R0_pl = data.hermite_locator.R0 + eps;
  const Real Z0_pl = data.hermite_locator.Z0 + eps;
  const Real R1_pl = data.hermite_locator.R1 - eps;
  const Real Z1_pl = data.hermite_locator.Z1 - eps;
  const Real dR_pl = (R1_pl - R0_pl) / static_cast<Real>(nR_pl - 1);
  const Real dZ_pl = (Z1_pl - Z0_pl) / static_cast<Real>(nZ_pl - 1);
  const Real dphi_pl = 2.0 * M_PI / static_cast<Real>(nphi_pl);

  FourierEvaluator ev{data.hermite_locator};
  Kokkos::parallel_reduce(
      "eval_non_axisymmetric_field",
      policy3D({0, 0, 0}, {nR_pl, nZ_pl, nphi_pl}),
      KOKKOS_LAMBDA(const int i, const int j, const int iphi, Real& err0,
                    Real& err1, Real& err2) {
        const Real R = R0_pl + dR_pl * static_cast<Real>(i);
        const Real Z = Z0_pl + dZ_pl * static_cast<Real>(j);
        const Real phi = (static_cast<Real>(iphi) + 0.37) * dphi_pl;

        Dim3 RB = {}, B_exact = {};
        ev.evalField(RB, R, Z, phi, data.hermite_data.view_device());
        af.eval(B_exact, R, Z, phi);

        const Real diff0 = B_exact[0] - RB[0] / R;
        const Real diff1 = B_exact[1] - RB[1] / R;
        const Real diff2 = B_exact[2] - RB[2] / R;
        err0 += diff0 * diff0;
        err1 += diff1 * diff1;
        err2 += diff2 * diff2;
      },
      l2err[0], l2err[1], l2err[2]);

  const Real volume = dR_pl * dZ_pl * dphi_pl;
  for (int i = 0; i < l2err.size(); ++i) {
    l2err[i] = std::sqrt(l2err[i] * volume);
  }
  hR = data.hermite_locator.dR;
}

}  // namespace

int main() {
  Kokkos::initialize();

  int NR = 64;
  Kokkos::Array<Real, 3> l2err = {};
  Real hR = 0.0;
  constexpr Real expected_order = 2 * 2 + 2;

  for (int ix = 0; ix < 4; ++ix) {
    Real hR_new = 0.0;
    Kokkos::Array<Real, 3> l2err_new = {};

    run(NR, 2 * NR, hR_new, l2err_new);
    if (ix > 0) {
      std::cout << std::format("{:.17g} {:.17g} {:.17g}\n", l2err[0], l2err[1], l2err[2]);
      std::cout << std::format("{:.17g} {:.17g} {:.17g}\n", l2err_new[0], l2err_new[1], l2err_new[2]);

      for (int d = 0; d < 3; ++d) {
        const Real order = std::log(l2err[d] / l2err_new[d]) / std::log(hR / hR_new);
        if (static_cast<int>(std::round(order)) < expected_order && l2err_new[d] > 1e-12) {
          std::fprintf(stderr, "B_%d interpolation did not converge with order %f %le\n",
                       d, expected_order, order);
          Kokkos::finalize();
          return d + 1;
        }
      }
    }

    l2err = l2err_new;
    hR = hR_new;
    NR = static_cast<int>(1.5 * NR);
  }

  Kokkos::finalize();
  return 0;
}

#include <cmath>
#include <cstdio>
#include <format>
#include <iostream>

#include "hFlux/FieldData3D.hpp"
#include "hFlux/FourierEvaluator.hpp"
#include "hFlux/Evaluator.hpp"
#include "hFlux/Interpolator.hpp"
#include "hFlux/MagneticAxis.hpp"
#include "hFlux/dopri.hpp"
#include "AnalyticField.hpp"

namespace {

// Field-line ODE in the toroidal angle phi:
//   dR/dphi   = (R B_R)   / (R B_phi) * ... = RB[0] / RB[1] * R
//   dZ/dphi   = (R B_Z)   / (R B_phi)       = RB[2] / RB[1] * R
// The evaluator returns R*B, so the ratios below already carry the extra R.
template <class FieldView>
struct FieldLine {
  FourierEvaluator ev;
  FieldView hermite_data;

  typedef Dim2 value_type;

  KOKKOS_INLINE_FUNCTION
  ErrorCode operator()(const Real phi, const Dim2 X, Dim2& dXdphi) const {
    ErrorCode ret = ev.locator.checkBounds(X[0], X[1]);
    if (ret == ErrorCode::Success) {
      Dim3 RB = {};
      ev.evalField(RB, X[0], X[1], phi, hermite_data);
      dXdphi[0] = RB[0] / RB[1] * X[0];
      dXdphi[1] = RB[2] / RB[1] * X[0];
    } else {
      dXdphi[0] = 0.0;
      dXdphi[1] = 0.0;
    }
    return ret;
  }
};


int run(const int nR_data, const int nZ_data) {
  static constexpr int m = 4;
  static constexpr int swidth = 9;
  static constexpr int nphi = 9;

  using exec_space = Kokkos::DefaultExecutionSpace;
  using policy2D = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>;
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

  // Unperturbed, axisymmetric analytic field. Sampled on all nphi phi-planes
  // so the full phi-dependent (Fourier) machinery is exercised even though
  // only the DC channel is non-zero.
  AnalyticField af(q0, q2, R_a, E_0);
  auto sample_data = data.data;
  Kokkos::parallel_for(
      "set_axisymmetric_field",
      policy3D({0, 0, 0}, {nR_data, nZ_data, nphi}),
      KOKKOS_LAMBDA(const int i, const int j, const int iphi) {
        const Real R = R0 + dR * static_cast<Real>(i);
        const Real Z = Z0 + dZ * static_cast<Real>(j);

        Dim3 B = {};
        af.eval(B, R, Z);
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

  // Layout is stride-4 per channel: quantities 0=R B_R, 1=R B_phi,
  // 2=R B_Z, 3=phi-correction. nphi channels total.
  constexpr int component_stride = 4;

  // Phi-dependent poloidal flux psi, one Hermite field per Fourier channel.
  // Channel axis is contiguous (stride 1) so it can be summed by
  // evalTaylorFourier just like B_phi.
  Kokkos::DualView<Real*****, Kokkos::LayoutLeft, exec_space> psi_fourier_data(
      "psi_fourier_data", 2 * m + 3, 2 * m + 3, nphi,
      data.hermite_locator.nR, data.hermite_locator.nZ);

  // Compute the flux before cleanDivergence, matching hermite_test ordering.
  itrp.computeFluxFourier(data.hermite_locator,
      data.hermite_data.view_device(),
      psi_fourier_data.view_device(),
      /*component0=*/0, /*nfields=*/nphi, component_stride);
  psi_fourier_data.modify_device();

  itrp.cleanDivergence(data.hermite_locator,
      data.hermite_data.view_device(),
      /*component0=*/0, /*nfields=*/nphi, component_stride);

  itrp.computeChi(data.hermite_locator,
      data.hermite_data.view_device(),
      /*component0=*/0, /*nfields=*/nphi, component_stride);

  data.DifferentiatePhiCorrection(data.hermite_data.view_device());
  data.hermite_data.modify_device();

  // Normalize psi to the magnetic axis. The field is axisymmetric, so the axis
  // is phi-independent; find it on the DC channel using the 2D Evaluator, then
  // subtract Psi_min from the DC constant coefficient across all cells.
  Real R_center = 3.2;
  Real Z_center = -0.7;
  int descent = -1;
  Real Psi_min = 0.0;

  Evaluator ev2d{data.hermite_locator};

  data.hermite_data.sync_host();
  psi_fourier_data.sync_host();
  auto psi_dc_host = Kokkos::subview(psi_fourier_data.view_host(),
      Kokkos::ALL, Kokkos::ALL, 0, Kokkos::ALL, Kokkos::ALL);
  findMagneticAxis(R_center, Z_center,
              data.hermite_data.view_host(),
              psi_dc_host,
              ev2d,
              descent,
              Psi_min);

  std::cout << std::format("Psi_min = {:.17g}", Psi_min) << std::endl;

  psi_fourier_data.sync_device();
  {
    const int nR_h = data.hermite_locator.nR;
    const int nZ_h = data.hermite_locator.nZ;
    auto psi_d = psi_fourier_data.view_device();
    const Real Psi_min_d = Psi_min;
    Kokkos::parallel_for(
        "normalize_psi",
        policy2D({0, 0}, {nR_h, nZ_h}),
        KOKKOS_LAMBDA(int iR, int iZ) {
          psi_d(0, 0, 0, iR, iZ) -= Psi_min_d;
        });
  }
  psi_fourier_data.modify_device();

  const int n_r = 4;
  const int n_theta = 3;
  const double dr = 0.05;
  const double dtheta = 2 * M_PI / (double) n_theta;
  const Real R_axis = 3.0;
  const Real Z_axis = 0.0;
  const int n_turn = 20;
  const int n_traces = n_r * n_theta;

  // poincare_data(q, trace, turn): q = 0 -> R, q = 1 -> Z, q = 2 -> psi.
  Kokkos::DualView<Real***> poincare_data("poincare_data", 3, n_traces,
                                          n_turn + 1);
  auto pd_d = poincare_data.view_device();
  Kokkos::parallel_for(
      "seed_traces",
      policy2D({0, 0}, {n_r, n_theta}),
      KOKKOS_LAMBDA(const int ir, const int itheta) {
        pd_d(0, ir + itheta * n_r, 0) =
            R_axis + (ir + 1) * dr * Kokkos::cos(itheta * dtheta);
        pd_d(1, ir + itheta * n_r, 0) =
            Z_axis + (ir + 1) * dr * Kokkos::sin(itheta * dtheta);
      });
  poincare_data.modify_device();

  FourierEvaluator ev{data.hermite_locator};
  using FieldLineT = FieldLine<decltype(data.hermite_data.view_device())>;
  FieldLineT f{ev, data.hermite_data.view_device()};

  auto psi_d = psi_fourier_data.view_device();
  Kokkos::parallel_for(
      "poincare", n_traces,
      KOKKOS_LAMBDA(int i) {
        Kokkos::Array<Dim2, 10> work;
        Dim2 trace = {pd_d(0, i, 0), pd_d(1, i, 0)};
        Real psi0 = 0.0;
        ev.evalPsi(psi0, trace[0], trace[1], 0.0, psi_d);
        pd_d(2, i, 0) = psi0;
        for (int it = 0; it < n_turn; ++it) {
          solve_dopri5(f, trace, 0.0, M_PI, 1e-10, 1e-12, 1e-6, 5e-11,
                       2000000, work);
          solve_dopri5(f, trace, 0.0, M_PI, 1e-10, 1e-12, 1e-6, 5e-11,
                       2000000, work);
          pd_d(0, i, it + 1) = trace[0];
          pd_d(1, i, it + 1) = trace[1];
          Real psi = 0.0;
          ev.evalPsi(psi, trace[0], trace[1], 0.0, psi_d);
          pd_d(2, i, it + 1) = psi;
        }
      });
  poincare_data.modify_device();

  // Check psi is conserved along each trajectory. For an axisymmetric field
  // psi is an exact invariant, so any drift comes from interpolation and ODE
  // tolerances.
  poincare_data.sync_host();
  auto pd_h = poincare_data.view_host();

  const Real psi_floor = 1e-6;
  Real max_rel_drift = 0.0;
  for (int i = 0; i < n_traces; ++i) {
    const Real psi0 = pd_h(2, i, 0);
    const Real denom = std::max(std::abs(psi0), psi_floor);
    for (int it = 0; it <= n_turn; ++it) {
      const Real drift = std::abs(pd_h(2, i, it) - psi0) / denom;
      max_rel_drift = std::max(max_rel_drift, drift);
    }
  }

  std::cout << std::format("max relative psi drift = {:.17g}", max_rel_drift)
            << std::endl;

  // psi is an exact invariant of the axisymmetric field; the residual drift is
  // set by the Hermite interpolation error at this resolution (absolute drift
  // ~1e-8 on psi values of O(1e-3)).
  const Real tol = 1.1e-8;
  if (max_rel_drift > tol) {
    std::fprintf(stderr,
        "psi not conserved along field line: max relative drift %.17g > %.17g\n",
        max_rel_drift, tol);
    return 1;
  }

  return 0;
}  // run

}  // namespace

int main() {
  Kokkos::initialize();

  const int NR = 128;
  const int rc = run(NR, 2 * NR);

  Kokkos::finalize();
  return rc;
}

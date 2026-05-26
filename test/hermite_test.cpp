#include <cmath>
#include <iostream>

#include "AnalyticField.hpp"
#include "hFlux/FieldInterpolation.hpp"
#include "hFlux/FiniteDifferenceWeights.hpp"



void run(int nR_data, int nZ_data, Real& hR, Dim3& l2err) {
  static const int m = 2;
  Real R0 = 1.525;
  Real Z0 = -2.975;
  Real dR = 0.0345;
  Real dZ = 0.02975;

  Real R1 = R0 + 99 * dR;
  Real Z1 = Z0 + 199 * dZ;

  dR = (R1 - R0) / (nR_data-1);
  dZ = (Z1 - Z0) / (nZ_data-1);

  Real q0 = 2.1;
  Real q2 = 2.0;
  Real R_a = 3.0;
  Real E_0 = 70.0;

  AnalyticField af(q0, q2, R_a, E_0);
  FieldInterpolation<m> field_interpolation(nR_data, nZ_data, R0, Z0, dR, dZ);
  auto field_data = field_interpolation.data;
  auto hermite_data = field_interpolation.hermite_data;

  using policy2D = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>;
  Kokkos::parallel_for("setfields",
  policy2D({0,0}, {nR_data,nZ_data}),
  KOKKOS_LAMBDA(int i, int j){
    // linearize: row-major numbering
    auto sbv = Kokkos::subview(field_data, i, j, Kokkos::ALL);

    Real R = R0 + dR * i, Z = Z0 + dZ * j;
    Dim5 X = {};
    X[2] = R; X[4] = Z;
    Dim3 vB = {}, dBdR = {}, dBdZ = {}, curlB = {}, E = {};
    Real t = 0.0;
    af(X, t, vB, curlB, dBdR, dBdZ, E);
    for (int di = 0; di < sbv.extent(0); ++di)
      sbv(di) = vB[di] * R;
  });

  field_interpolation.interpolate();

  int nR_pl = 400;
  int nZ_pl = 800;
  auto corners = field_interpolation.getCorners();
  Real eps = 1e-8;
  Real R0_pl = corners[0] + eps;
  Real Z0_pl = corners[2] + eps;
  Real dR_pl = (corners[1] - eps - (corners[0] + eps)) / (nR_pl-1);
  Real dZ_pl = (corners[3] - eps - (corners[2] + eps)) / (nZ_pl-1);

  Kokkos::View<Real***, Kokkos::LayoutRight, ExecSpace> view_pl("plot", nR_pl, nZ_pl, 3);

  l2err = {};
  Kokkos::parallel_reduce("eval",
  policy2D({0,0}, {nR_pl,nZ_pl}),
  KOKKOS_LAMBDA(int i, int j, Real& err0, Real& err1, Real& err2){
    // linearize: row-major numbering
    auto sbv = Kokkos::subview(view_pl, i, j, Kokkos::ALL);

    Real R = R0_pl + dR_pl * i, Z = Z0_pl + dZ_pl * j;
    Dim5 X = {};
    X[2] = R; X[4] = Z;
    field_interpolation(sbv, X);
    Dim3 vB = {}, dBdR = {}, dBdZ = {}, curlB = {}, E = {};
    Real t = 0.0;
    af(X, t, vB, curlB, dBdR, dBdZ, E);

    err0 += pow(vB[0] - sbv(0) / R, 2);
    err1 += pow(vB[1] - sbv(1) / R, 2);
    err2 += pow(vB[2] - sbv(2) / R, 2);
  }, l2err[0], l2err[1], l2err[2]);

  l2err[0] = sqrt(l2err[0] * dR_pl * dZ_pl);
  l2err[1] = sqrt(l2err[1] * dR_pl * dZ_pl);
  l2err[2] = sqrt(l2err[2] * dR_pl * dZ_pl);

  hR = field_interpolation.hR;
}


int main() {
  Kokkos::initialize();
  int NR = 100;
  Dim3 l2err;
  Real hR;
  Real order = (2*2+2);
  for (int ix = 0; ix < 5; ++ix) {
    Real hR_new;
    Dim3 l2err_new;

    run(NR, 2*NR, hR_new, l2err_new);
    if (ix > 0) {
      Real e0 = std::abs(pow(l2err[0] / l2err_new[0], 1.0 / (order)) - hR / hR_new);
      Real e2 = std::abs(pow(l2err[2] / l2err_new[2], 1.0 / (order-1.0)) - hR / hR_new);
      std::cout << std::format("{:.17g} {:.17g} {:.17g}\n", l2err[0], l2err[1], l2err[2]);
      std::cout << std::format("{:.17g} {:.17g} {:.17g}\n", l2err_new[0], l2err_new[1], l2err_new[2]);

      if (e0 > 5.e-3) {
        std::fprintf(stderr, "B_R interpolation did not converge with order %f %le\n", order, e0);
        Kokkos::finalize();
        return 1;
      }
      if (e2 > 4.e-2) {
        std::fprintf(stderr, "B_Z interpolation did not converge with order %f %le\n", order, e2);
        Kokkos::finalize();
        return 2;
      }
    }

    l2err = l2err_new;
    hR = hR_new;

    NR *= 1.5;
  }
  Kokkos::finalize();
  return 0;
}


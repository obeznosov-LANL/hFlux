#include <cmath>
#include <iostream>

#include "AnalyticField.hpp"
#include "hFlux.hpp"

void run(int nR_data, int nZ_data, Real& hR, Kokkos::Array<Real, 4>& l2err) {
  static const int m = 2;
  static const int swidth = 7;

  using exec_space = Kokkos::DefaultExecutionSpace;

  Real R0 = 1.525;
  Real Z0 = -2.975;
  Real dR = 0.0345;
  Real dZ = 0.02975;

  Real R1 = R0 + 99 * dR;
  Real Z1 = Z0 + 199 * dZ;

  dR = (R1 - R0) / (nR_data-1);
  dZ = (Z1 - Z0) / (nZ_data-1);

  FieldData<m, swidth, exec_space> data(nR_data, nZ_data, R0, Z0, dR, dZ);

  Real q0 = 2.1;
  Real q2 = 2.0;
  Real R_a = 3.0;
  Real E_0 = 70.0;

  AnalyticField af(q0, q2, R_a, E_0);

  auto field_data = data.data;
  auto hermite_data = data.hermite_data;

  int nR = hermite_data.extent_int(3);
  int nZ = hermite_data.extent_int(4);

  using policy2D = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>;
  Kokkos::parallel_for("setfields",
  policy2D({0,0}, {nR_data,nZ_data}),
  KOKKOS_LAMBDA(int i, int j){
    // linearize: row-major numbering
    auto sbv = Kokkos::subview(field_data, i, j, Kokkos::ALL);

    Real R = R0 + dR * i, Z = Z0 + dZ * j;
    Dim3 B = {};
    af.eval(B, R, Z);
    for (int di = 0; di < sbv.extent(0); ++di)
      sbv(di) = B[di] * R;
  });

  Interpolator<m, swidth> itrp;
  itrp.interpolate(data.fd_locator,
      data.hermite_locator,
      data.data,
      data.hermite_data);

  itrp.computeFlux(data.hermite_locator,
      data.hermite_data,
      data.psi_data);

  itrp.cleanDivergence(data.hermite_locator,
      data.hermite_data);



  Kokkos::fence();

  auto hermite_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, hermite_data);
  auto psi_hermite_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, data.psi_data);

  // Find MA center
  Real R_center = 3.2;
  Real Z_center = -0.7;
  int descent = -1;
  Real Psi_min;

  Evaluator ev{data.hermite_locator};

  findMagneticAxis(R_center, Z_center,
              hermite_host,
              psi_hermite_host,
              ev,
              descent,
              Psi_min);

  std::cout << std::format("Psi_min = {:.17g}", Psi_min) << std::endl;

  Kokkos::parallel_for("Normalize psi",
  policy2D({0,0}, {nR,nZ}),
  KOKKOS_LAMBDA(int i, int j){
    data.psi_data(0,0,i,j) -= Psi_min;
  });

  l2err = {};

  int nR_pl = 400;
  int nZ_pl = 800;
  Real eps = 1e-8;
  Real R0_pl = data.hermite_locator.R0 + eps;
  Real Z0_pl = data.hermite_locator.Z0 + eps;
  Real R1_pl = data.hermite_locator.R1 - eps;
  Real Z1_pl = data.hermite_locator.Z1 - eps;
  Real dR_pl = (R1_pl - R0_pl) / (nR_pl-1);
  Real dZ_pl = (Z1_pl - Z0_pl) / (nZ_pl-1);

  Kokkos::View<Real***, Kokkos::LayoutRight, exec_space> view_B("plot_B", nR_pl, nZ_pl, 3);
  Kokkos::View<Real**, Kokkos::LayoutRight, exec_space> view_psi("plot_psi", nR_pl, nZ_pl);
  Kokkos::View<Real**, Kokkos::LayoutRight, exec_space> view_psi_exact("plot_psi_exact", nR_pl, nZ_pl);

  Kokkos::parallel_reduce("eval",
  policy2D({0,0}, {nR_pl,nZ_pl}),
  KOKKOS_LAMBDA(int i, int j, Real& err0, Real& err1, Real& err2, Real& err_psi){
    Real R = R0_pl + dR_pl * i, Z = Z0_pl + dZ_pl * j;

    Dim3 B = {}, B_exact = {};
    Real Psi = 0., Psi_exact = 0.;

    ev.evalField(B, R, Z, data.hermite_data);
    ev.evalPsi(Psi, R, Z, data.psi_data);

    af.eval(B_exact, R, Z);
    Psi_exact = af.Psi(R, Z);

    for (int d = 0; d < 3; ++d) {
      view_B(i, j, d) = B[d];
    }
    view_psi(i, j) = Psi;

    view_psi_exact(i, j) = Psi_exact;

    err0 += pow(B_exact[0] - B[0] / R, 2);
    err1 += pow(B_exact[1] - B[1] / R, 2);
    err2 += pow(B_exact[2] - B[2] / R, 2);

    err_psi += pow(Psi_exact - Psi, 2);
  }, l2err[0], l2err[1], l2err[2], l2err[3]);

  Kokkos::fence();
//  auto psi_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},view_psi);
//  auto psi_exact_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},view_psi_exact);
//  auto B_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},view_B);
//
//  std::cout << "<<<<" << std::endl;
//  for (int i = 0; i < nR_pl; ++i) {
//    for (int j = 0; j < nZ_pl; ++j) {
//      Real R = R0_pl + dR_pl * i, Z = Z0_pl + dZ_pl * j;
//      std::cout << std::format("{:.17g} {:.17g} {:.17g} {:.17g} {:.17g} {:.17g} {:.17g} ",
//          R, Z,
//          B_host(i,j,0),
//          B_host(i,j,1),
//          B_host(i,j,2),
//          psi_host(i,j),
//          psi_exact_host(i,j));
//    }
//    std::cout << std::endl;
//  }
//  std::cout << ">>>>" << std::endl;

  for (int i = 0; i < l2err.size(); ++i)
    l2err[i] = sqrt(l2err[i] * dR_pl * dZ_pl);
  hR = data.hermite_locator.dR;
}


int main() {
  Kokkos::initialize();
  int NR = 100;
  Kokkos::Array<Real, 4> l2err;
  Real hR;
  Real order = (2*2+2);
  for (int ix = 0; ix < 5; ++ix) {
    Real hR_new;
    Kokkos::Array<Real, 4> l2err_new;

    run(NR, 2*NR, hR_new, l2err_new);
    if (ix > 0) {
      Real o1 = log(l2err[0] / l2err_new[0]) / log(hR / hR_new);
      Real o2 = log(l2err[2] / l2err_new[2]) / log(hR / hR_new);
      Real o3 = log(l2err[3] / l2err_new[3]) / log(hR / hR_new);
      std::cout << std::format("{:.17g} {:.17g} {:.17g} {:.17g}\n", l2err[0], l2err[1], l2err[2], l2err[3]);
      std::cout << std::format("{:.17g} {:.17g} {:.17g} {:.17g}\n", l2err_new[0], l2err_new[1], l2err_new[2], l2err_new[3]);

      if ((int)round(o1) < order && l2err_new[3] > 1e-12) {
        std::fprintf(stderr, "B_R interpolation did not converge with order %f %le\n", order, o1);
        Kokkos::finalize();
        return 1;
      }
      if ((int)round(o2) < order - 1 && l2err_new[3] > 1e-12) {
        std::fprintf(stderr, "B_Z interpolation did not converge with order %f %le\n", order - 1, o2);
        Kokkos::finalize();
        return 2;
      }
      if ((int)round(o3) < order && l2err_new[3] > 1e-10) {
        std::fprintf(stderr, "Psi did not converge with order %f %le %le\n", order, o3, hR/hR_new);
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


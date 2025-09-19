#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>
#include "hFlux/dopri.hpp"
#include "hFlux/FieldInterpolation.hpp"
#include "hFlux/c_wrapper.h"

constexpr int m = 2;

void hflux_kokkos_init() {
  Kokkos::initialize();
}

void hflux_kokkos_finalize() {
  Kokkos::finalize();
}

void hflux_init(
    const int nR_data,
    const int nZ_data,
    const int nfields,
    const int nphi_data,
    const int nt,
    const double R0,
    const double Z0,
    const double dR,
    const double dZ,
    void ** fi) {
  *fi = (void*) new FieldInterpolation<m>(nR_data, nZ_data, nfields, nphi_data, nt, R0, Z0, dR, dZ);
}

void hflux_interpolate(
    void* fi, double* raw_field_data) {

    auto pFi = static_cast<FieldInterpolation<m>*>(fi);

    Kokkos::View<double ******, Kokkos::LayoutLeft, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h_view(raw_field_data, pFi->nR_data, pFi->nZ_data, pFi->nfields,
               pFi->ndims, pFi->nphi_data, pFi->nt);
    Kokkos::deep_copy(pFi->getDataRef(), h_view);
    Kokkos::fence();

    pFi->interpolate();
}

void hflux_getcorners(void* fi, double* corners) {
    auto pFi = static_cast<FieldInterpolation<m>*>(fi);
    auto corners_ = pFi -> getCorners();
    for (int i = 0; i < 4; ++i) corners[i] = corners_[i];
}



void hflux_compute_poincare(
    void* fi,
    const double r0,
    const double dr,
    const int n_r,
    const int n_theta,
    const int n_turn,
    double* poincare_data) {
  const auto pFi = static_cast<FieldInterpolation<m>*>(fi);

  Kokkos::View<double******, ExecSpace> psi_hermite_data("psi",
      pFi->hermite_data.extent(0),
      pFi->hermite_data.extent(1),
      pFi->hermite_data.extent(2) + 1,
      pFi->hermite_data.extent(3),
      pFi->hermite_data.extent(6),
      pFi->hermite_data.extent(7));
  Kokkos::parallel_for("psi_compute",
  Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0,0}, {pFi->nphi_data,pFi->nt}),
  KOKKOS_LAMBDA(int k, int ti){
    auto sbv_hermite_data = Kokkos::subview((*pFi).hermite_data, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, 0, Kokkos::ALL, k, ti);
    auto sbv_psi_data = Kokkos::subview(psi_hermite_data, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, k, ti);
    computeFlux<m>(sbv_hermite_data, sbv_psi_data, (*pFi).hR, (*pFi).hZ);
  });

  Kokkos::fence();


  using HostMemSpace = Kokkos::HostSpace::memory_space;
  auto hermite_data_h = Kokkos::create_mirror_view_and_copy(HostMemSpace{}, (*pFi).hermite_data);
  auto psi_data_h = Kokkos::create_mirror_view_and_copy(HostMemSpace{}, psi_hermite_data);

  Dim5 X = {0.0, 0.0, 3.2, 0.0, 1.0};
  Real Psi0 = (*pFi).isd(X, hermite_data_h, psi_data_h);

  std::printf("Psi = %le\nX = %le %le %le %le %le\n", Psi0, X[0], X[1], X[2], X[3], X[4]);

  struct FieldLine {
    const FieldInterpolation<m> pFi;
    KOKKOS_INLINE_FUNCTION ERROR_CODE operator() (const Real phi, const Dim2 X, Dim2& dXdphi) const  {
      Dim3 B_;
      pFi.evalB(B_, {0.0, 0.0, X[0], phi, X[1]}, pFi.hermite_data);
      dXdphi[0] = (B_[0]) / B_[1] * X[0];
      dXdphi[1] = (B_[2]) / B_[1] * X[0];
      return ERROR_CODE::SUCCESS;
    }
    typedef Dim2 value_type;
  };

  using DevMemSpace = Kokkos::DefaultExecutionSpace::memory_space;

  Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
    X_trace_h(poincare_data, n_r * n_theta, 2,  n_turn+1);

  auto X_trace = create_mirror_view_and_copy(DevMemSpace{}, X_trace_h);
  Kokkos::fence();

  Kokkos::parallel_for("poincare", n_r * n_theta,
  KOKKOS_LAMBDA(int i){
    Kokkos::Array<Dim2, 10> work;
    FieldLine f(*pFi);
    auto sbv = Kokkos::subview(X_trace, i, Kokkos::ALL, 0);
    Real r = (i  % n_r) * dr;
    Real theta = (i  / n_r) * 2 * M_PI / n_theta;
    sbv(0) = X[2] + r * cos(theta);
    sbv(1) = X[4] + r * sin(theta);
    for (int it = 0; it < n_turn; ++it) {
      auto sbv = Kokkos::subview(X_trace, i, Kokkos::ALL, it);
      Dim2 trace = {sbv(0), sbv(1)};

      solve_dopri5(f, trace, 0.0, 2.0 * M_PI, 1e-8, 1e-9, 1e-6, 1e-9, 2000000, work);
      sbv = Kokkos::subview(X_trace, i, Kokkos::ALL, it+1);
      sbv(0) = trace[0];
      sbv(1) = trace[1];
    }
  });




  Kokkos::fence();
  Kokkos::deep_copy(X_trace_h, X_trace);
}



void hflux_field_eval(
    void* fi,
    const int N,
    const double* R_mesh,
    const double* phi_mesh,
    const double* Z_mesh,
    const double* t_mesh,
    double* mesh_value) {

  const auto pFi = static_cast<FieldInterpolation<m>*>(fi);
  using MeshView = Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using MeshValueView = Kokkos::View<double*****, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  const MeshView R_h(R_mesh, N);
  const MeshView phi_h(phi_mesh, N);
  const MeshView Z_h(Z_mesh, N);
  const MeshView t_h(t_mesh, N);

  using DevMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
  auto R = Kokkos::create_mirror_view_and_copy(DevMemSpace{}, R_h);
  auto phi = Kokkos::create_mirror_view_and_copy(DevMemSpace{}, phi_h);
  auto Z = Kokkos::create_mirror_view_and_copy(DevMemSpace{}, Z_h);
  auto t = Kokkos::create_mirror_view_and_copy(DevMemSpace{}, t_h);

  MeshValueView B_h(mesh_value, N, pFi->nfields, pFi->ndims, pFi->nphi_data, pFi->nt);
  auto B = Kokkos::create_mirror_view_and_copy(DevMemSpace{}, B_h);

  Kokkos::fence();
  Kokkos::parallel_for("eval", N,
  KOKKOS_LAMBDA(int i){
    auto sbv = Kokkos::subview(B,
             i, 0, Kokkos::ALL, 0, 0);
    Kokkos::Array<Real, 3> B_;
    (*pFi).evalB(B_, {0.0, 0.0, R(i), phi(i), Z(i)}, (*pFi).hermite_data);
    for (int di = 0; di < 3; ++di) sbv(di) = B_[di];
  });

  Kokkos::fence();
  Kokkos::deep_copy(B_h, B);

}

void hflux_psi_eval(
    void* fi,
    const int N,
    const double* R_mesh,
    const double* phi_mesh,
    const double* Z_mesh,
    const double* t_mesh,
    double* mesh_value,
    double* center_R, double* center_Z) {

  static const int m = 2;

  const auto pFi = static_cast<FieldInterpolation<m>*>(fi);
  using MeshView = Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using MeshValueView = Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  const MeshView R_h(R_mesh, N);
  const MeshView phi_h(phi_mesh, N);
  const MeshView Z_h(Z_mesh, N);
  const MeshView t_h(t_mesh, N);

  using DevMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
  auto R = Kokkos::create_mirror_view_and_copy(DevMemSpace{}, R_h);
  auto phi = Kokkos::create_mirror_view_and_copy(DevMemSpace{}, phi_h);
  auto Z = Kokkos::create_mirror_view_and_copy(DevMemSpace{}, Z_h);
  auto t = Kokkos::create_mirror_view_and_copy(DevMemSpace{}, t_h);

  MeshValueView Psi_h(mesh_value, N, pFi->nphi_data, pFi->nt);
  auto Psi = Kokkos::create_mirror_view_and_copy(DevMemSpace{}, Psi_h);

  Kokkos::View<double******, ExecSpace> psi_hermite_data("psi",
      pFi->hermite_data.extent(0),
      pFi->hermite_data.extent(1),
      pFi->hermite_data.extent(2) + 1,
      pFi->hermite_data.extent(3),
      pFi->hermite_data.extent(6),
      pFi->hermite_data.extent(7));
  Kokkos::parallel_for("psi_compute",
  Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0,0}, {pFi->nphi_data,pFi->nt}),
  KOKKOS_LAMBDA(int k, int ti){
    auto sbv_hermite_data = Kokkos::subview((*pFi).hermite_data, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, 0, Kokkos::ALL, k, ti);
    auto sbv_psi_data = Kokkos::subview(psi_hermite_data, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, k, ti);
    computeFlux<m>(sbv_hermite_data, sbv_psi_data, (*pFi).hR, (*pFi).hZ);
  });

  Kokkos::fence();
  Kokkos::parallel_for("eval", N,
  KOKKOS_LAMBDA(int i){
    (*pFi).evalPsi(Psi(i, 0, 0), {0.0, 0.0, R(i), phi(i), Z(i)}, psi_hermite_data);
  });

  *center_R = pFi->hR0 + (pFi->nR_hermite_data/2 + 0.5) * pFi->hR;
  *center_Z = pFi->hZ0 + (pFi->nZ_hermite_data/2 + 0.5) * pFi->hZ;

  Kokkos::fence();
  Kokkos::deep_copy(Psi_h, Psi);

}

void hflux_destroy(void* fi) {
  auto pFi = static_cast<FieldInterpolation<m> *>(fi);

  delete pFi;
}

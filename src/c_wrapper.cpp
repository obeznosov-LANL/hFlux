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
    const double R0,
    const double Z0,
    const double dR,
    const double dZ,
    void ** fi) {
  *fi = (void*) new FieldInterpolation<m>(nR_data, nZ_data, R0, Z0, dR, dZ);
}

void hflux_interpolate(
    void* fi, double* raw_field_data) {

    auto pFi = static_cast<FieldInterpolation<m>*>(fi);

    Kokkos::View<double ***, Kokkos::LayoutRight, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h_view(raw_field_data, pFi->nR_data, pFi->nZ_data,  pFi->ndims);
    Kokkos::deep_copy(pFi->data, h_view);
    pFi->interpolate();
}

double hflux_get_psi_extrema(
    void* fi, double* x, int sign) {
  auto pFi = static_cast<FieldInterpolation<m>*>(fi);

  // psi_hermite_data: (idR, idZ, k, ti, iR, iZ)
  Kokkos::View<double****, Kokkos::LayoutLeft, ExecSpace> psi_hermite_data("psi",
      pFi->hermite_data.extent(0),       // idR: 2*m+3
      pFi->hermite_data.extent(1),       // idZ: 2*m+3
      pFi->hermite_data.extent(3),       // iR: nR
      pFi->hermite_data.extent(4));      // iZ: nZ
  computeFlux<m>(pFi->hermite_data, psi_hermite_data, (*pFi).hR, (*pFi).hZ);

  using HostMemSpace = Kokkos::HostSpace::memory_space;
  auto hermite_data_h = Kokkos::create_mirror_view_and_copy(HostMemSpace{}, (*pFi).hermite_data);
  auto psi_data_h = Kokkos::create_mirror_view_and_copy(HostMemSpace{}, psi_hermite_data);

  Dim5 X0 = {0.0, 0.0, x[0], 0.0, x[1]};

  double psi0 = (*pFi).isd(X0, hermite_data_h, psi_data_h, sign);

  x[0] = X0[2];
  x[1] = X0[4];
  return psi0;
}

void hflux_getcorners(void* fi, double* corners) {
    auto pFi = static_cast<FieldInterpolation<m>*>(fi);
    auto corners_ = pFi -> getCorners();
    for (int i = 0; i < 4; ++i) corners[i] = corners_[i];
}

void hflux_compute_poincare(
    void* fi,
    const int n_traces,
    const int n_turn,
    double* poincare_data) {
  const auto pFi = static_cast<FieldInterpolation<m>*>(fi);


  struct FieldLine {
    const FieldInterpolation<m> pFi;
    KOKKOS_INLINE_FUNCTION ErrorCode operator() (const Real phi, const Dim2 X, Dim2& dXdphi) const  {
      Dim3 B_ = {};
      // pFi(B_, {0.0, 0.0, X[0], phi, X[1]});
      dXdphi[0] = (B_[0]) / B_[1] * X[0];
      dXdphi[1] = (B_[2]) / B_[1] * X[0];
      return ErrorCode::Success;
    }
    typedef Dim2 value_type;
  };

  using DevMemSpace = Kokkos::DefaultExecutionSpace::memory_space;

  Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
    X_trace_h(poincare_data, n_traces, 2,  n_turn+1);

  auto X_trace = create_mirror_view_and_copy(DevMemSpace{}, X_trace_h);
  Kokkos::parallel_for("poincare", n_traces,
  KOKKOS_LAMBDA(int i){
    Kokkos::Array<Dim2, 10> work;
    FieldLine f(*pFi);
    for (int it = 0; it < n_turn; ++it) {
      auto sbv = Kokkos::subview(X_trace, i, Kokkos::ALL, it);
      Dim2 trace = {sbv(0), sbv(1)};

      solve_dopri5(f, trace, 0.0, 2.0 * M_PI, 1e-10, 1e-12, 1e-6, 1e-10, 2000000, work);

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
    double* mesh_value) {

  const auto pFi = static_cast<FieldInterpolation<m>*>(fi);
  using MeshView = Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using MeshValueView = Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  const MeshView R_h(R_mesh, N);
  const MeshView phi_h(phi_mesh, N);
  const MeshView Z_h(Z_mesh, N);

  using DevMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
  auto R = Kokkos::create_mirror_view_and_copy(DevMemSpace{}, R_h);
  auto phi = Kokkos::create_mirror_view_and_copy(DevMemSpace{}, phi_h);
  auto Z = Kokkos::create_mirror_view_and_copy(DevMemSpace{}, Z_h);

  MeshValueView B_h(mesh_value, N, pFi->ndims);
  auto B = Kokkos::create_mirror_view_and_copy(DevMemSpace{}, B_h);

  Kokkos::fence();
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  Kokkos::parallel_for("eval",
  Kokkos::RangePolicy<ExecSpace>(0, N),
  KOKKOS_LAMBDA(int i){
    Dim3 B_ = {};
    // (*pFi)(B_, {0.0, 0.0, R(i), phi(i), Z(i)});
    for (int d = 0; d < 3; ++d) B(i, d) = B_[d];
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
    double* mesh_value,
    double* center_R, double* center_Z) {

  static const int m = 2;

  const auto pFi = static_cast<FieldInterpolation<m>*>(fi);
  using MeshView = Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using MeshValueView = Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  const MeshView R_h(R_mesh, N);
  const MeshView phi_h(phi_mesh, N);
  const MeshView Z_h(Z_mesh, N);

  using DevMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
  auto R = Kokkos::create_mirror_view_and_copy(DevMemSpace{}, R_h);
  auto phi = Kokkos::create_mirror_view_and_copy(DevMemSpace{}, phi_h);
  auto Z = Kokkos::create_mirror_view_and_copy(DevMemSpace{}, Z_h);

  MeshValueView Psi_h(mesh_value, N);
  auto Psi = Kokkos::create_mirror_view_and_copy(DevMemSpace{}, Psi_h);

  // psi_hermite_data: (idR, idZ, k, ti, iR, iZ)
  Kokkos::View<double****, Kokkos::LayoutLeft, ExecSpace> psi_hermite_data("psi",
      pFi->hermite_data.extent(0),       // idR: 2*m+3
      pFi->hermite_data.extent(1),       // idZ: 2*m+3
      pFi->hermite_data.extent(3),       // iR: nR
      pFi->hermite_data.extent(4));      // iZ: nZ
  computeFlux<m>(pFi->hermite_data, psi_hermite_data, (*pFi).hR, (*pFi).hZ);

  Kokkos::fence();
  Kokkos::parallel_for("eval", N,
  KOKKOS_LAMBDA(int i){
    pFi->evalPsi(Psi(i), {0.0, 0.0, R(i), phi(i), Z(i)}, psi_hermite_data);
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

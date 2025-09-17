#include <Kokkos_Core.hpp>
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
    const double* poincare_data) {

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
             i, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    (*pFi)(sbv, {0.0, 0.0, R(i), phi(i), Z(i)});
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
    auto sbv = Kokkos::subview(Psi,
             i, Kokkos::ALL, Kokkos::ALL);
    (*pFi).evalPsi(sbv, {0.0, 0.0, R(i), phi(i), Z(i)}, psi_hermite_data);
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

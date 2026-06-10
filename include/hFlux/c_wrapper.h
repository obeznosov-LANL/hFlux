//========================================================================================
// (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

#ifndef HFLUX_C_WRAPPER_H_
#define HFLUX_C_WRAPPER_H_

#ifdef __cplusplus
extern "C" {
#endif

  void hflux_kokkos_init();
  void hflux_kokkos_finalize();

  void hflux_init(const int nR_data, const int nZ_data, const double R0,
                  const double Z0, const double dR, const double dZ, void **fi);

  void hflux_interpolate(void *fi, double *raw_field_data);
  void hflux_getcorners(void *fi, double *corners);
  double hflux_get_psi_extrema(void* fi, double* x, int sign);

  void hflux_compute_poincare(void* fi, const int n_traces, const int n_turn, double* poincare_data);

  void hflux_field_eval(void *fi, const int N, const double *R_mesh,
                        const double *phi_mesh, const double *Z_mesh,
                        double *mesh_value);

  void hflux_psi_eval(void *fi, const int N, const double *R_mesh,
                        const double *phi_mesh, const double *Z_mesh,
                        double *mesh_value, double* center_R, double* center_Z);

  void hflux_destroy(void* fi);

#ifdef __cplusplus
}
#endif

#endif

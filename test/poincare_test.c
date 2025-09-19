#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hFlux/c_wrapper.h"

typedef double Dim3[3];

void run(int nR_data, int nZ_data, double* phR, Dim3 l2err) {
  void *fi_data;

  int nfields = 2;
  int nphi_data = 1;
  int nt = 1;
  int ndim  = 3;
  double R0 = 1.525;
  double Z0 = -2.975;
  double dR = 0.0345;
  double dZ = 0.02975;

  double R1 = R0 + 99 * dR;
  double Z1 = Z0 + 199 * dZ;

  dR = (R1 - R0) / (nR_data-1);
  dZ = (Z1 - Z0) / (nZ_data-1);

  double q0 = 2.1;
  double q2 = 2.0;
  double R_a = 3.0;
  double E_0 = 70.0;

  hflux_init(nR_data, nZ_data, nfields, nphi_data, nt, R0, Z0, dR, dZ,
             &fi_data);

  double * raw_field_data = (double*) malloc(nR_data * nZ_data * nfields * ndim * nphi_data * nt  * sizeof(double));

  for (int i = 0; i < nR_data; ++i)
    for (int j = 0; j < nZ_data; ++j) {
      double R = R0 + dR * i, Z = Z0 + dZ * j;
      double q = 2.1 + 2.0 * (R - 3.0) * (R - 3.0) + 2.0 * Z * Z;
      for (int fi = 0; fi < nfields; ++fi)
        for (int di = 0; di < ndim; ++di)
          for (int k = 0; k < nphi_data; ++k)
            for (int ti = 0; ti < nt; ++ti) {
              int ii = i + nR_data * (j + nZ_data * (fi + nfields * (di + ndim * (k + nphi_data * ti))));
              if (di == 0) raw_field_data[ii] = -Z / q;
              else if (di == 1) raw_field_data[ii] = 3.0;
              else  raw_field_data[ii] = (R - 3.0) / q;
            }
    }

  hflux_interpolate(fi_data, raw_field_data);

  int n_r = 100;
  int n_theta = 5;
  double dr = 0.01;
  int n_turn  = 1000;
  int N = n_r * n_theta * (n_turn+1);

  double * poincare_data = (double*) malloc(2*N * sizeof(double));

  hflux_compute_poincare(fi_data, 0.0, dr, n_r, n_theta, n_turn, poincare_data);


//  l2err[0] = 0.0;
//  l2err[1] = 0.0;
//  l2err[2] = 0.0;
//  for (int i = 0; i < nR_mesh; ++i)
//    for (int j = 0; j < nZ_mesh; ++j) {
//      int ii = i + j * nR_mesh;
//      double R = R_mesh[ii], Z = Z_mesh[ii];
//      int jj = i + nR_mesh * (j + nZ_mesh * (0 + nfields * (0 + ndim * (0 + nphi_mesh * 0))));
//      double q = 2.1 + 2.0 * (R - 3.0) * (R - 3.0) + 2.0 * Z * Z;
//      l2err[0] += pow(    -Z / q / R - mesh_value[jj], 2);
//      jj = i + nR_mesh * (j + nZ_mesh * (0 + nphi_mesh * 0));
//      l2err[1] += pow(log(q) / q2 * 0.5 - Psi0  - mesh_value_psi[jj], 2);
//      jj = i + nR_mesh * (j + nZ_mesh * (0 + nfields * (2 + ndim * (0 + nphi_mesh * 0))));
//      l2err[2] += pow((R-3.0)/ q / R - mesh_value[jj], 2);
//    }
//
//  l2err[0] = sqrt(l2err[0] * dR_mesh * dZ_mesh);
//  l2err[1] = sqrt(l2err[1] * dR_mesh * dZ_mesh);
//  l2err[2] = sqrt(l2err[2] * dR_mesh * dZ_mesh);
//
//  *phR = 6 * dR; // default stencil width is 7

  free(poincare_data);
  free(raw_field_data);
  hflux_destroy(fi_data);
}

int main(int argc, char **argv) {
  int NR = 100;
  Dim3 l2err;
  double hR;
  double order = 2 * 2 + 2;

  hflux_kokkos_init();

  for (int ix = 0; ix < 1; ++ix) {
    double hR_new;
    Dim3 l2err_new;

    run(NR, 2*NR, &hR_new, l2err_new);
    if (ix > 0) {
      if (fabs(pow(l2err[0] / l2err_new[0], 1.0 / (order - 0.0))  - hR / hR_new) > 5.e-3) {
        fprintf(stderr, "B_R interpolation did not converge with order %f\n", order);
        fprintf(stderr, "l2err[0] =%le, l2err_new[0] = %le\n", l2err[0], l2err_new[0]);
        hflux_kokkos_finalize();
        return 1;
      }
      if (fabs(pow(l2err[1] / l2err_new[1], 1.0 / (order + 0.0)) - hR / hR_new) > 4.e-2) {
        fprintf(stderr, "Psi interpolation did not converge with order %f\n", order + 0.0);
        fprintf(stderr, "l2err[0] =%le, l2err_new[0] = %le\n", l2err[1], l2err_new[1]);
        hflux_kokkos_finalize();
        return 2;
      }
      if (fabs(pow(l2err[2] / l2err_new[2], 1.0 / (order - 1.0)) - hR / hR_new) > 4.e-2) {
        fprintf(stderr, "B_Z interpolation did not converge with order %f\n", order - 1.0);
        hflux_kokkos_finalize();
        return 2;
      }
    }

    memcpy(l2err, l2err_new, sizeof l2err);
    hR = hR_new;

    NR *= 1.5;
  }

  hflux_kokkos_finalize();
  return 0;
}


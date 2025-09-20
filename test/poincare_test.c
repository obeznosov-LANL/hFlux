#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hFlux/c_wrapper.h"

typedef double Dim3[3];

void run(int nR_data, int nZ_data, double* l2err) {
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

  double * R_poincare = (double*) malloc(sizeof(double) * N);
  double * Z_poincare = (double*) malloc(sizeof(double) * N);
  double * phi_mesh = (double*) malloc(sizeof(double) * N);
  double * t_mesh = (double*) malloc(sizeof(double) * N);
  double * Psi_poincare = (double*) malloc(sizeof(double) * N);

  for (int i = 0; i < n_turn+1; ++i) {
    for (int j = 0; j < n_r * n_theta; ++j) {
      R_poincare[i + j * (n_turn+1)] = poincare_data[j + n_r * n_theta * (0 + 2 * i)];
      Z_poincare[i + j * (n_turn+1)] = poincare_data[j + n_r * n_theta * (1 + 2 * i)];
    }
  }

  double center_R, center_Z;

  hflux_psi_eval(fi_data, N, R_poincare, phi_mesh, Z_poincare, t_mesh, Psi_poincare, &center_R, &center_Z);

  (*l2err) = 0.0;
  for (int j = 0; j < n_r * n_theta; ++j) {//; j < n_r * n_theta; ++j) {
    double Psi0 = Psi_poincare[j * (n_turn+1)];
    double r = sqrt(pow(R_poincare[j * (n_turn+1)] - 3.0, 2) + pow(Z_poincare[j * (n_turn+1)], 2));
    (*l2err) += pow((Psi_poincare[n_turn + j * (n_turn+1)] - Psi0) * 2*M_PI * r * dr / n_theta, 2);
  }
  (*l2err) = sqrt(*l2err);

  free(R_poincare);
  free(Z_poincare);
  free(t_mesh);
  free(phi_mesh);
  free(Psi_poincare);
  free(poincare_data);
  free(raw_field_data);
  hflux_destroy(fi_data);
}

int main(int argc, char **argv) {
  int NR = 100;
  double l2err;

  hflux_kokkos_init();

  for (int ix = 0; ix < 5; ++ix) {
    run(NR, 2*NR, &l2err);
    if (fabs(l2err) > 1e-8) {
      fprintf(stderr, "Psi is not conserved %le\n", l2err);
      hflux_kokkos_finalize();
      return 1;
    }
    NR *= 1.5;
  }

  hflux_kokkos_finalize();
  return 0;
}


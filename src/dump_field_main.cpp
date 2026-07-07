#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <Kokkos_Core.hpp>

#include "hFlux/common.hpp"
#include "AnalyticField.hpp"

// Dumps the resonant analytic magnetic field into the plain-text B-field input
// format consumed by poincare_tool:
//
//   NR NZ Nphi R0 Z0 dR dZ            (header line)
//   B_R B_phi B_Z                     (NR*NZ*Nphi lines; i outer, j, iphi inner)
//
// Values are the physical B components (not R*B); poincare_tool multiplies by R
// on read. Grid and physics are hardcoded to the poincare_test resonant case.
int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <out.txt>\n", argv[0]);
    return 1;
  }
  const char* out_path = argv[1];

  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    // Grid (matches poincare_test.cpp).
    const int NR = 64;
    const int NZ = 128;
    const int Nphi = 9;

    const Real R0 = 1.525;
    const Real Z0 = -2.975;
    const Real R1 = R0 + 99.0 * 0.0345;
    const Real Z1 = Z0 + 199.0 * 0.02975;
    const Real dR = (R1 - R0) / static_cast<Real>(NR - 1);
    const Real dZ = (Z1 - Z0) / static_cast<Real>(NZ - 1);
    const Real dphi = 2.0 * M_PI / static_cast<Real>(Nphi);

    // Physics (matches poincare_test.cpp resonant case).
    const Real q0 = 1.98;
    const Real q2 = 2.0;
    const Real R_a = 3.0;
    const Real E_0 = 70.0;
    const Real perturb_amp = 0.05;

    AnalyticField af(q0, q2, R_a, E_0, perturb_amp, PerturbationKind::Resonant);

    std::FILE* out = std::fopen(out_path, "w");
    if (out == nullptr) {
      std::fprintf(stderr, "error: cannot open %s for writing\n", out_path);
      status = 2;
    } else {
      std::fprintf(out, "%d %d %d %.17g %.17g %.17g %.17g\n", NR, NZ, Nphi, R0,
                   Z0, dR, dZ);
      for (int i = 0; i < NR; ++i) {
        const Real R = R0 + dR * static_cast<Real>(i);
        for (int j = 0; j < NZ; ++j) {
          const Real Z = Z0 + dZ * static_cast<Real>(j);
          for (int iphi = 0; iphi < Nphi; ++iphi) {
            const Real phi = dphi * static_cast<Real>(iphi);
            Dim3 B = {};
            af.eval(B, R, Z, phi);
            std::fprintf(out, "%.17g %.17g %.17g\n", B[0], B[1], B[2]);
          }
        }
      }
      std::fclose(out);
    }
  }
  Kokkos::finalize();
  return status;
}

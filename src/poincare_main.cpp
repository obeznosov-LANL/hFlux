#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "hFlux/FieldData3D.hpp"
#include "hFlux/FourierEvaluator.hpp"
#include "hFlux/Interpolator.hpp"
#include "hFlux/dopri.hpp"

namespace {

// Field-line ODE in the toroidal angle phi (identical to poincare_test.cpp):
//   dR/dphi = (R B_R) / (R B_phi) * R = RB[0] / RB[1] * R
//   dZ/dphi = (R B_Z) / (R B_phi) * R = RB[2] / RB[1] * R
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

// Reads the plain-text B-field input file:
//   NR NZ Nphi R0 Z0 dR dZ            (header line)
//   B_R B_phi B_Z                     (NR*NZ*Nphi lines; i outer, j, iphi inner)
// B are the physical components; the caller multiplies by R for the sampler.
bool readBField(const std::string& path, int& NR, int& NZ, int& Nphi, Real& R0,
                Real& Z0, Real& dR, Real& dZ, std::vector<Real>& B) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "error: cannot open B-field file %s\n", path.c_str());
    return false;
  }
  if (!(in >> NR >> NZ >> Nphi >> R0 >> Z0 >> dR >> dZ)) {
    std::fprintf(stderr, "error: bad header in %s\n", path.c_str());
    return false;
  }
  if (NR < 2 || NZ < 2 || Nphi < 1) {
    std::fprintf(stderr, "error: invalid grid %d %d %d in %s\n", NR, NZ, Nphi,
                 path.c_str());
    return false;
  }

  const std::size_t ncomp = static_cast<std::size_t>(NR) * NZ * Nphi * 3;
  B.resize(ncomp);
  for (std::size_t k = 0; k < ncomp; ++k) {
    if (!(in >> B[k])) {
      std::fprintf(stderr,
                   "error: expected %zu B values in %s, got %zu\n", ncomp,
                   path.c_str(), k);
      return false;
    }
  }
  return true;
}

// Reads the seed file:
//   n_turn                            (first line)
//   R Z                               (one seed per line)
bool readSeeds(const std::string& path, int& n_turn,
               std::vector<Dim2>& seeds) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "error: cannot open seed file %s\n", path.c_str());
    return false;
  }
  if (!(in >> n_turn) || n_turn < 1) {
    std::fprintf(stderr, "error: bad n_turn in %s\n", path.c_str());
    return false;
  }
  Real R = 0.0, Z = 0.0;
  while (in >> R >> Z) {
    seeds.push_back({R, Z});
  }
  if (seeds.empty()) {
    std::fprintf(stderr, "error: no seeds in %s\n", path.c_str());
    return false;
  }
  return true;
}

// Appends ".fine.txt" to the output path, stripping a trailing ".txt".
std::string fineMeshPath(const std::string& out_path) {
  const std::string suffix = ".txt";
  std::string base = out_path;
  if (base.size() >= suffix.size() &&
      base.compare(base.size() - suffix.size(), suffix.size(), suffix) == 0) {
    base.erase(base.size() - suffix.size());
  }
  return base + ".fine.txt";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s <bfield_file> <seed_file> <output_txt>\n",
                 argv[0]);
    return 1;
  }
  const std::string bfield_path = argv[1];
  const std::string seed_path = argv[2];
  const std::string out_path = argv[3];

  // Interpolation order parameters are fixed at compile time (as in the tests).
  static constexpr int m = 4;
  static constexpr int swidth = 7;

  int NR = 0, NZ = 0, Nphi = 0, n_turn = 0;
  Real R0 = 0.0, Z0 = 0.0, dR = 0.0, dZ = 0.0;
  std::vector<Real> B_host;
  std::vector<Dim2> seeds;

  if (!readBField(bfield_path, NR, NZ, Nphi, R0, Z0, dR, dZ, B_host)) {
    return 2;
  }
  if (!readSeeds(seed_path, n_turn, seeds)) {
    return 3;
  }

  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    using exec_space = Kokkos::DefaultExecutionSpace;
    using policy2D = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>;
    using policy3D = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>;
    using FieldData = FieldData3D<m, swidth, exec_space>;

    const Real dphi = 2.0 * M_PI / static_cast<Real>(Nphi);

    FieldData data(NR, NZ, Nphi, R0, Z0, dR, dZ, dphi);
    Kokkos::DualView<Real***, Kokkos::LayoutRight, exec_space> fourier_data(
        "fourier_data", NR, NZ, (3 + 1) * Nphi);

    // Fill sample data on the host from the file, storing R*B.
    auto sample_host = data.data.view_host();
    for (int i = 0; i < NR; ++i) {
      const Real R = R0 + dR * static_cast<Real>(i);
      for (int j = 0; j < NZ; ++j) {
        for (int iphi = 0; iphi < Nphi; ++iphi) {
          const std::size_t base =
              ((static_cast<std::size_t>(i) * NZ + j) * Nphi + iphi) * 3;
          for (int d = 0; d < 3; ++d) {
            sample_host(i, j, FieldData::sample_component(iphi, d)) =
                R * B_host[base + d];
          }
        }
      }
    }
    data.data.modify_host();
    data.data.sync_device();

    data.sampleToFourier(data.data.view_device(), fourier_data.view_device());
    fourier_data.modify_device();

    Interpolator<m, swidth> itrp;
    itrp.interpolate(data.fd_locator, data.hermite_locator,
                     fourier_data.view_device(),
                     data.hermite_data.view_device());
    data.hermite_data.modify_device();


    int R_center = -1;
    int Z_center = -1;
  //  data.hermite_locator.locateCell(1.61, 0.0, R_center, Z_center);

    // Divergence clean each Fourier channel (stride-4 layout, Nphi channels).
    constexpr int component_stride = 4;
    itrp.cleanDivergence(data.hermite_locator, data.hermite_data.view_device(),
                         /*component0=*/0, /*nfields=*/Nphi, component_stride,
                         Z_center);
    itrp.computeChi(data.hermite_locator, data.hermite_data.view_device(),
                    /*component0=*/0, /*nfields=*/Nphi, component_stride,
                    R_center, Z_center);
    data.DifferentiatePhiCorrection(data.hermite_data.view_device());
    data.hermite_data.modify_device();

    FourierEvaluator ev{data.hermite_locator};

    // ----- Poincare trace -----
    const int n_traces = static_cast<int>(seeds.size());
    Kokkos::DualView<Real***> poincare_data("poincare_data", 2, n_traces,
                                            n_turn + 1);
    auto pd_h = poincare_data.view_host();
    for (int i = 0; i < n_traces; ++i) {
      pd_h(0, i, 0) = seeds[i][0];
      pd_h(1, i, 0) = seeds[i][1];
    }
    poincare_data.modify_host();
    poincare_data.sync_device();
    auto pd_d = poincare_data.view_device();

    using FieldLineT = FieldLine<decltype(data.hermite_data.view_device())>;
    FieldLineT f{ev, data.hermite_data.view_device()};

    Kokkos::parallel_for(
        "poincare", n_traces, KOKKOS_LAMBDA(int i) {
          Kokkos::Array<Dim2, 10> work;
          Dim2 trace = {pd_d(0, i, 0), pd_d(1, i, 0)};
          for (int it = 0; it < n_turn; ++it) {
            auto ret = solve_dopri5(f, trace, 0.0, 2.0 * M_PI, 1e-8, 1e-9, 1e-6, 1e-10,
                         2000000, work);
//            auto ret = solve_dopri5_fixed(f, trace, 0.0, 2.0 * M_PI, 1e-6, work);
            if (ret != ErrorCode::Success) break;
            pd_d(0, i, it + 1) = trace[0];
            pd_d(1, i, it + 1) = trace[1];
          }
        });
    poincare_data.modify_device();
    poincare_data.sync_host();

    std::FILE* out = std::fopen(out_path.c_str(), "w");
    if (out == nullptr) {
      std::fprintf(stderr, "error: cannot open %s for writing\n",
                   out_path.c_str());
      status = 4;
    } else {
      std::fprintf(out, "# trace turn R Z\n");
      for (int i = 0; i < n_traces; ++i) {
        for (int it = 0; it <= n_turn; ++it) {
          std::fprintf(out, "%d %d %.17g %.17g\n", i, it, pd_h(0, i, it),
                       pd_h(1, i, it));
        }
      }
      std::fclose(out);
    }

    // ----- Fine-mesh field dump (same mesh as fourier_test_cpp) -----
    const int nR_pl = 160;
    const int nZ_pl = 160;
    const int nphi_pl = 90;
    const Real eps = 1e-8;
    const Real R0_pl = data.hermite_locator.R0 + eps;
    const Real Z0_pl = data.hermite_locator.Z0 + eps;
    const Real R1_pl = data.hermite_locator.R1 - eps;
    const Real Z1_pl = data.hermite_locator.Z1 - eps;
    const Real dR_pl = (R1_pl - R0_pl) / static_cast<Real>(nR_pl - 1);
    const Real dZ_pl = (Z1_pl - Z0_pl) / static_cast<Real>(nZ_pl - 1);
    const Real dphi_pl = 2.0 * M_PI / static_cast<Real>(nphi_pl);

    Kokkos::DualView<Real****, Kokkos::LayoutRight, exec_space> fine(
        "fine_B", nR_pl, nZ_pl, nphi_pl, 3);
    auto fine_d = fine.view_device();
    Kokkos::parallel_for(
        "eval_fine",
        policy3D({0, 0, 0}, {nR_pl, nZ_pl, nphi_pl}),
        KOKKOS_LAMBDA(const int i, const int j, const int iphi) {
          const Real R = R0_pl + dR_pl * static_cast<Real>(i);
          const Real Z = Z0_pl + dZ_pl * static_cast<Real>(j);
          const Real phi = dphi_pl * iphi;
          Dim3 RB = {};
          ev.evalField(RB, R, Z, phi, data.hermite_data.view_device());
          fine_d(i, j, iphi, 0) = RB[0] / R;
          fine_d(i, j, iphi, 1) = RB[1] / R;
          fine_d(i, j, iphi, 2) = RB[2] / R;
        });
    fine.modify_device();
    fine.sync_host();
    auto fine_h = fine.view_host();

    const std::string fine_path = fineMeshPath(out_path);
    std::FILE* fout = std::fopen(fine_path.c_str(), "w");
    if (fout == nullptr) {
      std::fprintf(stderr, "error: cannot open %s for writing\n",
                   fine_path.c_str());
      status = 5;
    } else {
      std::fprintf(fout, "# R Z phi B_R B_phi B_Z\n");
      for (int i = 0; i < nR_pl; ++i) {
        const Real R = R0_pl + dR_pl * static_cast<Real>(i);
        for (int j = 0; j < nZ_pl; ++j) {
          const Real Z = Z0_pl + dZ_pl * static_cast<Real>(j);
          for (int iphi = 0; iphi < nphi_pl; ++iphi) {
            const Real phi = (static_cast<Real>(iphi) + 0.37) * dphi_pl;
            std::fprintf(fout, "%.17g %.17g %.17g %.17g %.17g %.17g\n", R, Z,
                         phi, fine_h(i, j, iphi, 0), fine_h(i, j, iphi, 1),
                         fine_h(i, j, iphi, 2));
          }
        }
      }
      std::fclose(fout);
    }
  }
  Kokkos::finalize();
  return status;
}
